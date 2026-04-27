//! Spatial index for repeated point-in-polygon queries over a fixed set of
//! `MultiPolygon`s. An `rstar` R-tree of bounding rectangles narrows
//! candidates; each polygon's per-polygon predicate runs the actual test.
//!
//! The predicate is generic via [`Predicate`]. Two impls are provided:
//! - [`IntervalTreePredicate`] (default, geo's `IntervalTreeMultiPolygon`)
//! - [`SimdMultiPolygon`] (linear NEON crossing-number, aarch64; scalar
//!   fallback elsewhere)
//!
//! [`DefaultIndex`] is whichever of the two is configured. Build once,
//! query many times from any number of threads: both predicates and
//! `MultiPolygonIndex<P>` are `Send + Sync`.

use std::ops::ControlFlow;
use std::sync::atomic::{AtomicU64, Ordering};

use geo::algorithm::Contains;
use geo::algorithm::indexed::IntervalTreeMultiPolygon;
use geo_types::{MultiPolygon, Point, Rect};
use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};

mod simd;
pub use simd::SimdMultiPolygon;

#[cfg(target_os = "macos")]
pub mod gpu;

pub trait Predicate: Send + Sync {
    fn build(mp: &MultiPolygon<f64>) -> Self;
    fn contains(&self, p: Point<f64>) -> bool;
}

/// Newtype around `geo`'s `IntervalTreeMultiPolygon` so we can give it our
/// own `Predicate` impl without colliding with `geo::Contains`.
pub struct IntervalTreePredicate(IntervalTreeMultiPolygon<f64>);

impl Predicate for IntervalTreePredicate {
    fn build(mp: &MultiPolygon<f64>) -> Self {
        Self(IntervalTreeMultiPolygon::new(mp))
    }
    fn contains(&self, p: Point<f64>) -> bool {
        Contains::contains(&self.0, &p)
    }
}

impl Predicate for SimdMultiPolygon {
    fn build(mp: &MultiPolygon<f64>) -> Self {
        SimdMultiPolygon::new(mp)
    }
    fn contains(&self, p: Point<f64>) -> bool {
        SimdMultiPolygon::contains(self, p)
    }
}

/// Edges-per-multipolygon at or below which the hybrid uses SIMD; above
/// which it uses the interval tree. Picked from the criterion bench:
/// SIMD wins 4x at n=16, breaks even ~n=120, loses 17x at n=4096. 64 is
/// inside the SIMD-wins zone with margin.
pub const HYBRID_EDGE_THRESHOLD: usize = 64;

fn count_edges(mp: &MultiPolygon<f64>) -> usize {
    mp.iter()
        .map(|poly| {
            std::iter::once(poly.exterior())
                .chain(poly.interiors().iter())
                .map(|ring| ring.0.len().saturating_sub(1))
                .sum::<usize>()
        })
        .sum()
}

/// Per-polygon dispatch: SIMD for small polygons, IntervalTree for large.
/// Match-based, no vtable.
pub enum HybridPredicate {
    Simd(SimdMultiPolygon),
    IntervalTree(IntervalTreePredicate),
}

impl Predicate for HybridPredicate {
    fn build(mp: &MultiPolygon<f64>) -> Self {
        if count_edges(mp) <= HYBRID_EDGE_THRESHOLD {
            Self::Simd(SimdMultiPolygon::build(mp))
        } else {
            Self::IntervalTree(IntervalTreePredicate::build(mp))
        }
    }
    fn contains(&self, p: Point<f64>) -> bool {
        match self {
            Self::Simd(s) => s.contains(p),
            Self::IntervalTree(t) => t.contains(p),
        }
    }
}

pub struct MultiPolygonIndex<P: Predicate = IntervalTreePredicate> {
    rtree: RTree<GeomWithData<Rectangle<(f64, f64)>, usize>>,
    trees: Vec<P>,
}

impl<P: Predicate> MultiPolygonIndex<P> {
    pub fn new(polys: &[MultiPolygon<f64>]) -> Self {
        use geo::algorithm::BoundingRect;
        let aabbs: Vec<_> = polys
            .iter()
            .enumerate()
            .filter_map(|(i, mp)| {
                let r: Rect<f64> = mp.bounding_rect()?;
                let rect = Rectangle::from_corners((r.min().x, r.min().y), (r.max().x, r.max().y));
                Some(GeomWithData::new(rect, i))
            })
            .collect();
        let trees = polys.par_iter().map(P::build).collect();
        Self {
            rtree: RTree::bulk_load(aabbs),
            trees,
        }
    }

    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    pub fn for_each_containing<F: FnMut(usize)>(&self, point: Point<f64>, mut f: F) {
        let envelope = AABB::from_point((point.x(), point.y()));
        // Internal-iteration variant of `locate_in_envelope_intersecting`
        // walks the R-tree on the call stack instead of pushing children onto
        // a SmallVec — avoids the per-query heap allocation that the iterator
        // form pays once the stack overflows its 24-slot inline buffer.
        let _ = self.rtree.locate_in_envelope_intersecting_int(
            &envelope,
            |hit| -> ControlFlow<()> {
                if self.trees[hit.data].contains(point) {
                    f(hit.data);
                }
                ControlFlow::Continue(())
            },
        );
    }

    pub fn count_points_par(&self, points: &[Point<f64>]) -> Vec<u64> {
        let counts: Vec<AtomicU64> = (0..self.len()).map(|_| AtomicU64::new(0)).collect();
        points.par_iter().for_each(|p| {
            self.for_each_containing(*p, |i| {
                counts[i].fetch_add(1, Ordering::Relaxed);
            });
        });
        counts.into_iter().map(AtomicU64::into_inner).collect()
    }
}

/// Whichever predicate is configured. The example uses this so the
/// `simd-predicate` and `hybrid-predicate` feature flags flip the whole
/// pipeline. `hybrid-predicate` takes precedence if both are enabled.
#[cfg(feature = "hybrid-predicate")]
pub type DefaultIndex = MultiPolygonIndex<HybridPredicate>;
#[cfg(all(feature = "simd-predicate", not(feature = "hybrid-predicate")))]
pub type DefaultIndex = MultiPolygonIndex<SimdMultiPolygon>;
#[cfg(all(not(feature = "simd-predicate"), not(feature = "hybrid-predicate")))]
pub type DefaultIndex = MultiPolygonIndex<IntervalTreePredicate>;

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Polygon};

    fn unit_square(min_x: f64, min_y: f64) -> MultiPolygon<f64> {
        let exterior = LineString::new(vec![
            Coord { x: min_x, y: min_y },
            Coord { x: min_x + 1.0, y: min_y },
            Coord { x: min_x + 1.0, y: min_y + 1.0 },
            Coord { x: min_x, y: min_y + 1.0 },
            Coord { x: min_x, y: min_y },
        ]);
        MultiPolygon(vec![Polygon::new(exterior, vec![])])
    }

    fn for_each_containing_visits_each_polygon_at_most_once<P: Predicate>() {
        let polys = vec![unit_square(0.0, 0.0), unit_square(1.0, 0.0)];
        let index = MultiPolygonIndex::<P>::new(&polys);

        let mut hits = Vec::new();
        index.for_each_containing(Point::new(0.5, 0.5), |i| hits.push(i));
        assert_eq!(hits, vec![0]);

        let mut hits = Vec::new();
        index.for_each_containing(Point::new(1.5, 0.5), |i| hits.push(i));
        assert_eq!(hits, vec![1]);
    }

    fn count_points_par_returns_one_count_per_polygon<P: Predicate>() {
        let polys = vec![
            unit_square(0.0, 0.0),
            unit_square(1.0, 0.0),
            unit_square(0.0, 1.0),
            unit_square(1.0, 1.0),
        ];
        let index = MultiPolygonIndex::<P>::new(&polys);
        let points = vec![
            Point::new(0.5, 0.5),
            Point::new(0.1, 0.1),
            Point::new(1.5, 0.5),
            Point::new(0.5, 1.5),
            Point::new(1.5, 1.5),
            Point::new(1.5, 1.5),
            Point::new(5.0, 5.0),
        ];

        let counts = index.count_points_par(&points);
        assert_eq!(counts, vec![2, 1, 1, 2]);
    }

    #[test]
    fn interval_tree_for_each_containing() {
        for_each_containing_visits_each_polygon_at_most_once::<IntervalTreePredicate>();
    }

    #[test]
    fn interval_tree_count_points_par() {
        count_points_par_returns_one_count_per_polygon::<IntervalTreePredicate>();
    }

    #[test]
    fn simd_for_each_containing() {
        for_each_containing_visits_each_polygon_at_most_once::<SimdMultiPolygon>();
    }

    #[test]
    fn simd_count_points_par() {
        count_points_par_returns_one_count_per_polygon::<SimdMultiPolygon>();
    }

    #[test]
    fn hybrid_for_each_containing() {
        for_each_containing_visits_each_polygon_at_most_once::<HybridPredicate>();
    }

    #[test]
    fn hybrid_count_points_par() {
        count_points_par_returns_one_count_per_polygon::<HybridPredicate>();
    }
}
