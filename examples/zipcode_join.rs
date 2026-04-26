//! The reusable piece is [`MultiPolygonIndex`]: hand it a slice
//! of `MultiPolygon<f64>` and it lets you ask which polygons contain a given
//! `Point<f64>`. It combines an `rstar::RTree` of bounding rectangles
//! (coarse candidate filter) with one `IntervalTreeMultiPolygon` per polygon
//! (the actual point-in-polygon predicate). Build once; query many times
//! from any number of threads: `MultiPolygonIndex` is `Send + Sync`.
//!
//! The rest of the file is the parquet driver: it loads 33k zipcode polygons
//! from a GeoParquet file, builds the index, then streams ~130M building
//! centroids from a plain Parquet file (one row group per rayon worker) and
//! aggregates hits into a `Vec<AtomicU64>`.
//!
//! Assumes `data/us-zip-codes.parquet` and
//! `data/microsoft-buildings_point.parquet` already exist.
//!
//! Run with `cargo run --release --example zipcode_join`.
//! To give you an idea of what is meant by "fast" in this context: the calculation runs in around 4.3 s

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use arrow_array::{Array, BinaryArray, RecordBatch, StringArray};
use geo::algorithm::indexed::IntervalTreeMultiPolygon;
use geo::algorithm::{BoundingRect, Contains};
use geo_traits::to_geo::ToGeoMultiPolygon;
use geo_types::{MultiPolygon, Point, Rect};
use geoarrow::array::{AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor, from_arrow_array};
use geoparquet::reader::{GeoParquetReaderBuilder, GeoParquetRecordBatchReader};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};

// =============================================================================
// Reusable spatial index
// =============================================================================

/// Spatial index for repeated point-in-polygon queries over many
/// `MultiPolygon`s.
///
/// Built from a slice of polygons; subsequent queries return the *position*
/// of each containing polygon in that slice. Threads can share one
/// `Arc<MultiPolygonIndex>`: both inner structures are `Send + Sync`.
pub struct MultiPolygonIndex {
    // rstar's bare `AABB` doesn't implement `RTreeObject`; `Rectangle` does.
    // `GeomWithData` pairs each rectangle with its polygon's slice index.
    rtree: RTree<GeomWithData<Rectangle<(f64, f64)>, usize>>,
    // One per polygon, in the same order. `IntervalTreeMultiPolygon` is
    // `geo`'s purpose-built point-in-multipolygon index: a Y-interval tree of
    // polygon edges plus a winding-number test on the candidates.
    trees: Vec<IntervalTreeMultiPolygon<f64>>,
}

impl MultiPolygonIndex {
    /// Build both indices over `polys`. The expensive piece is the per-polygon
    /// edge tree, so we build them in parallel.
    pub fn new(polys: &[MultiPolygon<f64>]) -> Self {
        let aabbs: Vec<_> = polys
            .iter()
            .enumerate()
            .filter_map(|(i, mp)| {
                let r: Rect<f64> = mp.bounding_rect()?;
                let rect = Rectangle::from_corners((r.min().x, r.min().y), (r.max().x, r.max().y));
                Some(GeomWithData::new(rect, i))
            })
            .collect();
        let trees = polys
            .par_iter()
            .map(IntervalTreeMultiPolygon::new)
            .collect();
        Self {
            rtree: RTree::bulk_load(aabbs),
            trees,
        }
    }

    /// Number of polygons the index was built over.
    pub fn len(&self) -> usize {
        self.trees.len()
    }

    pub fn is_empty(&self) -> bool {
        self.trees.is_empty()
    }

    /// Call `f(i)` for every polygon `i` that contains `point`.
    ///
    /// Allocation-free; this is the inner-loop primitive both bulk and
    /// streaming usage build on.
    pub fn for_each_containing<F: FnMut(usize)>(&self, point: Point<f64>, mut f: F) {
        let envelope = AABB::from_point((point.x(), point.y()));
        // RTree narrows ~33k polygons down to a handful of bbox candidates;
        // the IntervalTree then runs the precise predicate on each.
        for hit in self.rtree.locate_in_envelope_intersecting(&envelope) {
            if self.trees[hit.data].contains(&point) {
                f(hit.data);
            }
        }
    }

    /// For each polygon, count how many of `points` lie inside it. Returns a
    /// `Vec<u64>` of length [`len`](Self::len), indexed by polygon position.
    /// Parallelises across `points` via rayon.
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

// =============================================================================
// Parquet driver
// =============================================================================

// ZIPCODES is a GeoParquet file (geometry stored as GeoArrow MultiPolygon).
// BUILDINGS is plain Parquet — geometry is a `geometry` column of WKB bytes.
const ZIPCODES: &str = "data/us-zip-codes.parquet";
const BUILDINGS: &str = "data/microsoft-buildings_point.parquet";

fn main() -> Result<()> {
    let t0 = Instant::now();

    let (codes, polys) = load_zipcodes()?;
    let index = Arc::new(MultiPolygonIndex::new(&polys));

    // One counter per zipcode, indexed by position in `polys`. Dense indices
    // beat a HashMap here: no hashing, no per-thread map to merge.
    let counts: Arc<Vec<AtomicU64>> =
        Arc::new((0..index.len()).map(|_| AtomicU64::new(0)).collect());

    let total = join_from_parquet(&index, &counts)?;

    let dt = t0.elapsed().as_secs_f64();
    println!(
        "{total} points joined in {dt:.2}s ({:.1}M pts/s)",
        (total as f64 / 1.0e6) / dt
    );
    print_top5(&counts, &codes);
    Ok(())
}

fn load_zipcodes() -> Result<(Vec<String>, Vec<MultiPolygon<f64>>)> {
    let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(ZIPCODES)?)?;
    let geo_meta = builder.geoparquet_metadata().unwrap()?;
    let schema = builder.geoarrow_schema(&geo_meta, true, Default::default())?;
    let geom_col = geo_meta.primary_column.clone();
    let reader = GeoParquetRecordBatchReader::try_new(builder.build()?, schema)?;

    let mut codes = Vec::new();
    let mut polys = Vec::new();
    for batch in reader {
        let batch: RecordBatch = batch?;
        let zip_idx = batch.schema().index_of("zipcode")?;
        let geom_idx = batch.schema().index_of(&geom_col)?;
        let zips = batch
            .column(zip_idx)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let geoms = from_arrow_array(
            batch.column(geom_idx).as_ref(),
            batch.schema().field(geom_idx),
        )?;
        let mp = geoms.as_multi_polygon_opt().unwrap();
        for i in 0..batch.num_rows() {
            if zips.is_null(i) || mp.is_null(i) {
                continue;
            }
            polys.push(mp.value(i)?.to_multi_polygon());
            codes.push(zips.value(i).to_string());
        }
    }
    Ok((codes, polys))
}

fn join_from_parquet(index: &MultiPolygonIndex, counts: &[AtomicU64]) -> Result<u64> {
    // One probe to get the row-group count and confirm the column exists;
    // the actual data is read by the per-row-group workers below.
    let probe = ParquetRecordBatchReaderBuilder::try_new(File::open(BUILDINGS)?)?;
    let n_groups = probe.metadata().num_row_groups();
    let total = probe.metadata().file_metadata().num_rows() as u64;
    let geom_idx = probe.schema().index_of("geometry")?;
    drop(probe);

    // Dispatch one row group per rayon worker. Each worker opens its own
    // `File` handle (parquet readers aren't `Sync`) and reads only its
    // assigned row group via `with_row_groups`. With ~hundreds of row groups
    // and a small thread pool, this saturates all cores trivially.
    (0..n_groups)
        .into_par_iter()
        .try_for_each(|rg| -> Result<()> {
            let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(BUILDINGS)?)?
                .with_row_groups(vec![rg])
                .with_batch_size(65_536)
                .build()?;
            for batch in reader {
                let batch: RecordBatch = batch?;
                let wkb = batch
                    .column(geom_idx)
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap();
                for v in wkb.iter().flatten() {
                    let Some((x, y)) = decode_wkb_point(v) else {
                        continue;
                    };
                    // Same primitive a non-parquet caller would use; the
                    // Relaxed fetch_add is safe because counts don't
                    // synchronise other memory and rayon's join provides
                    // the final happens-before edge before we read them.
                    index.for_each_containing(Point::new(x, y), |i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    });
                }
            }
            Ok(())
        })?;
    Ok(total)
}

// WKB Point layout: byte 0 is the endianness flag (1 = little-endian),
// bytes 1..5 are the geometry-type tag (1 = Point — we don't bother to
// check), bytes 5..13 are x, bytes 13..21 are y. Calling this once per
// point is the inner loop, hence inlining and the hand-rolled byte parse.
#[inline(always)]
fn decode_wkb_point(buf: &[u8]) -> Option<(f64, f64)> {
    if buf.len() < 21 {
        return None;
    }
    let xs: [u8; 8] = buf[5..13].try_into().unwrap();
    let ys: [u8; 8] = buf[13..21].try_into().unwrap();
    Some(if buf[0] == 1 {
        (f64::from_le_bytes(xs), f64::from_le_bytes(ys))
    } else {
        (f64::from_be_bytes(xs), f64::from_be_bytes(ys))
    })
}

fn print_top5(counts: &[AtomicU64], codes: &[String]) {
    let mut by_count: Vec<(&str, u64)> = counts
        .iter()
        .enumerate()
        .map(|(i, n)| (codes[i].as_str(), n.load(Ordering::Relaxed)))
        .filter(|(_, n)| *n > 0)
        .collect();
    by_count.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (zip, n) in by_count.iter().take(5) {
        println!("{zip:<8} {n}");
    }
}

// =============================================================================
// API tests — show how MultiPolygonIndex is used without the parquet path.
// Run with `cargo test --release --example zipcode_join`.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use geo_types::{Coord, LineString, Polygon};

    fn unit_square(min_x: f64, min_y: f64) -> MultiPolygon<f64> {
        let exterior = LineString::new(vec![
            Coord { x: min_x, y: min_y },
            Coord {
                x: min_x + 1.0,
                y: min_y,
            },
            Coord {
                x: min_x + 1.0,
                y: min_y + 1.0,
            },
            Coord {
                x: min_x,
                y: min_y + 1.0,
            },
            Coord { x: min_x, y: min_y },
        ]);
        MultiPolygon(vec![Polygon::new(exterior, vec![])])
    }

    #[test]
    fn for_each_containing_visits_each_polygon_at_most_once() {
        // Two adjacent unit squares sharing the edge x=1.
        let polys = vec![unit_square(0.0, 0.0), unit_square(1.0, 0.0)];
        let index = MultiPolygonIndex::new(&polys);

        let mut hits = Vec::new();
        index.for_each_containing(Point::new(0.5, 0.5), |i| hits.push(i));
        assert_eq!(hits, vec![0]);

        let mut hits = Vec::new();
        index.for_each_containing(Point::new(1.5, 0.5), |i| hits.push(i));
        assert_eq!(hits, vec![1]);
    }

    #[test]
    fn count_points_par_returns_one_count_per_polygon() {
        let polys = vec![
            unit_square(0.0, 0.0),
            unit_square(1.0, 0.0),
            unit_square(0.0, 1.0),
            unit_square(1.0, 1.0),
        ];
        let index = MultiPolygonIndex::new(&polys);
        let points = vec![
            Point::new(0.5, 0.5), // poly 0
            Point::new(0.1, 0.1), // poly 0
            Point::new(1.5, 0.5), // poly 1
            Point::new(0.5, 1.5), // poly 2
            Point::new(1.5, 1.5), // poly 3
            Point::new(1.5, 1.5), // poly 3
            Point::new(5.0, 5.0), // outside everything
        ];

        let counts = index.count_points_par(&points);
        assert_eq!(counts, vec![2, 1, 1, 2]);
    }
}
