//! Instruction-count benchmark of the per-polygon point-in-polygon predicate.
//! Linux + Valgrind only. Install the runner once:
//!
//!     cargo install --version 0.18.1 gungraun-runner
//!
//! Run with `cargo bench --bench predicate_gungraun`.

use std::f64::consts::TAU;
use std::hint::black_box;

use geo_types::{Coord, LineString, MultiPolygon, Point, Polygon};
use gungraun::{Callgrind, LibraryBenchmarkConfig, library_benchmark, library_benchmark_group, main};
use polygon::{IntervalTreePredicate, Predicate, SimdMultiPolygon};

fn regular_ngon(n: usize) -> MultiPolygon<f64> {
    let coords: Vec<Coord<f64>> = (0..=n)
        .map(|i| {
            let t = TAU * (i as f64) / (n as f64);
            Coord { x: t.cos(), y: t.sin() }
        })
        .collect();
    MultiPolygon(vec![Polygon::new(LineString::new(coords), vec![])])
}

fn itree_inside(n: usize) -> (IntervalTreePredicate, Point<f64>) {
    (IntervalTreePredicate::build(&regular_ngon(n)), Point::new(0.0, 0.0))
}
fn itree_outside(n: usize) -> (IntervalTreePredicate, Point<f64>) {
    (IntervalTreePredicate::build(&regular_ngon(n)), Point::new(0.99, 0.99))
}
fn simd_inside(n: usize) -> (SimdMultiPolygon, Point<f64>) {
    (SimdMultiPolygon::build(&regular_ngon(n)), Point::new(0.0, 0.0))
}
fn simd_outside(n: usize) -> (SimdMultiPolygon, Point<f64>) {
    (SimdMultiPolygon::build(&regular_ngon(n)), Point::new(0.99, 0.99))
}

#[library_benchmark]
#[bench::n16(itree_inside(16))]
#[bench::n256(itree_inside(256))]
#[bench::n4096(itree_inside(4096))]
fn interval_tree_inside(input: (IntervalTreePredicate, Point<f64>)) -> bool {
    let (p, point) = input;
    black_box(p.contains(point))
}

#[library_benchmark]
#[bench::n16(itree_outside(16))]
#[bench::n256(itree_outside(256))]
#[bench::n4096(itree_outside(4096))]
fn interval_tree_outside_in_bbox(input: (IntervalTreePredicate, Point<f64>)) -> bool {
    let (p, point) = input;
    black_box(p.contains(point))
}

#[library_benchmark]
#[bench::n16(simd_inside(16))]
#[bench::n256(simd_inside(256))]
#[bench::n4096(simd_inside(4096))]
fn simd_inside_bench(input: (SimdMultiPolygon, Point<f64>)) -> bool {
    let (p, point) = input;
    black_box(p.contains(point))
}

#[library_benchmark]
#[bench::n16(simd_outside(16))]
#[bench::n256(simd_outside(256))]
#[bench::n4096(simd_outside(4096))]
fn simd_outside_in_bbox(input: (SimdMultiPolygon, Point<f64>)) -> bool {
    let (p, point) = input;
    black_box(p.contains(point))
}

library_benchmark_group!(
    name = predicate;
    benchmarks =
        interval_tree_inside,
        interval_tree_outside_in_bbox,
        simd_inside_bench,
        simd_outside_in_bbox
);

main!(
    config = LibraryBenchmarkConfig::default().tool(Callgrind::default());
    library_benchmark_groups = predicate
);
