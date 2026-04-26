//! Wall-clock benchmark of the per-polygon point-in-polygon predicate,
//! comparing geo's `IntervalTreeMultiPolygon` against the linear-SIMD
//! crossing-number implementation.
//!
//! Run with `cargo bench --bench predicate`.

use std::f64::consts::TAU;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use geo_types::{Coord, LineString, MultiPolygon, Point, Polygon};
use polygon::{HybridPredicate, IntervalTreePredicate, Predicate, SimdMultiPolygon};

fn regular_ngon(n: usize) -> MultiPolygon<f64> {
    // n distinct points; geo's `Polygon::new` appends an exact closing copy,
    // so the polygon has exactly n edges. (Generating 0..=n + relying on
    // cos(TAU) == 1.0 doesn't work — f64 rounding leaves first != last.)
    let coords: Vec<Coord<f64>> = (0..n)
        .map(|i| {
            let t = TAU * (i as f64) / (n as f64);
            Coord { x: t.cos(), y: t.sin() }
        })
        .collect();
    MultiPolygon(vec![Polygon::new(LineString::new(coords), vec![])])
}

fn bench_contains(c: &mut Criterion) {
    // (0,0) is dead-center inside; (0.99, 0.99) is inside the [-1,1]^2 bbox
    // but outside the inscribed unit polygon, so it forces the full predicate
    // (not a bbox-rejection).
    let inside = Point::new(0.0, 0.0);
    let outside_in_bbox = Point::new(0.99, 0.99);

    for &n in &[16usize, 32, 64, 128, 256, 4096] {
        let mp = regular_ngon(n);
        let itree = IntervalTreePredicate::build(&mp);
        let simd = SimdMultiPolygon::build(&mp);
        let hybrid = HybridPredicate::build(&mp);

        let mut g = c.benchmark_group(format!("contains/n{n}"));
        g.bench_function(BenchmarkId::new("interval_tree", "inside"), |b| {
            b.iter(|| black_box(itree.contains(black_box(inside))))
        });
        g.bench_function(BenchmarkId::new("simd", "inside"), |b| {
            b.iter(|| black_box(simd.contains(black_box(inside))))
        });
        g.bench_function(BenchmarkId::new("hybrid", "inside"), |b| {
            b.iter(|| black_box(hybrid.contains(black_box(inside))))
        });
        g.bench_function(BenchmarkId::new("interval_tree", "outside_in_bbox"), |b| {
            b.iter(|| black_box(itree.contains(black_box(outside_in_bbox))))
        });
        g.bench_function(BenchmarkId::new("simd", "outside_in_bbox"), |b| {
            b.iter(|| black_box(simd.contains(black_box(outside_in_bbox))))
        });
        g.bench_function(BenchmarkId::new("hybrid", "outside_in_bbox"), |b| {
            b.iter(|| black_box(hybrid.contains(black_box(outside_in_bbox))))
        });
        g.finish();
    }
}

criterion_group!(benches, bench_contains);
criterion_main!(benches);
