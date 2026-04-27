# polygon

Joins ~141M Microsoft building centroids against ~33k US ZIP code polygons and
prints the top 5 ZIPs by building count. The interesting piece is
`MultiPolygonIndex` in `src/lib.rs` — an `rstar` R-tree of bounding boxes
layered over one `IntervalTreeMultiPolygon` per polygon. Builds once, queries
in parallel.

## Run

```
cargo run --release --example zipcode_join
```

Expects two files under `data/`:

- `data/us-zip-codes.parquet` — GeoParquet, one row per ZIP, columns `zipcode`
  (string) + a GeoArrow `MultiPolygon` geometry column.
- `data/microsoft-buildings_point.parquet` — plain Parquet (not GeoParquet),
  one row per building, column `geometry` of WKB `Point` bytes.

Neither is a canonical published file; you build them once from the sources
below. `scripts/fetch_data.sh` automates the ZCTA step and runs the DuckDB
recipe for the buildings step (you provide the source via `$BUILDINGS_SRC`).

## Building `data/us-zip-codes.parquet`

Source: US Census Bureau **ZCTA5** TIGER/Line shapefile. Download the latest
ZCTA5 nation-wide shapefile from the Census and convert with GDAL ≥ 3.5:

```
ogr2ogr -f Parquet data/us-zip-codes.parquet \
  -sql "SELECT ZCTA5CE20 AS zipcode FROM tl_2020_us_zcta520" \
  tl_2020_us_zcta520.shp
```

The `SELECT ... AS zipcode` rename matters — `load_zipcodes` looks the column
up by that exact name. If your shapefile is the 2010 vintage, use
`ZCTA5CE10` instead.

Census ZCTA page: https://www.census.gov/programs-surveys/geography/guidance/geo-areas/zctas.html

## Building `data/microsoft-buildings_point.parquet`

Source: Microsoft's open US building footprints. The HDX repackaging on
Source Cooperative is the easiest cloud-native form:
https://source.coop/hdx/microsoft-open-buildings — filter to USA partitions.
The original per-state GeoJSON release also works:
https://github.com/microsoft/USBuildingFootprints.

The driver wants centroids as WKB Points in a plain Parquet column called
`geometry`. DuckDB streams the USA partition straight from Source Cooperative
and writes the centroids locally:

```sql
INSTALL spatial; LOAD spatial;
INSTALL httpfs;  LOAD httpfs;
CREATE SECRET (
  TYPE S3, PROVIDER CONFIG,
  ENDPOINT 'data.source.coop', URL_STYLE 'path',
  REGION 'us-west-2', USE_SSL true
);
COPY (
  SELECT ST_AsWKB(ST_Centroid(geometry)) AS geometry
  FROM read_parquet('s3://hdx/microsoft-open-buildings/parquet/RegionName=UnitedStates/**/*.parquet')
) TO 'data/microsoft-buildings_point.parquet' (FORMAT PARQUET);
```

The WKB decoder (`decode_wkb_point`) only reads bytes 5..21 and assumes a
Point geometry — make sure the centroid step actually produces points, not
multipoints.

## Expected output

```
141611717 points joined in ~3.4s (~41M pts/s)
77494    42147
73099    41611
78641    40843
78660    39730
60629    38910
```

on a recent multi-core box with both files on local SSD (M2 Pro reference).

## Predicate variants

The predicate is generic over a `Predicate` trait (see `src/lib.rs`).
Three impls and feature-flagged defaults:

- **IntervalTree** (`IntervalTreePredicate`, default) — `geo`'s
  `IntervalTreeMultiPolygon`. Y-interval tree of edges + winding-number test.
  Sub-linear in edge count.
- **SIMD** (`SimdMultiPolygon`, behind `--features simd-predicate`) —
  NEON crossing-number sweep across all edges. Scalar fallback off-aarch64.
- **Hybrid** (`HybridPredicate`, behind `--features hybrid-predicate`) —
  Picks SIMD or IntervalTree per polygon at build time, threshold
  `HYBRID_EDGE_THRESHOLD = 64`.

Per-call wall-clock from `cargo bench --bench predicate` on Apple Silicon:

| edges | IntervalTree | SIMD | Hybrid |
|------:|-------------:|-----:|-------:|
| 16    | 25.5 ns | 6.0 ns | 6.3 ns |
| 64    | 31.0 ns | 18.1 ns | 18.3 ns |
| 128   | 39.7 ns | 36.2 ns | 39.7 ns |
| 4096  | 60.8 ns | 1048 ns | 61.0 ns |

End-to-end on the full 141M-point dataset, the predicate variants are a
wash because dense urban ZIPs (where most points fall) all have hundreds
of edges and run via the IntervalTree path either way. SIMD-only
*regresses* 2.9× because rural Western ZIPs with 5k–15k edges get hit
linearly. Hybrid matches IntervalTree to within noise.

A Gungraun (Valgrind callgrind) bench for instruction-count comparisons
lives in `benches/predicate_gungraun.rs`. Linux + valgrind only:

```
cargo install --version 0.18.1 gungraun-runner
cargo bench --bench predicate_gungraun
```

## Metal GPU experiment (dead end, kept for documentation)

`src/gpu.rs` + `src/pip.metal` + `examples/zipcode_join_gpu.rs` build a
Metal compute kernel for the predicate, with a CPU-side rstar AABB filter
producing per-(point, polygon) tasks. Run with
`cargo run --release --example zipcode_join_gpu`.

End-to-end **6.78s** vs **3.42s** for the CPU IntervalTree path, with
phase breakdown:

| phase | time |
|-------|-----:|
| rstar filter (CPU) | 2.00s — 267M tasks |
| pack f32 points | 0.05s |
| alloc Metal buffers | 0.88s |
| GPU dispatch + wait | 1.93s — 138M tasks/s |
| readback (shared mem) | 0.00s |

Even with zero GPU compute the CPU side already runs over 4.87s, exceeding
the IntervalTree baseline. Three changes would make GPU competitive but
each is significant work:
1. Sort tasks by polygon size before dispatch (kills warp divergence).
2. No-copy Metal buffers (page-aligned `newBufferWithBytesNoCopy`).
3. Move the AABB filter onto the GPU (uniform grid hash) — the big one.

Without (3), the CPU rstar filter alone exceeds the entire CPU
IntervalTree wall time. The unstated assumption that "GPU = faster" fails
because the IntervalTree's edge filter cuts per-call work to ~5 candidate
edges, while a GPU brute-force linear scan does the full polygon every
time. Compute parity, not speed-up.
