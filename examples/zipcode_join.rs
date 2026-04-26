//! Parquet driver around `polygon::DefaultIndex`. Loads ~33k zipcode
//! polygons from a GeoParquet file, builds the index, then streams ~141M
//! Microsoft building centroids from a plain Parquet file (one row group per
//! rayon worker) and aggregates hits into a `Vec<AtomicU64>`.
//!
//! Assumes `data/us-zip-codes.parquet` and
//! `data/microsoft-buildings_point.parquet` already exist (see README and
//! `scripts/fetch_data.sh`).
//!
//! Run with `cargo run --release --example zipcode_join`. End-to-end ~4.3s.

use std::fs::File;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use anyhow::Result;
use arrow_array::{Array, BinaryArray, RecordBatch, StringArray};
use geo_traits::to_geo::ToGeoMultiPolygon;
use geo_types::{MultiPolygon, Point};
use geoarrow::array::{AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor, from_arrow_array};
use geoparquet::reader::{GeoParquetReaderBuilder, GeoParquetRecordBatchReader};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use polygon::DefaultIndex;
use rayon::prelude::*;

const ZIPCODES: &str = "data/us-zip-codes.parquet";
const BUILDINGS: &str = "data/microsoft-buildings_point.parquet";

fn main() -> Result<()> {
    let t0 = Instant::now();

    let (codes, polys) = load_zipcodes()?;
    let index = Arc::new(DefaultIndex::new(&polys));

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

fn join_from_parquet(index: &DefaultIndex, counts: &[AtomicU64]) -> Result<u64> {
    let probe = ParquetRecordBatchReaderBuilder::try_new(File::open(BUILDINGS)?)?;
    let n_groups = probe.metadata().num_row_groups();
    let total = probe.metadata().file_metadata().num_rows() as u64;
    let geom_idx = probe.schema().index_of("geometry")?;
    drop(probe);

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
                    index.for_each_containing(Point::new(x, y), |i| {
                        counts[i].fetch_add(1, Ordering::Relaxed);
                    });
                }
            }
            Ok(())
        })?;
    Ok(total)
}

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
