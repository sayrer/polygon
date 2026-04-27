//! Metal GPU variant of zipcode_join. Loads all building centroids into a
//! single `Vec<Point<f64>>`, then runs the rstar AABB filter + Metal compute
//! kernel in `polygon::gpu::GpuPipIndex` to produce per-zipcode counts.
//!
//! Apple-silicon only. Run with:
//!     cargo run --release --example zipcode_join_gpu

#![cfg(target_os = "macos")]

use std::fs::File;
use std::time::Instant;

use anyhow::Result;
use arrow_array::{Array, BinaryArray, RecordBatch, StringArray};
use geo_traits::to_geo::ToGeoMultiPolygon;
use geo_types::{MultiPolygon, Point};
use geoarrow::array::{AsGeoArrowArray, GeoArrowArray, GeoArrowArrayAccessor, from_arrow_array};
use geoparquet::reader::{GeoParquetReaderBuilder, GeoParquetRecordBatchReader};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use polygon::gpu::GpuPipIndex;
use rayon::prelude::*;

const ZIPCODES: &str = "data/us-zip-codes.parquet";
const BUILDINGS: &str = "data/microsoft-buildings_point.parquet";

fn main() -> Result<()> {
    let t_total = Instant::now();

    let t = Instant::now();
    let (codes, polys) = load_zipcodes()?;
    eprintln!("load zipcodes:      {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let index = GpuPipIndex::new(&polys);
    eprintln!("build GPU index:    {:.2}s", t.elapsed().as_secs_f64());

    let t = Instant::now();
    let points = load_all_building_points()?;
    eprintln!(
        "load {} points:    {:.2}s",
        points.len(),
        t.elapsed().as_secs_f64()
    );

    let t = Instant::now();
    let counts = index.count_points(&points);
    eprintln!("count_points (GPU): {:.2}s", t.elapsed().as_secs_f64());

    let total_dt = t_total.elapsed().as_secs_f64();
    let total_pts = points.len() as f64;
    println!(
        "{} points joined in {:.2}s ({:.1}M pts/s)",
        points.len(),
        total_dt,
        total_pts / 1.0e6 / total_dt
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

fn load_all_building_points() -> Result<Vec<Point<f64>>> {
    let probe = ParquetRecordBatchReaderBuilder::try_new(File::open(BUILDINGS)?)?;
    let n_groups = probe.metadata().num_row_groups();
    let geom_idx = probe.schema().index_of("geometry")?;
    drop(probe);

    let chunks: Vec<Vec<Point<f64>>> = (0..n_groups)
        .into_par_iter()
        .map(|rg| -> Result<Vec<Point<f64>>> {
            let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(BUILDINGS)?)?
                .with_row_groups(vec![rg])
                .with_batch_size(65_536)
                .build()?;
            let mut out = Vec::new();
            for batch in reader {
                let batch: RecordBatch = batch?;
                let wkb = batch
                    .column(geom_idx)
                    .as_any()
                    .downcast_ref::<BinaryArray>()
                    .unwrap();
                for v in wkb.iter().flatten() {
                    if let Some((x, y)) = decode_wkb_point(v) {
                        out.push(Point::new(x, y));
                    }
                }
            }
            Ok(out)
        })
        .collect::<Result<_>>()?;

    let total: usize = chunks.iter().map(|c| c.len()).sum();
    let mut all = Vec::with_capacity(total);
    for c in chunks {
        all.extend(c);
    }
    Ok(all)
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

fn print_top5(counts: &[u64], codes: &[String]) {
    let mut by_count: Vec<(&str, u64)> = counts
        .iter()
        .enumerate()
        .map(|(i, &n)| (codes[i].as_str(), n))
        .filter(|(_, n)| *n > 0)
        .collect();
    by_count.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
    for (zip, n) in by_count.iter().take(5) {
        println!("{zip:<8} {n}");
    }
}
