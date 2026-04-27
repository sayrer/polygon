//! Metal-based batched point-in-polygon. Apple-silicon only.
//!
//! Architecture: CPU runs the rstar AABB filter to produce a flat list of
//! `(point_idx, poly_idx)` candidate tasks; the GPU runs the crossing-number
//! predicate on each task and atomic-adds into a per-polygon u32 counter.
//! Apple's unified memory makes the buffer uploads zero-copy.

use std::mem::size_of;
use std::sync::Arc;

use geo::algorithm::BoundingRect;
use geo_types::{LineString, MultiPolygon, Point};
use metal::*;
use rayon::prelude::*;
use rstar::primitives::{GeomWithData, Rectangle};
use rstar::{AABB, RTree};

const MSL: &str = include_str!("pip.metal");

pub struct GpuPipIndex {
    device: Device,
    pipeline: ComputePipelineState,
    queue: CommandQueue,
    rtree: Arc<RTree<GeomWithData<Rectangle<(f64, f64)>, usize>>>,
    edges: Buffer,
    ranges: Buffer,
    n_polys: usize,
}

// Metal device handles are Send/Sync via objc retain semantics; rstar is.
unsafe impl Send for GpuPipIndex {}
unsafe impl Sync for GpuPipIndex {}

impl GpuPipIndex {
    pub fn new(polys: &[MultiPolygon<f64>]) -> Self {
        let device = Device::system_default().expect("no Metal device available");
        let lib = device
            .new_library_with_source(MSL, &CompileOptions::new())
            .expect("MSL compile failed");
        let func = lib.get_function("pip_predicate", None).unwrap();
        let pipeline = device
            .new_compute_pipeline_state_with_function(&func)
            .unwrap();
        let queue = device.new_command_queue();

        // Flatten edges per polygon, contiguous.
        let mut edges: Vec<[f32; 4]> = Vec::new();
        let mut ranges: Vec<[u32; 2]> = Vec::with_capacity(polys.len());
        for mp in polys {
            let start = edges.len() as u32;
            for poly in mp {
                push_ring(&mut edges, poly.exterior());
                for hole in poly.interiors() {
                    push_ring(&mut edges, hole);
                }
            }
            let len = edges.len() as u32 - start;
            ranges.push([start, len]);
        }

        // CPU-side rstar for the AABB candidate filter.
        let aabbs: Vec<_> = polys
            .iter()
            .enumerate()
            .filter_map(|(i, mp)| {
                let r = mp.bounding_rect()?;
                let rect = Rectangle::from_corners((r.min().x, r.min().y), (r.max().x, r.max().y));
                Some(GeomWithData::new(rect, i))
            })
            .collect();
        let rtree = Arc::new(RTree::bulk_load(aabbs));

        let edges_buf = device.new_buffer_with_data(
            edges.as_ptr().cast(),
            (edges.len() * size_of::<[f32; 4]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ranges_buf = device.new_buffer_with_data(
            ranges.as_ptr().cast(),
            (ranges.len() * size_of::<[u32; 2]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        Self {
            device,
            pipeline,
            queue,
            rtree,
            edges: edges_buf,
            ranges: ranges_buf,
            n_polys: polys.len(),
        }
    }

    pub fn len(&self) -> usize {
        self.n_polys
    }

    pub fn is_empty(&self) -> bool {
        self.n_polys == 0
    }

    /// Counts points-per-polygon. Returns a `Vec<u64>` indexed by polygon
    /// position. CPU rstar filter feeds a single big GPU dispatch. Prints
    /// per-phase timings to stderr so the cost breakdown is visible.
    pub fn count_points(&self, points: &[Point<f64>]) -> Vec<u64> {
        let t0 = std::time::Instant::now();

        // 1. CPU pre-filter (parallel): emit (point_idx, poly_idx) per candidate.
        let rtree = Arc::clone(&self.rtree);
        let tasks: Vec<[u32; 2]> = points
            .par_iter()
            .enumerate()
            .flat_map_iter(|(pi, p)| {
                let env = AABB::from_point((p.x(), p.y()));
                rtree
                    .locate_in_envelope_intersecting(&env)
                    .map(move |hit| [pi as u32, hit.data as u32])
                    .collect::<Vec<_>>()
            })
            .collect();
        eprintln!(
            "  rstar filter:       {:.2}s ({} tasks)",
            t0.elapsed().as_secs_f64(),
            tasks.len()
        );

        // 2. Pack points as f32x2.
        let t = std::time::Instant::now();
        let points_f32: Vec<[f32; 2]> = points
            .par_iter()
            .map(|p| [p.x() as f32, p.y() as f32])
            .collect();
        eprintln!("  pack f32 points:    {:.2}s", t.elapsed().as_secs_f64());

        // 3. Allocate GPU buffers.
        let t = std::time::Instant::now();
        let pts_buf = self.device.new_buffer_with_data(
            points_f32.as_ptr().cast(),
            (points_f32.len() * size_of::<[f32; 2]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tasks_buf = self.device.new_buffer_with_data(
            tasks.as_ptr().cast(),
            (tasks.len() * size_of::<[u32; 2]>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let counts_buf = self.device.new_buffer(
            (self.n_polys * size_of::<u32>()) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        // SAFETY: shared-mode buffer; we own it; size matches.
        unsafe {
            std::ptr::write_bytes(
                counts_buf.contents() as *mut u8,
                0,
                self.n_polys * size_of::<u32>(),
            );
        }
        eprintln!("  alloc buffers:      {:.2}s", t.elapsed().as_secs_f64());

        // 4. Encode and dispatch.
        let t = std::time::Instant::now();
        let cmd = self.queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&self.pipeline);
        enc.set_buffer(0, Some(&pts_buf), 0);
        enc.set_buffer(1, Some(&self.edges), 0);
        enc.set_buffer(2, Some(&self.ranges), 0);
        enc.set_buffer(3, Some(&tasks_buf), 0);
        enc.set_buffer(4, Some(&counts_buf), 0);
        let task_count = tasks.len() as u32;
        enc.set_bytes(
            5,
            size_of::<u32>() as u64,
            (&task_count as *const u32).cast(),
        );

        let max_per_tg = self.pipeline.max_total_threads_per_threadgroup();
        let tg = MTLSize::new(max_per_tg.min(64), 1, 1);
        let grid = MTLSize::new(tasks.len() as u64, 1, 1);
        enc.dispatch_threads(grid, tg);
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
        eprintln!("  GPU dispatch+wait:  {:.2}s", t.elapsed().as_secs_f64());

        // 5. Read counts back (zero-copy on shared buffer).
        let t = std::time::Instant::now();
        let counts_ptr = counts_buf.contents() as *const u32;
        // SAFETY: shared-mode buffer is alive; size is n_polys u32s.
        let counts_slice = unsafe { std::slice::from_raw_parts(counts_ptr, self.n_polys) };
        let out: Vec<u64> = counts_slice.iter().map(|&c| c as u64).collect();
        eprintln!("  readback:           {:.2}s", t.elapsed().as_secs_f64());
        out
    }
}

fn push_ring(edges: &mut Vec<[f32; 4]>, ring: &LineString<f64>) {
    let pts = &ring.0;
    if pts.len() < 2 {
        return;
    }
    for w in pts.windows(2) {
        edges.push([
            w[0].x as f32,
            w[0].y as f32,
            w[1].x as f32,
            w[1].y as f32,
        ]);
    }
}

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

    #[test]
    fn gpu_count_matches_cpu() {
        let polys = vec![
            unit_square(0.0, 0.0),
            unit_square(1.0, 0.0),
            unit_square(0.0, 1.0),
            unit_square(1.0, 1.0),
        ];
        let index = GpuPipIndex::new(&polys);
        let points = vec![
            Point::new(0.5, 0.5),
            Point::new(0.1, 0.1),
            Point::new(1.5, 0.5),
            Point::new(0.5, 1.5),
            Point::new(1.5, 1.5),
            Point::new(1.5, 1.5),
            Point::new(5.0, 5.0),
        ];
        let counts = index.count_points(&points);
        assert_eq!(counts, vec![2, 1, 1, 2]);
    }
}
