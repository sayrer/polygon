//! SIMD-friendly point-in-multipolygon predicate.
//!
//! All edges across all rings (exterior + interior) of all polygons are
//! flattened into four SoA `f32` buffers, padded to a multiple of 4 with
//! degenerate edges (`y1 == y2 == 0`) so the inner loop has no tail.
//!
//! Crossing-number parity gives correct in/out for arbitrary multipolygons
//! with holes — ring identity is irrelevant. The cross-product sign trick
//! avoids a per-lane division: that division would lower to `vdivq_f32` on
//! aarch64, which has poor throughput on Firestorm/Avalanche cores and would
//! become the bottleneck.

use geo_types::{LineString, MultiPolygon, Point};

pub struct SimdMultiPolygon {
    x1: Vec<f32>,
    y1: Vec<f32>,
    x2: Vec<f32>,
    y2: Vec<f32>,
}

impl SimdMultiPolygon {
    pub fn new(mp: &MultiPolygon<f64>) -> Self {
        let mut x1 = Vec::new();
        let mut y1 = Vec::new();
        let mut x2 = Vec::new();
        let mut y2 = Vec::new();
        let mut push_ring = |ring: &LineString<f64>| {
            let pts = &ring.0;
            if pts.len() < 2 {
                return;
            }
            for w in pts.windows(2) {
                x1.push(w[0].x as f32);
                y1.push(w[0].y as f32);
                x2.push(w[1].x as f32);
                y2.push(w[1].y as f32);
            }
        };
        for poly in mp {
            push_ring(poly.exterior());
            for hole in poly.interiors() {
                push_ring(hole);
            }
        }
        while x1.len() % 4 != 0 {
            x1.push(0.0);
            y1.push(0.0);
            x2.push(0.0);
            y2.push(0.0);
        }
        Self { x1, y1, x2, y2 }
    }

    #[inline]
    pub fn contains(&self, p: Point<f64>) -> bool {
        let px = p.x() as f32;
        let py = p.y() as f32;
        #[cfg(target_arch = "aarch64")]
        // SAFETY: NEON is baseline on aarch64; lengths are equal multiples of 4.
        let crossings = unsafe { count_crossings_neon(&self.x1, &self.y1, &self.x2, &self.y2, px, py) };
        #[cfg(not(target_arch = "aarch64"))]
        let crossings = count_crossings_scalar(&self.x1, &self.y1, &self.x2, &self.y2, px, py);
        crossings & 1 == 1
    }
}

#[cfg(target_arch = "aarch64")]
#[inline]
#[allow(unsafe_op_in_unsafe_fn)]
unsafe fn count_crossings_neon(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    px: f32,
    py: f32,
) -> u32 {
    use std::arch::aarch64::*;
    debug_assert_eq!(x1.len() % 4, 0);
    debug_assert!(x1.len() == y1.len() && x1.len() == x2.len() && x1.len() == y2.len());

    let vpx = vdupq_n_f32(px);
    let vpy = vdupq_n_f32(py);
    let mut acc = vdupq_n_u32(0);

    let mut i = 0;
    while i < x1.len() {
        let vx1 = vld1q_f32(x1.as_ptr().add(i));
        let vy1 = vld1q_f32(y1.as_ptr().add(i));
        let vx2 = vld1q_f32(x2.as_ptr().add(i));
        let vy2 = vld1q_f32(y2.as_ptr().add(i));

        // Straddle: (y1 > py) ^ (y2 > py).  Each lane is 0 or 0xFFFFFFFF.
        let m1 = vcgtq_f32(vy1, vpy);
        let m2 = vcgtq_f32(vy2, vpy);
        let straddle = veorq_u32(m1, m2);

        // cross = (py - y1) * (x2 - x1) - (px - x1) * (y2 - y1)
        // Crossing iff straddle && sign(cross) == sign(dy).
        let dx = vsubq_f32(vx2, vx1);
        let dy = vsubq_f32(vy2, vy1);
        let pmy1 = vsubq_f32(vpy, vy1);
        let pmx1 = vsubq_f32(vpx, vx1);
        let cross = vfmsq_f32(vmulq_f32(pmy1, dx), pmx1, dy);

        // Same-sign test: top bit of (cross XOR dy) is 0 ⇔ signs match.
        let xored = veorq_u32(vreinterpretq_u32_f32(cross), vreinterpretq_u32_f32(dy));
        let diff_mask = vreinterpretq_u32_s32(vshrq_n_s32(vreinterpretq_s32_u32(xored), 31));
        let same_sign = vmvnq_u32(diff_mask);

        let crossing = vandq_u32(straddle, same_sign);
        acc = vaddq_u32(acc, vshrq_n_u32(crossing, 31));
        i += 4;
    }
    vaddvq_u32(acc)
}

#[cfg(not(target_arch = "aarch64"))]
fn count_crossings_scalar(
    x1: &[f32],
    y1: &[f32],
    x2: &[f32],
    y2: &[f32],
    px: f32,
    py: f32,
) -> u32 {
    let mut c = 0u32;
    for i in 0..x1.len() {
        let straddle = (y1[i] > py) != (y2[i] > py);
        let cross = (py - y1[i]) * (x2[i] - x1[i]) - (px - x1[i]) * (y2[i] - y1[i]);
        if straddle && cross.signum() == (y2[i] - y1[i]).signum() {
            c += 1;
        }
    }
    c
}
