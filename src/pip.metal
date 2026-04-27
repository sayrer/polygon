#include <metal_stdlib>
using namespace metal;

// One thread per (point, candidate-polygon) task. Each thread walks the
// polygon's edges and accumulates the crossing-number parity.
kernel void pip_predicate(
    device const float2*  points       [[buffer(0)]],
    device const float4*  edges        [[buffer(1)]],
    device const uint2*   poly_ranges  [[buffer(2)]],  // (edge_offset, edge_count)
    device const uint2*   tasks        [[buffer(3)]],  // (point_idx, poly_idx)
    device atomic_uint*   counts       [[buffer(4)]],
    constant uint&        task_count   [[buffer(5)]],
    uint tid                           [[thread_position_in_grid]]
) {
    if (tid >= task_count) return;

    uint2  task  = tasks[tid];
    float2 p     = points[task.x];
    uint2  range = poly_ranges[task.y];

    uint crossings = 0u;
    for (uint i = 0u; i < range.y; i++) {
        float4 e  = edges[range.x + i];
        float2 a  = e.xy;
        float2 b  = e.zw;

        // Crossing-number test, no division: cross-product sign vs. dy sign.
        bool   straddle = (a.y > p.y) != (b.y > p.y);
        float  dy       = b.y - a.y;
        float  cross    = (p.y - a.y) * (b.x - a.x) - (p.x - a.x) * dy;
        // Top bit of (cross XOR dy) is 0 iff signs match.
        bool   right    = (as_type<uint>(cross) ^ as_type<uint>(dy)) < 0x80000000u;
        crossings += (straddle && right) ? 1u : 0u;
    }
    if ((crossings & 1u) == 1u) {
        atomic_fetch_add_explicit(&counts[task.y], 1u, memory_order_relaxed);
    }
}
