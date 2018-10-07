#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global const unsigned int *data,
                  __global unsigned int *res,
                  __local unsigned int *local_xs,
                  unsigned int size) {
    size_t global_id = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t wg_size = get_local_size(0);

    if (global_id < size) {
        local_xs[local_id] = data[global_id];
    } else {
        local_xs[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = 1; step < wg_size; step *= 2) {
        unsigned int start_idx = 2 * local_id * step;
        unsigned int end_idx = (2 * local_id + 1) * step;
        if (end_idx < wg_size) {
            local_xs[start_idx] = local_xs[start_idx] + local_xs[end_idx];
        }
        if (WARP_SIZE <= wg_size / (2 * step)) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (local_id == 0) {
        atomic_add(res, local_xs[0]);
    }
}