#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void max_prefix_sum(__global int *numbers,
                             __global int *max_prefixes,
                             __global int *indexes,
                             unsigned int size,
                             __local  int *numbers_local,
                             __local  int *max_prefixes_local,
                             __local  int *indexes_local) {

    size_t global_id = get_global_id(0);
    size_t local_id = get_local_id(0);
    size_t wg_size = get_local_size(0);
    size_t local_data_size = size < wg_size ? size : wg_size;

    if (global_id < size) {
        numbers_local[local_id] = numbers[global_id];
        max_prefixes_local[local_id] = max_prefixes[global_id];
        indexes_local[local_id] = indexes[global_id];

    } else {
        numbers_local[local_id] = 0;
        max_prefixes_local[local_id] = 0;
        indexes_local[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = 1; step < wg_size; step *= 2) {
        unsigned int start_idx = 2 * local_id * step;
        unsigned int end_idx = (2 * local_id + 1) * step;
        if (end_idx < local_data_size) {

            int next_prefix_sum = max_prefixes_local[end_idx] + numbers_local[start_idx];

            if (next_prefix_sum >= max_prefixes_local[start_idx]) {
                max_prefixes_local[start_idx] = next_prefix_sum;
                indexes_local[start_idx] = indexes_local[end_idx];
            }
            numbers_local[start_idx] += numbers_local[end_idx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        numbers[global_id] = numbers_local[0];
        max_prefixes[global_id] = max_prefixes_local[0];
        indexes[global_id] = indexes_local[0];
    }
}


__kernel void shift_to_start(__global int *numbers,
                             __global int *max_prefixes,
                             __global int *indexes,
                             unsigned int size) {

    size_t global_id = get_global_id(0);
    size_t wg_size = get_local_size(0);

    size_t shift_from = global_id * wg_size;
    if (shift_from < size) {
        numbers[global_id] = numbers[shift_from];
        max_prefixes[global_id] = max_prefixes[shift_from];
        indexes[global_id] = indexes[shift_from];
    }
}
