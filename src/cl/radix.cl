#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void radix(__global unsigned int *as,
                    __global unsigned int *sorted,
                    __global unsigned int *zerosCount,
                    __global unsigned int *new_bits,
                    unsigned int n,
                    unsigned int mask) {
    unsigned int globalId = get_global_id(0);
    if (globalId < n) {
        unsigned int a = as[globalId];
        unsigned int resultIdx = zerosCount[globalId];
        if (a & mask) {
            unsigned int totalZeros = zerosCount[n - 1] + ((as[n - 1] & mask) ? 0 : 1);
            resultIdx = totalZeros + globalId - resultIdx;
        }
        sorted[resultIdx] = a;
        if (mask << 1)
            new_bits[resultIdx] = (a & (mask << 1)) ? 0 : 1;
    }
}

__kernel void bit_lookup(__global unsigned int *as,
                         __global unsigned int *bits,
                         unsigned int n,
                         unsigned int mask) {
    size_t globalId = get_global_id(0);

    if (globalId < n) {
        bits[globalId] = (as[globalId] & mask) ? 0 : 1;
    }
}

__kernel void prefix_sum(__global unsigned int *as,
                         __global unsigned int *sums,
                         __local unsigned int *local_xs,
                         unsigned int n,
                         unsigned int is_last) {
    size_t localId = get_local_id(0);
    size_t globalId = get_global_id(0);
    size_t wgSize = get_local_size(0);
    unsigned int size = n < 2 * wgSize ? n : 2 * wgSize;


    local_xs[2 * localId] = (2 * globalId >= n) ? 0 : as[2 * globalId];
    local_xs[2 * localId + 1] = (2 * globalId + 1 >= n) ? 0 : as[2 * globalId + 1];

    unsigned last_x = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId == 0) {
        last_x = local_xs[wgSize * 2 - 1];
    }

    unsigned int offset = 1;
    for (unsigned int groupSize = size >> 1; groupSize > 0; groupSize >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < groupSize) {
            unsigned int leftIdx = offset * (2 * localId + 1) - 1;
            unsigned int rightIdx = offset * (2 * localId + 2) - 1;
            local_xs[rightIdx] += local_xs[leftIdx];
        }

        offset *= 2;
    }

    if (localId == 0) {
        local_xs[size - 1] = 0;
    }

    for (unsigned groupSize = 1; groupSize <  size; groupSize *= 2) {
        offset >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId < groupSize) {
            unsigned leftIdx = offset * (2 * localId + 1) - 1;
            unsigned rightIdx = offset * (2 * localId + 2) - 1;
            unsigned x = local_xs[leftIdx];
            local_xs[leftIdx] = local_xs[rightIdx];
            local_xs[rightIdx] += x;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (2 * globalId + 1 < n) {
        as[2 * globalId] = local_xs[2 * localId];
        as[2 * globalId + 1] = local_xs[2 * localId + 1];
    }

    if (localId == 0 && !is_last) {
        sums[get_group_id(0)] = local_xs[wgSize * 2 - 1] + last_x;
    }
}

__kernel void add_prev_pref_sums(__global unsigned int *xs,
                                 __global const unsigned *sums) {
    const size_t groupId = get_group_id(0);
    const size_t globalId = get_global_id(0);
    unsigned s = sums[groupId];
    xs[globalId * 2] += s;
    xs[globalId * 2 + 1] += s;
}
