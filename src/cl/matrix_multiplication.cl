#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_multiplication(__global const float *as, // M * K
                                    __global const float *bs, // K * N
                                    __global float *cs, // M * N
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N,
                                    __local float *aTile, // workGroupSize * workGroupSize (wg(0) == wg(1))
                                    __local float *bTile) {
    unsigned int nGlobalIdx = get_global_id(0); // [0; N)
    unsigned int mGlobalIdx = get_global_id(1); // [0; M)

    unsigned int nLocalIdx = get_local_id(0); // [0; wgSize)
    unsigned int mLocalIdx = get_local_id(1); // [0; wgSize)

    unsigned int workGroupSize = get_local_size(0);

    unsigned int stepCount = (K + workGroupSize - 1) / workGroupSize;

    float sum = 0.0f;

    for (unsigned int step = 0; step < stepCount; ++step) {
        aTile[mLocalIdx * workGroupSize + nLocalIdx] = as[mGlobalIdx * K + nLocalIdx + step * workGroupSize];
        bTile[mLocalIdx * workGroupSize + nLocalIdx] = bs[nGlobalIdx + (mLocalIdx + step * workGroupSize) * K];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int i = 0; i < workGroupSize; ++i) {
            sum += aTile[mLocalIdx * workGroupSize + i] * bTile[nLocalIdx + i * workGroupSize];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[mGlobalIdx * N + nGlobalIdx] = sum;
}