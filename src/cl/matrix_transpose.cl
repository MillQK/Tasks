#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_transpose(__global const float *as,
                               __global float *as_t,
                               unsigned int M,
                               unsigned int K,
                               __local float *localMem) {
    unsigned int mGlobalIdx = get_global_id(0); // [0; M)
    unsigned int kGlobalIdx = get_global_id(1); // [0; K)

    unsigned int mLocalIdx = get_local_id(0);
    unsigned int kLocalIdx = get_local_id(1);

    unsigned int mGroupSize = get_local_size(0);

    if (mGlobalIdx < M && kGlobalIdx < K) {
        localMem[kLocalIdx * mGroupSize + mLocalIdx] = as[kGlobalIdx * M + mGlobalIdx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int kTransposeIdx = get_group_id(0) * mGroupSize + kLocalIdx;
    const int mTransposeIdx = get_group_id(1) * mGroupSize + mLocalIdx;
    if (mTransposeIdx < M && kTransposeIdx < K) {
        as_t[kTransposeIdx * M + mTransposeIdx] = localMem[mLocalIdx * mGroupSize + kLocalIdx];
    }

}