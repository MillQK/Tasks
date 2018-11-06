#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

int diagonalEdge(__global float *firstPart,
                 __global float *secondPart,
                 int partSize,
                 int diagonalPos) {
    int left = max(0, diagonalPos - partSize);
    int right = min(diagonalPos, partSize);

    while (left < right) {
        int medium = (left + right) / 2;
        if (firstPart[medium] <= secondPart[diagonalPos - medium - 1]) {
            left = medium + 1;
        } else {
            right = medium;
        }
    }

    return left;
}

__kernel void merge(__global float *as,
                    __global float *asOut,
                    unsigned int n,
                    unsigned int partSize,
                    unsigned int diagonalSize) {

    size_t globalIdx = get_global_id(0);
//    size_t debugGlobalIdx = 3;

    int partWorkers = partSize / diagonalSize;
    int partsPairNum = 2 * (globalIdx / partWorkers);
    int partLocalIdx = globalIdx % partWorkers;

    __global float *firstPart = as + partsPairNum * partSize;
    __global float *secondPart = firstPart + partSize;

    int diagonalLeft = 2 * partLocalIdx * diagonalSize;
    int diagonalRight = diagonalLeft + 2 * diagonalSize;

    int firstStart = diagonalEdge(firstPart, secondPart, partSize, diagonalLeft);
    int firstEnd = diagonalEdge(firstPart, secondPart, partSize, diagonalRight);

//    if (globalIdx == debugGlobalIdx) {
//        printf("here ok\n");
//    }

    int secondStart = diagonalLeft - firstStart;
    int secondEnd = diagonalRight - firstEnd;

    __global float *resultPart = asOut + partsPairNum * partSize;

//    if (globalIdx == debugGlobalIdx) {
//        printf("asOut offset: %d\n", partsPairNum * partSize);
//        printf("fS: %d, sS: %d\n", firstStart, secondStart);
//        printf("fE: %d, sE: %d\n", firstEnd, secondEnd);
//    }

    int firstIdx = firstStart;
    int secondIdx = secondStart;
//    if (partsPairNum * partSize == 0 && firstIdx == 0 && secondIdx == 0) {
//        printf("gIdx: %d\n", globalIdx);
//    }
    while (firstIdx < firstEnd || secondIdx < secondEnd) {
        bool isFromFirstPart =
                secondIdx == secondEnd || (firstIdx < firstEnd && firstPart[firstIdx] <= secondPart[secondIdx]);

//        if (globalIdx == debugGlobalIdx) {
//            printf("fI: %d, sI: %d\n", firstIdx, secondIdx);
//            printf("elem: %f\n", isFromFirstPart ? firstPart[firstIdx] : secondPart[secondIdx]);
//        }

        if (isFromFirstPart) {
            resultPart[firstIdx + secondIdx] = firstPart[firstIdx];
            ++firstIdx;
        } else {
            resultPart[firstIdx + secondIdx] = secondPart[secondIdx];
            ++secondIdx;
        }

    }

}

__kernel void bitonic_local(__global float *as,
                            __local float *locals) {
    size_t localIdx = get_local_id(0);
    size_t wgSize = get_local_size(0);

    int offset = get_group_id(0) * wgSize;
    as += offset;

    locals[localIdx] = as[localIdx];
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int length = 1; length < wgSize; length <<= 1) {

        bool isDesc = ((localIdx & (length << 1)) != 0);
        for (int inc = length; inc > 0; inc >>= 1) {
            int jIdx = localIdx ^inc;

            float a = locals[localIdx];
            float b = locals[jIdx];

            bool isSmaller = (b < a) || (b == a && jIdx < localIdx);
            bool isSwap = isSmaller ^(jIdx < localIdx) ^isDesc;
            barrier(CLK_LOCAL_MEM_FENCE);
            locals[localIdx] = (isSwap) ? b : a;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    as[localIdx] = locals[localIdx];
}
