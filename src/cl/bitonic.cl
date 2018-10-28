#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as,
                      int stage,
                      int passOfStage) {
    int globalId = get_global_id(0);
    int pairDistance = 1 << (stage - passOfStage);
    int blockWidth = 2 * pairDistance;
    int leftIdx = (globalId & (pairDistance - 1)) + (globalId >> (stage - passOfStage)) * blockWidth;
    int rightIdx = leftIdx + pairDistance;

    float leftElem = as[leftIdx];
    float rightElem = as[rightIdx];

    int sameDirection = (globalId >> stage) & 0x1;

    int temp = sameDirection ? rightIdx : temp;
    rightIdx = sameDirection ? leftIdx : rightIdx;
    leftIdx = sameDirection ? temp : leftIdx;

    bool compare = (leftElem < rightElem);

    float greaterElem = compare ? rightElem : leftElem;
    float lesserElem = compare ? leftElem : rightElem;

    as[leftIdx] = lesserElem;
    as[rightIdx] = greaterElem;
}
