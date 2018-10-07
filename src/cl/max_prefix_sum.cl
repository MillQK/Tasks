#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void max_prefix_sum(__global int *mutable_data,
                             unsigned int step,
                             unsigned int size) {
    size_t global_id = get_global_id(0);

    size_t part_size = 2 * step;
    size_t part_num = global_id / step;
    size_t part_pos = global_id % step;
    size_t part_elem_pos = step + part_num * part_size - 1;
    size_t our_element_pos = part_elem_pos + part_pos + 1;
    if (our_element_pos < size) {
        mutable_data[our_element_pos] += mutable_data[part_elem_pos];
    }
}
