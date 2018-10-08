#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include "cl/max_prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = 0;
            int sum = 0;
            int result = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")"
                  << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = 0;
                int sum = 0;
                int result = 0;
                for (int i = 0; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            gpu::Device device = gpu::chooseGPUDevice(argc, argv);

            gpu::Context context;
            context.init(device.device_id_opencl);
            context.activate();
            {
                ocl::Kernel max_prefix_kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "max_prefix_sum"),
                        shift_to_start_kernel(max_prefix_sum_kernel, max_prefix_sum_kernel_length, "shift_to_start");

                bool printLog = false;
                max_prefix_kernel.compile(printLog);
                shift_to_start_kernel.compile(printLog);

                std::vector<int> numbers_ram(as), max_prefixes_ram(as), indexes_ram(n, 0);
                for (int i = 0; i < n; ++i) {
                    indexes_ram[i] = i + 1;
                }

                gpu::gpu_mem_32i numbers_vram, max_prefixes_vram, indexes_vram;

                numbers_vram.resizeN(n);
                max_prefixes_vram.resizeN(n);
                indexes_vram.resizeN(n);


                unsigned int workGroupSize = 128;
                ocl::LocalMem numbers_local(workGroupSize * sizeof(int)),
                        max_prefixes_local(workGroupSize * sizeof(int)), indexes_local(workGroupSize * sizeof(int));

                timer t;
                for (int i = 0; i < benchmarkingIters; ++i) {
                    numbers_vram.writeN(numbers_ram.data(), n);
                    max_prefixes_vram.writeN(max_prefixes_ram.data(), n);
                    indexes_vram.writeN(indexes_ram.data(), n);
                    for (unsigned int elemCount = n;
                         elemCount > 1;
                         elemCount = (elemCount + workGroupSize - 1) / workGroupSize) {

                        unsigned int workSize = (elemCount + workGroupSize - 1) / workGroupSize * workGroupSize;

                        max_prefix_kernel.exec(gpu::WorkSize(workGroupSize, workSize),
                                               numbers_vram,
                                               max_prefixes_vram,
                                               indexes_vram,
                                               elemCount,
                                               numbers_local,
                                               max_prefixes_local,
                                               indexes_local);

                        shift_to_start_kernel.exec(gpu::WorkSize(workGroupSize, workSize),
                                                   numbers_vram,
                                                   max_prefixes_vram,
                                                   indexes_vram,
                                                   elemCount);

                    }
                    int max_prefix_sum = 0;
                    int max_prefix_sum_index = 0;
                    max_prefixes_vram.readN(&max_prefix_sum, 1);
                    indexes_vram.readN(&max_prefix_sum_index, 1);
                    if (max_prefix_sum < 0) {
                        max_prefix_sum = 0;
                        max_prefix_sum_index = 0;
                    }
                    EXPECT_THE_SAME(reference_max_sum, max_prefix_sum, "GPU sum result should be consistent!");
                    EXPECT_THE_SAME(reference_result, max_prefix_sum_index, "GPU sum pos result should be consistent!");
                    t.nextLap();
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
