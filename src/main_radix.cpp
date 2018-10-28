#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void calc_prefix_sum(gpu::gpu_mem_32u &as,
                     unsigned int n,
                     ocl::Kernel &prefix_sum,
                     ocl::Kernel &add_prev_pref_sums,
                     unsigned int wgSize) {

    ocl::LocalMem as_local(2 * wgSize);

    if (n <= 4 * wgSize) {
        prefix_sum.exec(gpu::WorkSize(wgSize, n / 2), as, as, as_local, n, 1);
    } else {
        size_t sums_size = n / (2 * wgSize);
        gpu::gpu_mem_32u sums = gpu::gpu_mem_32u::createN(sums_size);

        prefix_sum.exec(gpu::WorkSize(wgSize, n / 2), as, sums, as_local, n, 0);
        calc_prefix_sum(sums, sums_size, prefix_sum, add_prev_pref_sums, wgSize);
        add_prev_pref_sums.exec(gpu::WorkSize(wgSize, n / 2), as, sums);
    }

}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 16 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        ocl::Kernel bit_lookup(radix_kernel, radix_kernel_length, "bit_lookup");
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        ocl::Kernel add_prev_pref_sums(radix_kernel, radix_kernel_length, "add_prev_pref_sums");
        radix.compile();
        bit_lookup.compile();
        prefix_sum.compile();
        add_prev_pref_sums.compile();

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, global_work_size);

        gpu::gpu_mem_32u as_bits, as_bits_updated, as_updated;
        as_bits.resizeN(n);
        as_bits_updated.resizeN(n);
        as_updated.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            bit_lookup.exec(workSize, as_gpu, as_bits, n, 1);
            for (unsigned int mask = 1; mask != 0; mask <<= 1) {
                calc_prefix_sum(as_bits, n, prefix_sum, add_prev_pref_sums, workGroupSize);
                radix.exec(workSize, as_gpu, as_updated, as_bits, as_bits_updated, n, mask);

                as_gpu.swap(as_updated);
                as_bits.swap(as_bits_updated);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
