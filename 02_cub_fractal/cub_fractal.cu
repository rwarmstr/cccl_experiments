#include <iostream>
#include <cub/cub.cuh>
#include <thrust/device_vector.h>
#include <thrust/complex.h>
#include <opencv2/opencv.hpp>

struct fractal_t
{
    std::uint8_t *d_ptr;
    int width;
    int height;
    thrust::complex<float> min;
    thrust::complex<float> max;
    int iterations;

    __device__ void operator()(int i)
    {
        if (i >= width * height)
        {
            return;
        }

        thrust::complex<float> z(0, 0), z_sqr(0, 0);

        int x = i % width;
        int y = i / width;

        thrust::complex<float> c(x * ((max.real() - min.real()) / width) + min.real(),
                                 max.imag() - y * ((max.imag() - min.imag()) / height));

        int it = 0;

        while (thrust::abs(z) < 2.0f && it < iterations)
        {
            z = (z * z) + c;
            it++;
        }

        if (it == iterations)
        {
            d_ptr[i] = 0;
        }
        else
        {
            d_ptr[i] = (uint8_t)((255.0f * it) / iterations);
        }
    }
};

int main(void)
{
    int width = 1920;
    int height = 1080;
    int size = width * height;

    // Force creation of a CUDA context
    cudaFree(0);

    // Allocate a device vector to hold our image
    thrust::device_vector<std::uint8_t> d_image(size);

    // Create a fractal object and set its parameters
    fractal_t fractal;
    fractal.d_ptr = thrust::raw_pointer_cast(d_image.data());
    fractal.width = width;
    fractal.height = height;
    fractal.min = std::complex<float>(-2.5f, -1.0f);
    fractal.max = std::complex<float>(1.0f, 1.0f);
    fractal.iterations = 256;

    std::uint8_t *d_temp_storage{};
    std::size_t temp_storage_bytes{};

    // Get and allocate temp storage
    cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, d_image.size(), fractal);
    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
    d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

    // Perform the operation
    cub::DeviceFor::Bulk(d_temp_storage, temp_storage_bytes, d_image.size(), fractal);

    cv::Mat image(height, width, CV_8UC1);
    std::cout << image.step << std::endl;

    // Copy from device to host
    cudaMemcpy(image.data, thrust::raw_pointer_cast(d_image.data()), size * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cv::imwrite("mandelbrot.png", image);

    return EXIT_SUCCESS;
}
