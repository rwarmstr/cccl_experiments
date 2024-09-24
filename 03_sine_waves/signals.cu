#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <cufft.h>
#include <array>
#include <nvtx3/nvToolsExt.h>

#define NUM_SAMPLES (1024 * 1024)

#define FREQ_A (440.0f)
#define FREQ_E (659.255f)

#define SAMPLE_RATE (44100.0f)

struct sine_wave_functor
{
    float amplitude;
    float frequency;
    float phase;

    __host__ __device__
    sine_wave_functor(float _amplitude, float _frequency, float _phase)
        : amplitude(_amplitude), frequency(_frequency), phase(_phase) {}

    __host__ __device__ float operator()(const int &index) const
    {
        return amplitude * sinf(frequency * index + phase);
    }
};

struct add_waves
{
    __host__ __device__ float operator()(const thrust::tuple<float, float> &t) const
    {
        return thrust::get<0>(t) + thrust::get<1>(t);
    }
};

void write_results(std::string filename, std::vector<float> &data)
{
    nvtxRangePushA(("Write results: " + filename).c_str());
    std::ofstream outfile("filename");
    if (outfile.is_open())
    {
        for (auto &val : data)
        {
            outfile << val << "\n";
        }
        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file " << filename << std::endl;
    }
    nvtxRangePop();
}

int main(void)
{
    // Quick call to cudaFree to ensure context creation
    nvtxRangePushA("Context creation");
    cudaFree(0);
    nvtxRangePop();

    // Create a device vector to hold out input waveform
    nvtxRangePushA("Memory Initialization");
    auto wave1 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), sine_wave_functor(1, 2 * M_PI * FREQ_A / SAMPLE_RATE, 0));
    auto wave2 = thrust::make_transform_iterator(thrust::counting_iterator<int>(0), sine_wave_functor(0.5, 2 * M_PI * FREQ_E / SAMPLE_RATE, 0));
    const auto waves = thrust::make_zip_iterator(thrust::make_tuple(wave1, wave2));
    const auto initializer = thrust::make_transform_iterator(waves, add_waves());
    thrust::device_vector<float> d_combined(initializer, initializer + NUM_SAMPLES);

    nvtxRangePop();

    // Now let's do an FFT on the combined waveform
    cufftHandle plan;
    cufftResult result;

    int complex_size = NUM_SAMPLES / 2 + 1;
    thrust::device_vector<cufftComplex> d_fft(complex_size);

    nvtxRangePushA("FFT");
    nvtxRangePushA("Create Plan");
    result = cufftPlan1d(&plan, NUM_SAMPLES, CUFFT_R2C, 1);
    if (result != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT Error: Plan creation failed" << std::endl;
        return EXIT_FAILURE;
    }
    nvtxRangePop();
    nvtxRangePushA("Execute");

    result = cufftExecR2C(plan, d_combined.data().get(), d_fft.data().get());
    if (result != CUFFT_SUCCESS)
    {
        std::cout << "CUFFT Error: ExecR2C failed" << std::endl;
        return EXIT_FAILURE;
    }
    nvtxRangePop();
    nvtxRangePop();

    // On-device magnitude spectrum
    nvtxRangePushA("FFT bin magnitudes");
    auto mag = thrust::transform_iterator(d_fft.begin(), [] __host__ __device__(cufftComplex c)
                                          { return sqrtf(c.x * c.x + c.y * c.y) / NUM_SAMPLES; });
    thrust::device_vector<float> d_mags{mag, mag + d_fft.size()};
    nvtxRangePop();

    nvtxRangePushA("Copy results");
    std::vector<float> h_combined(d_combined.size());
    std::vector<float> h_mags(d_mags.size());

    thrust::copy(d_combined.begin(), d_combined.end(), h_combined.begin());
    thrust::copy(d_mags.begin(), d_mags.end(), h_mags.begin());
    nvtxRangePop();

    // Write the results to a file
    write_results("combined.csv", h_combined);
    write_results("mags.csv", h_mags);

    return EXIT_SUCCESS;
}
