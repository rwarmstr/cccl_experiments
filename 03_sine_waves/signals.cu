#include <iostream>
#include <fstream>
#include <thrust/device_vector.h>
#include <thrust/tabulate.h>
#include <array>

#define NUM_SAMPLES (1024 * 1024)

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

int main(void)
{
    // Quick call to cudaFree to ensure context creation
    cudaFree(0);

    // Create a device vector to hold out input waveform
    thrust::device_vector<float> d_vec(NUM_SAMPLES);
    thrust::device_vector<float> d_vec2(NUM_SAMPLES);
    thrust::device_vector<float> d_combined(NUM_SAMPLES);

    // Initialize the two device vectors
    thrust::tabulate(
        d_vec.begin(),
        d_vec.end(),
        sine_wave_functor(1, 2 * M_PI / NUM_SAMPLES, 0));
    thrust::tabulate(
        d_vec2.begin(),
        d_vec2.end(),
        sine_wave_functor(0.5, 4 * M_PI / NUM_SAMPLES, 0));

    // Add all of the component waves
    thrust::transform(d_vec.begin(), d_vec.end(), d_vec2.begin(), d_combined.begin(), thrust::plus<float>());

    std::vector<float> h_combined(d_combined.size());

    thrust::copy(d_combined.begin(), d_combined.end(), h_combined.begin());

    std::ofstream outfile("output.csv");
    if (outfile.is_open())
    {
        for (auto &val : h_combined)
        {
            outfile << val << "\n";
        }
        outfile.close();
    }
    else
    {
        std::cout << "Unable to open file" << std::endl;
    }

    return EXIT_SUCCESS;
}
