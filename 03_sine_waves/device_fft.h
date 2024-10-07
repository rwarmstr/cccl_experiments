#ifndef __DEVICE_FFT
#define __DEVICE_FFT

#include <cub/cub.cuh>
#include <cufft.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

// Template type for Thrust pinned host vector
template <class T>
using pinned_vector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, thrust::system::cuda::universal_host_pinned_memory_resource>>;

template <typename CubFunction>
void cub_helper(cudaStream_t stream, CubFunction cub_function)
{
    void *d_temp_storage      = nullptr;
    size_t temp_storage_bytes = 0;

    // CUB "double call" idiom with temporary storage allocation
    cub_function(d_temp_storage, temp_storage_bytes);
    cudaMallocAsync(&d_temp_storage, temp_storage_bytes, stream);
    cub_function(d_temp_storage, temp_storage_bytes);
    cudaFreeAsync(d_temp_storage, stream);
}

class DeviceFFT
{
private:
    // Device-side buffers for FFT results and intermediate results
    thrust::device_vector<float> _fft_buf;
    thrust::device_vector<float> _preprocess_buf;
    thrust::device_vector<float> _hann;
    thrust::device_vector<float> _ones;
    thrust::device_vector<cufftComplex> _fft_results;
    thrust::device_vector<double> _postprocess_buf;
    thrust::device_vector<double> _chunk_max_buf;

    pinned_vector<float> _host_input_buf;

    // CUDA streams
    cudaStream_t _data_xfer_stream = 0;
    cudaStream_t _compute_stream   = 0;
    cudaEvent_t _transfer_complete;
    cudaEvent_t _compute_complete;

    // Other parameters
    size_t _window_size; // Size per FFT
    size_t _hop_size;    // Data offset between sample ranges
    size_t _batch_size;  // Number of batches to process simultaneously
    size_t _last_batch_size = 0;
    float _hz_per_bin       = 0.0f;
    double *_d_output_max   = nullptr;
    double *_h_output_max   = nullptr;


    size_t _host_buf_pos = 0;
    size_t _chunk_count  = 0;

    int _sample_rate;
    int _freq_bin_end;

    // FFT plan
    cufftHandle _plan;


public:
    // Host-side output buffer for FFT results
    pinned_vector<double> _host_output_buf;

    DeviceFFT(size_t window_size, size_t hop_size, size_t batch_size, size_t full_length, int sample_rate, int cutoff_freq)
        : _fft_buf(window_size + (batch_size - 1) * hop_size),
          _preprocess_buf(batch_size * window_size),
          _hann(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), hann_functor(window_size)),
                thrust::make_transform_iterator(thrust::counting_iterator<int>(window_size), hann_functor(window_size))),
          _ones(window_size, 1.0f),
          _host_input_buf(2 * window_size + (batch_size - 1) * hop_size),
          _hop_size(hop_size),
          _window_size(window_size),
          _batch_size(batch_size),
          _sample_rate(sample_rate)
    {
        NVTX3_FUNC_RANGE();
        cudaStreamCreateWithFlags(&_data_xfer_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&_compute_stream, cudaStreamNonBlocking);

        cudaEventCreateWithFlags(&_transfer_complete, cudaEventDisableTiming);
        cudaEventCreateWithFlags(&_compute_complete, cudaEventDisableTiming);

        // Calculate the maximum frequency bin we care about given the sample rate
        _hz_per_bin   = static_cast<float>(sample_rate) / window_size;
        _freq_bin_end = static_cast<int>(std::ceil(cutoff_freq / _hz_per_bin));

        // Create an output buffer on the host to hold the results
        _postprocess_buf.resize((_batch_size + 1) * _freq_bin_end);
        _host_output_buf.resize((full_length / hop_size) * _freq_bin_end);

        // Allocate intermediate space for max chunks
        size_t chunk_size      = (_window_size + (_batch_size - 1) * _hop_size);
        size_t expected_chunks = (full_length + chunk_size - 1) / chunk_size;
        _chunk_max_buf.resize(expected_chunks);

        cudaMalloc(&_d_output_max, sizeof(double));
        cudaMallocHost(&_h_output_max, sizeof(double));
    }

    ~DeviceFFT()
    {
        NVTX3_FUNC_RANGE();
        cudaStreamDestroy(_data_xfer_stream);
        cudaStreamDestroy(_compute_stream);
        cudaEventDestroy(_transfer_complete);
        cudaEventDestroy(_compute_complete);

        cudaFree(_d_output_max);
        cudaFreeHost(_h_output_max);

        cufftDestroy(_plan);
    }


    void normalize_values()
    {
        NVTX3_FUNC_RANGE();

        float *d_max = nullptr;
        {
            nvtx3::scoped_range find_max("Find max");
            auto abs_it = thrust::transform_iterator(_fft_buf.begin(),
                                                     [] __host__ __device__(float x) {
                                                         return cuda::std::abs(x);
                                                     });


            cudaMallocAsync(&d_max, sizeof(float), _compute_stream);
            cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
                cub::DeviceReduce::Max(
                    d_temp_storage,
                    temp_storage_bytes,
                    abs_it,          // Input iterator
                    d_max,           // Output pointer
                    _fft_buf.size(), // Number of items
                    _compute_stream  // CUDA stream
                );
            });
        }

        {
            nvtx3::scoped_range norm("Normalization");

            auto normalize_op = [d_max] __device__(float &x) {
                x = x / *d_max;
            };

            cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
                cub::DeviceFor::ForEach(
                    d_temp_storage,
                    temp_storage_bytes,
                    _fft_buf.begin(), // Range begin
                    _fft_buf.end(),   // Range end
                    normalize_op,     // Operator
                    _compute_stream   // CUDA stream
                );
            });
            cudaFreeAsync(d_max, _compute_stream);
        }
    }


    struct hann_functor
    {
        size_t window_size;

        __host__ __device__
        hann_functor(size_t _window_size) : window_size(_window_size) {}

        __host__ __device__ float operator()(const int &index) const
        {
            return 0.5f * (1.0f - cosf((2.0f * M_PI * index) / window_size));
        }
    };


    bool add_sample(float *data, int length, bool last)
    {
        NVTX3_FUNC_RANGE();
        static size_t total_length = 0;
        const size_t buffer_length = _window_size + (_batch_size - 1) * _hop_size;

        nvtx3::scoped_range r("Copy to bounce");
        std::copy(data, data + length, _host_input_buf.begin() + total_length);
        //_host_input_buf.insert(_host_input_buf.begin() + total_length, data, data + length);
        total_length += length;


        if ((total_length >= buffer_length) || last) {
            nvtx3::scoped_range r("Copy to device");

            cudaMemcpyAsync(thrust::raw_pointer_cast(_fft_buf.data()),
                            thrust::raw_pointer_cast(_host_input_buf.data()),
                            buffer_length * sizeof(float),
                            cudaMemcpyHostToDevice,
                            _data_xfer_stream);
            if (last) {
                _batch_size = total_length / _hop_size;
            }
            if (total_length > buffer_length) {
                cudaMemcpyAsync(thrust::raw_pointer_cast(_host_input_buf.data()),
                                thrust::raw_pointer_cast(_host_input_buf.data() + buffer_length),
                                (total_length - buffer_length) * sizeof(float),
                                cudaMemcpyHostToHost,
                                _data_xfer_stream);
                total_length -= buffer_length;
            }
            else {
                total_length = 0;
            }
            // std::cout << "cudaEventQuery is " << cudaEventQuery(_transfer_complete) << std::endl;
            cudaEventRecord(_transfer_complete, _data_xfer_stream);
            // std::cout << "cudaEventQuery is " << cudaEventQuery(_transfer_complete) << std::endl;

            return true;
        }

        return false;
    }

    std::vector<float> get_window_vector()
    {
        std::vector<float> window(_hann.size());
        thrust::copy(_hann.begin(), _hann.end(), window.begin());

        return window;
    }

    struct WindowTransformFunctor
    {
        float *in_buf;
        float *out_buf;
        float *window;
        int batch_size;
        int hop_size;
        int window_size;

        // Custom functor for the transformation
        __device__ __host__ void operator()(int index) const
        {
            const int chunk_index = index / window_size;
            const int in_offset   = chunk_index * hop_size;
            const int i           = index % window_size;
            out_buf[index]        = in_buf[in_offset + i] * window[i];
        }
    };

    cufftResult run_batch_fft(bool normalize = true, bool window = true)
    {
        NVTX3_FUNC_RANGE();
        cufftResult result;

        {
            nvtx3::scoped_range r("Wait for data transfer");
            // std::cout << "cudaEventQuery is " << cudaEventQuery(_transfer_complete) << std::endl;
            //  cudaStreamWaitEvent(_compute_stream, _transfer_complete);
            cudaEventSynchronize(_transfer_complete);
            // std::cout << "-- cudaEventQuery is " << cudaEventQuery(_transfer_complete) << std::endl;
        }
        if (_batch_size != _last_batch_size) {
            nvtx3::scoped_range r("Allocate intermediate memory");
            _fft_results.resize(_batch_size * _window_size);
        }

        if (normalize) {
            normalize_values();
        }

        {
            nvtx3::scoped_range r("Transform input buffer");
            float *window_ptr = thrust::raw_pointer_cast(_ones.data());
            if (window) {
                window_ptr = thrust::raw_pointer_cast(_hann.data());
            }
            WindowTransformFunctor f{thrust::raw_pointer_cast(_fft_buf.data()),
                                     thrust::raw_pointer_cast(_preprocess_buf.data()),
                                     window_ptr,
                                     static_cast<int>(_batch_size),
                                     static_cast<int>(_hop_size),
                                     static_cast<int>(_window_size)};
            // Apply the transformation to each element in the input buffer
            cub::CountingInputIterator<int> cnt(0);

            cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
                cub::DeviceFor::ForEachN(d_temp_storage,
                                         temp_storage_bytes,
                                         cnt,
                                         _window_size * _batch_size,
                                         f,
                                         _compute_stream);
            });
        }
        // Create cuFFT plan
        if (_last_batch_size != _batch_size) {
            nvtx3::scoped_range plan_creation("Create FFT Plan");
            result = cufftPlan1d(&_plan, _window_size, CUFFT_R2C, _batch_size);
            if (result != CUFFT_SUCCESS) {
                std::cout << "cuFFT error: Plan creation failed - " << result << std::endl;
                return result;
            }
            cufftSetStream(_plan, _compute_stream);
        }

        {
            nvtx3::scoped_range rng_exec("Execute");
            result = cufftExecR2C(_plan, _preprocess_buf.data().get(), _fft_results.data().get());
            if (result != CUFFT_SUCCESS) {
                std::cout << "CUFFT Error: ExecR2C failed: " << result << std::endl;
                return result;
            }
        }
        _last_batch_size = _batch_size;
        this->magnitudes_to_host();
        return result;
    }

    struct BinMagnitudeFunctor
    {
        cufftComplex *fft_results;
        double *magnitudes;
        int window_size;
        int bins_to_keep;

        __device__ __host__ void operator()(int index) const
        {
            const int batch      = index / bins_to_keep;
            const int i          = index % bins_to_keep;
            const int step       = window_size / 2 + 1;
            const cufftComplex c = fft_results[batch * step + i];
            magnitudes[index]    = sqrt(c.x * c.x + c.y * c.y);
        }
    };

    void magnitudes_to_host()
    {
        NVTX3_FUNC_RANGE();
        {
            nvtx3::scoped_range r("Convert bins to magnitude and trim");

            size_t total_items = _batch_size * _freq_bin_end;
            cub::CountingInputIterator<int> counting_iterator(0);

            cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
                cub::DeviceFor::ForEachN(d_temp_storage,
                                         temp_storage_bytes,
                                         counting_iterator,
                                         total_items,
                                         BinMagnitudeFunctor{_fft_results.data().get(),
                                                             _postprocess_buf.data().get(),
                                                             static_cast<int>(_window_size),
                                                             _freq_bin_end},
                                         _compute_stream);
            });
            cudaEventRecord(_compute_complete, _compute_stream);
        }

        {
            nvtx3::scoped_range r("Find max output magnitude");
            cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
                cub::DeviceReduce::Max(d_temp_storage,
                                       temp_storage_bytes,
                                       _postprocess_buf.data().get(),
                                       _chunk_max_buf.data().get() + _chunk_count,
                                       _postprocess_buf.size(),
                                       _compute_stream);
            });
            _chunk_count++;
        }

        {
            nvtx3::scoped_range r(("Fill output: " + std::to_string(_host_buf_pos) + " to " + std::to_string(_host_buf_pos + _last_batch_size * _freq_bin_end)));
            cudaStreamWaitEvent(_data_xfer_stream, _compute_complete);
            cudaMemcpyAsync(thrust::raw_pointer_cast(_host_output_buf.data().get()) + _host_buf_pos,
                            thrust::raw_pointer_cast(_postprocess_buf.data().get()),
                            sizeof(double) * _batch_size * _freq_bin_end,
                            cudaMemcpyDeviceToHost,
                            _data_xfer_stream);
            _host_buf_pos += _last_batch_size * _freq_bin_end;
        }
    }

    size_t get_num_bins() const { return _freq_bin_end; }
    size_t get_sample_rate() const { return _sample_rate; }
    size_t get_window_size() const { return _window_size; }
    float get_hz_per_bin() const { return _hz_per_bin; }
    size_t get_hop_size() const { return _hop_size; }
    double get_output_max() const
    {
        cub_helper(_compute_stream, [&](void *d_temp_storage, size_t &temp_storage_bytes) {
            cub::DeviceReduce::Max(d_temp_storage,
                                   temp_storage_bytes,
                                   _chunk_max_buf.data().get(),
                                   _d_output_max,
                                   _chunk_count,
                                   _compute_stream);
        });
        cudaMemcpyAsync(_h_output_max, _d_output_max, sizeof(double), cudaMemcpyDeviceToHost, _compute_stream);
        cudaStreamSynchronize(_compute_stream);
        return *_h_output_max;
    }

    void synchronize()
    {
        cudaStreamSynchronize(_compute_stream);
        cudaStreamSynchronize(_data_xfer_stream);
    }
};

#endif
