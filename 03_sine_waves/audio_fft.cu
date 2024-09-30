#include <cub/cub.cuh>
#include <cufft.h>
#include <fstream>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <thrust/device_vector.h>

// Include FFMPeg headers
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#define WINDOW_SIZE (4096)
#define HOP_SIZE (1024)

class DeviceFFT
{
private:
    thrust::device_vector<float> _fft_buf;
    thrust::device_vector<float> _hann;
    thrust::device_vector<cufftComplex> _fft_results;
    thrust::device_vector<float> _mags;

    cudaStream_t _data_xfer_stream = 0;
    cudaStream_t _compute_stream   = 0;

    size_t _buf_depth     = 0;
    int _last_window_size = 0;


public:
    void
    normalize_values()
    {
        NVTX3_FUNC_RANGE();

        float *d_max              = nullptr;
        void *d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;

        {
            nvtx3::scoped_range find_max("Find max");
            auto abs_it = thrust::transform_iterator(_fft_buf.begin(),
                                                     [] __host__ __device__(float x) {
                                                         return cuda::std::abs(x);
                                                     });

            cudaMalloc(&d_max, sizeof(float));
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, abs_it, d_max, _fft_buf.size());
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, abs_it, d_max, _fft_buf.size());
            cudaFree(d_temp_storage);
        }

        {
            nvtx3::scoped_range norm("Normalization");

            auto normalize_op = [d_max] __device__(float &x) {
                x = x / *d_max;
            };

            temp_storage_bytes = 0;
            d_temp_storage     = nullptr;
            cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, _fft_buf.begin(), _fft_buf.end(), normalize_op);
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, _fft_buf.begin(), _fft_buf.end(), normalize_op);
            cudaFree(d_temp_storage);

            cudaFree(d_max);
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


    DeviceFFT(size_t buffer_size, size_t window_size) : _fft_buf(buffer_size),
                                                        _hann(thrust::make_transform_iterator(thrust::counting_iterator<int>(0), hann_functor(window_size)),
                                                              thrust::make_transform_iterator(thrust::counting_iterator<int>(window_size), hann_functor(window_size)))
    {
        NVTX3_FUNC_RANGE();
        cudaStreamCreateWithFlags(&_data_xfer_stream, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&_compute_stream, cudaStreamNonBlocking);
    }

    ~DeviceFFT()
    {
        NVTX3_FUNC_RANGE();
        cudaStreamDestroy(_data_xfer_stream);
        cudaStreamDestroy(_compute_stream);
    }

    void add_sample(float *data, int length)
    {
        if (_buf_depth + length <= _fft_buf.size()) {
            nvtx3::scoped_range copy_in("Copy in " + std::to_string(length) + " from " + std::to_string(_buf_depth).c_str());
            cudaMemcpyAsync(thrust::raw_pointer_cast(_fft_buf.data()) + _buf_depth,
                            data,
                            sizeof(float) * length,
                            cudaMemcpyHostToDevice,
                            _data_xfer_stream);
            _buf_depth += length;
        }
        else {
            nvtx3::scoped_range("Shuffle and copy in");

            cudaMemcpyAsync(thrust::raw_pointer_cast(_fft_buf.data()),
                            thrust::raw_pointer_cast(_fft_buf.data() + length),
                            (_buf_depth - length) * sizeof(float),
                            cudaMemcpyDeviceToDevice, _data_xfer_stream);
            cudaMemcpyAsync(thrust::raw_pointer_cast(_fft_buf.data()) + _buf_depth - length,
                            data,
                            sizeof(float) * length,
                            cudaMemcpyHostToDevice,
                            _data_xfer_stream);
        }
    }

    std::vector<float> get_window_vector()
    {
        std::vector<float> window(_hann.size());
        thrust::copy(_hann.begin(), _hann.end(), window.begin());

        return window;
    }

    cufftResult run_batch_fft(int length, int window_size, int hop, bool normalize = true)
    {
        NVTX3_FUNC_RANGE();
        cufftResult result;

        nvtxRangePushA("Ensure data transfer completion");
        cudaStreamSynchronize(_data_xfer_stream);
        nvtxRangePop();


        int _num_batches  = (length - window_size) / hop + 1;
        _last_window_size = window_size;
        {
            nvtx3::scoped_range r("Allocate intermediate memory");
            _fft_results.resize(_num_batches * window_size);
            _mags.resize(_num_batches * window_size);
        }

        if (normalize) {
            normalize_values();
        }

        // Create cuFFT plan
        cufftHandle plan;
        {
            nvtx3::scoped_range plan_creation("Create FFT Plan");
            int n[1]       = {window_size}; // FFT size is 4096
            int inembed[1] = {length};      // Input size per batch
            int istride    = 1;             // Stride between samples in a batch
            int idist      = hop;           // Distance between the start of each batch (hop size)
            int onembed[1] = {window_size}; // Output size per batch (same as input in 1D FFT)
            int ostride    = 1;
            int odist      = window_size; // Distance between the start of each output batch

            // Batch mode FFT plan creation
            cufftPlanMany(&plan, 1, n, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, _num_batches);
            cufftSetStream(plan, _compute_stream);
        }

        {
            nvtx3::scoped_range rng_exec("Execute");
            result = cufftExecR2C(plan, _fft_buf.data().get(), _fft_results.data().get());
            if (result != CUFFT_SUCCESS) {
                std::cout << "CUFFT Error: ExecR2C failed" << std::endl;
                return result;
            }
        }

        return result;
    }

    std::vector<float> get_magnitudes()
    {
        NVTX3_FUNC_RANGE();
        cudaStreamSynchronize(_compute_stream);

        {
            nvtx3::scoped_range r("Convert bins to magnitude");
            auto mag = thrust::transform_iterator(_fft_results.begin(),
                                                  [this] __host__ __device__(cufftComplex c) {
                                                      return sqrtf(c.x * c.x + c.y * c.y) / _last_window_size;
                                                  });

            thrust::copy(mag, mag + _fft_results.size(), _mags.begin());
        }

        nvtxRangePushA("Allocate host vector");
        std::vector<float> h_mags(_mags.size());
        nvtxRangePop();

        nvtxRangePushA("Copy to Host");
        thrust::copy(_mags.begin(), _mags.end(), h_mags.begin());
        nvtxRangePop();

        return h_mags;
    }

    std::vector<std::uint8_t> get_img_magnitudes()
    {
        NVTX3_FUNC_RANGE();
        cudaStreamSynchronize(_compute_stream);
        nvtxRangePushA("Transform FFT results");
        {
            nvtx3::scoped_range r("Transform FFT results");

            auto mag = thrust::transform_iterator(_fft_results.begin(),
                                                  [this] __host__ __device__(cufftComplex c) {
                                                      return sqrtf(c.x * c.x + c.y * c.y) / _last_window_size;
                                                  });

            thrust::copy(mag, mag + _fft_results.size(), _mags.begin());
        }

        float *d_max              = nullptr;
        void *d_temp_storage      = nullptr;
        size_t temp_storage_bytes = 0;
        {
            nvtx3::scoped_range r("Find Maximum");

            cudaMalloc(&d_max, sizeof(float));
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, _mags.data().get(), d_max, _mags.size());
            cudaMalloc(&d_temp_storage, temp_storage_bytes);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, _mags.data().get(), d_max, _mags.size());
            cudaFree(d_temp_storage);
        }

        nvtxRangePushA("Convert to normalized uchar");
        const int k        = 25;
        auto img_transform = thrust::transform_iterator(_mags.begin(),
                                                        [d_max, k] __host__ __device__(float x) {
                                                            return (std::uint8_t)((logf(1 + (k * (x / *d_max))) / logf(1 + k)) * 255);
                                                        });
        thrust::device_vector<std::uint8_t> img_mags(img_transform, img_transform + _mags.size());
        nvtxRangePop();

        nvtxRangePushA("Allocate host memory");
        std::vector<std::uint8_t> img(img_mags.size());
        nvtxRangePop();

        nvtxRangePushA("Copy to Host");
        thrust::copy(img_mags.begin(), img_mags.end(), img.begin());
        nvtxRangePop();

        cudaFree(d_max);

        return img;
    }
};

void write_csv(std::string filename, std::vector<float> &data)
{
    nvtx3::scoped_range r(("Write results: " + filename).c_str());
    std::ofstream outfile(filename);
    if (outfile.is_open()) {
        std::ostringstream buffer;
        for (const auto &val : data) {
            buffer << val << "\n";
        }
        outfile << buffer.str();
        outfile.close();
    }
    else {
        std::cout << "Unable to open file " << filename << std::endl;
    }
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize defice FFT
    std::unique_ptr<DeviceFFT> fft = nullptr;

    std::string video_file = argv[1];

    std::cout << "Processing data for file: " << video_file << std::endl;

    // Initialize FFmpeg libraries
    AVFormatContext *format_context = nullptr;

    // Open the video
    if (avformat_open_input(&format_context, video_file.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Error opening input file" << std::endl;
        return EXIT_FAILURE;
    }

    // Retrieve stream information
    if (avformat_find_stream_info(format_context, nullptr) < 0) {
        std::cerr << "Error finding stream info" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    const AVCodec *codec          = nullptr;
    AVCodecContext *codec_context = nullptr;
    int audio_stream_index        = -1;

    // Find the first audio stream
    for (int i = 0; i < format_context->nb_streams; i++) {
        if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_index = i;
            break;
        }
    }

    // Check if an audio stream was found
    if (audio_stream_index < 0) {
        std::cerr << "No audio stream found" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    AVCodecParameters *codec_parameters = format_context->streams[audio_stream_index]->codecpar;
    // Find the decoder for the audio stream
    codec = avcodec_find_decoder(codec_parameters->codec_id);
    if (!codec) {
        std::cerr << "Error finding decoder" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    // Allocate a codec context for the decoder
    codec_context = avcodec_alloc_context3(codec);
    if (!codec_context) {
        std::cerr << "Error allocating codec context" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    // Set the codec parameters to the allocated codec context
    if (avcodec_parameters_to_context(codec_context, codec_parameters) < 0) {
        std::cerr << "Error setting codec parameters" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    // Open the decoder
    if (avcodec_open2(codec_context, codec, nullptr) < 0) {
        std::cerr << "Error opening decoder" << std::endl;
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }

    AVPacket *packet = av_packet_alloc();
    if (!packet) {
        std::cerr << "Error allocating packet" << std::endl;
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }
    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        std::cerr << "Error allocating frame" << std::endl;
        av_packet_free(&packet);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
        return EXIT_FAILURE;
    }


    bool first         = true;
    size_t full_length = 0;


    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index == audio_stream_index) {
            int response = avcodec_send_packet(codec_context, packet);
            if (response < 0) {
                std::cerr << "Error sending packet to decoder" << std::endl;
                break;
            }

            while (response >= 0) {
                response = avcodec_receive_frame(codec_context, frame);
                if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                    break;
                }
                else if (response < 0) {
                    std::cerr << "Error during decoding" << std::endl;
                    av_packet_unref(packet);
                    return EXIT_FAILURE;
                }

                if (first) {
                    AVSampleFormat sample_fmt = static_cast<AVSampleFormat>(frame->format);
                    std::cout << "Sample format: " << av_get_sample_fmt_name(sample_fmt);
                    std::cout << " - " << (av_sample_fmt_is_planar(sample_fmt) ? "Planar" : "Non-Planar") << std::endl;

                    // Determine full memory requirement
                    full_length = format_context->streams[audio_stream_index]->nb_frames *
                                  frame->nb_samples;
                    std::cout << "Total: " << full_length << " samples across " << format_context->streams[audio_stream_index]->nb_frames << " frames" << std::endl;
                    fft   = std::make_unique<DeviceFFT>(full_length, WINDOW_SIZE);
                    first = false;
                }

                size_t data_size = av_get_bytes_per_sample(codec_context->sample_fmt) * frame->nb_samples;

                fft->add_sample(reinterpret_cast<float *>(frame->data[0]), frame->nb_samples);
            }
        }
        av_packet_unref(packet);
    }

    avformat_close_input(&format_context);


    // Now all data resides on the GPU. We can calculate the FFT
    cufftResult result;

    // Save the Hann window for visualization later

    result = fft->run_batch_fft(full_length, WINDOW_SIZE, HOP_SIZE);
    if (result != CUFFT_SUCCESS) {
        return EXIT_FAILURE;
    }


    std::vector<float> float_mags = fft->get_magnitudes();
    write_csv("video_mags.csv", float_mags);

    std::vector<std::uint8_t> mags = fft->get_img_magnitudes();

    std::vector<float> window = fft->get_window_vector();
    write_csv("window.csv", window);

    /*
        nvtxRangePushA("Image creation");

        cv::Mat image(full_length / HOP_SIZE, WINDOW_SIZE, CV_8UC1, mags.data());
        nvtxRangePushA("Crop");
        cv::Mat cropped_image = image(cv::Rect(0, 0, 400, image.rows));
        nvtxRangePop();
        nvtxRangePushA("Transpose");
        // cv::rotate(cropped_image, cropped_image, cv::ROTATE_90_COUNTERCLOCKWISE);
        cv::Mat color_image;

        // Apply a colormap. Options include COLORMAP_JET, COLORMAP_HOT, COLORMAP_COOL, etc.
        // cv::applyColorMap(cropped_image, color_image, cv::COLORMAP_JET);
        nvtxRangePop();

        nvtxRangePushA("Write");

        cv::imwrite("fft.png", cropped_image);
        nvtxRangePop();
        nvtxRangePop();
    */
    return EXIT_SUCCESS;
}
