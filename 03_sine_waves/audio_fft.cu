#include "piano_scale_draw.h"

#include <QApplication>
#include <QProcessEnvironment>
#include <QString>
#include <cub/cub.cuh>
#include <cufft.h>
#include <fstream>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <opencv2/opencv.hpp>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_renderer.h>
#include <qwt_scale_draw.h>
#include <qwt_scale_engine.h>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>

// Include FFMPeg headers
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

#define WINDOW_SIZE (8192)
#define BATCH_SIZE (1024)
#define FREQUENCY_CUTOFF (4200) // In Hz, piano C8 = 4186.01

// Template type for Thrust pinned host vector
template <class T>
using pinned_vector = thrust::host_vector<T, thrust::mr::stateless_resource_allocator<T, thrust::system::cuda::universal_host_pinned_memory_resource>>;

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

    pinned_vector<float> _host_input_buf;

    // CUDA streams
    cudaStream_t _data_xfer_stream = 0;
    cudaStream_t _compute_stream   = 0;
    cudaEvent_t _transfer_complete;

    // Other parameters
    size_t _window_size; // Size per FFT
    size_t _hop_size;    // Data offset between sample ranges
    size_t _batch_size;  // Number of batches to process simultaneously
    size_t _last_batch_size = 0;
    float _hz_per_bin       = 0.0f;
    double output_max       = 0.0;


    size_t _host_buf_pos = 0;

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

        // Calculate the maximum frequency bin we care about given the sample rate
        _hz_per_bin   = static_cast<float>(sample_rate) / window_size;
        _freq_bin_end = static_cast<int>(std::ceil(cutoff_freq / _hz_per_bin));

        // Create an output buffer on the host to hold the results
        _postprocess_buf.resize((_batch_size + 1) * _freq_bin_end);
        _host_output_buf.resize((full_length / hop_size) * _freq_bin_end);
    }

    ~DeviceFFT()
    {
        NVTX3_FUNC_RANGE();
        cudaStreamDestroy(_data_xfer_stream);
        cudaStreamDestroy(_compute_stream);
        cudaEventDestroy(_transfer_complete);

        cufftDestroy(_plan);
    }


    void normalize_values()
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

            cudaMallocAsync(&d_max, sizeof(float), _compute_stream);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, abs_it, d_max, _fft_buf.size(), _compute_stream);
            cudaMallocAsync(&d_temp_storage, temp_storage_bytes, _compute_stream);
            cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, abs_it, d_max, _fft_buf.size(), _compute_stream);
            cudaFreeAsync(d_temp_storage, _compute_stream);
        }

        {
            nvtx3::scoped_range norm("Normalization");

            auto normalize_op = [d_max] __device__(float &x) {
                x = x / *d_max;
            };

            temp_storage_bytes = 0;
            d_temp_storage     = nullptr;
            cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, _fft_buf.begin(), _fft_buf.end(), normalize_op, _compute_stream);
            cudaMallocAsync(&d_temp_storage, temp_storage_bytes, _compute_stream);
            cub::DeviceFor::ForEach(d_temp_storage, temp_storage_bytes, _fft_buf.begin(), _fft_buf.end(), normalize_op, _compute_stream);
            cudaFreeAsync(d_temp_storage, _compute_stream);

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
            float *d_temp_storage     = nullptr;
            size_t temp_storage_bytes = 0;

            cub::CountingInputIterator<int> cnt(0);

            cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, cnt, _window_size * _batch_size, f, _compute_stream);
            cudaMallocAsync(&d_temp_storage, temp_storage_bytes, _compute_stream);
            cub::DeviceFor::ForEachN(d_temp_storage, temp_storage_bytes, cnt, _window_size * _batch_size, f, _compute_stream);
            cudaFreeAsync(d_temp_storage, _compute_stream);
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

            size_t temp_storage_bytes = 0;
            void *d_temp_storage      = nullptr;
            cub::DeviceFor::ForEachN(d_temp_storage,
                                     temp_storage_bytes,
                                     counting_iterator,
                                     total_items,
                                     BinMagnitudeFunctor{_fft_results.data().get(),
                                                         _postprocess_buf.data().get(),
                                                         static_cast<int>(_window_size),
                                                         _freq_bin_end},
                                     _compute_stream);
            cudaMallocAsync(&d_temp_storage, temp_storage_bytes, _compute_stream);
            cub::DeviceFor::ForEachN(d_temp_storage,
                                     temp_storage_bytes,
                                     counting_iterator,
                                     total_items,
                                     BinMagnitudeFunctor{_fft_results.data().get(),
                                                         _postprocess_buf.data().get(),
                                                         static_cast<int>(_window_size),
                                                         _freq_bin_end},
                                     _compute_stream);
            cudaFreeAsync(d_temp_storage, _compute_stream);
        }

        {
            nvtx3::scoped_range r("Find max output magnitude");
        }

        {
            nvtx3::scoped_range r(("Fill output: " + std::to_string(_host_buf_pos) + " to " + std::to_string(_host_buf_pos + _last_batch_size * _freq_bin_end)));

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

int draw_spectrogram(const std::string &filename, const std::unique_ptr<DeviceFFT> &fft)
{
    int num_bins = fft->get_num_bins();

    std::vector<double> frequencies(num_bins);
    for (int i = 0; i < frequencies.size(); i++) {
        frequencies[i] = i * fft->get_hz_per_bin();
    }

    double max = 0.0;
    max        = std::max(max, *std::max_element(fft->_host_output_buf.begin(), fft->_host_output_buf.end()));

    nvtxRangePushA("Set up plot");
    // Create the plot
    QwtPlot plot;
    plot.setCanvasBackground(Qt::white);
    plot.setTitle("Frequency");
    plot.setAxisTitle(QwtPlot::xBottom, "Frequency (Hz)");
    plot.setAxisTitle(QwtPlot::yLeft, "Magnitude");
    plot.setAxisScaleEngine(QwtPlot::xBottom, new QwtLogScaleEngine());

    QwtScaleDiv scale(27.0, 4200.0, QList<double>(), QList<double>(), notes);
    plot.setAxisScale(QwtPlot::xBottom, 27, 4200); // Frequency range
    plot.setAxisScaleDiv(QwtPlot::xBottom, scale);
    plot.setAxisScaleDraw(QwtPlot::xBottom, new PianoScaleDraw());
    plot.setAxisScale(QwtPlot::yLeft, 0, max);
    plot.setFixedSize(1920, 1080);

    std::unique_ptr<QwtPlotGrid> grid = std::make_unique<QwtPlotGrid>();

    // Step 2: Customize the grid
    // Set the pen for the major grid lines (solid line, black color)
    grid->setMajorPen(QPen(Qt::gray, 0, Qt::DashLine));

    // Set the pen for the minor grid lines (dotted line, gray color)
    grid->setMinorPen(QPen(Qt::gray, 0, Qt::DashLine));

    // Step 3: Attach the grid to the plot
    grid->attach(&plot);

    std::unique_ptr<QwtPlotCurve> curve = std::make_unique<QwtPlotCurve>();
    curve->setPen(QPen(Qt::blue, 3, Qt::SolidLine));
    curve->attach(&plot);

    QwtPlotRenderer renderer;
    renderer.setDiscardFlag(QwtPlotRenderer::DiscardBackground, false);
    nvtxRangePop();

    nvtxRangePushA("Initialize video writer");
    cv::VideoWriter video(filename,
                          cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                          (double)fft->get_sample_rate() / (double)fft->get_hop_size(),
                          cv::Size(1920, 1080));
    if (!video.isOpened()) {
        std::cerr << "Could not open the video writer" << std::endl;
        return EXIT_FAILURE;
    }
    nvtxRangePop();

    for (int frame = 0; frame < fft->_host_output_buf.size() / num_bins; ++frame) {
        nvtx3::scoped_range r("Process frame " + std::to_string(frame));
        if (frame % 100 == 0) {
            std::cout << "Processing frame: " << frame << "/" << fft->_host_output_buf.size() / num_bins << std::endl;
        }

        curve->setSamples(frequencies.data(), static_cast<double *>(thrust::raw_pointer_cast(fft->_host_output_buf.data())) + frame * num_bins, num_bins);

        plot.replot();
        QPixmap pixmap(plot.size());
        QPainter painter(&pixmap);

        renderer.render(&plot, &painter, plot.geometry());
        QImage image = pixmap.toImage();

        {
            nvtx3::scoped_range r2("Write mat to video");
            cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar *>(image.bits()), image.bytesPerLine());
            cv::Mat bgr_mat;
            cv::cvtColor(mat, bgr_mat, cv::COLOR_BGRA2BGR);
            video.write(bgr_mat);
        }
    }

    return EXIT_SUCCESS;
}


int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }

    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);

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
    size_t frame_count = 0;
    int hop_size       = 0;

    int num_frames = format_context->streams[audio_stream_index]->nb_frames;

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
                    std::cout << "Sample rate: " << frame->sample_rate << " kHz" << std::endl;
                    std::cout << "Total: " << full_length << " samples across " << format_context->streams[audio_stream_index]->nb_frames << " frames" << std::endl;
                    hop_size = frame->sample_rate / 60;
                    fft      = std::make_unique<DeviceFFT>(WINDOW_SIZE, hop_size, BATCH_SIZE, full_length, frame->sample_rate, FREQUENCY_CUTOFF);
                    first    = false;
                }

                size_t data_size = av_get_bytes_per_sample(codec_context->sample_fmt) * frame->nb_samples;
                frame_count++;
                bool last_frame = (frame_count == (num_frames - 1)) ? true : false;
                if (fft->add_sample(reinterpret_cast<float *>(frame->data[0]), frame->nb_samples, last_frame)) {
                    fft->run_batch_fft(true, true);
                }
            }
        }
        av_packet_unref(packet);
    }

    avformat_close_input(&format_context);

    // Wait for all threads to finish
    cudaDeviceSynchronize();

    // write_csv("raw.csv", fft->_host_output_buf);

    draw_spectrogram("animation.mp4", fft);

    // Uncomment below to save raw values to file for external processing
    /*
    std::ofstream outfile("raw.csv");
    if (outfile.is_open()) {
        std::ostringstream buffer;
        for (const auto &val : fft->_host_output_buf) {
            buffer << val << "\n";
        }
        outfile << buffer.str();
        outfile.close();
    }
    else {
        std::cout << "Unable to open file" << std::endl;
    }
    */

    // Uncomment below to save the entire spectrogram as a PNG image
    {
        nvtx3::scoped_range("Image Creation");

        cv::Mat image(full_length / hop_size, fft->get_num_bins(), CV_32F, fft->_host_output_buf.data().get());
        cv::Mat normalized_image;
        cv::normalize(image, normalized_image, 0, 255, cv::NORM_MINMAX, CV_32F);

        cv::Mat grayscale;
        normalized_image.convertTo(grayscale, CV_8U);
        cv::Mat colorized;
        cv::applyColorMap(grayscale, colorized, cv::COLORMAP_JET);

        cv::imwrite("fft.png", colorized);
    }
    return EXIT_SUCCESS;
}
