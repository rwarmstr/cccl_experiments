#include <cufft.h>
#include <fstream>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <thrust/device_vector.h>
#include <string>

// Include FFMPeg headers
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

class DeviceFFT
{
    private:
        thrust::device_vector<float> _fft_buf;
        int _buf_depth = 0;

    public:
        DeviceFFT() : _fft_buf(4096) {
            nvtxRangePushA("Context creation");
            cudaFree(0);
            nvtxRangePop();
        }

        void add_sample(float *data, int length)
        {
            if (_buf_depth + length <= 4096) {
                nvtxRangePushA(("Copy in " + std::to_string(length) + " from " + std::to_string(_buf_depth)).c_str());
                thrust::copy(data, data + length, _fft_buf.begin() + _buf_depth);
                _buf_depth += length;
                nvtxRangePop();
            }
            else {
                nvtxRangePushA("Shuffle");
                nvtxRangePushA(("Move " + std::to_string(length) + " to " + std::to_string(_buf_depth) + " to 0").c_str());
                thrust::copy(_fft_buf.begin() + length, _fft_buf.begin() + _buf_depth, _fft_buf.begin());
                nvtxRangePop();
                nvtxRangePushA("Copy new");
                thrust::copy(data, data + length, _fft_buf.begin() + _buf_depth - length);
                nvtxRangePop();
                nvtxRangePop();
            }
        }
};

int main(int argc, char **argv)
{
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize defice FFT
    DeviceFFT fft;

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

    // Save the time base of hte audio stream
    AVRational time_base = format_context->streams[audio_stream_index]->time_base;

    bool first = true;
    std::uint64_t total_samples = 0;
    int n = 0;
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
                    first = false;
                }

                double time_in_seconds = 0.0;
                if (frame->pts != AV_NOPTS_VALUE) {
                    time_in_seconds = frame->pts * av_q2d(time_base);
                }
                else if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
                    time_in_seconds = frame->best_effort_timestamp * av_q2d(time_base);
                }
                else {
                    // Estimate time based on total samples
                    total_samples += frame->nb_samples;
                    time_in_seconds = static_cast<double>(total_samples) / frame->sample_rate;
                }

                size_t data_size = av_get_bytes_per_sample(codec_context->sample_fmt) * frame->nb_samples;

                std::cout << "Current: " << time_in_seconds << " - " << data_size << " bytes" << std::endl; // * frame->nb_samples * frame->channels;
                n++;
                fft.add_sample(reinterpret_cast<float *>(frame->data[0]), frame->nb_samples);
                
            }
        }
        if (n > 10) {
            break;
        }
        av_packet_unref(packet);
    }

    

    avformat_close_input(&format_context);


    return EXIT_SUCCESS;
}
