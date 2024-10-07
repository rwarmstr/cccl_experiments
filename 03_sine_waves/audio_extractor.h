#ifndef __AUDIO_EXTRACTOR
#define __AUDIO_EXTRACTOR

#include <iostream>
#include <memory>

// Include FFMPeg headers
extern "C"
{
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/samplefmt.h>
}

class AudioExtractor
{
    AVFormatContext *format_context;
    AVCodecContext *codec_context;
    AVPacket *packet;
    AVFrame *frame;
    int audio_stream_index;

    bool initialize(const std::string &file_name)
    {
        if (avformat_open_input(&format_context, file_name.c_str(), nullptr, nullptr) < 0) {
            std::cerr << "Error opening input file." << std::endl;
            return false;
        }

        if (avformat_find_stream_info(format_context, nullptr) < 0) {
            std::cerr << "Error finding stream info." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        for (int i = 0; i < format_context->nb_streams; i++) {
            if (format_context->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                audio_stream_index = i;
                break;
            }
        }

        if (audio_stream_index < 0) {
            std::cerr << "No audio stream found." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        const AVCodec *codec = avcodec_find_decoder(format_context->streams[audio_stream_index]->codecpar->codec_id);
        if (!codec) {
            std::cerr << "Error finding decoder." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        codec_context = avcodec_alloc_context3(codec);
        if (!codec_context) {
            std::cerr << "Error allocating codec context." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        if (avcodec_parameters_to_context(codec_context, format_context->streams[audio_stream_index]->codecpar) < 0) {
            std::cerr << "Error setting codec parameters." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        if (avcodec_open2(codec_context, codec, nullptr) < 0) {
            std::cerr << "Error opening decoder." << std::endl;
            avformat_close_input(&format_context);
            return false;
        }

        packet = av_packet_alloc();
        if (!packet) {
            std::cerr << "Error allocating packet." << std::endl;
            return false;
        }

        frame = av_frame_alloc();
        if (!frame) {
            std::cerr << "Error allocating frame." << std::endl;
            return false;
        }

        return true;
    }


public:
    AudioExtractor(const std::string &file_name) : format_context(nullptr), codec_context(nullptr), packet(nullptr), frame(nullptr), audio_stream_index(-1)
    {
        if (!initialize(file_name)) {
            throw std::runtime_error("Failed to initialize video decoder.");
        }
    }

    ~AudioExtractor()
    {
        av_packet_free(&packet);
        av_frame_free(&frame);
        avcodec_free_context(&codec_context);
        avformat_close_input(&format_context);
    }

    bool read_frame(AVFrame *&out_frame)
    {
        while (av_read_frame(format_context, packet) >= 0) {
            if (packet->stream_index == audio_stream_index) {
                int response = avcodec_send_packet(codec_context, packet);
                if (response < 0) {
                    std::cerr << "Error sending packet to decoder." << std::endl;
                    av_packet_unref(packet);
                    return false;
                }

                while (response >= 0) {
                    response = avcodec_receive_frame(codec_context, frame);
                    if (response == AVERROR(EAGAIN) || response == AVERROR_EOF) {
                        break;
                    }
                    else if (response < 0) {
                        std::cerr << "Error during decoding." << std::endl;
                        av_packet_unref(packet);
                        return false;
                    }

                    out_frame = frame;
                    av_packet_unref(packet);
                    return true;
                }
            }
            av_packet_unref(packet);
        }
        return false;
    }

    int total_frames() const
    {
        return format_context->streams[audio_stream_index]->nb_frames;
    }

    size_t bytes_per_sample() const
    {
        return av_get_bytes_per_sample(codec_context->sample_fmt);
    }
};


#endif
