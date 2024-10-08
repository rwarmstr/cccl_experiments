// Header libraries - CUDA part lives in "device_fft.h"
#include "audio_extractor.h"
#include "device_fft.h"
#include "piano_scale_draw.h"

#include <QApplication>
#include <QProcessEnvironment>
#include <QString>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_renderer.h>
#include <qwt_scale_draw.h>
#include <qwt_scale_engine.h>
#include <string>

#define WINDOW_SIZE (8192)
#define BATCH_SIZE (1024)
#define FREQUENCY_CUTOFF (4200) // In Hz, piano C8 = 4186.01

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

template <typename T>
T condense_fft_to_notes(const std::unique_ptr<DeviceFFT> &fft, const std::vector<T> &piano_notes, std::vector<T> &condensed_data)
{
    nvtx3::scoped_range r("Condense FFT to notes");
    constexpr int num_notes = 88;
    condensed_data.resize((fft->_host_output_buf.size() / fft->get_num_bins()) * num_notes);

    float hz_per_bin = fft->get_hz_per_bin();

    T max = 0.0;

#pragma omp parallel for schedule(dynamic) reduction(max : max)
    for (int window = 0; window < (fft->_host_output_buf.size() / fft->get_num_bins()); ++window) {
        for (auto it = piano_notes.begin(); it != piano_notes.end(); it++) {
            auto next_it = std::next(it);

            // Upper and lower bound as frequencies
            float lower_bound = (it == piano_notes.begin()) ? 0 : (*it + *std::prev(it)) / 2;
            float upper_bound = (next_it == piano_notes.end()) ? fft->get_hz_per_bin() * fft->get_num_bins() : (*it + *next_it) / 2;

            // Upper and lower bounds as bin indices
            int lower_bin = static_cast<int>(std::floor(lower_bound / hz_per_bin));
            int upper_bin = static_cast<int>(std::ceil(upper_bound / hz_per_bin));

            if (upper_bin > fft->get_num_bins()) {
                upper_bin = fft->get_num_bins();
            }

            // Iterate over the bins and sum up their values
            T note_value = 0.0;
            for (int bin = lower_bin; bin < upper_bin; bin++) {
                note_value += fft->_host_output_buf[window * fft->get_num_bins() + bin];
            }

#pragma omp critical
            {
                max                                       = std::max(note_value, max);
                auto note_offset                          = std::distance(piano_notes.begin(), it);
                condensed_data[window * 88 + note_offset] = note_value;
            }
        }
    }

    return max;
}

void draw_spectrogram_animation(const std::string &filename, const std::unique_ptr<DeviceFFT> &fft, bool condense_to_notes = false)
{
    int num_bins = fft->get_num_bins();

    std::vector<double> frequencies;
    double *data_ptr;

    std::vector<double> condensed_data;
    double max;

    if (condense_to_notes) {
        for (int i = 0; i < 88; i++) {
            float note_freq = 27.5f * pow(2.0f, static_cast<float>(i) / 12.0f);
            frequencies.push_back(note_freq);
        }
        num_bins = 88;
        max      = condense_fft_to_notes<double>(fft, frequencies, condensed_data);
        data_ptr = condensed_data.data();
    }
    else {
        frequencies.resize(fft->get_num_bins());
        for (int i = 0; i < frequencies.size(); i++) {
            frequencies[i] = i * fft->get_hz_per_bin();
        }
        data_ptr = static_cast<double *>(thrust::raw_pointer_cast(fft->_host_output_buf.data()));
        max      = fft->get_output_max();
    }

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

    // Initialize grid: dotted gray lines for major, attach to plot
    std::unique_ptr<QwtPlotGrid> grid = std::make_unique<QwtPlotGrid>();
    grid->setMajorPen(QPen(Qt::gray, 0, Qt::DashLine));
    grid->attach(&plot);

    std::unique_ptr<QwtPlotCurve> curve = std::make_unique<QwtPlotCurve>();
    curve->setPen(QPen(Qt::blue, 3, Qt::SolidLine));
    if (condense_to_notes) {
        // Draw curve as steps for better clarity at low res
        curve->setStyle(QwtPlotCurve::Steps);
    }
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
        return;
    }
    nvtxRangePop();

    QPixmap pixmap(plot.size());
    QPainter painter(&pixmap);
    QImage image = pixmap.toImage();
    cv::Mat mat(image.height(), image.width(), CV_8UC4);
    cv::Mat bgr_mat;

#pragma omp parallel for schedule(dynamic) private(mat, bgr_mat, image)
    for (int frame = 0; frame < fft->_host_output_buf.size() / fft->get_num_bins(); ++frame) {
        nvtx3::scoped_range r("Process frame " + std::to_string(frame));
        if (frame % 100 == 0) {
#pragma omp critical
            std::cout << "Processing frame: " << frame << "/" << fft->_host_output_buf.size() / fft->get_num_bins() << std::endl;
        }

        // Set samples for the curve (this may involve allocations depending on curve implementation)
#pragma omp critical
        curve->setSamples(frequencies.data(),
                          data_ptr + frame * num_bins,
                          num_bins);

        // Render plot without recreating pixmap, painter, or QImage
#pragma omp critical
        plot.replot();
#pragma omp critical
        renderer.render(&plot, &painter, plot.geometry());
        image = pixmap.toImage(); // Reset the image without reallocating pixmap

        {
            nvtx3::scoped_range r2("Write mat to video");
            mat = cv::Mat(image.height(), image.width(), CV_8UC4, const_cast<uchar *>(image.bits()), image.bytesPerLine());
            cv::cvtColor(mat, bgr_mat, cv::COLOR_BGRA2BGR); // Reuse bgr_mat
#pragma omp critical
            video.write(bgr_mat); // Write the frame to the video
        }
    }
}

void draw_spectrogram_image(const std::string &filename, const std::unique_ptr<DeviceFFT> &fft, size_t full_length, size_t hop_size)
{
    nvtx3::scoped_range r("Image Creation");

    // Create a new vector to hold consolidated note values
    std::vector<float> piano_spectrum((fft->_host_output_buf.size() / fft->get_num_bins()) * 88, 0.0f);
    std::vector<float> piano_notes;
    for (int i = 0; i < 88; i++) {
        float note_freq = 27.5f * pow(2.0f, static_cast<float>(i) / 12.0f);
        piano_notes.push_back(note_freq);
    }

    condense_fft_to_notes<float>(fft, piano_notes, piano_spectrum);

    cv::Mat image(piano_spectrum.size() / 88, 88, CV_32F, piano_spectrum.data());
    cv::Mat normalized_image;
    cv::normalize(image, normalized_image, 0, 255, cv::NORM_MINMAX, CV_32F);

    cv::Mat grayscale;
    normalized_image.convertTo(grayscale, CV_8U);
    cv::Mat colorized;
    cv::applyColorMap(grayscale, colorized, cv::COLORMAP_JET);

    cv::imwrite(filename, colorized);
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

    bool first         = true;
    size_t full_length = 0;
    size_t frame_count = 0;
    int hop_size       = 0;

    try {
        // Initialize the audio extractor and read frames. The decoder will throw an exception on error.
        AudioExtractor extractor(video_file);
        AVFrame *frame = nullptr;

        while (extractor.read_frame(frame)) {
            if (first) {
                // On first frame we'll learn the data sizes and formats, so we can initialize the FFT
                AVSampleFormat sample_fmt = static_cast<AVSampleFormat>(frame->format);
                std::cout << "Sample format: " << av_get_sample_fmt_name(sample_fmt);
                std::cout << " - " << (av_sample_fmt_is_planar(sample_fmt) ? "Planar" : "Non-Planar") << std::endl;

                if (!av_sample_fmt_is_planar(sample_fmt)) {
                    std::cerr << "ERROR: Only planar audio formats are supported" << std::endl;
                    return EXIT_FAILURE;
                }

                // Determine full memory requirement, dependent parameters, and initialize
                full_length = extractor.total_frames() * frame->nb_samples;
                std::cout << "Sample rate: " << frame->sample_rate << " kHz" << std::endl;
                std::cout << "Total: " << full_length << " samples across " << extractor.total_frames() << " frames" << std::endl;
                hop_size = frame->sample_rate / 60;
                fft      = std::make_unique<DeviceFFT>(WINDOW_SIZE, hop_size, BATCH_SIZE, full_length, frame->sample_rate, FREQUENCY_CUTOFF);
                first    = false;
            }

            size_t data_size = extractor.bytes_per_sample() * frame->nb_samples;
            frame_count++;
            bool last_frame = (frame_count == (extractor.total_frames() - 1)) ? true : false;
            if (fft->add_sample(reinterpret_cast<float *>(frame->data[0]), frame->nb_samples, last_frame)) {
                fft->run_batch_fft(false, true);
            }
        }
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Wait for all threads to finish
    fft->synchronize();

    // Uncomment below to write FFT output values to CSV
    // write_csv("raw.csv", fft->_host_output_buf);

    // Uncomment below to write an animation of all FFT output values across frames
    draw_spectrogram_animation("animation.mp4", fft, true);

    // Uncomment below to save the entire spectrogram as a PNG image
    draw_spectrogram_image("spectrogram.png", fft, full_length, hop_size);

    return EXIT_SUCCESS;
}
