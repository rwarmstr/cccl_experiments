#include <QApplication>
#include <QProcessEnvironment>
#include <QString>
#include <algorithm>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <qwt_plot.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_renderer.h>
#include <qwt_scale_draw.h>
#include <qwt_scale_engine.h>
#include <sstream>
#include <unordered_map>
#include <vector>


class CustomScaleDraw : public QwtScaleDraw
{
    std::unordered_map<double, std::string> noteMap = {
        {27.5, "A0"},
        {4186, "C8"},
        {30.868, "B"},
        {32.7, "C1"},
        {36.7, "D"},
        {41.2, "E"},
        {43.7, "F"},
        {49, "G"},
        {55, "A"},
        {61.7, "B"},
        {65.4, "C2"},
        {73.4, "D"},
        {82.4, "E"},
        {87.3, "F"},
        {98, "G"},
        {110, "A"},
        {123.5, "B"},
        {130.8, "C3"},
        {146.8, "D"},
        {164.8, "E"},
        {174.6, "F"},
        {196, "G"},
        {220, "A"},
        {246.9, "B"},
        {261.6, "C4"},
        {293.7, "D"},
        {329.6, "E"},
        {349.2, "F"},
        {392, "G"},
        {440, "A"},
        {493.9, "B"},
        {523.3, "C5"},
        {587.3, "D"},
        {659.3, "E"},
        {698.5, "F"},
        {784, "G"},
        {880, "A"},
        {987.8, "B"},
        {1046.5, "C6"},
        {1174.7, "D"},
        {1318.5, "E"},
        {1396.9, "F"},
        {1568, "G"},
        {1760, "A"},
        {1979.5, "B"},
        {2093, "C7"},
        {2349.3, "D"},
        {2637, "E"},
        {2793.8, "F"},
        {3136, "G"},
        {3520, "A"},
        {3951.1, "B"}};

public:
    virtual QwtText label(double value) const
    {
        QwtText text;
        // For example, format labels in scientific notation
        auto it = noteMap.find(value);
        if (it != noteMap.end()) {
            text = QString::fromStdString(it->second); // Safely return the string
            if (text.text().startsWith("C")) {
                text.setColor(Qt::red);
            }
        }
        else
            return QwtText(""); // Return an empty string if not found
        return text;
    }
};


// Read in FFT values from the output CSV
std::vector<std::vector<double>> load_fft_data(const std::string &filename, int samples_per_fft)
{
    std::vector<std::vector<double>> fftData;
    std::ifstream file(filename);
    std::string line;

    std::vector<double> frameData;
    while (std::getline(file, line)) {
        frameData.push_back(std::stof(line));
        if (frameData.size() == samples_per_fft) {
            fftData.push_back(frameData);
            frameData.clear();
        }
    }
    return fftData;
}

int main(int argc, char *argv[])
{
    qputenv("QT_QPA_PLATFORM", QByteArray("offscreen"));
    QApplication app(argc, argv);

    // Constant values for the FFT
    const int sample_rate      = 48000;
    const int n_bins           = 8192;
    const int step             = 800;
    const int frequency_cutoff = 4200;
    const float time_per_fft   = step / sample_rate;
    const float hz_per_bin     = (float)sample_rate / (float)n_bins;
    const int samples_per_fft  = int(ceil(frequency_cutoff / hz_per_bin));

    std::vector<double> frequencies(samples_per_fft);
    for (int i = 0; i < frequencies.size(); i++) {
        frequencies[i] = i * hz_per_bin;
    }

    std::vector<std::vector<double>> fft_data;
    fft_data = load_fft_data("/home/ra/Projects/cccl_experiments/build/03_sine_waves/raw.csv", samples_per_fft);

    double max = 0.0;
    for (int frame = 0; frame < fft_data.size(); frame++) {
        max = std::max(max, *std::max_element(fft_data[frame].begin(), fft_data[frame].end()));
    }

    QList<double> notes = {
        27.5, 30.868,
        32.7, 36.7, 41.2, 43.7, 49, 55, 61.7,
        65.4, 73.4, 82.4, 87.3, 98, 110, 123.5,
        130.8, 146.8, 164.8, 174.6, 196, 220, 246.9,
        261.6, 293.7, 329.6, 349.2, 392, 440, 493.9,
        523.3, 587.3, 659.3, 698.5, 784, 880, 987.8,
        1046.5, 1174.7, 1318.5, 1396.9, 1568, 1760, 1979.5,
        2093, 2349.3, 2637, 2793.8, 3136, 3520, 3951.1,
        4186};

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
    plot.setAxisScaleDraw(QwtPlot::xBottom, new CustomScaleDraw());
    plot.setAxisScale(QwtPlot::yLeft, 0, max);
    plot.setFixedSize(1920, 1080);

    QwtPlotGrid *grid = new QwtPlotGrid();

    // Step 2: Customize the grid
    // Set the pen for the major grid lines (solid line, black color)
    grid->setMajorPen(QPen(Qt::gray, 0, Qt::DashLine));

    // Set the pen for the minor grid lines (dotted line, gray color)
    grid->setMinorPen(QPen(Qt::gray, 0, Qt::DashLine));

    // Step 3: Attach the grid to the plot
    grid->attach(&plot);

    QwtPlotCurve *curve = new QwtPlotCurve();
    curve->setPen(QPen(Qt::blue, 3, Qt::SolidLine));
    curve->attach(&plot);


    QwtPlotRenderer renderer;
    renderer.setDiscardFlag(QwtPlotRenderer::DiscardBackground, false);

    cv::VideoWriter video("animation.mp4", cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
                          (double)sample_rate / (double)step, cv::Size(1920, 1080));
    if (!video.isOpened()) {
        std::cerr << "Could not open the video writer" << std::endl;
        return EXIT_FAILURE;
    }

    for (int frame = 0; frame < fft_data.size(); frame++) {
        if (frame % 100 == 0) {
            std::cout << "Frame " << frame << std::endl;
        }

        curve->setSamples(frequencies.data(), fft_data[frame].data(), frequencies.size());

        plot.replot();

        QPixmap pixmap(plot.size());
        QPainter painter(&pixmap);

        renderer.render(&plot, &painter, plot.geometry());
        QImage image = pixmap.toImage();

        cv::Mat mat(image.height(), image.width(), CV_8UC4, const_cast<uchar *>(image.bits()), image.bytesPerLine());
        cv::Mat bgr_mat;
        cv::cvtColor(mat, bgr_mat, cv::COLOR_BGRA2BGR);
        video.write(bgr_mat);
    }

    delete curve;
    delete grid;

    video.release();

    return EXIT_SUCCESS;
}
