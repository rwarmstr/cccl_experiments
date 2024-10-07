#ifndef __PIANO_SCALE_DRAW
#define __PIANO_SCALE_DRAW

#include <QColor>
#include <QString>
#include <qwt_scale_draw.h>
#include <unordered_map>

class PianoScaleDraw : public QwtScaleDraw
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

#endif
