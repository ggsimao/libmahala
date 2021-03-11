#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class PointCollector {
public:
    PointCollector(Mat& input, Mat& output);
    PointCollector(const char* path, cv::ImreadModes flags, Mat& output);
    PointCollector();
    virtual ~PointCollector();

    /*
     * GETTERS
     */
    Mat& inputImage();
    Mat& paintedImage();
    Mat& collectedPoints();

    void run();
private:
    Mat _inputImage;
    Mat _paintedImage;
    Mat _collectedPoints;
    bool _showCollectedPoints;
    bool _pressed;

    struct CallbackParams {
        Mat& img;
        Mat& paintedImg;
        Mat& points;
        bool& pressed;
    };

    static void onMouse(int event, int x, int y, int flags, void* param);
};