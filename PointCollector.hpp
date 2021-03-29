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
    PointCollector(Mat& input);
    PointCollector(const char* path, cv::ImreadModes flags);
    PointCollector();
    virtual ~PointCollector();

    /*
     * GETTERS
     */
    Mat& inputImage();
    Mat& paintedImage();
    Mat& collectedPixels();
    Mat& collectedCoordinates();
    Mat& referencePixel();
    Mat& referenceCoordinate();

    void run();
private:
    Mat _inputImage;
    Mat _paintedImage;
    Mat _collectedPixels;
    Mat _collectedCoordinates;
    bool _showCollectedPoints;
    bool _pressed;
    bool _pressedRight;
    Mat _referencePixel;
    Mat _referenceCoordinate;

    struct CallbackParams {
        Mat& img;
        Mat& paintedImg;
        Mat& pixels;
        Mat& coordinates;
        bool& pressed;
        bool& pressedRight;
        Mat& referencePixel;
        Mat& referenceCoordinate;
    };

    static void onMouse(int event, int x, int y, int flags, void* param);
};