#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class CoordinateCollector {
public:
    CoordinateCollector(Mat& output, Mat& reference);
    CoordinateCollector();
    virtual ~CoordinateCollector();

    /*
     * GETTERS
     */
    Mat& paintedImage();
    Mat& collectedPoints();
    Mat& reference();

    void run();
private:
    Mat _inputImage;
    Mat _paintedImage;
    Mat _collectedPoints;
    bool _showCollectedPoints;
    bool _pressed;
    bool _pressedRight;
    Mat _reference;

    struct CallbackParams {
        Mat& img;
        Mat& paintedImg;
        Mat& points;
        bool& pressed;
        bool& pressedRight;
        Mat& reference;
    };

    static void onMouse(int event, int x, int y, int flags, void* param);
};