#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "imageUtils.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class DummyMahalaDist {
public:
    DummyMahalaDist(const Mat& input, Mat reference = Mat());
    DummyMahalaDist();
    virtual ~DummyMahalaDist();

    double pointTo(Mat& point1, Mat& point2);
    double pointToReference(Mat& point);
    Mat pointsTo(Mat& points, Mat& ref);
    Mat pointsToReference(Mat& points);
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    template <typename T> Mat imageToReference(Mat& image);
private:
    int _dimension;
    int _numberOfPoints;
    Mat _cInv;
    Mat _reference;
};