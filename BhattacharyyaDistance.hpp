#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "imageUtils.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class BhattaDist {
public:
    /*! \brief Constructor
        \param input The point neighborhood used to build the metric
        \param smin Will be the _smin class member
    */
    BhattaDist(const Mat& input, double smin);
    BhattaDist();
    virtual ~BhattaDist();

    // getters (need to be updated)
    // Mat inputMatrix();
    Mat mi();
    double smin();
    int dimension();
    bool dirty();
    Mat c();
    const Mat u() const;
    Mat w();
    Mat cSigma2Inv();
    double sigma2();

    // setter
    void smin(double smin);

    /*! \brief Builds the parameters that are used on the calculation
               of the distances and are based on the _smin class member
    */
    void build();

    double pointsTo(Mat& points);
    double metricTo(BhattaDist& metric);
    template <typename T> double imageTo(Mat& image);

private:
    Mat _c;
    Mat _u;
    Mat _w;
    Mat _mi;
    Mat _cSigma2Inv;
    bool _dirty;
    double _smin;
    double _sigma2;
    int _dimension;
    int _numberOfPoints;
};