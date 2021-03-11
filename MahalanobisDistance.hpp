#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class MahalaDist {
public:
    /*
   TODO:
   1. Incluir mais getters/setters;
   2. Possivelmente melhorar funções pointsTo() e pointsToReference();
   3. Cálculos diferentes para quando d > m;
   4. Diferente critério de descarte de colunas da matriz U (variância);
   5. Asserts nas funções imageTo e imageToReference;
   */

    MahalaDist(const Mat& input, double smin, Mat reference = Mat());
    MahalaDist();
    virtual ~MahalaDist();

    Mat inputMatrix();
    Mat reference();
    double smin();
    int dimension();
    // int numberOfPoints();
    bool dirty();

    const Mat u() const;
    Mat w();
    Mat c();
    double w(int k);
    double c(int k);
    double sigma2();

    void setSmin(double smin);
    void build();

    double pointTo(Mat& point1, Mat& point2);
    double pointToReference(Mat& point);
    Mat pointsTo(Mat& points, Mat& ref);
    Mat pointsToReference(Mat& points);
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    template <typename T> Mat imageToReference(Mat& image);
    Mat coordinateTo(Mat& ref);
    Mat coordinateToReference();
private:
    Mat _inputMatrix;
    Mat _a;
    Mat _c;
    Mat _u;
    Mat _w;
    Mat _reference;
    double _smin;
    double _sigma2;
    int _dimension;
    int _k;
    int _numberOfPoints;
    bool _dirty;
    bool _setReference;
};
