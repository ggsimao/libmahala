#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "imageUtils.cpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;
// using namespace classifiers;

/*! \brief A class that holds the parameters required to calculate the Mahalanobis
           distance between two points and can perform the calculation. It follows
           the convention of single points being stored as a D x 1 matrix while
           collections of points (even if they might be unitary) being stored
           as a N x D matrix, with D being the number of features and N being the
           number of samples
    \param _c The covariance matrix of the point neighborhood
    \param _u Lateral matrix resultant of performing an SVD operation on A^T * A, where
              A is the difference between each row of the point neighborhood matrix and _reference
    \param _w Central matrix resultant of performing an SVD operation on A^T * A, where
              A is the difference between each row of the point neighborhood matrix and _reference
    \param _reference Reference point (usually the mean of the point neighborhood)
    \param _cSigma2Inv Inverse of the covariance matrix when _sigma2 is added to _w
                       (it's used to calculate the distances instead of the traditional
                       inverse because the latter might not be obtainable)
    \param _smin User-set parameter used to guarantee that _c will be inversible but also slightly
                 changes the result (can also be used to make the calculation more efficient in
                 a formula that is currently not being used)
    \param _sigma2 _smin-derived parameter that is actually used on the calculations
    \param _dimension Number of dimensions of the point neighborhood
    \param _numberOfPoints Number of points in the point neighborhood (not necessarily
                           including the reference)
    \param _dirty True if the build() method needs to be called before calculating a distance
*/
class MahalaDist {
public:
    /*! \brief Constructor
        \param input The point neighborhood used to build the metric
        \param smin Will be the _smin class member
        \param reference Will be the _reference class member (which will be the mean of
                         the point neighborhood if empty)
    */
    MahalaDist(const Mat& input, double smin, Mat reference = Mat());
    MahalaDist();
    virtual ~MahalaDist();

    // getters (need to be updated)
    // Mat inputMatrix();
    Mat reference();
    double smin();
    int dimension();
    // int numberOfPoints();
    bool dirty();
    const Mat u() const;
    // const Mat uK() const;
    Mat w();
    Mat c();
    Mat cSigma2Inv();
    // double w(int k);
    // double wSigma2(int k);
    // double c(int k);
    double sigma2();

    // setter
    void smin(double smin);

    /*! \brief Builds the parameters that are used on the calculation
               of the distances and are based on the _smin class member
    */
    void build();

    /*! \brief Calculates the Mahalanobis distance between
               the point1 and point2 arguments
        \param point1 A single point
        \param point2 A single point
        \return The value of the Mahalanobis distance
    */
    double pointTo(Mat& point1, Mat& point2);
    /*! \brief Calculates the Mahalanobis distance between
               the _reference class member and the point argument
        \param point A single point
        \return The value of the Mahalanobis distance
    */
    double pointToReference(Mat& point);
    /*! \brief Calculates the Mahalanobis distance between
               ref and each point in points
        \param points A collection of points
        \param ref A single point
        \return A N x 1 matrix containing the calculated distances
                where N is the number of points in points
    */ 
    Mat pointsTo(Mat& points, Mat& ref);
    /*! \brief Calculates the Mahalanobis distance between
               the _reference class member and each point in points
        \param points A collection of points
        \return A N x 1 matrix containing the calculated distances
                where N is the number of points in points
    */ 
    Mat pointsToReference(Mat& points);
    /*! \brief Transforms an image into a collection of points
               and calculates the Mahalanobis distance between
               each of the points and ref
        \param image An image
        \param ref A single point
        \return An image where each pixel is the distance between the equivalent
                pixel in the image argument and ref (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    /*! \brief Transforms an image into a collection of points
               and calculates the Mahalanobis distance between
               each of the points and the _reference class member
        \param image An image
        \return An image where each pixel is the distance between the equivalent
                pixel in the image argument and _reference (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageToReference(Mat& image);
private:
    // Mat _inputMatrix;
    // Mat _a;
    Mat _c;
    // Mat _cInv;
    Mat _u;
    // Mat _uK;
    Mat _w;
    // Mat _wSigma2;
    Mat _reference;
    Mat _cSigma2Inv;
    double _smin;
    double _sigma2;
    int _dimension;
    int _numberOfPoints;
    bool _dirty;
    // bool _setReference;
};
