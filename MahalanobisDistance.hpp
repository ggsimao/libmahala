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
           distance between two vectors and can perform the calculation. It follows
           the convention of single vectors being stored as a D x 1 matrix while
           collections of vectors (even if they might be unitary) being stored
           as a N x D matrix, with D being the number of features and N being the
           number of samples
    \param _inputMatrix Vector neighborhood used to build the metric
    \param _a Matrix that holds the difference between the vector neighborhood
              and the reference vector
    \param _c The covariance matrix of the vector neighborhood (currently not used)
    \param _cInv The inverse of _c (currently not used)
    \param _u U matrix resultant of performing an SVD operation on _a^T * _a
    \param _uK U matrix with its last _k columns removed (currently not used)
    \param _w W matrix resultant of performing an SVD operation on _a^T * _a
    \param _wSigma2 _w + _sigma2
    \param _reference Reference vector (usually the mean of the vector neighborhood)
    \param _cSigma2Inv Inverse of c when _sigma2 is added to _w (it's used to calculate the distances
                       instead of the traditional _cInv because the latter might not be obtainable)
    \param _smin User-set parameter used to guarantee that _c will be inversible but also slightly
                 changes the result (can also be used to make the calculation more efficient in
                 a formula that is currently not being used)
    \param _sigma2 _smin-derived parameter that is actually used on the calculations
    \param _dimension Number of dimensions of the vector neighborhood
    \param _k Number of null dimensions of the vector neighborhood (currently not used)
    \param _numberOfPoints Number of vectors in the vector neighborhood (not necessarily
                           including the reference)
    \param _dirty True if the build() method needs to be called before calculating a distance
    \param _setReference True if the reference vector was set by the user when instanciating the object
*/
class MahalaDist {
public:
    /*! \brief Constructor
        \param input will be the _inputMatrix class member
        \param smin will be the _smin class member
        \param reference will be the _reference class member (which will be the mean of _inputMatrix if empty)
    */
    MahalaDist(const Mat& input, double smin, Mat reference = Mat());
    MahalaDist();
    virtual ~MahalaDist();

    // getters (need to be updated)
    Mat inputMatrix();
    Mat reference();
    double smin();
    int dimension();
    int numberOfPoints();
    bool dirty();
    const Mat u() const;
    const Mat uK() const;
    Mat w();
    Mat c();
    double w(int k);
    double wSigma2(int k);
    // double c(int k);
    double sigma2();

    // setter
    void smin(double smin);

    /*! \brief builds the parameters that are used on the calculation
               of the distances and are based on the _smin class member
    */
    void build();

    /*
     * The following two methods are not currently being worked on
     * and they might not have any reason to continue to exist, since
     * they are a simple particular case of the two methods that come after
     */
    double pointTo(Mat& point1, Mat& point2);
    double pointToReference(Mat& point);
    /*! \brief calculates the Mahalanobis distance between
               ref and each vector in points
        \param points a collection of vectors
        \param ref a single vector
        \return a N x 1 matrix containing the calculated distances
                where N is the number of vectors in points
    */ 
    Mat pointsTo(Mat& points, Mat& ref);
    /*! \brief calculates the Mahalanobis distance between
               the _reference class member and each vector in points
        \param points a collection of vectors
        \return a N x 1 matrix containing the calculated distances
                where N is the number of vectors in points
    */ 
    Mat pointsToReference(Mat& points);
    /*! \brief transforms an image into a collection of vectors
               and calculates the Mahalanobis distance between
               each of the vectors and ref
        \param image an image
        \param ref a single vector
        \return an image where each pixel is the distance between the equivalent
                pixel in the image argument and ref (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    /*! \brief transforms an image into a collection of vectors
               and calculates the Mahalanobis distance between
               each of the vectors and the _reference class member
        \param image an image
        \return an image where each pixel is the distance between the equivalent
                pixel in the image argument and _reference (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageToReference(Mat& image);
private:
    Mat _inputMatrix;
    Mat _a;
    Mat _c;
    Mat _cInv;
    Mat _u;
    Mat _uK;
    Mat _w;
    Mat _wSigma2;
    Mat _reference;
    Mat _cSigma2Inv;
    double _smin;
    double _sigma2;
    int _dimension;
    int _k;
    int _numberOfPoints;
    bool _dirty;
    bool _setReference;
};
