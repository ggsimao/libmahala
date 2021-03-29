#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MahalanobisDistance.hpp"
#include "imageUtils.cpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

/*! \brief A class that holds the parameters required to calculate the polynomial
           Mahalanobis distance between two vectors and can perform the calculation. 
           It follows the convention of single vectors being stored as a D x 1 matrix
           while collections of vectors (even if they might be unitary) being stored
           as a N x D matrix, with D being the number of features and N being the
           number of samples
    \param _inputMatrix Vector neighborhood used to build the metric
    \param _reference Reference vector (usually the mean of the vector neighborhood)
    \param _smin User-set parameter used to guarantee that _c of the used MahalaDist objects
                 are inversible inversible but also slightly changes the result (can also be
                 used to make the calculation more efficient in a formula that is currently
                 not being used)
    \param _l log_2 of the order of the polynomial terms
    \param _dimension Number of dimensions of the vector neighborhood
    \param _numberOfPoints Number of vectors in the vector neighborhood (not necessarily
                           including the reference)
    \param _polynomialDimension Final number of dimensions after the polynomial expansions
    \param _expandedReferences References for each element in _expandedDists
    \param _expandedUs _u matrices for each element in _expandedDists
    \param _expandedDists MahalaDist objects with their neighborhoods and references expanded
                          polynomially (one object for each 0 <= l < _l)
    \param _indexesVector Indexes of the non-null dimensions of each element in _expandedDists
    \param _maxAbsVector
    \param _baseDist MahalaDist object of vectors in the 0-order polynomial expansion
    \param _dirty True if the build() method needs to be called before calculating a distance
    \param _setReference True if the reference vector was set by the user when instanciating the object
*/
class PolyMahalaDist {
public:


    /*! \brief Constructor
        \param input will be the _inputMatrix class member
        \param smin will be the _smin class member
        \param l will be the _l class member
        \param reference will be the _reference class member (which will be the mean of _inputMatrix if empty)
    */
    PolyMahalaDist(Mat input, double smin, int l, Mat reference = Mat());
    PolyMahalaDist();
    virtual ~PolyMahalaDist();

    // getters (need to be updated)
    Mat inputMatrix();
    Mat reference();
    double smin();
    // int dimension();
    // int numberOfPoints();
    int l();

    // setters
    void smin(double smin);
    void l(int l);

    /*! \brief Projects a collection of vectors into its second-order polynomial space
        \param vec Collection of vectors to be projected
        \return result of projecting each of vec's rows into its second-order polynomial space 
    */
    Mat polynomialProjection(Mat vec);
    /*! \brief Select the non-null dimensions of a collection of vectors by comparing their
               variances to the _smin class member
        \param vec Collection of vectors to be analyzed
        \return The indexes of the non-null dimensions of vec
    */
    vector<int> filterByVariance(Mat vec);
    /*! \brief written by PC
    */
    Mat filterByVariancePC(const Mat &data, std::vector<int>& outIndes);
    /*! \brief Multiplies two collections of vectors and then filter out undesired dimensions
        \param slaveVec The first matrix of the multiplication.
        \param masterVec The second matrix of the multiplication.
        \param indexes The desired indexes of the desired features of the result of the multiplication.
        \return The matrix resultant from the multiplication.
    */
    Mat filteringMultiplication(Mat slaveVec, Mat masterVec, vector<int> indexes);
    /*! \brief written by PC
    */
    Mat removeNullDimensions(Mat& data, vector<int>& validDimesions);

    /*! \brief builds the parameters that are used on the calculation
               of the distances and are based on the _smin and _l class members
    */
    void build();
    /*! \brief builds the parameters that are used on the calculation
               of the distances and are based on the _smin and _l
               class members (written by PC)
    */
    void buildPC();

    /*
     * The following three methods are not currently being worked on
     * and they might not have any reason to continue to exist, since
     * they are a simple particular case of the three methods that come after
     */
    double pointTo(Mat point1, Mat point2);
    double pointToReference(Mat point);
    double pointToReferencePC(Mat point);
    /*! \brief calculates the polynomial Mahalanobis distance between
               ref and each vector in points
        \param points a collection of vectors
        \param ref a single vector
        \return a N x 1 matrix containing the calculated distances
                where N is the number of vectors in points
    */
    Mat pointsTo(Mat points, Mat point);
    /*! \brief calculates the polynomial Mahalanobis distance between
               the _reference class member and each vector in points
        \param points a collection of vectors
        \return a N x 1 matrix containing the calculated distances
                where N is the number of vectors in points
    */
    Mat pointsToReference(Mat points);
    /*! \brief calculates the polynomial Mahalanobis distance between
               the _reference class member and each vector in points
               (written by PC)
        \param points a collection of vectors
        \return a N x 1 matrix containing the calculated distances
                where N is the number of vectors in points
    */
    Mat pointsToReferencePC(Mat points);
    /*! \brief transforms an image into a collection of vectors
               and calculates the polynomial Mahalanobis distance
               between each of the vectors and ref
        \param image an image
        \param ref a single vector
        \return an image where each pixel is the distance between the equivalent
                pixel in the image argument and ref (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    /*! \brief transforms an image into a collection of vectors
               and calculates the polynomial Mahalanobis distance between
               each of the vectors and the _reference class member
        \param image an image
        \return an image where each pixel is the distance between the equivalent
                pixel in the image argument and _reference (might need transformations
                before being properly visualized)
    */
    template <typename T> Mat imageToReference(Mat& image);
private:
    Mat _inputMatrix;
    Mat _reference;
    double _smin;
    int _l;
    int _dimension;
    int _numberOfPoints;
    int _polynomialDimension;
    vector<Mat> _expandedReferences;
    vector<Mat> _expandedUs;
    vector<MahalaDist> _expandedDists;
    vector<vector<int>> _indexesVector;
    vector<double> _maxAbsVector;
    MahalaDist _baseDist;
    bool _dirty;
    bool _setReference;
};

