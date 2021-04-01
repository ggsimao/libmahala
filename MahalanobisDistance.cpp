#include "MahalanobisDistance.hpp"

MahalaDist::MahalaDist(const Mat& input, double smin, Mat reference)
    : _smin(smin), _reference(reference)
{
    assert(input.data);
    assert(input.type() == CV_64FC1);

    _u = 0;
    _w = 0;
    _sigma2 = 0;

    _dimension = input.cols;
    _numberOfPoints = input.rows;


    if(!_reference.data){
        _reference = Mat::zeros(_dimension, 1, CV_64FC1);

        for (int i = 0; i < _dimension; i++) {
            _reference.at<double>(i) += (mean(input.col(i)))[0];
        }
    }

    Mat a = Mat(input.size(), input.type());

    Mat refT = _reference.t();
    for (int i = 0; i < _numberOfPoints; i++) {
        a.row(i) = input.row(i) - refT;
    }

    /*
     * The following if section was supposed to make calculations more
     * efficient by reducing the size of the _c matrix, but it doesn't
     * work with the used formula
     */

    // if (_dimension < _numberOfPoints) {
        _c = (a.t() * a);
    // } else {
    //     _c = (_a * _a.t());
    // }
    // SVD::compute(_c, _w, _u, discard);
    Mat discard;
    SVD::compute(_c, _w, discard, _u);
    // discard.release();
    
    // _c is not directly used but it's still in the code for completeness' sake
    _c /= (_numberOfPoints - 1);

    // _cInv = _c.inv();

    _dirty = 1;
    assert(input.data);
    assert(input.type() == CV_64FC1);
}

MahalaDist::MahalaDist() {}

MahalaDist::~MahalaDist() {}

/*------------------------------*/

// Mat MahalaDist::inputMatrix() {
//     return _inputMatrix;
// }

Mat MahalaDist::reference() {
    return _reference;
}

double MahalaDist::smin() {
    return _smin;
}

int MahalaDist::dimension() {
    return _dimension;
}

// int MahalaDist::numberOfPoints() {
//     return _numberOfPoints;
// }

bool MahalaDist::dirty() {
    return _dirty;
}

/*------------------------------*/

const Mat MahalaDist::u() const {
    return _u;
}

// const Mat MahalaDist::uK() const {
//     return _uK;
// }

Mat MahalaDist::w() {
    return _w;
}

Mat MahalaDist::c() {
    return _c;
}

// double MahalaDist::w(int k) {
//     return _w.at<double>(k);
// }

// double MahalaDist::wSigma2(int k) {
//     assert(!_dirty);

//     return _wSigma2.at<double>(k);
// }


// double MahalaDist::c(int k) {
//     return _c.at<double>(k,k);
// }

double MahalaDist::sigma2() {
    assert(!_dirty);

    return _sigma2;
}

/*------------------------------*/

void MahalaDist::smin(double smin) {
    _smin = smin;
    _dirty = 1;
}

void MahalaDist::build() {
    if (!_dirty) return;

    double w0 = _w.at<double>(0);
    _sigma2 = _smin * w0;


    /*
     * Old way of doing Mat wSigma2 = _w + _sigma2;
     */
    // for(int k = 0; k < _w.rows; k++) {
    //     if (_w.at<double>(k) >= _sigma2) {
    //         // _w.at<double>(k) += _sigma2;
    //         _k++;
    //     }
    //     else{
    //         // _w.at<double>(k) = 0;
    //     }
    // }

    /*
     * The following if sections were supposed to make calculations more
     * efficient by reducing the size of the _uK matrix, but they don't
     * work with the used formula
     */
    // if (_dimension < _numberOfPoints) {
        // _uK = Mat(_u, Rect(0,0, _k, _u.rows)).clone();
    // } else {
    //     Mat b = (_u.t() * _a).t();
    //     for (int k = 0; k < _k; k++) {
    //         b.col(k) /= cv::norm(b.col(k));
    //     }
    //     _uK = Mat(b, Rect(0,0, _k, b.rows)).clone();
    //     // _u = b.t();
    // }

    Mat wSigma2 = Mat::diag(_w + _sigma2);

    _cSigma2Inv = (_u.t() * wSigma2.inv() * _u) * (_numberOfPoints-1);
    
    // if (_dimension < _numberOfPoints) {
        // assert(_k <= _dimension);
        // assert(_uK.cols == _k);
        // assert(_w.rows == _dimension && _w.cols == 1);
    // } else {
        // assert(_k <= _numberOfPoints);
        // assert(_uK.cols == _k);
        // assert(_w.rows == _numberOfPoints && _w.cols == 1);
    // }
    _dirty = 0;
}

/*------------------------------*/

double MahalaDist::pointTo(Mat& point1, Mat& point2) {
    // assert(!_dirty);
    assert(point1.rows == _dimension && point2.rows == _dimension);
    assert(point1.cols == 1 && point2.cols == 1);
    assert(point1.type() == CV_64FC1 && point2.type() == CV_64FC1);
    Mat point1T = point1.t();
    return pointsTo(point1T, point2).at<double>(0);
}

double MahalaDist::pointToReference(Mat& point) {
    assert(point.rows == _dimension);
    assert(point.cols == 1);
    assert(point.type() == CV_64FC1);
    Mat pointT = point.t();
    return pointsTo(pointT, _reference).at<double>(0);
}

Mat MahalaDist::pointsTo(Mat& points, Mat& ref) {
    assert(!_dirty);
    assert(points.cols == ref.rows);
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);
    assert(ref.type() == CV_64FC1);

    Mat result = Mat::zeros(points.rows, 1, CV_64FC1);

    Mat diff = Mat(points.size(), CV_64FC1);
    Mat refT = ref.t();
    // Mat rowDiffTemp;//matriz temporaria pra guardas a diferença de cada linha.
    for(int i = 0; i < points.rows; i++){
        // rowDiffTemp = (points.row(i)-refT);
        // rowDiffTemp.copyTo(diff.row(i));
        diff.row(i) = (points.row(i)-refT);
    }
    
    for (int i = 0; i < points.rows; i++) {
        Mat diffrow = diff.row(i);
        result.row(i) = (diffrow * _cSigma2Inv * diffrow.t());
        // result.row(i) = (diffrow * _cInv * diffrow.t());
    }
    pow(result, 0.5, result);

    return result;
}

Mat MahalaDist::pointsToReference(Mat& points) {
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);
    return pointsTo(points, _reference);
}

template <typename T> Mat MahalaDist::imageTo(Mat& image, Mat& ref) {
    assert(!_dirty);

    Mat linearized = linearizeImage<T>(image);
    linearized.convertTo(linearized, CV_64FC1);
    
    Mat distMat = pointsTo(linearized, ref);

    Mat result = delinearizeImage<double>(distMat, image.rows, image.cols);

    return result;
}

template <typename T> Mat MahalaDist::imageToReference(Mat& image) {
    return imageTo<T>(image, _reference);
}

template Mat MahalaDist::imageTo<uchar>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<schar>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<ushort>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<short>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<int>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<float>(Mat& image, Mat& ref);
template Mat MahalaDist::imageTo<double>(Mat& image, Mat& ref);
template Mat MahalaDist::imageToReference<uchar>(Mat& image);
template Mat MahalaDist::imageToReference<schar>(Mat& image);
template Mat MahalaDist::imageToReference<ushort>(Mat& image);
template Mat MahalaDist::imageToReference<short>(Mat& image);
template Mat MahalaDist::imageToReference<int>(Mat& image);
template Mat MahalaDist::imageToReference<float>(Mat& image);
template Mat MahalaDist::imageToReference<double>(Mat& image);