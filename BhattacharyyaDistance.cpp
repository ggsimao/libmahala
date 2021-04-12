#include "BhattacharyyaDistance.hpp"

BhattaDist::BhattaDist(const Mat& input, double smin)
    : _smin(smin)
{
    assert(input.data);
    assert(input.type() == CV_64FC1);

    Mat inputUnique = input;

    // for (int i = 1; i < input.rows; ++i) {
    //     int isInside = false;
    //     for (int j = 0; j < inputUnique.rows; ++j) {
    //         int count = 0;
    //         for (int k = 0; k < inputUnique.cols; ++k) // checks by element of 
    //             if(input.at<double>(i,k) == inputUnique.at<double>(j,k)) 
    //                 ++count;
    //         if (count == input.cols) {
    //             isInside = true;
    //             break;
    //         }   
    //     }
    //     if (isInside == false) inputUnique.push_back( input.row(i) );
    // }

    _dimension = input.cols;
    _numberOfPoints = inputUnique.rows;


    _mi = Mat::zeros(_dimension, 1, CV_64FC1);

    for (int i = 0; i < _dimension; i++) {
        _mi.at<double>(i) += (mean(inputUnique.col(i)))[0];
    }

    Mat a = Mat(inputUnique.size(), input.type());

    Mat refT = _mi.t();
    for (int i = 0; i < _numberOfPoints; i++) {
        a.row(i) = inputUnique.row(i) - refT;
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
    Mat discard;
    SVD::compute(_c, _w, discard, _u);
    // SVD::compute(_c, _w, _u, discard);
    // discard.release();
    
    // _c is not directly used but it's still in the code for completeness' sake
    _c /= (_numberOfPoints - 1);

    // _cInv = _c.inv();

    assert(input.data);
    assert(input.type() == CV_64FC1);

    _dirty = 1;
}

BhattaDist::BhattaDist() {}

BhattaDist::~BhattaDist() {}

/*------------------------------*/

// Mat BhattaDist::inputMatrix() {
//     return _inputMatrix;
// }

Mat BhattaDist::mi() {
    return _mi;
}

double BhattaDist::smin() {
    return _smin;
}

int BhattaDist::dimension() {
    return _dimension;
}

Mat BhattaDist::c() {
    return _c;
}


bool BhattaDist::dirty() {
    return _dirty;
}

const Mat BhattaDist::u() const {
    return _u;
}

Mat BhattaDist::w() {
    assert(!_dirty);
    return _w;
}

Mat BhattaDist::cSigma2Inv() {
    assert(!_dirty);
    return _cSigma2Inv;
}

double BhattaDist::sigma2() {
    assert(!_dirty);
    return _sigma2;
}

/*------------------------------*/

void BhattaDist::smin(double smin) {
    _smin = smin;
    _dirty = 1;
}

void BhattaDist::build() {
    if (!_dirty) return;

    double w0 = _w.at<double>(0);
    _sigma2 = _smin * w0;

    Mat wSigma2 = Mat::diag(_w + _sigma2);

    _cSigma2Inv = (_u.t() * wSigma2.inv() * _u) * (_numberOfPoints-1);
    
    _dirty = 0;
}

/*------------------------------*/

double BhattaDist::pointsTo(Mat& points) {
    BhattaDist metric = BhattaDist(points, _smin);
    metric.build();
    return metricTo(metric);
}

double BhattaDist::metricTo(BhattaDist& dist) {
    assert(!_dirty);

    Mat c2 = dist.c(), cSigma2Inv2 = dist.cSigma2Inv();
    Mat mi2 = dist.mi(), miDiff = _mi - mi2;

    // cout << c2 << endl;
    // cout << _c << endl;
    // cout << c2 << endl;
    Mat prod1 = cSigma2Inv2 * _c;
    Mat prod2 = _cSigma2Inv * c2;
    Mat eye2 = 2 * Mat::eye(_dimension, _dimension, CV_64FC1);
    double det = determinant(prod1 + prod2 + eye2);
    cout << eye2 << endl;

    double result = log(det/4)/4;
    // Mat diffMult = miDiff*miDiff.t();
    // result = trace((_c+c2).inv()*diffMult)[0] / 4;

    return result;
}

template <typename T> double BhattaDist::imageTo(Mat& image) {
    // Mat result = Mat(image.size(), CV_64FC1);
    int numberOfChannels = image.channels();

    Mat linearized = linearizeImage<T>(image);
    linearized.convertTo(linearized, CV_64FC1);

    double result = pointsTo(linearized);

    return result;
}

template double BhattaDist::imageTo<uchar>(Mat& image);
template double BhattaDist::imageTo<schar>(Mat& image);
template double BhattaDist::imageTo<ushort>(Mat& image);
template double BhattaDist::imageTo<short>(Mat& image);
template double BhattaDist::imageTo<int>(Mat& image);
template double BhattaDist::imageTo<float>(Mat& image);
template double BhattaDist::imageTo<double>(Mat& image);