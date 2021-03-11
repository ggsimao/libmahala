#include "MahalanobisDistance.hpp"

MahalaDist::MahalaDist(const Mat& input, double smin, Mat reference)
    : _inputMatrix(input), _smin(smin), _reference(reference)
{
    assert(_inputMatrix.data);
    assert(_inputMatrix.type() == CV_64FC1);

    _u = 0;
    _w = 0;
    _sigma2 = 0;

    _k = 0;
    _dimension = input.cols;
    _numberOfPoints = input.rows;
    if(!_reference.data){
        _reference = Mat::zeros(_dimension, 1, CV_64FC1);

        for (int i = 0; i < _numberOfPoints; i++) {
            for (int j = 0; j < _dimension; j++) {
                _reference.at<double>(j) += _inputMatrix.at<double>(i,j);
            }
        }
        _reference /= _numberOfPoints;
    }

    Mat discard;
    _a = Mat(_inputMatrix.size(), _inputMatrix.type());

    double sum = 0;

    Mat refT = _reference.t();
    for (int i = 0; i < _numberOfPoints; i++) {
        _a.row(i) = _inputMatrix.row(i) - refT;
    }

    if (_dimension < _numberOfPoints) {
        _c = (_a.t() * _a) / (_numberOfPoints - 1);
    } else {
        _c = (_a * _a.t()) / (_numberOfPoints - 1);
    }
    SVD::compute(_c, _w, _u, discard);
    discard.release();

    _dirty = 1;
    _setReference = 1;
    assert(_inputMatrix.data);
    assert(_inputMatrix.type() == CV_64FC1);
}

MahalaDist::MahalaDist() {}

MahalaDist::~MahalaDist() {}

/*------------------------------*/

Mat MahalaDist::inputMatrix() {
    return _inputMatrix;
}

Mat MahalaDist::reference() {
    return _reference;
}

double MahalaDist::smin() {
    return _smin;
}

int MahalaDist::dimension() {
    return _dimension;
}

bool MahalaDist::dirty() {
    return _dirty;
}

/*------------------------------*/

const Mat MahalaDist::u() const {
    assert(!_dirty);

    return _u;
}

Mat MahalaDist::w() {
    assert(!_dirty);

    return _w;
}

Mat MahalaDist::c() {
    assert(!_dirty);

    return _c;
}

double MahalaDist::w(int k) {
    assert(!_dirty);

    return _w.at<double>(k);
}

double MahalaDist::c(int k) {
    assert(!_dirty);

    return _c.at<double>(k,k);
}

double MahalaDist::sigma2() {
    assert(!_dirty);

    return _sigma2;
}

/*------------------------------*/

void MahalaDist::setSmin(double smin) {
    _smin = smin;
    _dirty = 1;
}

void MahalaDist::build() {
    if (!_dirty) return;

    double w0 = _w.at<double>(0);
    _sigma2 = _smin * w0;

    for(int k = 0; k < _w.rows; k++) {
        if (_w.at<double>(k) >= _sigma2) {
            _k++;
        }
        else{
            _w.at<double>(k) = 0;
        }
    }
    if (_dimension < _numberOfPoints) {
        _u = Mat(_u, Rect(0,0, _k, _u.rows)).clone();
    } else {
        Mat b = (_u.t() * _a).t();
        for (int k = 0; k < _k; k++) {
            b.col(k) /= cv::norm(b.col(k));
        }
        _u = Mat(b, Rect(0,0, _k, b.rows)).clone();
    }
    
    if (_dimension < _numberOfPoints) {
        assert(_k <= _dimension);
        assert(_u.cols == _k);
        assert(_w.rows == _dimension && _w.cols == 1);
    } else {
        assert(_k <= _numberOfPoints);
        assert(_u.cols == _k);
        assert(_w.rows == _numberOfPoints && _w.cols == 1);
    }
    _dirty = 0;
}

/*------------------------------*/

double MahalaDist::pointTo(Mat& point1, Mat& point2) {
    assert(!_dirty);
    assert(point1.cols == point2.cols && point1.cols == 1);
    assert(point1.rows == point2.rows && point1.rows == _dimension);

    Mat diff = point1-point2;
    double diffSquareSum = cv::sum(diff.mul(diff))[0];
    double ksum = 0;

    Mat proj;
    for (int k = 0; k < _k; k++) {
        proj = _u.col(k).t() * diff;
        double squareSum = cv::sum(proj.mul(proj))[0];
        ksum += (-w(k) * squareSum) / (_sigma2 * (w(k) + _sigma2));
    }

    double distance = (diffSquareSum/_sigma2) + ksum;
    distance = sqrt(distance*(_numberOfPoints-1));

    return (distance < 0) ? 0 : distance;
}

double MahalaDist::pointToReference(Mat& point) {
    assert(point.rows == _dimension);
    assert(point.cols == 1);
    assert(point.type() == CV_64FC1);
    return pointTo(point, _reference);
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
    Mat rowDiffTemp;//matriz temporaria pra guardas a diferença de cada linha.
    for(int i = 0; i < points.rows; i++){
        rowDiffTemp = (points.row(i)-refT);
        rowDiffTemp.copyTo(diff.row(i));
        // diff.row(i) = (points.row(i)-refT);
    }
    rowDiffTemp.release();
    Mat proj = diff * (_u);
    diff = diff.mul(diff);
    for(int i = 0; i < diff.rows; i++){
        result.row(i) = cv::sum(diff.row(i));
    }
    result /= _sigma2;
    proj = proj.mul(proj);
    assert(proj.cols == _k);
    for(int k = 0; k < _k; k++) {
        proj.col(k) *= ((-w(k))/(_sigma2 * (w(k) + _sigma2)));
    }
    for(int i = 0; i < proj.rows; i++){
        result.row(i) += cv::sum(proj.row(i));
        result.row(i) *= (_numberOfPoints-1);
        result.row(i) = Mat(1,1,CV_64FC1,{pow(result.at<double>(i), 1/2)});
    }
    return result;
}

Mat MahalaDist::pointsToReference(Mat& points) {
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);
    return pointsTo(points, _reference);
}

template <typename T> Mat MahalaDist::imageTo(Mat& image, Mat& ref) {
    assert(!_dirty);

    Mat result = Mat(image.size(), CV_64FC1);
    int numberOfChannels = image.channels();

    Mat linearized = Mat(image.rows * image.cols, numberOfChannels, CV_64FC1);
    Mat distMat = Mat(image.rows * image.cols, 1, CV_64FC1);
    Mat bgrArray[numberOfChannels];
    split(image, bgrArray);
    
    for (int c = 0; c < numberOfChannels; c++) {
        Mat a = bgrArray[c];
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int currentIndex = i*image.cols + j;
                linearized.at<double>(currentIndex, c) = (double)a.at<T>(i,j);
            }
        }
    }

    distMat = pointsTo(linearized, ref);
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int currentIndex = i*image.cols + j;
            for (int c = 0; c < numberOfChannels; c++) {
                result.at<double>(i, j) = distMat.at<double>(currentIndex, c);
            }
        }
    }

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