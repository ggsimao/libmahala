#include "DummyMahalanobisDistance.hpp"

DummyMahalaDist::DummyMahalaDist(const Mat& input, Mat reference)
    : _reference(reference)
{
    assert(input.data);
    assert(input.type() == CV_64FC1);

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

    Mat c = a.t() * a / (_numberOfPoints - 1);

    _cInv = c.inv();
}

DummyMahalaDist::DummyMahalaDist() {}

DummyMahalaDist::~DummyMahalaDist() {}

/*------------------------------*/

double DummyMahalaDist::pointTo(Mat& point1, Mat& point2) {
    // assert(!_dirty);
    assert(point1.rows == _dimension && point2.rows == _dimension);
    assert(point1.cols == 1 && point2.cols == 1);
    assert(point1.type() == CV_64FC1 && point2.type() == CV_64FC1);
    Mat point1T = point1.t();
    return pointsTo(point1T, point2).at<double>(0);
}

double DummyMahalaDist::pointToReference(Mat& point) {
    assert(point.rows == _dimension);
    assert(point.cols == 1);
    assert(point.type() == CV_64FC1);
    Mat pointT = point.t();
    return pointsTo(pointT, _reference).at<double>(0);
}

Mat DummyMahalaDist::pointsTo(Mat& points, Mat& ref) {
    assert(points.cols == ref.rows);
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);
    assert(ref.type() == CV_64FC1);

    Mat result = Mat(points.rows, 1, CV_64FC1);

    Mat refT = ref.t();

    // cout << "_cInv: " << endl << _cInv << endl;
    // cout << "refT: " << endl << refT << endl;
    
    for (int i = 0; i < points.rows; i++) {
        // cout << "points.row("<< i << "): " << endl << points.row(i) << endl;
        result.row(i) = Mahalanobis(points.row(i), refT, _cInv);
    }
    return result;
}

Mat DummyMahalaDist::pointsToReference(Mat& points) {
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);
    return pointsTo(points, _reference);
}

template <typename T> Mat DummyMahalaDist::imageTo(Mat& image, Mat& ref) {

    Mat linearized = linearizeImage<T>(image);
    linearized.convertTo(linearized, CV_64FC1);
    
    Mat distMat = pointsTo(linearized, ref);

    Mat result = delinearizeImage<double>(distMat, image.rows, image.cols);

    return result;
}

template <typename T> Mat DummyMahalaDist::imageToReference(Mat& image) {
    return imageTo<T>(image, _reference);
}

template Mat DummyMahalaDist::imageTo<uchar>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<schar>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<ushort>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<short>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<int>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<float>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageTo<double>(Mat& image, Mat& ref);
template Mat DummyMahalaDist::imageToReference<uchar>(Mat& image);
template Mat DummyMahalaDist::imageToReference<schar>(Mat& image);
template Mat DummyMahalaDist::imageToReference<ushort>(Mat& image);
template Mat DummyMahalaDist::imageToReference<short>(Mat& image);
template Mat DummyMahalaDist::imageToReference<int>(Mat& image);
template Mat DummyMahalaDist::imageToReference<float>(Mat& image);
template Mat DummyMahalaDist::imageToReference<double>(Mat& image);