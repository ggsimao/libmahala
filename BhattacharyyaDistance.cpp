#include "BhattacharyyaDistance.hpp"

BhattaDist::BhattaDist(const Mat& input) : _inputMatrix(input) {
    assert(_inputMatrix.data);
    assert(_inputMatrix.type() == CV_64FC1);

    _dimension = input.cols;
    _numberOfPoints = input.rows;

    calcCovarMatrix(_inputMatrix, _sigma, _mi, COVAR_COLS | COVAR_SCRAMBLED, _inputMatrix.type());
    _mi = Mat(1, _dimension, _inputMatrix.type());
    for (int i = 0; i < _dimension; i++) {
        _mi.at<double>(i) = (mean(_inputMatrix.col(i)))[0];
    }

    assert(_inputMatrix.data);
    assert(_inputMatrix.type() == CV_64FC1);
}

BhattaDist::BhattaDist() {}

BhattaDist::~BhattaDist() {}

/*------------------------------*/

int BhattaDist::dimension() {
    return _dimension;
}

double BhattaDist::pointsTo(Mat& points) {
    Mat mi, sigma;
    int numberOfPoints = points.rows;


    // // double preresult = 0;

    // // for (int i = 0; i < numberOfPoints; i++) {
    // //     double frequency_points = 0;
    // //     for (int j = 0; j < numberOfPoints; j++) {
    // //         if (countNonZero(points.row(i)-points.row(j)) == 0) {
    // //             frequency_points++;
    // //         }
    // //     }
    // //     frequency_points /= numberOfPoints;
    // //     double frequency_this = 0;
    // //     for (int j = 0; j < _numberOfPoints; j++) {
    // //         if (countNonZero(points.row(i)-_inputMatrix.row(j)) == 0) {
    // //             frequency_this++;
    // //         }
    // //     }
    // //     frequency_this /= _numberOfPoints;

    // //     preresult += sqrt(frequency_points*frequency_this);
    // // }

    // // return -log(preresult);
    // // return preresult;

    // mi = Mat::zeros(1, _dimension, CV_64FC1);
    // for (int i = 0; i < numberOfPoints; i++) {
    //     for (int j = 0; j < _dimension; j++) {
    //         mi.at<double>(j) += points.at<double>(i,j);
    //     }
    // }
    // mi /= numberOfPoints;

    // // sigma = Mat::zeros(1, _dimension, points.type());

    // // for (int i = 0; i < numberOfPoints; i++) {
    // //     for (int j = 0; j < _dimension; j++) {
    // //         sigma.at<double>(j) += pow(points.at<double>(i,j) - mi.at<double>(0,j), 2);
    // //     }
    // // }

    // // sigma /= numberOfPoints;

    // // Mat result = Mat(1, _dimension, CV_64FC1);
    // // for (int j = 0; j < _dimension; j++) {
    // //     double sigmai2 = _sigma.at<double>(j)*_sigma.at<double>(j);
    // //     double sigmaj2 = sigma.at<double>(j)*sigma.at<double>(j);
    // //     double diffmi2 = pow(_mi.at<double>(j) - mi.at<double>(j),2);
    // //     result.at<double>(j) = 1/4 * log(1/4 * (sigmai2/sigmaj2+sigmaj2/sigmai2+2))
    // //                          + 1/4 * (diffmi2/(sigmai2+sigmaj2));
    // // }
    // // return result;

    // // cout << "kkk" << endl;

    // sigma = Mat::zeros(points.rows, points.rows, points.type());
    // Mat a = Mat(points.size(), points.type());

    // // Mat refT = mi.t();
    // for (int i = 0; i < numberOfPoints; i++) {
    //     a.row(i) = points.row(i) - mi;
    // }

    // mi = mi.t();

    // sigma = (a.t() * a) / (numberOfPoints);
    // cout << "_inputMatrix size: " << _inputMatrix.size() << endl;
    // cout << "_sigma size: " << _sigma.size() << endl;
    // cout << "_mi size: " << _mi.size() << endl;
    // cout << "points size:" << points.size() << endl;

    calcCovarMatrix(points, sigma, mi, COVAR_COLS | COVAR_SCRAMBLED, points.type());
    // cout << "sigma size: " << sigma.size() << endl;

    mi = Mat(1, _dimension, points.type());
    for (int i = 0; i < _dimension; i++) {
        mi.at<double>(i) = (mean(points.col(i)))[0];
    }

    // cout << "mi size: " << mi.size() << endl;

    Mat diff_mi = _mi - mi;
    Mat sigma_avg_inv = ((_sigma + sigma) / 2).inv();
    // cout << "diff_mi size: " << diff_mi.size() << endl;
    // cout << "sigma_avg_inv size: " << sigma_avg_inv.size() << endl;
    Mat first_term = diff_mi * sigma_avg_inv * diff_mi.t() / 8;
    // cout << "kkk" << endl;
    double det_sigma_average = determinant(_sigma + sigma / 2);
    double sqrt_sigma_dets = sqrt(determinant(_sigma))*sqrt(determinant(sigma));
    double second_term = log(det_sigma_average / sqrt_sigma_dets) / 2;
    return first_term.at<double>(0,0) + second_term;
}

double BhattaDist::metricTo(BhattaDist& dist) {
    Mat diff_mi = _mi - dist.mi();
    Mat first_term = diff_mi.t() * (_sigma + dist.sigma() / 2).inv() * diff_mi / 8;
    double det_sigma_average = determinant(_sigma + dist.sigma() / 2);
    double sqrt_sigma_dets = sqrt(determinant(_sigma)*determinant(dist.sigma()));
    double second_term = log(det_sigma_average / sqrt_sigma_dets) / 2;
    return first_term.at<double>(0,0) + second_term;
}

template <typename T> double BhattaDist::imageTo(Mat& image) {
    // Mat result = Mat(image.size(), CV_64FC1);
    int numberOfChannels = image.channels();

    Mat linearized = linearizeImage<T>(image);

    double result = pointsTo(linearized);

    return result;
}

/*------------------------------*/

Mat BhattaDist::mi() {
    return _mi;
}

Mat BhattaDist::sigma() {
    return _sigma;
}

template double BhattaDist::imageTo<uchar>(Mat& image);
template double BhattaDist::imageTo<schar>(Mat& image);
template double BhattaDist::imageTo<ushort>(Mat& image);
template double BhattaDist::imageTo<short>(Mat& image);
template double BhattaDist::imageTo<int>(Mat& image);
template double BhattaDist::imageTo<float>(Mat& image);
template double BhattaDist::imageTo<double>(Mat& image);