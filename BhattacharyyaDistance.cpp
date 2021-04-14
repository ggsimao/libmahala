#include "BhattacharyyaDistance.hpp"

BhattaDist::BhattaDist(vector<int> channels, vector<int> histSize, vector<float> ranges)
    : _channels(channels), _histSize(histSize), _ranges(ranges) {}

BhattaDist::BhattaDist() {}

BhattaDist::~BhattaDist() {}

vector<int> BhattaDist::channels() {
    return _channels;
}

vector<int> BhattaDist::histSize() {
    return _histSize;
}

vector<float> BhattaDist::ranges() {
    return _ranges;
}

template <typename T> double BhattaDist::calcBetweenPoints(Mat& points1, Mat& points2) {
    Mat img1 = delinearizeImage<T>(points1, points1.rows, 1);
    Mat img2 = delinearizeImage<T>(points2, points2.rows, 1);
    return calcBetweenImg(img1, img2);
}

double BhattaDist::calcBetweenImg(const Mat& image1, const Mat& image2, 
                                  const Mat& mask1, const Mat& mask2) {
    assert(image1.channels() == _channels.size());
    assert(image2.channels() == _channels.size());
    vector<Mat> imageVec1, imageVec2;
    imageVec1.push_back(image1);
    imageVec2.push_back(image2);
    Mat hist1, hist2;
    cv::calcHist(imageVec1, _channels, mask1, hist1, _histSize, _ranges);
    cv::calcHist(imageVec2, _channels, mask2, hist2, _histSize, _ranges);
    return calcBetweenHist(hist1, hist2);
}

double BhattaDist::calcBetweenHist(const Mat &hist1, const Mat &hist2) {
    double hellinger = compareHist(hist1, hist2, HISTCMP_BHATTACHARYYA);
    double bhatCoeff = -(hellinger * hellinger) + 1;
    return -log(bhatCoeff);
}

template double BhattaDist::calcBetweenPoints<uchar>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<schar>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<ushort>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<short>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<int>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<float>(Mat& points1, Mat& points2);
template double BhattaDist::calcBetweenPoints<double>(Mat& points1, Mat& points2);