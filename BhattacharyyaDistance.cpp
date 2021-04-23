#include "BhattacharyyaDistance.hpp"

BhattaDist::BhattaDist(vector<int> channels, vector<int> histSize, vector<float> ranges)
    : _channels(channels), _histSize(histSize), _ranges(ranges) {
        assert(channels.size() == histSize.size());
        assert(2*channels.size() == ranges.size());
}

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

double BhattaDist::calcBetweenPoints(Mat& points1, Mat& points2) {
    Mat img1 = delinearizeImage<double>(points1, points1.rows, 1);
    Mat img2 = delinearizeImage<double>(points2, points2.rows, 1);
    return calcBetweenImg(img1, img2);
}

double BhattaDist::calcBetweenImg(const Mat& image1, const Mat& image2, 
                                  const Mat& mask1, const Mat& mask2) {
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

// *! \brief Transforma uma matriz de C canais, N linhas e M colunas em uma matriz
//            de (N*M) linhas e C colunas
//     \param image A matriz a ser transformada
//     \return A matriz linearizada
// *//
template <typename T> Mat BhattaDist::linearizeImage(Mat& image) {
    int numberOfChannels = image.channels();

    Mat linearized = Mat(numberOfChannels, image.rows * image.cols, image.type() % 8);
    vector<Mat> bgrArray;
    split(image, bgrArray);
    
    for (int c = 0; c < numberOfChannels; c++) {
        Mat a = bgrArray[c];
        for (int i = 0; i < image.rows; i++) {
            for (int j = 0; j < image.cols; j++) {
                int currentIndex = i*image.cols + j;
                linearized.at<T>(c, currentIndex) = (double)a.at<T>(i,j);
            }
        }
    }

    linearized = linearized.t();

    return linearized;
}

// /*! \brief Transforma uma matriz de 1 canal, (N*M) linhas e C colunas em uma matriz
//            de C canais, N linhas e M colunas
//     \param image A matriz a ser transformada
//     \param rows Número de linhas da matriz resultante
//     \param rows Número de colunas da matriz resultante
//     \return A matriz delinearizada
// */
template <typename T> Mat BhattaDist::delinearizeImage(Mat& linearized, int rows, int cols) {
    assert(linearized.rows == rows*cols);
    int numberOfChannels = linearized.cols;

    Mat result;
    vector<Mat> channels;
    linearized = linearized.t();

    Mat a = Mat(rows, cols, linearized.type());
    for (int c = 0; c < numberOfChannels; c++) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                int currentIndex = i*cols + j;
                a.at<T>(i, j) = linearized.at<T>(c, currentIndex);
            }
        }
        channels.push_back(a.clone());
    }

    linearized = linearized.t();

    merge(channels, result);

    return result;
}