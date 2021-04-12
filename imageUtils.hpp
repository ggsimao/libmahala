#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

/*! \brief Transforms a C-channel NxM matrix into a single-channel (N*M)xC matrix.
           This is useful for functions in this library that interpret collections
           of points as NxD matrices, where N is the number of points and D is the
           number of dimensions.
    \param image The matrix to be linearized
    \return The linearized matrix
*/
template <typename T> Mat linearizeImage(Mat& image) {
    cout << "linearizing " << image.rows << "x" << image.cols << " into " << image.rows * image.cols << endl;
    int numberOfChannels = image.channels();

    Mat linearized = Mat(numberOfChannels, image.rows * image.cols, image.type() % 8);
    Mat bgrArray[numberOfChannels];
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

/*! \brief Transforms a single-channel (N*M)xC matrix into a C-channels NxM matrix.
    \param linearized The matrix to be delinearized
    \param rows The number of rows the result must have
    \param cols The number of cols the result must have
    \return The delinearized matrix
*/
template <typename T> Mat delinearizeImage(Mat& linearized, int rows, int cols) {
    cout << "delinearizing " << linearized.rows * linearized.cols << " into " << rows << "x" << cols << endl;
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
        // cout << a.size() << endl;
        channels.push_back(a.clone());
    }

    linearized = linearized.t();

    merge(channels, result);

    return result;
}