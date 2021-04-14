#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;

/*! \brief Transforma uma matriz de C canais, N linhas e M colunas em uma matriz
           de (N*M) linhas e C colunas
    \param image A matriz a ser transformada
    \return A matriz linearizada
*/
template <typename T> Mat linearizeImage(Mat& image) {
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

/*! \brief Transforma uma matriz de 1 canal, (N*M) linhas e C colunas em uma matriz
           de C canais, N linhas e M colunas
    \param image A matriz a ser transformada
    \param rows Número de linhas da matriz resultante
    \param rows Número de colunas da matriz resultante
    \return A matriz delinearizada
*/
template <typename T> Mat delinearizeImage(Mat& linearized, int rows, int cols) {
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