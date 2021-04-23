#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

/*! \brief Cria uma janela interativa com uma imagem na qual o usuário pode clicar para
           extrair os pixels (coordenadas) e seus valores (apenas uma vez por coordenada)
    \param collectedPixels Valores dos pixels extraídos
                           em que cada linha é um pixel e cada coluna é um canal (somente leitura)
    \param collectedCoordinates Coordenadas dos pixels extraídos, em que cada linha é um pixel (somente leitura)
    \param referencePixel Valores do pixel central (somente leitura)
    \param referenceCoordinate Coordenadas do pixel central (somente leitura)
*/
class PointCollector {
public:
    /*! \brief Construtor que gera e mantém uma janela interativa
        \param input Imagem que será exibida para usuário extrair pontos
    */
    PointCollector(Mat& input);
    /*! \brief Construtor que gera e mantém uma janela interativa
        \param path Caminho para um arquivo contendo uma imagem que será exibida para usuário extrair pontos
        \param flags flags usadas pela função de leitura de arquivo do OpenCV
    */
    PointCollector(const char* path, int flags);
    PointCollector();
    virtual ~PointCollector();

    /*
     * GETTERS
     */
    Mat& collectedPixels();
    Mat& collectedCoordinates();
    Mat& referencePixel();
    Mat& referenceCoordinate();
private:
    Mat _collectedPixels;
    Mat _collectedCoordinates;
    Mat _referencePixel;
    Mat _referenceCoordinate;

    /*! !brief Struct used to pass variables back-and-forth between the constructor
               and the onMouse callback method
        !param img The input image
        !param paintedImg The input image with the selected points highlighted
        !param mask Keeps track of which pixels were already extracted
        !param pixels The values of the collected pixels organized in a NxD matrix
                      where N is the number of collected points and D is the number
                      of channels in the image
        !param coordinates The row and column indexes of the collected pixels organized
                           in a Nx2 matrix where N is the number of collected points;
                           the row index is stored in the second column of the matrix
                           while the column index is stored in the first column
        !param pressedLeft Determines whether the left mouse button is pressed
        !param pressedRight Determines whether the right mouse button is pressed
        !param referencePixel The pixel values of the single point chosen with the right mouse button
        !param referenceCoordinate The row and column indexes of the single point chosen with
                                   the right mouse button
        !param color Color in which the selected points will be painted on paintedImg
    */
    struct CallbackParams {
        Mat& img;
        Mat& paintedImg;
        Mat& mask;
        Mat& pixels;
        Mat& coordinates;
        bool& pressedLeft;
        bool& pressedRight;
        Mat& referencePixel;
        Mat& referenceCoordinate;
        Scalar& color;
    };

    static void redraw(const Mat& originalImage, Mat& imageToPaint, const Mat& points, const Mat& reference, const Scalar& color);

    static void onMouse(int event, int x, int y, int flags, void* param);
};