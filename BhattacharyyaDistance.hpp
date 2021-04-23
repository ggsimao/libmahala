#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

/*! \brief Classe que contém os parâmetros necessários para transformar matrizes de vizinhanças
           de pontos em histogramas para permitir o cálculo da distância de Bhattacharyya entre elas
    \param channels Dimensões dos pontos a serem consideradas (somente leitura)
    \param histSize Vetor de tamanho do histograma para cada dimensão (somente leitura)
    \param ranges Vetor com os limites inferiores e superiores dos valores para cada dimensão (somente leitura)
*/
class BhattaDist {
public:
    /*! \brief Construtor
        \param channels Dimensões dos pontos a serem consideradas
        \param histSize Vetor de tamanho do histograma para cada dimensão
        \param ranges Vetor com os limites inferiores e superiores dos valores para cada dimensão
    */
    BhattaDist(vector<int> channels, vector<int> histSize, vector<float> ranges);
    BhattaDist();
    virtual ~BhattaDist();

    // getters (need to be updated)
    vector<int> channels();
    vector<int> histSize();
    vector<float> ranges();

    /*! \brief Calcula a distância de Bhattacharyya entre dois conjuntos de pontos
        \param points1 Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                       coluna é uma dimensão
        \param points2 Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                       coluna é uma dimensão
        \return O valor da distância de Bhattacharyya entre os dois conjuntos de pontos
    */
    double calcBetweenPoints(Mat& points1, Mat& points2);
    /*! \brief Calcula a distância de Bhattacharyya entre duas imagens
        \param image1 Uma imagem
        \param image2 Uma imagem
        \param mask1 Máscara a ser aplicada na primeira imagem a fim de filtrar quais pontos
                     serão escolhidos para construir o histograma
        \param mask2 Máscara a ser aplicada na segunda imagem a fim de filtrar quais pontos
                     serão escolhidos para construir o histograma
        \return O valor da distância de Bhattacharyya entre as duas imagens
    */
    double calcBetweenImg(const Mat& image1, const Mat& image2, const Mat& mask1 = Mat(), const Mat& mask2 = Mat());

    /*! \brief Calcula a distância de Bhattacharyya entre dois histogramas
        \param hist1 Um histograma
        \param hist2 Um histograma
        \return O valor da distância de Bhattacharyya entre os dois histogramas
    */
    static double calcBetweenHist(const Mat &hist1, const Mat &hist2);

private:
    vector<int> _channels;
    vector<int> _histSize;
    vector<float> _ranges;

    template <typename T> Mat linearizeImage(Mat& image);
    template <typename T> Mat delinearizeImage(Mat& linearized, int rows, int cols);
};