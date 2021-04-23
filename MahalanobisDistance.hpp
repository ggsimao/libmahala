#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

// #include <libmahala>

using namespace cv;
using namespace std;

/*! \brief Classe que contém os parâmetros necessários para calcular a distância de
           Mahalanobis entre pontos e realiza o cálculo
    \param c A matriz de covariância da vizinhança de pontos (somente leitura)
    \param u Matriz lateral resultante de uma operação de SVD em C * (N-1), em que
             C é a matriz de covariância associada à classe e N é a cardinalidade
             da vizinhança de pontos (somente leitura)
    \param w Matriz central resultante de uma operação de SVD em C * (N-1), em que
             C é a matriz de covariância associada à classe e N é a cardinalidade
             da vizinhança de pontos (somente leitura)
    \param reference Centro da métrica (média da vizinhança de pontos, por padrão) (somente leitura)
    \param cSigma2Inv Matriz usada para realizar o cálculo da distância no lugar da
                      inversa da matriz de covariância da vizinhança de pontos C, calculada
                      a partir de C e smin (somente leitura)
    \param smin Parâmetro do usuário, usado para garantir a inversão da matriz de
                covariância da vizinhança de pontos C, alterando o resultado
                proporcionalmente ao seu valor (leitura e escrita)
    \param sigma2 Valor derivado de smin, usado diretamente para o cálculo de cSigma2Inv (somente leitura)
    \param dimension Número de dimensões dos pontos usados para construir a métrica (somente leitura)
    \param dirty Indica se o método build() precisa ser chamado antes de o objeto poder
                 realizar o cálculo da distância (somente leitura)
*/
class MahalaDist {
public:
    /*! \brief Construtor
        \param input Matriz da vizinhança de pontos, em que cada linha é um ponto
                     e cada coluna uma dimensão
        \param smin Parâmetro usado para garantir a inversão da matriz de
                    covariância da vizinhança de pontos C, alterando o resultado
                    proporcionalmente ao seu valor
        \param reference Centro da métrica (média da vizinhança de pontos, por padrão)
                         na forma de uma matriz coluna
    */
    MahalaDist(const Mat& input, double smin = 4e-6, Mat reference = Mat());
    MahalaDist();
    virtual ~MahalaDist();

    // getters
    Mat reference();
    double smin();
    int dimension();
    bool dirty();
    const Mat u() const;
    Mat w();
    Mat c();
    Mat cSigma2Inv();
    double sigma2();

    // setter
    void smin(double smin);

    /*! \brief Calcula os parâmetros necessários para o cálculo
               da distância que são dependentes do atributo smin
    */
    void build();

    /*! \brief Calcula a distância de Mahalanobis entre dois pontos
        \param point1 Um ponto na forma de uma matriz coluna
        \param point2 Um ponto na forma de uma matriz coluna
        \return O valor da distância de Mahalanobis entre os pontos
    */
    double pointTo(Mat& point1, Mat& point2);
    /*! \brief Calcula a distância de Mahalanobis entre um ponto e o centro da métrica
        \param point Um ponto na forma de uma matriz coluna
        \return O valor da distância de Mahalanobis entre os pontos
    */
    double pointToReference(Mat& point);
    /*! \brief Calcula a distância de Mahalanobis entre um conjunto de pontos e um único ponto
        \param points Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                      coluna é uma dimensão
        \param ref Um único ponto na forma de uma matriz coluna
        \return Matriz de todos os valores da distância entre o ponto correspondente
                no conjunto de ponto passada como argumento e o ponto singular de referência
    */ 
    Mat pointsTo(Mat& points, Mat& ref);
    /*! \brief Calcula a distância de Mahalanobis entre um conjunto de pontos e o centro da métrica
        \param points Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                      coluna é uma dimensão
        \return Matriz de todos os valores da distância entre o ponto correspondente
                no conjunto de ponto passada como argumento e o centro da métrica
    */ 
    Mat pointsToReference(Mat& points);
    /*! \brief Transforma uma imagem em um conjunto de pontos e calcula a distância
               de Mahalanobis entre ela e um ponto de referência
        \tparam T O mesmo tipo que seria usado no método at<T>() da classe cv::Mat para acessar um elemento de image
        \param image Uma imagem
        \param ref Um ponto na forma de uma matriz coluna
        \return Uma imagem em que o valor de cada pixel é a distância de Mahalanobis entre
                o pixel correspondente na imagem passada como argumento e o ponto de referência
    */
    template <typename T> Mat imageTo(Mat& image, Mat& ref);
    /*! \brief Transforma uma imagem em um conjunto de pontos e calcula a distância
               de Mahalanobis entre ela e o centro da métrica
        \tparam T O mesmo tipo que seria usado no método at<T>() da classe cv::Mat para acessar um elemento de image
        \param image Uma imagem
        \return Uma imagem em que o valor de cada pixel é a distância de Mahalanobis entre
                o pixel correspondente na imagem passada como argumento e o centro da métrica
    */
    template <typename T> Mat imageToReference(Mat& image);
private:
    Mat _c;
    Mat _u;
    Mat _w;
    Mat _reference;
    Mat _cSigma2Inv;
    double _smin;
    double _sigma2;
    int _dimension;
    int _numberOfPoints;
    bool _dirty;

    template <typename T> Mat linearizeImage(Mat& image);
    template <typename T> Mat delinearizeImage(Mat& linearized, int rows, int cols);
};
