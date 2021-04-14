#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MahalanobisDistance.hpp"
#include "imageUtils.hpp"
#include "mathUtils.hpp"

#include <iostream>
#include <vector>
#include <math.h>
// #include <omp.h>

using namespace cv;
using namespace std;

/*! \brief Classe que contém os parâmetros necessários para calcular a distância de
           Mahalanobis polinomial entre pontos e realiza o cálculo
    \param reference Centro da métrica (média da vizinhança de pontos, por padrão) (somente leitura)
    \param eps_svd Parâmetro do usuário, usado para garantir a inversão da matriz de
                   covariância da vizinhança de pontos C, alterando o resultado
                   proporcionalmente ao seu valor (leitura e escrita)
    \param order Ordem dos termos polinomiais (somente leitura)
    \param dimension Número inicial de dimensões da vizinhança de pontos (somente leitura)
*/
class PolyMahalaDist {
public:


    /*! \brief Construtor
        \param input Matriz da vizinhança de pontos, em que cada linha é um ponto
                     e cada coluna uma dimensão
        \param order Ordem dos termos polinomiais
        \param smin Parâmetro do usuário, usado para garantir a inversão da matriz de
                    covariância da vizinhança de pontos C, alterando o resultado
                    proporcionalmente ao seu valor
        \param reference Centro da métrica (média da vizinhança de pontos, por padrão)
                         na forma de uma matriz coluna
    */
    PolyMahalaDist(const Mat& input, int order, double sig_max = 4e-6, Mat reference = Mat());
    PolyMahalaDist();
    virtual ~PolyMahalaDist();

    // getters
    Mat reference();
    double eps_svd();
    int dimension();
    int order();

    // // setters
    // void eps_svd(double eps_svd);
    // void order(int order);

    // /*! \brief builds the parameters that are used on the calculation
    //            of the distances and are based on the _smin and _l class members
    // */
    // void build();

    /*! \brief Calcula a distância de Mahalanobis polinomial entre dois pontos
        \param point1 Um ponto na forma de uma matriz coluna
        \param point2 Um ponto na forma de uma matriz coluna
        \return O valor da distância de Mahalanobis polinomial entre os pontos
    */
    double pointTo(Mat& im_data, Mat& refVector);
    /*! \brief Calcula a distância de Mahalanobis polinomial entre um ponto e o centro da métrica
        \param point Um ponto na forma de uma matriz coluna
        \return O valor da distância de Mahalanobis polinomial entre os pontos
    */
    double pointToReference(Mat& im_data);
    /*! \brief Calcula a distância de Mahalanobis polinomial entre um conjunto de pontos e um único ponto
        \param points Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                      coluna é uma dimensão
        \param ref Um único ponto na forma de uma matriz coluna
        \return Matriz de todos os valores da distância de Mahalanobis polinomial entre o ponto correspondente
                no conjunto de ponto passada como argumento e o ponto singular de referência
    */ 
    Mat pointsTo(Mat& im_data, Mat& refVector);
    /*! \brief Calcula a distância de Mahalanobis polinomial entre um conjunto de pontos e o centro da métrica
        \param points Uma matriz que contém conjunto de pontos, onde cada linha é um ponto e cada
                      coluna é uma dimensão
        \return Matriz de todos os valores da distância de Mahalanobis polinomial entre o ponto correspondente
                no conjunto de ponto passada como argumento e o centro da métrica
    */ 
    Mat pointsToReference(Mat& im_data);
    /*! \brief Transforma uma imagem em um conjunto de pontos e calcula a distância
               de Mahalanobis polinomial entre ela e um ponto de referência
        \param image Uma imagem
        \param ref Um ponto na forma de uma matriz coluna
        \return Uma imagem em que o valor de cada pixel é a distância de Mahalanobis polinomial entre
                o pixel correspondente na imagem passada como argumento e o ponto de referência
    */
    template <typename T> Mat imageTo(Mat& image, Mat& refVector);
    /*! \brief Transforma uma imagem em um conjunto de pontos e calcula a distância
               de Mahalanobis polinomial entre ela e o centro da métrica
        \param image Uma imagem
        \return Uma imagem em que o valor de cada pixel é a distância de Mahalanobis polinomial entre
                o pixel correspondente na imagem passada como argumento e o centro da métrica
    */
    template <typename T> Mat imageToReference(Mat& image);
private:
    Mat _reference;
    double _eps_svd;
    int _order;
    int _max_level;
    int _dimension;
    int _numberOfPoints;
    bool _dirty;

    /*
     * struct usada para armazenar os parâmetros de cada nível da expansão polinomial 
     */
    struct lev_basis {
        Mat A_basis;
        double max_aP;
        int ind_usesize;
        vector<int> ind_use;

        int d_proj;
        int dmssize;
        vector<double> dms;
        
        double sigma_inv;
    };

    /*
     * Distância de Mahalanobis linear, realiza o cálculo quando a ordem do polinômio é 1 (linear)
     */
    MahalaDist _baseMaha;

    vector<lev_basis> _basisVec;
};

