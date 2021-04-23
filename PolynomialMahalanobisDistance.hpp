#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MahalanobisDistance.hpp"

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
        \param eps_svd Parâmetro do usuário, usado para garantir a inversão da matriz de
                       covariância da vizinhança de pontos C, alterando o resultado
                       proporcionalmente ao seu valor
        \param reference Centro da métrica (média da vizinhança de pontos, por padrão)
                         na forma de uma matriz coluna
    */
    PolyMahalaDist(const Mat& input, int order, double eps_svd = 4e-6, Mat reference = Mat());
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
        \tparam T O mesmo tipo que seria usado no método at<T>() da classe cv::Mat para acessar um elemento de image
        \param image Uma imagem
        \param ref Um ponto na forma de uma matriz coluna
        \return Uma imagem em que o valor de cada pixel é a distância de Mahalanobis polinomial entre
                o pixel correspondente na imagem passada como argumento e o ponto de referência
    */
    template <typename T> Mat imageTo(Mat& image, Mat& refVector);
    /*! \brief Transforma uma imagem em um conjunto de pontos e calcula a distância
               de Mahalanobis polinomial entre ela e o centro da métrica
        \tparam T O mesmo tipo que seria usado no método at<T>() da classe cv::Mat para acessar um elemento de image
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

    template <typename T> Mat linearizeImage(Mat& image);
    template <typename T> Mat delinearizeImage(Mat& linearized, int rows, int cols);

    // /*! \brief Calcula a média de uma vizinhança de pontos 
    //     \param data Vizinhança de pontos
    //     \return Vetor de médias
    // */
    Mat calc_mean(Mat data);

    // /*! \brief Procura o maior valor em um dataset
    //     \param in Ponteiro para o início de um dataset
    //     \param size Tamanho do dataset 
    //     \return Valor máximo encontrado
    // */
    double getMaxValue(double *in, uint size);

    // /*! \brief Procura o maior valor absoluto em um dataset
    //     \param in Ponteiro para o início de um dataset
    //     \param size Tamanho do dataset 
    //     \return Valor absoluto máximo encontrado
    // */
    double getMaxAbsValue(double *in, uint size);

    // /*! \brief Realiza a expansão polinomial de um conjunto de pontos em segunda ordem
    //     \param vec Conjunto de pontos a serem expandidos polinomialmente
    //     \return Conjunto de pontos expandidos polinomialmente
    // */
    Mat polynomialProjection(Mat vec);

    // /*! \brief Encontra em um dataset valores iguais, maiores ou menores que 0
    //     \param opt Escolha de valores (0 para iguais a 0, negativo para menores que 0 e positivo para maiores que 0)
    //     \param in Ponteiro para o início de um dataset
    //     \param size Tamanho do dataset
    //     \return Vetor de valores encontrados
    // */
    vector<int> find_eq(int opt, double *in, uint size);

    // /*! \brief Calcula a variância de um conjunto de pontos
    //     \param A Matriz que guarda um conjunto de pontos, em que cada linha é um ponto e cada coluna é uma dimensão
    //     \param column Dimensão escolhida para calcular a variância
    //                 (valor negativo para calcular a variância de todos os elementos da matriz)
    //     \return Variância calculada
    // */
    double calcVarianceScalar(Mat A, int column);

    // /*! \brief Calcula a variância de um conjunto de pontos
    //     \param A Matriz que guarda um conjunto de pontos, em que cada linha é um ponto e cada coluna é uma dimensão
    //     \return Vetor das variâncias de cada dimensão dos pontos do conjunto
    // */
    Mat calcVarianceVector(Mat A);

    // /*! \brief Remove colunas de uma matriz, mantendo apenas as especificadas
    //     \param A Matriz a ter colunas removidas
    //     \param ind_use Vetor de colunas a serem mantidas
    //     \return Matriz com colunas removidas
    // */
    Mat removeNullIndexes(Mat A, vector<int> ind_use);

    // /*! \brief Remove colunas de uma matriz que possuem uma variância menor que 1e-8 vezes a maior variância da matriz
    //     \param A Matriz a ter colunas removidas
    //     \param ind_use Vetor que ao término da função armazenará as colunas que foram mantidas
    //     \return Matriz com colunas removidas
    // */
    Mat removeNullDimensions(Mat A, vector<int> &ind_use);
};

