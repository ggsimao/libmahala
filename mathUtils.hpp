#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


/*! \brief Calcula a média de uma vizinhança de pontos 
    \param data Vizinhança de pontos
    \return Vetor de médias
*/
Mat calc_mean(Mat data);

/*! \brief Procura o maior valor em um dataset
    \param in Ponteiro para o início de um dataset
    \param size Tamanho do dataset 
    \return Valor máximo encontrado
*/
double getMaxValue(double *in, uint size);

/*! \brief Procura o maior valor absoluto em um dataset
    \param in Ponteiro para o início de um dataset
    \param size Tamanho do dataset 
    \return Valor absoluto máximo encontrado
*/
double getMaxAbsValue(double *in, uint size);

/*! \brief Realiza a expansão polinomial de um conjunto de pontos em segunda ordem
    \param vec Conjunto de pontos a serem expandidos polinomialmente
    \return Conjunto de pontos expandidos polinomialmente
*/
Mat polynomialProjection(Mat vec);

/*! \brief Encontra em um dataset valores iguais, maiores ou menores que 0
    \param opt Escolha de valores (0 para iguais a 0, negativo para menores que 0 e positivo para maiores que 0)
    \param in Ponteiro para o início de um dataset
    \param size Tamanho do dataset
    \return Vetor de valores encontrados
 */
vector<int> find_eq(int opt, double *in, uint size);

/*! \brief Calcula a variância de um conjunto de pontos
    \param A Matriz que guarda um conjunto de pontos, em que cada linha é um ponto e cada coluna é uma dimensão
    \param column Dimensão escolhida para calcular a variância
                  (valor negativo para calcular a variância de todos os elementos da matriz)
    \return Variância calculada
*/
double calcVarianceScalar(Mat A, int column);

/*! \brief Calcula a variância de um conjunto de pontos
    \param A Matriz que guarda um conjunto de pontos, em que cada linha é um ponto e cada coluna é uma dimensão
    \return Vetor das variâncias de cada dimensão dos pontos do conjunto
*/
Mat calcVarianceVector(Mat A);

/*! \brief Remove colunas de uma matriz, mantendo apenas as especificadas
    \param A Matriz a ter colunas removidas
    \param ind_use Vetor de colunas a serem mantidas
    \return Matriz com colunas removidas
*/
Mat removeNullIndexes(Mat A, vector<int> ind_use);

/*! \brief Remove colunas de uma matriz que possuem uma variância menor que 1e-8 vezes a maior variância da matriz
    \param A Matriz a ter colunas removidas
    \param ind_use Vetor que ao término da função armazenará as colunas que foram mantidas
    \return Matriz com colunas removidas
*/
Mat removeNullDimensions(Mat A, vector<int> &ind_use);