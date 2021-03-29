#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "imageUtils.cpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class BhattaDist {
public:
    /*
     * input: distribuição que será usada para construir a métrica
     *        as colunas representam as dimensões dos vetores, as linhas representam cada vetor individualmente
     */
    BhattaDist(const Mat& input);
    BhattaDist();
    virtual ~BhattaDist();

    /* Retorna a dimensão dos vetores usados para construir a métrica */
    int dimension();
    // int numberOfPoints();
    /* Aplica o cálculo da distância entre distribuição usada para construir a métrica e uma distribuição */
    double pointsTo(Mat& points);
    /* Aplica o cálculo da distância entre distribuição usada para construir a métrica e uma usada para outra métrica */
    double metricTo(BhattaDist& dist);
    template <typename T> double imageTo(Mat& image);

    /* Retorna a média da distribuição usada para construir a métrica */
    Mat mi();
    /* Retorna a matriz de covariância da distribuição usada para construir a métrica */
    Mat sigma();

    double smin(double smin);
    void build();

private:
    Mat _inputMatrix;
    Mat _mi;
    Mat _sigma;
    int _dimension;
    int _numberOfPoints;
    bool _dirty;
};