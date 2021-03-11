#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "MahalanobisDistance.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

class PolyMahalaDist {
public:
    /*
   TODO:
   1. Incluir mais getters/setters;
   2. Terminar função build();
   3. Melhorar funções de cálculo de distância;
   4. Cálculos diferentes para quando d > m;
   5. Possivelmente mudar o locar das normalizações.

   BUG:
   1. Ordem alta causa distância = inf;
   2. Ainda puxa um pouco pra origem;
   */


    //input -> matriz MxN com os dados para construir a distancia.
    //         Cada linha da matriz deve ser uma sample, e cada coluna uma feature.
    //l     -> Ordem do polinomio.
    PolyMahalaDist(Mat input, double smin, int l, Mat reference = Mat());
    virtual ~PolyMahalaDist();

    Mat inputMatrix();
    Mat reference();
    double smin();
    // int dimension();
    // int numberOfPoints();
    int l();

    void setSmin(double smin);
    void setInputMatrix(Mat input);
    void setL(int l);

    Mat polynomialProjection(Mat vec);
    vector<int> filterByVariance(Mat vec);
    Mat filterByVariancePC(const Mat &data, std::vector<int>& outIndes);
    Mat filteringMultiplication(Mat slaveVec, Mat masterVec, vector<int> indexes);
    Mat removeNullDimensions(Mat& data, vector<int>& validDimesions);
    Mat u(MahalaDist md);
    void build();
    void buildPC();

    double pointTo(Mat point1, Mat point2);
    double pointToReference(Mat point);
    double pointToReferencePC(Mat point);
    Mat pointsTo(Mat points, Mat point);
    Mat pointsToReference(Mat points);
    Mat pointsToReferencePC(Mat points);
private:
    Mat _inputMatrix;
    Mat _reference;
    double _smin;
    int _l;
    int _dimension;
    int _numberOfPoints;
    int _polynomialDimension;
    vector<Mat> _expandedReferences;
    vector<Mat> _expandedUs;
    vector<MahalaDist> _expandedDists;
    vector<vector<int>> _indexesVector;
    vector<double> _maxAbsVector;
    MahalaDist _baseDist;
    bool _dirty;
    bool _setReference;
};

