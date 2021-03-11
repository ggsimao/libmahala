#include "PolynomialMahalanobisDistance.hpp"

#define NORMALIZE 0

PolyMahalaDist::PolyMahalaDist(Mat input, double smin, int l, Mat reference)
    : _inputMatrix(input), _smin(smin), _l(l), _reference(reference)
{
    assert(_inputMatrix.data);
    assert(_inputMatrix.type() == CV_64FC1);
    _dimension = input.cols;
    _numberOfPoints = input.rows;
    _polynomialDimension = pow(2, l);

    if(!_reference.data){
        _reference = Mat::zeros(_dimension, 1, CV_64FC1);
        for (int i = 0; i < _numberOfPoints; i++) {
            for (int j = 0; j < _dimension; j++) {
                _reference.at<double>(j) += input.at<double>(i,j);
            }
        }
        _reference /= _numberOfPoints;
    }

    _dirty = 1;
    _setReference = 1;

    assert(_reference.rows == _dimension);
    assert(_reference.cols == 1);
    assert(_reference.type() == CV_64FC1);
}

PolyMahalaDist::~PolyMahalaDist() {}

/*----------------------------------------------------------------------------*/

Mat PolyMahalaDist::inputMatrix() {
    return _inputMatrix;
}



Mat PolyMahalaDist::reference() {
    return _reference;
}



double PolyMahalaDist::smin() {
    return _smin;
}



int PolyMahalaDist::l() {
    return _l;
}

/*----------------------------------------------------------------------------*/

void PolyMahalaDist::setSmin(double smin) {
    _smin = smin;
}



void PolyMahalaDist::setInputMatrix(Mat input) {
    _inputMatrix = input;
    _dimension = input.rows;
    _numberOfPoints = input.cols;
    _setReference = 1;
}



void PolyMahalaDist::setL(int l) {
    _l = l;
    _polynomialDimension = pow(2, l);
}

/*----------------------------------------------------------------------------*/

Mat PolyMahalaDist::polynomialProjection(Mat vec) {
    Mat gvec = Mat(vec.rows, (vec.cols + 2) * (vec.cols + 1) / 2 - 1, vec.type());

    Mat mats[2] = {vec, Mat::ones(vec.rows, 1, vec.type())};

    Mat projinput;
    hconcat(mats, 2, projinput);

    int k = 0;

    //TODO - o for de n = 0 até vec.rows esta por dentro. Ele devia ser o for mais externo. Ajuda a dar menos miss em cach.
    for (int i = 0; i < vec.cols; i++) {
        for (int j = i; j < projinput.cols; j++) {
            for (int n = 0; n < vec.rows; n++) {
                gvec.at<double>(n, k) = vec.at<double>(n, i) * projinput.at<double>(n, j);
            }
            k++;
        }
    }

    assert(k == (vec.cols + 2) * (vec.cols + 1) / 2 - 1);

    //WHAT THE FUCK IS THIS!!!??? GIULIO!!!
    double sum = 0;
    for (int i = 0; i < vec.rows; i++) {
        for (int j = i; j < projinput.rows; j++) {
            //TA VAZIO!!!
        }
    }

    return gvec;
}



vector<int> PolyMahalaDist::filterByVariance(Mat vec) {
    vector<int> indexes;
    Mat c = (vec * vec.t()) / (vec.cols - 1);

    for (int i = 0; i < vec.rows; i++) {
        if (c.at<double>(i,i) > _smin) indexes.push_back(i);
    }

    return indexes;
}

Mat PolyMahalaDist::filterByVariancePC(const Mat &data, std::vector<int> &outIndes)
{
    //calcular a matriz de covariancia.
    Mat variations = Mat(data.size(), data.type());
    Mat means = Mat::zeros(1, data.cols, data.type());
    for(int i = 0; i < data.rows; i++){
        means += data.row(i);
    }
    means /= data.rows;
    for(int i = 0; i < data.rows; i++){
        variations.row(i) = data.row(i) + means;
    }

    //Cada elemento da diagonal principal dessa matriz, representa a variancia da respectiva coluna da "Mat data".
    Mat covariance = variations.t() * variations;

    //pegar a maior variancia.
    double maxVar = std::numeric_limits<double>::min();
    for(int i = 0; i < covariance.rows; i++){
        maxVar = std::max(maxVar, covariance.at<double>(i,i));
    }

    //Remover as colunas de data, que possuem variancia menor que "maxVar*parametro_smim".
    //Guardar os indices das colunas que permaneceram.
    //TODO - colocar um atributo para essa constante. Nao sei se devo usar a mesma variavel smim, que é usada para filtrar as colunas da matriz W, que sai do SVD.
    static const double parametro_smim = 1e-8;
    outIndes.clear();//so pra garantir que ta vazio.
    for(int j = 0; j < covariance.cols; j++){
        if(covariance.at<double>(j,j) > maxVar*parametro_smim){
            outIndes.push_back(j);//guardando os indices das colunas.
        }
    }
    Mat filteredMat = Mat(data.rows, outIndes.size(), data.type());//essa matriz que vai ser retornada.
    int count = 0;//variavel pra colocar as colunas nos lugares certos.
    for(int i : outIndes){
        data.col(i).copyTo(filteredMat.col(count++));
    }
    assert(count == filteredMat.cols);

    return filteredMat;
}

Mat PolyMahalaDist::filteringMultiplication(Mat slaveVec, Mat masterVec, vector<int> indexes) {
    Mat op;

    for (int i = 0; i < indexes.size(); i++) {
        op.push_back(masterVec.row(indexes[i]));
    }

    return slaveVec * op;
}

Mat PolyMahalaDist::removeNullDimensions(Mat &data, vector<int> &validDimesions)
{
    assert(data.cols >= validDimesions.size());
    Mat result = Mat(data.rows, validDimesions.size(), data.type());

    for(int j = 0; j < validDimesions.size(); j++){
        data.col(validDimesions[j]).copyTo(result.col(j));
    }
    return result;
}

Mat PolyMahalaDist::u(MahalaDist md) {
    assert (!md.dirty());
//    Mat us, ut = md.u().t();

//    for (int i = 0; i < md.w().rows; i++) {
//        if (md.c(i) >= md.sigma2()) {
//            us.push_back(ut.col(i).t());
//        }
//    }

//    assert(us.cols == md.dimension());

//    //us = Mat(us.t());

//    return us;
    return md.u();
}



void PolyMahalaDist::build() {
    if (!_dirty) return;

    _baseDist = MahalaDist(_inputMatrix, _smin, _reference);
    _baseDist.build();

    _expandedReferences.clear();
    _expandedUs.clear();
    _expandedDists.clear();
    _indexesVector.clear();

    double min, max, maxAbs;
    vector<int> indexes;

    MahalaDist nextDist = _baseDist;
    Mat nextInput = _inputMatrix.clone();
    Mat nextReference = _reference.clone();
    Mat nextBaseReference = _baseDist.reference().clone();

    for (int i = 0; i < nextInput.cols; i++)
        indexes.push_back(i);

    for (int l = 0; l < _l; l++) {
        Mat nextU = u(nextDist);

        _indexesVector.push_back(indexes);

        nextInput = filteringMultiplication(nextU, nextInput, indexes);
#if NORMALIZE
        minMaxLoc(nextInput, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextInput /= maxAbs;
#endif
        nextInput = polynomialProjection(nextInput);

        nextReference = filteringMultiplication(nextU, nextReference, indexes);
#if NORMALIZE
        minMaxLoc(nextReference, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextReference /= maxAbs;
#endif
        nextReference = polynomialProjection(nextReference);

        cout << "oi" << endl << nextInput.rows << endl << nextInput.cols << endl;

        if (_setReference) {
            nextBaseReference = filteringMultiplication(nextU, nextBaseReference, indexes);
#if NORMALIZE
            minMaxLoc(nextBaseReference, &min, &max);
            maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
            if (maxAbs > _smin) nextBaseReference /= maxAbs;
#endif
            nextBaseReference = polynomialProjection(nextBaseReference);

            indexes = filterByVariance(nextInput);
            nextInput = filteringMultiplication(Mat::eye(indexes.size(), indexes.size(), nextU.type()), nextInput, indexes);
            nextDist = MahalaDist(nextInput, _smin, nextBaseReference);
        } else {
            indexes = filterByVariance(nextInput);
            nextInput = filteringMultiplication(Mat::eye(indexes.size(), indexes.size(), nextU.type()), nextInput, indexes);
            nextDist = MahalaDist(nextInput, _smin);
        }

        nextDist.build();

        //        if (nextDist.maximumVariance() <= nextDist.smin()) break;

        _expandedReferences.push_back(nextReference);
        _expandedUs.push_back(nextU);
        _expandedDists.push_back(nextDist);
    }

    _dirty = 0;
}

void PolyMahalaDist::buildPC()
{
    if (!_dirty) return;

    _baseDist = MahalaDist(_inputMatrix, _smin, _reference);
    _baseDist.build();

    _expandedReferences.clear();
    _expandedUs.clear();
    _expandedDists.clear();
    _indexesVector.clear();
    _maxAbsVector.clear();

    double min, max, maxAbs;

    MahalaDist nextDist = _baseDist;
    Mat nextInput = _inputMatrix.clone();
    Mat nextReference = _reference.clone();
    Mat nextBaseReference = _baseDist.reference().clone();

    for (int l = 0; l < _l-1; l++) {
        const Mat nextU = nextDist.u();
        assert(nextInput.cols == nextU.rows);
        assert(nextReference.rows == nextU.rows);//next reference will be transposed

        //Projetando a matriz de dados de treino.
        Mat inputProj = nextInput * nextU;
        Mat refProj = nextReference.t() * nextU;

        //Normalizar os dados.
        cv::minMaxLoc(inputProj, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        _maxAbsVector.push_back(maxAbs);
        if (maxAbs > _smin) {
            inputProj /= maxAbs;
            refProj /= maxAbs;
        }

        //expandir polynomialmente.
        Mat expanded = polynomialProjection(inputProj);
        Mat refExpanded = polynomialProjection(refProj);

        //remover as colunas com pouca variancia e guardar os indices.
        vector<int> usedIndex;
        expanded = filterByVariancePC(expanded, usedIndex);
        refExpanded = removeNullDimensions(refExpanded, usedIndex);
        nextReference = refExpanded.t();
        _indexesVector.push_back(usedIndex);

        nextInput = expanded;
        nextDist = MahalaDist(nextInput, _smin, nextReference);
        nextDist.build();
        _expandedDists.push_back(nextDist);

    }

    _dirty = 0;
}

/*----------------------------------------------------------------------------*/

double PolyMahalaDist::pointTo(Mat point1, Mat point2) {
    assert(!_dirty);

    double sum = 0;

    Mat nextPoint1 = point1.clone();
    Mat nextPoint2 = point2.clone();

    double min, max, maxAbs;

    for (int l = 0; l < _expandedUs.size(); l++) {
        nextPoint1 = filteringMultiplication(_expandedUs[l], nextPoint1, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoint1, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoint1 /= maxAbs;
#endif
        nextPoint1 = polynomialProjection(nextPoint1);

        nextPoint2 = filteringMultiplication(_expandedUs[l], nextPoint2, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoint2, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoint2 /= maxAbs;
#endif
        nextPoint2 = polynomialProjection(nextPoint2);

        MahalaDist nextDist = _expandedDists[l];

        sum += nextDist.pointTo(nextPoint1, nextPoint2);
    }

    double finalSum = sum + _baseDist.pointTo(point1, point2);

    return finalSum;
}



double PolyMahalaDist::pointToReference(Mat point) {
    assert(!_dirty);

    double sum = 0;

    Mat nextPoint = point.clone();

    double min, max, maxAbs;

    for (int l = 0; l < _expandedUs.size(); l++) {
        filteringMultiplication(_expandedUs[l], nextPoint, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoint, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoint /= maxAbs;
#endif
        nextPoint = polynomialProjection(nextPoint);

        MahalaDist nextDist = _expandedDists[l];

        sum += nextDist.pointTo(nextPoint, _expandedReferences[l]);
    }

    double finalSum = sum + _baseDist.pointTo(point, _reference);

    return finalSum;
}

double PolyMahalaDist::pointToReferencePC(Mat point)
{
    assert(point.rows == _dimension);
    assert(point.cols == 1);
    assert(point.type() == CV_64FC1);

    double base = _baseDist.pointToReference(point);

    MahalaDist nextDist = _baseDist;
    Mat nextPoint = point.t();
    double sum = 0;
    for(int i = 0; i < _expandedDists.size(); i++){
        nextPoint = nextPoint * nextDist.u();

        //normaliza
        nextPoint /= _maxAbsVector[i];

        //expande
        nextPoint = polynomialProjection(nextPoint);

        //filtra
        nextPoint = removeNullDimensions(nextPoint, _indexesVector[i]);
        nextDist = _expandedDists[i];
        sum += nextDist.pointToReference(nextPoint);
    }
    return base + sum;
}

Mat PolyMahalaDist::pointsTo(Mat points, Mat point) {
    assert(!_dirty);

    Mat sum = Mat::zeros(1, points.cols, points.type());

    Mat nextPoints = points.clone();
    Mat nextPoint = point.clone();

    double min, max, maxAbs;

    for (int l = 0; l < _expandedUs.size(); l++) {
        nextPoints = filteringMultiplication(_expandedUs[l], nextPoints, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoints, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoints /= maxAbs;
#endif
        nextPoints = polynomialProjection(nextPoints);

        nextPoint = filteringMultiplication(_expandedUs[l], nextPoint, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoint, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoint /= maxAbs;
#endif
        nextPoint = polynomialProjection(nextPoint);

        MahalaDist nextDist = _expandedDists[l];

        sum += nextDist.pointsTo(nextPoints, nextPoint);
    }

    Mat finalSum = sum + _baseDist.pointsTo(points, point);

    return finalSum;
}



Mat PolyMahalaDist::pointsToReference(Mat points) {
    assert(!_dirty);

    Mat sum = Mat::zeros(1, points.cols, points.type());

    Mat nextPoints = points.clone();

    double min, max, maxAbs;

    for (int l = 0; l < _expandedUs.size(); l++) {
        nextPoints = filteringMultiplication(_expandedUs[l], nextPoints, _indexesVector[l]);
#if NORMALIZE
        minMaxLoc(nextPoints, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoints /= maxAbs;
#endif
        nextPoints = polynomialProjection(nextPoints);

        MahalaDist nextDist = _expandedDists[l];

        sum += nextDist.pointsTo(nextPoints, _expandedReferences[l]);
    }

    Mat finalSum = sum + _baseDist.pointsTo(points, _reference);

    return finalSum;
}

Mat PolyMahalaDist::pointsToReferencePC(Mat points)
{
    assert(points.cols == _dimension);
    assert(points.type() == CV_64FC1);

    Mat base = _baseDist.pointsToReference(points);

    MahalaDist nextDist = _baseDist;
    Mat nextPoints = points;
    Mat sum = Mat::zeros(base.size(), CV_64FC1);
    for(int i = 0; i < _expandedDists.size(); i++){
        nextPoints = nextPoints * nextDist.u();

        //normaliza
        nextPoints /= _maxAbsVector[i];

        //expande
        nextPoints = polynomialProjection(nextPoints);

        //filtra
        nextPoints = removeNullDimensions(nextPoints, _indexesVector[i]);
        nextDist = _expandedDists[i];
        sum += nextDist.pointsToReference(nextPoints);
    }
    return base + sum;
}
