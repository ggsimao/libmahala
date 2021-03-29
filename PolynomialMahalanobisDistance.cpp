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

    _dirty = 1;
    _setReference = 1;

    if(!_reference.data){
        _reference = Mat::zeros(_dimension, 1, CV_64FC1);
        for (int i = 0; i < _numberOfPoints; i++) {
            for (int j = 0; j < _dimension; j++) {
                _reference.at<double>(j) += input.at<double>(i,j);
            }
        }
        _reference /= _numberOfPoints;
        _setReference = 0;
    }

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

void PolyMahalaDist::smin(double smin) {
    _smin = smin;
    _dirty = 1;
}

void PolyMahalaDist::l(int l) {
    _l = l;
    _polynomialDimension = pow(2, l);
    _dirty = 1;
}

/*----------------------------------------------------------------------------*/

Mat PolyMahalaDist::polynomialProjection(Mat vec) {
    Mat gvec; //= Mat(vec.rows, (vec.cols + 2) * (vec.cols + 1) / 2 - 1, vec.type());

    cout << "vec.size(): " << endl;
    cout << vec.size() << endl;

    Mat vec_sq = vec.mul(vec); // square each of vec's elements
    Mat vec_cross = Mat(vec.rows, vec.cols * (vec.cols - 1) / 2, vec.type()); // cross products between vec's elements

    int k = 0;
    // calculates the cross products:
    for (int i = 0; i < vec.cols; i++) {
        for (int j = i+1; j < vec.cols; j++) {
            vec_cross.col(k) = vec.col(i).mul(vec.col(j));
            k++;
        }
    }

    Mat mats[3] = {vec, vec_sq, vec_cross};
    hconcat(mats, 3, gvec);

    cout << "gvec.size(): " << endl;
    cout << gvec.size() << endl;

    return gvec;
}



vector<int> PolyMahalaDist::filterByVariance(Mat vec) {
    vector<int> indexes;
    Mat c = (vec.t() * vec) / (vec.rows - 1); // calculate covariance matrix

    cout << "vec.size(): " << endl;
    cout << vec.size() << endl;
    cout << "c.size(): " << endl;
    cout << c.size() << endl;

    for (int i = 0; i < vec.cols; i++) {
        if (c.at<double>(i,i) > _smin) indexes.push_back(i); // choose the indexes
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
    Mat result;
    // cout << "filteringMultiplication" << endl;

    cout << "indexes.size(): " << endl;
    cout << indexes.size() << endl;

    // cout << "slaveVec: " << endl;
    // cout << slaveVec << endl;
    cout << "slaveVec.size(): " << endl;
    cout << slaveVec.size() << endl;

    // cout << "masterVec: " << endl;
    // cout << masterVec << endl;
    cout << "masterVec.size(): " << endl;
    cout << masterVec.size() << endl;

    masterVec = masterVec.t();

    Mat temp = slaveVec * masterVec;

    for (int i = 0; i < indexes.size(); i++) {
        result.push_back(temp.row(indexes[i]));
    }

    // op = op.t();

    // cout << "op: " << endl;
    // cout << op << endl;
    cout << "op.size(): " << endl;
    cout << temp.size() << endl;

    // Mat result = slaveVec * op;

    // cout << "result: " << endl;
    // cout << result << endl;
    cout << "result.size(): " << endl;
    cout << result.size() << endl;

    return result.t();
}

Mat PolyMahalaDist::removeNullDimensions(Mat &data, vector<int> &validDimesions)
{
    assert(data.cols >= validDimesions.size());
    Mat result = Mat(data.rows, validDimesions.size(), data.type());

    cout << "data.size(): " << endl;
    cout << data.size() << endl;

    for(int j = 0; j < validDimesions.size(); j++){
        data.col(validDimesions[j]).copyTo(result.col(j));
    }

    cout << "result.size(): " << endl;
    cout << result.size() << endl;
    return result;
}

void PolyMahalaDist::build() {
    if (!_dirty) return;

    /*
     * A GOOD PORTION OF THIS METHOD WAS WRITTEN WITH A
     * LINEAR MAHALANOBIS DISTANCE FORMULA THAT FILTERED OUT
     * DIMENSIONS BASED ON _SMIN, BUT THAT FORMULA IS NOT
     * USED IN THIS LIBRARY AT THE MOMENT, SO SOME STEPS
     * MIGHT EITHER BE REDUNDANT, USELESS OR STRAIGHT UP
     * NOT MAKE SENSE.
     */

    _baseDist = MahalaDist(_inputMatrix, _smin, _reference); // 0-order polynomial Mahalanobis distance
    _baseDist.build();

    _expandedReferences.clear();
    _expandedUs.clear();
    _expandedDists.clear();
    _indexesVector.clear();

    double min, max, maxAbs;

    // iteration variables:
    vector<int> indexes;
    MahalaDist nextDist = _baseDist;
    Mat nextInput = _inputMatrix.clone();
    Mat nextReference = _reference.clone();
    Mat nextBaseReference = _baseDist.reference().clone();

    for (int i = 0; i < nextInput.cols; i++)
        indexes.push_back(i);

    for (int l = 0; l < _l; l++) {
        Mat nextU = nextDist.u(); // gets filtered U matrix

        cout << "_indexesVector.push_back(indexes) -> begin" << endl;
        _indexesVector.push_back(indexes); // select non-null U indexes
        cout << "_indexesVector.push_back(indexes) -> end" << endl;

        // cout << nextU << endl;
        cout << "nextInput = filteringMultiplication(nextU, nextInput, indexes) -> begin" << endl;
        nextInput = filteringMultiplication(nextU, nextInput, indexes); // multiplies U by the vector neighborhood, then filter out dimensions
        cout << "nextInput = filteringMultiplication(nextU, nextInput, indexes) -> end" << endl;
#if NORMALIZE
        minMaxLoc(nextInput, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextInput /= maxAbs;
#endif
        cout << "nextInput = polynomialProjection(nextInput); -> begin" << endl;
        nextInput = polynomialProjection(nextInput); // projects the current neighborhood
        cout << "nextInput = polynomialProjection(nextInput); -> end" << endl;

        nextReference = nextReference.t();

        cout << "nextReference = filteringMultiplication(nextU, nextReference, indexes) -> begin" << endl;
        nextReference = filteringMultiplication(nextU, nextReference, indexes); // same operation as above but with the reference vector
        cout << "nextReference = filteringMultiplication(nextU, nextReference, indexes) -> end" << endl;
#if NORMALIZE
        minMaxLoc(nextReference, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextReference /= maxAbs;
#endif
        cout << "nextReference = polynomialProjection(nextReference) -> begin" << endl;
        nextReference = polynomialProjection(nextReference); // same operation as above but with the reference vector
        cout << "nextReference = polynomialProjection(nextReference) -> end" << endl;

        nextReference = nextReference.t();

        cout << "indexes = filterByVariance(nextInput) -> begin" << endl;
        indexes = filterByVariance(nextInput); // choose dimensions of projected input to be filtered out
        cout << "indexes = filterByVariance(nextInput) -> end" << endl;

        cout << "nextInput = filteringMultiplication(Mat::eye(indexes.size(), indexes.size(), nextU.type()), nextInput, indexes) -> begin" << endl;
        nextInput = filteringMultiplication(Mat::eye(nextInput.cols, nextInput.cols, nextU.type()), nextInput, indexes); // filters out chosen dimensions
        cout << "nextInput = filteringMultiplication(Mat::eye(indexes.size(), indexes.size(), nextU.type()), nextInput, indexes) -> end" << endl;

        // cout << "oi" << endl << nextInput.rows << endl << nextInput.cols << endl;
        // cout << _setReference << endl;

        cout << "nextDist = MahalaDist(nextInput, _smin, nextReference) -> begin" << endl;
        nextDist = MahalaDist(nextInput, _smin, nextReference); // create projected Mahalanobis metric
        cout << "nextDist = MahalaDist(nextInput, _smin, nextReference) -> end" << endl;

        cout << "nextDist.build() -> begin" << endl;
        nextDist.build();
        cout << "nextDist.build() -> end" << endl;

        // cout << "nextDist buildada" << endl;
        cout << "l=" << l << "/_l=" << _l << endl;

        // if (nextDist.maximumVariance() <= nextDist.smin()) break;

        // save expanded variables for later when calculating distances
        _expandedReferences.push_back(nextReference);
        _expandedUs.push_back(nextU);
        _expandedDists.push_back(nextDist);
        cout << "push_back" << endl;
    }

    cout << "build terminado" << endl;

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
        assert(nextInput.cols == nextU.cols);
        assert(nextReference.rows == nextU.cols);//next reference will be transposed

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

    Mat sum = Mat::zeros(points.rows, 1, points.type()); // matrix in which the results will be stored

    // iteration variables
    Mat nextPoints = points.clone();
    Mat nextPoint = point.clone();

    double min, max, maxAbs;

    // This for loop perform the same operations the build method does on the U, input and reference
    // matrices, but on the points and point matrices instead. It then calculates the l-th Mahalanobis
    // distance for each l that was stored in the vectors calculated by the build method and adds it
    // to the partial sum. See G. Grudic et al (2006) for reference.
    for (int l = 0; l < _expandedUs.size(); l++) {
        cout << "nextPoints = filteringMultiplication(_expandedUs[l], nextPoints, _indexesVector[l]) -> begin" << endl;
        nextPoints = filteringMultiplication(_expandedUs[l], nextPoints, _indexesVector[l]);
        cout << "nextPoints = filteringMultiplication(_expandedUs[l], nextPoints, _indexesVector[l]) -> end" << endl;
#if NORMALIZE
        minMaxLoc(nextPoints, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoints /= maxAbs;
#endif
        cout << "nextPoints = polynomialProjection(nextPoints) -> begin" << endl;
        nextPoints = polynomialProjection(nextPoints);
        cout << "nextPoints = polynomialProjection(nextPoints) -> end" << endl;

        nextPoint = nextPoint.t();
        cout << "nextPoint = filteringMultiplication(_expandedUs[l], nextPoint, _indexesVector[l]) -> begin" << endl;
        nextPoint = filteringMultiplication(_expandedUs[l], nextPoint, _indexesVector[l]);
        cout << "nextPoint = filteringMultiplication(_expandedUs[l], nextPoint, _indexesVector[l]) -> end" << endl;
#if NORMALIZE
        minMaxLoc(nextPoint, &min, &max);
        maxAbs = abs(min) > abs(max) ? abs(min) : abs(max);
        if (maxAbs > _smin) nextPoint /= maxAbs;
#endif
        cout << "nextPoint = polynomialProjection(nextPoint) -> begin" << endl;
        nextPoint = polynomialProjection(nextPoint);
        cout << "nextPoint = polynomialProjection(nextPoint) -> end" << endl;
        nextPoint = nextPoint.t();
        
        MahalaDist nextDist = _expandedDists[l];

        cout << "sum += nextDist.pointsTo(nextPoints, nextPoint) -> begin" << endl;
        sum += nextDist.pointsTo(nextPoints, nextPoint);
        cout << "sum += nextDist.pointsTo(nextPoints, nextPoint) -> end" << endl;
    }

    Mat finalSum = sum + _baseDist.pointsTo(points, point); // Add the 0-order distance

    return finalSum;

    // assert(points.cols == _dimension);
    // assert(points.type() == CV_64FC1);

    // Mat base = _baseDist.pointsTo(points, point);

    // MahalaDist nextDist = _baseDist;
    // Mat nextPoints = points;
    // Mat nextPoint = point;
    // Mat sum = Mat::zeros(base.size(), CV_64FC1);
    // cout << "base.size(): " << endl;
    // cout << base.size() << endl;
    // cout << "base.rows: " << endl;
    // cout << base.rows << endl;
    // cout << "base.cols: " << endl;
    // cout << base.cols << endl;
    // for(int i = 0; i < _expandedDists.size(); i++){
    //     Size presize = nextPoint.t().size(), presizes = nextPoints.size();
    //     cout << "nextPoints = nextPoints * nextDist.u();" << endl;
    //     cout << "nextPoint = nextPoint * nextDist.u();" << endl;
    //     nextPoints = nextPoints * nextDist.u();
    //     nextPoint = nextPoint.t();
    //     nextPoint = nextPoint * nextDist.u();
    //     nextPoint = nextPoint.t();
    //     cout << nextPoints.size() << " = " << presizes << " * " << nextDist.u().size() << endl;
    //     cout << nextPoint.size() << " = " << presize << " * " << nextDist.u().size() << endl;



    //     //normaliza
    //     nextPoints /= _maxAbsVector[i];
    //     nextPoint /= _maxAbsVector[i];

    //     //expande
    //     nextPoints = polynomialProjection(nextPoints);
    //     nextPoint = nextPoint.t();
    //     nextPoint = polynomialProjection(nextPoint);
    //     nextPoint = nextPoint.t();

    //     //filtra
    //     nextPoints = removeNullDimensions(nextPoints, _indexesVector[i]);
    //     nextPoint = nextPoint.t();
    //     nextPoint = removeNullDimensions(nextPoint, _indexesVector[i]);
    //     nextPoint = nextPoint.t();
    //     nextDist = _expandedDists[i];
    //     sum += nextDist.pointsTo(nextPoints, nextPoint);
    // }
    // return base + sum;
}


// Same as above, but the operations for the reference vector were
// already performed by the build method, so they're skipped here.
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

template <typename T> Mat PolyMahalaDist::imageTo(Mat& image, Mat& ref) {
    assert(!_dirty);

    Mat linearized = linearizeImage<T>(image);
    linearized.convertTo(linearized, CV_64FC1);
    
    Mat distMat = pointsTo(linearized, ref);

    Mat result = delinearizeImage<double>(distMat, image.rows, image.cols);

    return result;
}

template <typename T> Mat PolyMahalaDist::imageToReference(Mat& image) {
    return imageTo<T>(image, _reference);
}

template Mat PolyMahalaDist::imageTo<uchar>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<schar>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<ushort>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<short>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<int>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<float>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageTo<double>(Mat& image, Mat& ref);
template Mat PolyMahalaDist::imageToReference<uchar>(Mat& image);
template Mat PolyMahalaDist::imageToReference<schar>(Mat& image);
template Mat PolyMahalaDist::imageToReference<ushort>(Mat& image);
template Mat PolyMahalaDist::imageToReference<short>(Mat& image);
template Mat PolyMahalaDist::imageToReference<int>(Mat& image);
template Mat PolyMahalaDist::imageToReference<float>(Mat& image);
template Mat PolyMahalaDist::imageToReference<double>(Mat& image);