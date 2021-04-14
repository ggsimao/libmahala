#pragma once

#include "PolynomialMahalanobisDistance.hpp"

PolyMahalaDist::PolyMahalaDist(const Mat& input, int order, double sig_max, Mat reference) {
    // static const int numThread = 4;//omp_get_max_threads();

    assert(order > 0);

    _max_level = 0;
    _order = order;
    _numberOfPoints = input.rows;
    _dimension = input.cols;
    _eps_svd = sig_max;

    if (order == 1) {
        _baseMaha = MahalaDist(input, sig_max, reference);
        _baseMaha.build();
        return;
    }

    if (!reference.data) {
        _reference = calc_mean(input);
    } else {
        _reference = reference;
    }

    Mat a = Mat(input.rows, input.cols, CV_64FC1);

    Mat refT = _reference.t();
    for (int i = 0; i < _numberOfPoints; i++) {
        a.row(i) = input.row(i) - refT;
    }

    Mat uCont = Mat();

    vector<double> my_lambda;
    vector<int> ind_null, ind_basis;
    int indn_length, indb_length;
    double s_max, s_min;


    if (_numberOfPoints > _dimension) {
        Mat aT = a.t();
        Mat aTa = aT * a;

        Mat uContTmp, s, v;
        // SVD::compute(aTa, s, uContTmp, v);
        SVD::compute(aTa, s, v, uContTmp);

        Mat uContTmpT = uContTmp.t();
        // uContTmp.release();
        uContTmp = uContTmpT;

        vector<double> s_val;

        for (uint i = 0; i < _dimension; i++) {
            s_val.push_back(s.at<double>(i));
        }

        s_max = getMaxValue(s_val.data(), s_val.size());
        s_min = _eps_svd * s_max;

        for (uint i = 0; i < _dimension; i++) {
            s_val[i] = (s_val[i] < s_min) ? 0 : s_val[i];
        }
        
        ind_null = find_eq(0, s_val.data(), s_val.size());
        indn_length = ind_null.size();
        ind_basis = find_eq(1, s_val.data(), s_val.size());
        indb_length = ind_basis.size();

        if (my_lambda.size()) my_lambda.clear();
        int my_lambdaSize = indb_length;
        for (uint i = 0; i < indb_length; i++) {
            my_lambda.push_back(s_val[ind_basis[i]]);
        }

        assert(my_lambdaSize > 0);

        // if (uCont.data) uCont.release();
        uCont = Mat(_dimension, my_lambdaSize, CV_64FC1);
        for (uint i = 0; i < _dimension; i++) {
            for (uint j = 0; j < my_lambdaSize; j++) {
                uCont.at<double>(i, j) = uContTmp.at<double>(i, j);
            }
        }
    } else {
        Mat aT = a.t();
        Mat aTa = a * aT;

        Mat uTmp, s, v;
        // SVD::compute(aTa, s, uTmp, v);
        SVD::compute(aTa, s, v, uTmp);

        Mat uT = uTmp.t();

        vector<double> s_val;

        for (uint i = 0; i < _numberOfPoints; i++) {
            s_val.push_back(s.at<double>(i));
        }

        s_max = getMaxValue(s_val.data(), s_val.size());
        s_min = _eps_svd * s_max;

        for (uint i = 0; i < _numberOfPoints; i++) {
            s_val[i] = (s_val[i] < s_min) ? 0 : s_val[i];
        }

        ind_null = find_eq(0, s_val.data(), s_val.size());
        indn_length = ind_null.size();
        ind_basis = find_eq(1, s_val.data(), s_val.size());
        indb_length = ind_basis.size();

        if (my_lambda.size()) my_lambda.clear();

        if (indb_length == 0) {
            std::cout << "Problema!" << endl;
            std::cout << "S = " << endl << s << endl;
            std::cout << "A = " << endl << a << endl;
            std::cout << "UTmp = " << endl << uTmp << endl;
        }
        int my_lambdaSize = indb_length;
        for (uint i = 0; i < indb_length; i++) {
            my_lambda.push_back(s_val[ind_basis[i]]);
        }

        assert(my_lambdaSize > 0);

        Mat u = Mat(_numberOfPoints, my_lambdaSize, CV_64FC1);
        for (uint i = 0; i < _numberOfPoints; i++) {
            for (uint j = 0; j < my_lambdaSize; j++) {
                u.at<double>(i, j) = uT.at<double>(i, j);
            }
        }

        // uT.release();
        uT = u.t();
        Mat uContTmp = uT * a;
        Mat uCont = uContTmp.t();

        // vector<double> uCont_dist;

        for (uint i = 0; i < uCont.cols; i++) {
            // uCont_dist.push_back(norm(uCont.col(i)));
            uCont.col(i) /= norm(uCont.col(i));//uCont_dist[i];
        }

    }

    Mat proj_A = a * uCont;
    proj_A = proj_A.clone();
    double max_aP = getMaxAbsValue(proj_A.ptr<double>(0), proj_A.rows * proj_A.cols);

    if (max_aP > _eps_svd) {
        proj_A /= max_aP;
    } else {
        max_aP = 1;
    }

    uint n_proj = proj_A.rows;
    uint d_proj = proj_A.cols;

    int num_svds = 1;

    lev_basis newBasis;

    newBasis.A_basis = uCont;
    newBasis.max_aP = max_aP;
    newBasis.ind_usesize = indb_length;
    newBasis.ind_use = ind_basis;
    newBasis.d_proj = d_proj;
    newBasis.dmssize = indb_length;
    for (uint i = 0; i < indb_length; i++) {
        newBasis.dms.push_back(-my_lambda[i] / (s_min * (my_lambda[i] + s_min)));
    }
    newBasis.sigma_inv = 1 / s_min;

    // m_model->levBegin = newBasis; // TROCAR
    _basisVec.push_back(newBasis);

    bool cont = false;
    Mat new_dim;

    if (d_proj > 1) {
        uint sizeC;
        new_dim = polynomialProjection(proj_A);
        double var = calcVarianceScalar(new_dim, -1);

        if (var > _eps_svd)
            cont = true;
    }


    if (_order == 1) {
        cont = false;
    }

    lev_basis currBasis = _basisVec[0];
    int nt, dt;
    //*** 	THE BIG WHILE !!!!!!!
    while (cont) {
        n_proj = new_dim.rows;
        d_proj = new_dim.cols;

        // if (a.data) a.release();

        vector<int> ind_use;
        if (d_proj == 1) {
            a = new_dim.clone();
        } else {
            vector<int> ind_useTmp;
            a = removeNullDimensions(new_dim, ind_useTmp);

            for (uint i = 0; i < ind_useTmp[0]; i++) {
                ind_use.push_back(ind_useTmp[i+1]);
            }
        }

        //%%%%%%%%%%%%%%%%%%%%%%%%
        //%%%%%%%%%%%%%%%%%%%%%%%%
        //%%%%%%%%%%%%%%%%%%%%%%%%
        //%%% Find PCA basis
        //%%%%%%%%%%%%%%%%%%%%%%%%
        //%%%%%%%%%%%%%%%%%%%%%%%%
        //%%%%%%%%%%%%%%%%%%%%%%%%

        nt = a.rows;
        dt = a.cols;

        // if (aTa.data) aTa.release();
        
        if (nt >= dt) {
            Mat aT = a.t();
            Mat aTa = aT * a;

            Mat uContTmp, s, v;
            // SVD::compute(aTa, s, uContTmp, v); // A = U S V^T
            SVD::compute(aTa, s, v, uContTmp); // A = U S V^T

            Mat uContTmpT = uContTmp.t();
            // uContTmp.release();
            uContTmp = uContTmpT;

            vector<double> s_val;
            for (uint i = 0; i < dt; i++) {
                s_val.push_back(s.at<double>(i));
            }

            s_max = getMaxValue(s_val.data(), s_val.size());
            s_min = _eps_svd * s_max;

            for (uint i = 0; i < dt; i++) {
                s_val[i] = (s_val[i] < s_min) ? 0 : s_val[i];
            }

            ind_null = find_eq(0, s_val.data(), s_val.size());
            indn_length = ind_null.size();
            ind_basis = find_eq(1, s_val.data(), s_val.size());
            indb_length = ind_basis.size();

            if (my_lambda.size()) my_lambda.clear();
            int my_lambdaSize = indb_length;
            for (uint i = 0; i < indb_length; i++) {
                my_lambda.push_back(s_val[ind_basis[i]]);
            }

            assert(my_lambdaSize > 0);

            uCont = Mat(dt, my_lambdaSize, CV_64FC1);
            for (uint i = 0; i < dt; i++) {
                for (uint j = 0; j < my_lambdaSize; j++) {
                    uCont.at<double>(i, j) = uContTmp.at<double>(i, j);
                }
            }
        } else {
            Mat aT = a.t();
            Mat aTa = a * aT;

            Mat uTmp, s, v;
            // SVD::compute(aTa, s, uTmp, v);
            SVD::compute(aTa, s, v, uTmp);

            Mat uT = uTmp.t();

            vector<double> s_val;
            for (uint i = 0; i < nt; i++) {
                s_val.push_back(s.at<double>(i));
            }

            s_max = getMaxValue(s_val.data(), s_val.size());
            s_min = _eps_svd * s_max;

            for (uint i = 0; i < nt; i++) {
                s_val[i] = (s_val[i] < s_min) ? 0 : s_val[i];
            }

            ind_null = find_eq(0, s_val.data(), s_val.size());
            indn_length = ind_null.size();
            ind_basis = find_eq(1, s_val.data(), s_val.size());
            indb_length = ind_basis.size();

            if (indb_length == 0)
                std::cout << "Problema!" << endl;

            if (my_lambda.size()) my_lambda.clear();
            int my_lambdaSize = indb_length;
            for (uint i = 0; i < indb_length; i++) {
                my_lambda.push_back(s_val[ind_basis[i]]);
            }

            assert(my_lambdaSize > 0);

            Mat u = Mat(nt, my_lambdaSize, CV_64FC1);
            for (uint i = 0; i < nt; i++) {
                for (uint j = 0; j < my_lambdaSize; j++) {
                    u.at<double>(i, j) = uT.at<double>(i, j);
                }
            }

            // uT.release();
            uT = u.t();
            Mat uContTmp = uT * a;
            uCont = uContTmp.t();

            // vector<double> uCont_dist;
            for (uint i = 0; i < uCont.cols; i++) {
                // uCont_dist.push_back(norm(uCont.col(i)));
                uCont.col(i) /= norm(uCont.col(i));//uCont_dist[i];
            }
        }

        // if (proj_A.data) proj_A.release();
        proj_A = a * uCont;
        proj_A = proj_A.clone();
        max_aP = getMaxAbsValue(proj_A.ptr<double>(0), proj_A.rows * proj_A.cols);

        if (max_aP > _eps_svd) {
            proj_A /= max_aP;
        } else {
            max_aP = 1;
        }        

        n_proj = proj_A.rows;
        d_proj = proj_A.cols;

        num_svds++;

        lev_basis nextBasis;
        nextBasis.A_basis = uCont;
        nextBasis.max_aP = max_aP;
        nextBasis.ind_usesize = dt;
        nextBasis.ind_use = ind_use;
        nextBasis.d_proj = d_proj;
        nextBasis.dmssize = indb_length;
        for (uint i = 0; i < indb_length; i++) {
            nextBasis.dms.push_back(-my_lambda[i] / (s_min * (my_lambda[i] + s_min)));
        }
        nextBasis.sigma_inv = 1 / s_min;
        _basisVec.push_back(nextBasis);
        currBasis = nextBasis;

        if (d_proj > 1) {
            new_dim = polynomialProjection(proj_A);
            double var = calcVarianceScalar(new_dim, -1);

            if (var > _eps_svd)
                cont = true;
            else
                cont = false;
        } else {
            cont = false;
        }

        if (num_svds >= _order)
            cont = false;
    }
}

PolyMahalaDist::PolyMahalaDist() {}

PolyMahalaDist::~PolyMahalaDist() {}

//-----------------------------------------------------------------------------

Mat PolyMahalaDist::reference() {
    return _reference;
}
double PolyMahalaDist::eps_svd() {
    return _eps_svd;
}
int PolyMahalaDist::dimension() {
    return _dimension;
}
int PolyMahalaDist::order() {
    return _order;
}

//-----------------------------------------------------------------------------

double PolyMahalaDist::pointTo(Mat& im_data, Mat& refVector) {
    assert(im_data.cols == 1);
    assert(im_data.rows == _dimension);
    Mat im_dataT = im_data.t();
    return pointsTo(im_dataT, refVector).at<double>(0,0);
}

double PolyMahalaDist::pointToReference(Mat& im_data) {
    return pointTo(im_data, _reference);
}

Mat PolyMahalaDist::pointsTo(Mat& im_data, Mat& refVector) {
    if (_order == 1) {
        return _baseMaha.pointsTo(im_data, refVector);
    }

    int size = im_data.rows;
    int dimensions = im_data.cols;

    _max_level = 0;

    Mat refT = refVector.t();
    Mat x = Mat(size, dimensions, CV_64FC1);
    for (uint i = 0; i < size; i++) {
        x.row(i) = im_data.row(i) - refT;
    }

    int projCount = 0;
    lev_basis currBasis = _basisVec[0];

    Mat proj_A_sq = x * currBasis.A_basis;

    proj_A_sq = proj_A_sq.mul(proj_A_sq);

    Mat a_sq = x.mul(x);

    Mat a_sqTmp = a_sq * currBasis.sigma_inv;

    Mat a_sqTmpT = a_sqTmp.t();

    Mat q1 = Mat::zeros(a_sqTmpT.cols, 1, CV_64FC1);
    for (int i = 0; i < a_sqTmpT.rows; i++) {
        for (int j = 0; j < a_sqTmpT.cols; j++) {
            q1.at<double>(j) += a_sqTmpT.at<double>(i, j);
        }
    }

    Mat q2 = Mat::zeros(proj_A_sq.rows, 1, CV_64FC1);
    for (int i = 0; i < proj_A_sq.rows; i++) {
        for (int j = 0; j < proj_A_sq.cols; j++) {
            q2.at<double>(i) += proj_A_sq.at<double>(i, j) * currBasis.dms[j];
        }
    }

    Mat q_in = max(q1 + q2, 0);

    Mat output_m_intensValues = q_in.clone();

    _max_level++;

    Mat new_dim = Mat();
    if (_order > 1) {
        Mat proj_A = x * currBasis.A_basis;

        proj_A /= currBasis.max_aP;

        new_dim = polynomialProjection(proj_A);
    }

    for (projCount = 1; projCount < _basisVec.size(); projCount++) {
        currBasis = _basisVec[projCount];

        Mat new_dim_used = removeNullIndexes(new_dim, currBasis.ind_use);

        // if (proj_A_sq.data) proj_A_sq.release();
        proj_A_sq = new_dim_used * currBasis.A_basis;

        proj_A_sq = proj_A_sq.mul(proj_A_sq);

        // if (a_sq.data) a_sq.release();
        a_sq = new_dim_used.mul(new_dim_used);

        Mat a_sqTmp = a_sq * currBasis.sigma_inv;

        Mat a_sqTmpT = a_sqTmp.t();

        Mat q1 = Mat::zeros(a_sqTmpT.cols, 1, CV_64FC1);
        for (uint i = 0; i < a_sqTmpT.rows; i++) {
            for (uint j = 0; j < a_sqTmpT.cols; j++) {
                q1.at<double>(j) += a_sqTmpT.at<double>(i, j);
            }
        }

        Mat q2 = Mat::zeros(proj_A_sq.rows, 1, CV_64FC1);
        for (uint i = 0; i < proj_A_sq.rows; i++) {
            for (uint j = 0; j < proj_A_sq.cols; j++) {
                q1.at<double>(i) += proj_A_sq.at<double>(i, j) * currBasis.dms[j];
            }
        }

        Mat q_in = max(q1 + q2, 0);

        Mat proj = new_dim_used * currBasis.A_basis;
        proj /= currBasis.max_aP;

        new_dim = polynomialProjection(proj);

        output_m_intensValues += q_in;

        _max_level++;
    }

    return output_m_intensValues;
}

Mat PolyMahalaDist::pointsToReference(Mat& im_data) {
    return pointsTo(im_data, _reference);
}

template <typename T> Mat PolyMahalaDist::imageTo(Mat& image, Mat& refVector) {
    Mat linearized = linearizeImage<T>(image);
    linearized.convertTo(linearized, CV_64FC1);
    
    Mat distMat = pointsTo(linearized, refVector);

    Mat result = delinearizeImage<double>(distMat, image.rows, image.cols);

    return result;
}

template <typename T> Mat PolyMahalaDist::imageToReference(Mat& image) {
    return imageTo<T>(image, _reference);
}

template Mat PolyMahalaDist::imageTo<uchar>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<schar>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<ushort>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<short>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<int>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<float>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageTo<double>(Mat& image, Mat& refVector);
template Mat PolyMahalaDist::imageToReference<uchar>(Mat& image);
template Mat PolyMahalaDist::imageToReference<schar>(Mat& image);
template Mat PolyMahalaDist::imageToReference<ushort>(Mat& image);
template Mat PolyMahalaDist::imageToReference<short>(Mat& image);
template Mat PolyMahalaDist::imageToReference<int>(Mat& image);
template Mat PolyMahalaDist::imageToReference<float>(Mat& image);
template Mat PolyMahalaDist::imageToReference<double>(Mat& image);