#include "mathUtils.hpp"

Mat calc_mean(Mat data) {
    int d = data.cols;

    Mat m = Mat(d, 1, CV_64FC1);

    for (uint i = 0; i < d; i++) {
        m.at<double>(i) = (mean(data.col(i)))[0];
    }

    return m;
}

double getMaxValue(double *in, uint size) {
    double max = 0;
    for (uint i = 0; i < size; i++) {
        if (in[i] > max) max = in[i];
    }
    return max;
}

double getMaxAbsValue(double *in, uint size) {
    double max = 0;
    for (uint i = 0; i < size; i++) {
        if (abs(in[i]) > max) max = abs(in[i]);
    }
    return max;
}

Mat polynomialProjection(Mat vec) {
    Mat gvec; //= Mat(vec.rows, (vec.cols + 2) * (vec.cols + 1) / 2 - 1, vec.type());

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

    return gvec;
}

vector<int> find_eq(int opt, double *in, uint size) {
    vector<int> indexes;
    if (opt < 0) {
        for (uint i = 0; i < size; i++) {
            if (in[i] < 0) indexes.push_back(i);
        }
    } else if (opt > 0) {
        for (uint i = 0; i < size; i++) {
            if (in[i] > 0) indexes.push_back(i);
        }
    } else {
        for (uint i = 0; i < size; i++) {
            if (in[i] == 0) indexes.push_back(i);
        }
    }

    return indexes;
}

double calcVariance(Mat A, int column) {
    assert(A.type() == CV_64FC1);
    double var = 0;
    uint nt = A.rows;
    uint dt = A.cols;

    if (column == -1) {
        double sum = 0;
        for (uint i = 0; i < nt; i++) {
            for (uint j = 0; j < dt; j++) {
                sum += A.at<double>(i, j);
            }
        }
        double mean = sum / (nt * dt);

        sum = 0;
        for (uint i = 0; i < nt; i++) {
            for (uint j = 0; j < dt; j++) {
                sum += (A.at<double>(i, j) - mean) * (A.at<double>(i, j) - mean);
            }
        }
        var = 1 / ((double) (nt * dt) - 1) * sum;
    } else {
        double sum = 0;
        for (unsigned int i = 0; i < nt; i++) {
            sum += A.at<double>(i, column);
        }
        double mean = sum / nt;

        sum = 0;
        for (uint i = 0; i < nt; i++) {
            sum += (A.at<double>(i, column) - mean) * (A.at<double>(i, column) - mean);
        }
        var = 1 / ((double) nt - 1) * sum;
    }
    return var;
}

Mat calcVariance(Mat A) {
    Mat var_dim = Mat::zeros(A.rows, 1, A.type());

    uint nt = A.rows;
    uint dt = A.cols;

    Mat mean = Mat::zeros(A.rows, 1, A.type());
    for (uint k = 0; k < dt; k++) {
        for (uint i = 0; i < nt; i++) {
            mean.at<double>(k) += A.at<double>(i, k);
        }
    }
    mean /= nt;

    for (uint k = 0; k < dt; k++) {
        for (uint i = 0; i < nt; i++) {
            var_dim.at<double>(k) += (A.at<double>(i, k) - mean.at<double>(k)) * (A.at<double>(i, k) - mean.at<double>(k));
        }
    }
    var_dim /= (nt - 1);

    return var_dim;
}

Mat removeNullIndexes(Mat A, vector<int> ind_use) {
    Mat new_dim_usedT, aT = A.t();

    for (uint i = 0; i < ind_use.size(); i++) {
        new_dim_usedT.push_back(aT.row(ind_use[i]));
    }

    Mat new_dim_used = new_dim_usedT.t();

    return new_dim_used;
}

Mat removeNullDimensions(Mat A, vector<int> &ind_use) {

    uint nt = A.rows;
    uint dt = A.cols;

    Mat var_new_dim = calcVariance(A);

    uint N = 0;
    double maxVar = getMaxValue(var_new_dim.clone().ptr<double>(0), dt);
    for (uint i = 0; i < dt; i++) {
        if (var_new_dim.at<double>(i) > 1e-8 * maxVar) N++;
    }
    ind_use.push_back(N);

    Mat newMat = Mat(nt, N, CV_64FC1);
    int k = 0;
    for (uint i = 0; i < dt; i++) {
        if (var_new_dim.at<double>(i) > 1e-8 * maxVar) {
            for (uint j = 0; j < nt; j++) {
                newMat.at<double>(j, k) = A.at<double>(j, i);
            }
            ind_use.push_back(i);
            k++;
        }
    }

    return newMat;
}