double MahalaDist::pointTo(Mat& point1, Mat& point2) {
    assert(!_dirty);
    assert(point1.cols == point2.cols && point1.cols == 1);
    assert(point1.rows == point2.rows && point1.rows == _dimension);

    Mat diff = point1-point2;
    double diffSquareSum = cv::sum(diff.mul(diff))[0];
    double ksum = 0;

    Mat proj;
    for (int k = 0; k < _k; k++) {
        proj = _u.col(k).t() * diff;
        double squareSum = cv::sum(proj.mul(proj))[0];
        ksum += (-w(k) * squareSum) / (_sigma2 * (w(k) + _sigma2));
    }

    double distance = (diffSquareSum/_sigma2) + ksum;
    distance = sqrt(distance*(_numberOfPoints-1));

    return (distance < 0) ? 0 : distance;
}

void classifiers::polyMahalanobis::evaluateToVector(double *im_data, double *refVector, unsigned int size, unsigned int dimensions, double* output_m_intensValues) {
    //allocating the return of the method
//    double *output_m_intensValues = (double*) calloc(size * m_model->num_levels, sizeof (double));
    m_model->max_level = 0;

    Mat diff = point1 - point2;

    int projCount = 1;
    classifiers::lev_basis *currBasis = m_model->levBegin;

    //*** proj_A = X * model.lev(i).A_basis;
    //CvMat *cvm_proj_A = cvCreateMat(cvm_X->rows,currBasis->cvm_A_basis->cols,CV_32FC1);
    CvMat *cvm_proj_A_sq = cvCreateMat(cvm_X->rows, currBasis->cvm_A_basis->cols, CV_32FC1);
    cvMatMul(cvm_X, currBasis->cvm_A_basis, cvm_proj_A_sq);

    //*** proj_A_sq = proj_A.^2;
    //CvMat *cvm_proj_A_sq = cvCreateMat(cvm_X->rows,currBasis->cvm_A_basis->cols,CV_32FC1);
    for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
        float v;
        for (int j = 0; j < cvm_proj_A_sq->cols; j++) {
            v = cvmGet(cvm_proj_A_sq, i, j);
            cvmSet(cvm_proj_A_sq, i, j, v * v);
        }
    }

    //*** A_sq = X.^2; 
    CvMat *cvm_A_sq = cvCreateMat(size, dimensions, CV_32FC1);
    for (int i = 0; i < cvm_A_sq->rows; i++) {
        float v;
        for (int j = 0; j < cvm_A_sq->cols; j++) {
            v = cvmGet(cvm_X, i, j);
            cvmSet(cvm_A_sq, i, j, v * v);
        }
    }

    //*** q1 = (sum((A_sq * model.lev(i).sigma_inv)'))';
    CvMat* cvm_A_sqtmp;
    cvm_A_sqtmp = cvCloneMat(cvm_A_sq);
    for (int i = 0; i < cvm_A_sq->rows; i++) {
        for (int j = 0; j < cvm_A_sq->cols; j++) {
            cvmSet(cvm_A_sqtmp, i, j, cvm_A_sq->data.fl[i * cvm_A_sq->cols + j] * currBasis->sigma_inv);
        }
    }

    CvMat* cvm_A_sqtmpT = cvCreateMat(cvm_A_sq->cols, cvm_A_sq->rows, CV_32FC1);
    cvTranspose(cvm_A_sqtmp, cvm_A_sqtmpT);

    double *q1 = (double*) calloc(cvm_A_sqtmpT->cols, sizeof (double));
    for (int i = 0; i < cvm_A_sqtmpT->rows; i++) {
        for (int j = 0; j < cvm_A_sqtmpT->cols; j++) {
            q1[j] += cvmGet(cvm_A_sqtmpT, i, j);
        }
    }

    //releasing local memory
    if (cvm_A_sqtmp) cvReleaseMat(&cvm_A_sqtmp);
    if (cvm_A_sqtmpT) cvReleaseMat(&cvm_A_sqtmpT);


    /***
    if size(proj_A_sq,2) > 1
            q2 = (sum((proj_A_sq.*repmat(dms,size(proj_A_sq,1),1))'))';
    else
            q2 = (proj_A_sq.*repmat(dms,size(proj_A_sq,1),1));
    end
    //*/

    //if(cvm_proj_A_sq->cols > 1)
    //{
    double *q2 = (double*) calloc(cvm_proj_A_sq->rows, sizeof (double));
    for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
        for (int j = 0; j < cvm_proj_A_sq->cols; j++) {
            q2[i] += cvmGet(cvm_proj_A_sq, i, j) * currBasis->dms[j];
        }
    }

    //}
    //*** q_in = q1 + q2; q_in(q_in<0) = 0;
    double *q_in = (double*) calloc(cvm_proj_A_sq->rows, sizeof (double));
    for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
        double v = q1[i] + q2[i];
        q_in[i] = (v < 0) ? 0 : v;
    }

    /***
    if DEBUG == 1
            val_all = q_in;
    end
    val = q_in;
     */
    for (unsigned int i = 0; i < size; i++) {
        output_m_intensValues[(projCount - 1)*size + i] = q_in[i];
    }
    if (q1) free(q1);
    if (q2) free(q2);
    if (q_in) free(q_in);
    m_model->max_level++;

    /***
    if model.num_levels > 1
            proj = X * model.lev(i).A_basis;
            proj = proj/model.lev(i).max_aP;
            if model.lev(i).d_proj > 1
                    C = nchoosek(1:model.lev(i).d_proj,2);
                    cross_terms = proj(:,C(:,1)).*proj(:,C(:,2));
                    new_dim = [proj,proj.^2,cross_terms];
                    %new_dim = [proj.^2,cross_terms];
            else
                    new_dim = [proj,proj.^2];
                    %new_dim = [proj.^2];
            end
    end
     */

    CvMat *cvm_new_dim = NULL;
    if (m_model->num_levels > 1) {
        //*** proj = X * model.lev(i).A_basis;
        //*** proj = proj/model.lev(i).max_aP;
        CvMat *cvm_proj_A = cvCreateMat(cvm_proj_A_sq->rows, cvm_proj_A_sq->cols, CV_32FC1);
        cvMatMul(cvm_X, currBasis->cvm_A_basis, cvm_proj_A);

        for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
            for (int j = 0; j < cvm_proj_A_sq->cols; j++) {
                cvm_proj_A->data.fl[i * cvm_proj_A_sq->cols + j] /= currBasis->max_aP;
            }
        }
        if (currBasis->d_proj > 1) {
            //*** C = nchoosek(1:d_proj,2);
            unsigned int sizeC;
            unsigned int *C = nchoosek(currBasis->d_proj, sizeC);

            //*** cross_terms = proj_A(:,C(:,1)).*proj_A(:,C(:,2));
            CvMat *cvm_cross_terms = cvCreateMat(cvm_proj_A_sq->rows, sizeC, CV_32FC1);
            for (unsigned int k = 0; k < sizeC * 2; k += 2) {
                float v;
                for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
                    v = cvmGet(cvm_proj_A, i, C[k] - 1) * cvmGet(cvm_proj_A, i, C[k + 1] - 1);
                    cvmSet(cvm_cross_terms, i, k / 2, v);
                }
            }

            //***  new_dim = [proj_A,proj_A.^2,cross_terms];
            //*** %new_dim = [proj_A.^2,cross_terms];
            cvm_new_dim = newProjection(cvm_proj_A, cvm_cross_terms);

            free(C);
            if (cvm_cross_terms) cvReleaseMat(&cvm_cross_terms);
            if (cvm_proj_A) cvReleaseMat(&cvm_proj_A);
        } else {
            //***  new_dim = [proj_A,proj_A.^2];
            cvm_new_dim = newProjection(cvm_proj_A, NULL);
        }
    }

    /*
    for i = 2:model.num_levels
    new_dim_used = new_dim(:,model.lev(i).ind_use);
    proj_A = new_dim_used * model.lev(i).A_basis;
    proj_A_sq = proj_A.^2;
     */
    for (projCount = 2; projCount < m_model->num_levels + 1; projCount++) {
        currBasis = currBasis->next;
        if (!currBasis) break;

        //new_dim_used = new_dim(:,model.lev(i).ind_use);
        CvMat *new_dim_used = removeNullIndexes(cvm_new_dim, currBasis->ind_use, currBasis->ind_usesize);

        //*** proj_A = X * model.lev(i).A_basis;
        if (cvm_proj_A_sq) cvReleaseMat(&cvm_proj_A_sq);
        cvm_proj_A_sq = cvCreateMat(new_dim_used->rows, currBasis->cvm_A_basis->cols, CV_32FC1);
        cvMatMul(new_dim_used, currBasis->cvm_A_basis, cvm_proj_A_sq);

        float v;
        for (int i = 0; i < new_dim_used->rows; i++) {
            for (int j = 0; j < currBasis->cvm_A_basis->cols; j++) {
                v = cvmGet(cvm_proj_A_sq, i, j);
                cvmSet(cvm_proj_A_sq, i, j, v * v);
            }
        }

        //*** dms = model.lev(i).dms;			
        //*** A_sq = new_dim_used.^2;
        if (cvm_A_sq) cvReleaseMat(&cvm_A_sq);
        CvMat *cvm_A_sq = cvCreateMat(new_dim_used->rows, new_dim_used->cols, CV_32FC1);
        for (int i = 0; i < new_dim_used->rows; i++) {
            for (int j = 0; j < new_dim_used->cols; j++) {
                v = cvmGet(new_dim_used, i, j);
                cvmSet(cvm_A_sq, i, j, v * v);
            }
        }

        /***
        q1 = (sum((A_sq * model.lev(i).sigma_inv)'))';
        if size(proj_A_sq,2) > 1
                q2 = (sum((proj_A_sq.*repmat(dms,size(proj_A_sq,1),1))'))';
        else
                q2 = (proj_A_sq.*repmat(dms,size(proj_A_sq,1),1));
        end
        q_in = q1 + q2;
        q_in(q_in<0) = 0;
         */
        CvMat* cvm_A_sqtmp;
        cvm_A_sqtmp = cvCloneMat(cvm_A_sq);
        for (int i = 0; i < cvm_A_sq->rows; i++) {
            for (int j = 0; j < cvm_A_sq->cols; j++) {
                cvmSet(cvm_A_sqtmp, i, j, cvm_A_sq->data.fl[i * cvm_A_sq->cols + j] * currBasis->sigma_inv);
            }
        }

        CvMat* cvm_A_sqtmpT = cvCreateMat(cvm_A_sq->cols, cvm_A_sq->rows, CV_32FC1);
        cvTranspose(cvm_A_sqtmp, cvm_A_sqtmpT);

        double *q1 = (double*) calloc(cvm_A_sqtmpT->cols, sizeof (double));
        for (int i = 0; i < cvm_A_sqtmpT->rows; i++) {
            for (int j = 0; j < cvm_A_sqtmpT->cols; j++) {
                q1[j] += cvmGet(cvm_A_sqtmpT, i, j);
            }
        }

        //releasing local memory
        if (cvm_A_sqtmp) cvReleaseMat(&cvm_A_sqtmp);
        if (cvm_A_sqtmpT) cvReleaseMat(&cvm_A_sqtmpT);


        double *q2 = (double*) calloc(cvm_proj_A_sq->rows, sizeof (double));
        for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
            for (int j = 0; j < cvm_proj_A_sq->cols; j++) {
                q2[i] += cvmGet(cvm_proj_A_sq, i, j) * currBasis->dms[j];
            }
        }

        //}
        //*** q_in = q1 + q2; q_in(q_in<0) = 0;
        double *q_in = (double*) calloc(cvm_proj_A_sq->rows, sizeof (double));
        for (int i = 0; i < cvm_proj_A_sq->rows; i++) {
            double v = q1[i] + q2[i];
            q_in[i] = (v < 0) ? 0 : v;
        }



        //*** proj = new_dim_used * model.lev(i).A_basis;
        //*** proj = proj/model.lev(i).max_aP;
        CvMat *cvm_proj = cvCreateMat(new_dim_used->rows, currBasis->cvm_A_basis->cols, CV_32FC1);
        cvMatMul(new_dim_used, currBasis->cvm_A_basis, cvm_proj);

        for (int i = 0; i < cvm_proj->rows; i++) {
            for (int j = 0; j < cvm_proj->cols; j++) {
                cvm_proj->data.fl[i * cvm_proj->cols + j] /= currBasis->max_aP;
            }
        }

        if (currBasis->d_proj > 1) {
            //*** C = nchoosek(1:d_proj,2);
            unsigned int sizeC;
            unsigned int *C = nchoosek(currBasis->d_proj, sizeC);

            //*** cross_terms = proj_A(:,C(:,1)).*proj_A(:,C(:,2));
            CvMat *cvm_cross_terms = cvCreateMat(size, sizeC, CV_32FC1);
            for (unsigned int k = 0; k < sizeC * 2; k += 2) {
                float v;
                for (int i = 0; i < cvm_proj->rows; i++) {
                    v = cvmGet(cvm_proj, i, C[k] - 1) * cvmGet(cvm_proj, i, C[k + 1] - 1);
                    cvmSet(cvm_cross_terms, i, k / 2, v);
                }
            }

            //***  new_dim = [proj_A,proj_A.^2,cross_terms];
            //*** %new_dim = [proj_A.^2,cross_terms];
            if (cvm_new_dim) cvReleaseMat(&cvm_new_dim);
            cvm_new_dim = newProjection(cvm_proj, cvm_cross_terms);

            free(C);
            if (cvm_cross_terms) cvReleaseMat(&cvm_cross_terms);
            if (cvm_proj) cvReleaseMat(&cvm_proj);
        } else {
            //***  new_dim = [proj_A,proj_A.^2];
            if (cvm_new_dim) cvReleaseMat(&cvm_new_dim);
            cvm_new_dim = newProjection(cvm_proj, NULL);
        }


        /*** val = [val,q_in];
        val = (sum(val'))';
		
        if DEBUG == 1
                val_all = [val_all,val];
        end
         */
        //if(DEBUG) showMatrixValues(cvm_new_dim);	  
        for (unsigned int i = 0; i < size; i++) {
            output_m_intensValues[(projCount - 1) * size + i] = q_in[i] + output_m_intensValues[(projCount - 2) * size + i];
        }
        if (q1) free(q1);
        if (q2) free(q2);
        if (q_in) free(q_in);

        //releasing local memory
        if (new_dim_used) cvReleaseMat(&new_dim_used);
        if (cvm_A_sq) cvReleaseMat(&cvm_A_sq);
        if (cvm_proj) cvReleaseMat(&cvm_proj);

        m_model->max_level++;
    }

    //releasing local memory
    if (cvm_X) cvReleaseMat(&cvm_X);
    //if(cvm_proj_A)	  cvReleaseMat(&cvm_proj_A);
    if (cvm_proj_A_sq) cvReleaseMat(&cvm_proj_A_sq);
    if (cvm_A_sq) cvReleaseMat(&cvm_A_sq);
    if (cvm_new_dim) cvReleaseMat(&cvm_new_dim);

    //if(q1) free(q1);
    //if(q2) free(q2);
    //if(q_in) free(q_in);
}