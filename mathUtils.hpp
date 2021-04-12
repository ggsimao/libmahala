#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>

using namespace cv;
using namespace std;


/*! \brief Method that calculates the mean value of the distribution 
  \param data dataset
  \param size the size of the dataset 
  \param d dimension of the values of the dataset
  \result float the mean value 
 */
Mat calc_mean(Mat data);

/*! \brief Gets the maximum value of a dataset
  \param in dataset
  \param size the size of the give dataset 
  \result float maximum value found 
 */
double getMaxValue(double *in, uint size);

/*! \brief Gets the absolute maximum value of a dataset
  \param in dataset
  \param size the size of the give dataset 
  \result float absolute maximum value found 
 */
double getMaxAbsValue(double *in, uint size);

Mat polynomialProjection(Mat vec);

/*! \brief Finds in an array values equal to zero (0), lower or bigger than, and return a vector of indexes
  \param opt reference range to search (<0 for <0, 0 for 0, >0 for >0)
  \param in input vector reference
  \param size vector length
  \result uint list of value indexes that satisfy opt restriction
 */
vector<int> find_eq(int opt, double *in, uint size);

/*! \brief Calculates the variance of a give set
  \param A set which will be calculated the variance
  \param column chosen column to calculate variance (-1 for all elements)
  \result double precision float variance
 */
double calcVariance(Mat A, int column);

/*! \brief Calculates the variance  
  \param A dataset which will be calculated the variance
  \return variance for each feature of the dataset
 */
Mat calcVariance(Mat A);

/*! \brief Method that fits the matrix A according to the ind_use vector (see below)
  \param A projection matrix
  \param ind_use vector of non-repeated dimension in a combinatory explosion
  \param size size of ind_use vector
  \result CvMat* matrix of non-repeated dimensional terms of a polynomial 
 */
Mat removeNullIndexes(Mat A, vector<int> ind_use);

/*! \brief Method that removes null dimensions, e.g.lower than a variance error, and return a new one 
  \param A projection matrix
  \param ind_use vector of non-repeated dimension in a combinatory explosion
  \result Mat matrix of non-repeated dimensional terms of a polynomial 
 */
Mat removeNullDimensions(Mat A, vector<int> &ind_use);