#include "MahalanobisDistance.hpp"
#include "PolynomialMahalanobisDistance.hpp"
#include "BhattacharyyaDistance.hpp"
#include "PointCollector.hpp"
#include "imageUtils.cpp"
// #include "mahalaLibWrapper.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "images/coelho.jpg";

    bool coordinates = false;
    Mat mat = Mat();
    Mat result;
    PointCollector pc;
    Mat resultCV, resultDiff;
    MahalaDist md;
    // PolyMahalaDist pmd;
    Mat img;
    Mat average = Mat();
    int size = 512;
    // Mat referenceCoordinates;

    if (!coordinates) {
        img = imread(filename, IMREAD_COLOR);
        pc = PointCollector(img);
        mat = pc.collectedPixels();
        average = pc.referencePixel();
    } else {
        Mat img = Mat::zeros(size, size, CV_8UC1);
        pc = PointCollector(img);
        mat = pc.collectedCoordinates();
        // double data[6] = {501, 304, 457, 352, 406, 409};
        // mat = Mat(3, 2, CV_64FC1, data);
        average = pc.referenceCoordinate();
    }

    md = MahalaDist(mat, 0.00000000000001, average);
    md.build();
    PolyMahalaDist pmd = PolyMahalaDist(mat, 0.000001, 2, average);
    pmd.build();
    average = md.reference();

    if (coordinates) {
        img = Mat(size, size, CV_64FC2);

        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                img.at<Vec2d>(i, j) = Vec2d(j, i);
            }
        }
    }
    result = coordinates ? md.imageToReference<double>(img) : md.imageToReference<uchar>(img);
    Mat resultPoly = coordinates ? pmd.imageToReference<double>(img) : pmd.imageToReference<uchar>(img);
    Mat distMat = Mat(img.rows * img.cols, 1, CV_64FC1);

    Mat averageT = average.t();
    cout << averageT << endl;
    cout << pmd.polynomialProjection(averageT) << endl;
    cout << averageT.at<double>(0) * averageT.at<double>(0) << endl;
    cout << averageT.at<double>(1) * averageT.at<double>(1) << endl;
    cout << averageT.at<double>(0) * averageT.at<double>(1) << endl;
    
    Mat linearized = coordinates ? linearizeImage<double>(img) : linearizeImage<uchar>(img);
    linearized.convertTo(linearized, CV_64FC1);
    // cout << pmd.pointsToReference(linearized) << endl;

    Mat icovar = md.c().inv();
    Mat averageCV = md.reference().t();

    for (int i = 0; i < linearized.rows; i++) {
        distMat.at<double>(i) = Mahalanobis(linearized.row(i), averageCV, icovar);
    }
    resultCV = delinearizeImage<double>(distMat, img.rows, img.cols);

    resultDiff = result - resultCV;
    // sqrt(resultDiff, resultDiff);
    // resultDiff = resultDiff.mul(resultDiff);

    double min, max;
    cv::minMaxLoc(result, &min, &max);
    std::cout << "min" << min << endl;
    std::cout << "max" << max << endl;
    result = (result - min) / (max - min);

    double minCV, maxCV;
    cv::minMaxLoc(resultCV, &minCV, &maxCV);
    std::cout << "minCV" << minCV << endl;
    std::cout << "maxCV" << maxCV << endl;
    resultCV = (resultCV - minCV) / (maxCV - minCV);

    double minPoly, maxPoly;
    cv::minMaxLoc(resultPoly, &minPoly, &maxPoly);
    std::cout << "minPoly" << minPoly << endl;
    std::cout << "maxPoly" << maxPoly << endl;
    resultDiff = (resultPoly - minPoly) / (maxPoly - minPoly);

    double minDiff, maxDiff;
    cv::minMaxLoc(resultDiff, &minDiff, &maxDiff);
    std::cout << "minDiff" << minDiff << endl;
    std::cout << "maxDiff" << maxDiff << endl;
    resultDiff = (resultDiff - minDiff) / (maxDiff - minDiff);

    if (coordinates) {
        for (int i = 0; i < mat.rows; i++) {
            circle(img, Point(mat.row(i)), 3, 1, -1);
            circle(resultCV, Point(mat.row(i)), 3, 1, -1);
            circle(result, Point(mat.row(i)), 3, 1, -1);
            circle(resultPoly, Point(mat.row(i)), 3, 1, -1);
            circle(resultDiff, Point(mat.row(i)), 3, 1, -1);
        }
        circle(img, Point(md.reference()), 5, 1, -1);
        circle(resultCV, Point(md.reference()), 5, 1, -1);
        circle(result, Point(md.reference()), 5, 1, -1);
        circle(resultPoly, Point(md.reference()), 5, 1, -1);
        circle(resultDiff, Point(md.reference()), 5, 1, -1);
    }

    // double alpha = 1/(max-min);
    // double beta = -alpha*min;

    // result*=alpha;
    // result+=beta;

    // result.convertTo(result, result.type(), alpha, beta);

    // for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //     result.at<double>(i,j) = pow(M_E,-result.at<double>(i,j)*average/(pow(2, 16)));
    //     }
    // }

    // result.convertTo(result, result.type(), alpha, beta);

    // minMaxLoc(result, &min, &max);

    // alpha = 255/(max-min);
    // beta = -alpha*min;

    // img = linearizeImage<uchar>(img);
    // img = delinearizeImage<uchar>(img, size, size);

    //imshow("kkk", result_old);
    while (true){
        imshow("kkkkkkkkkkk", img);
        imshow("cvresultado", resultCV);
        imshow("meuresultado", result);
        imshow("meuresultadoPoly", resultPoly);
        // imshow("diffresultado", resultDiff);
        if (waitKey(1) == 27) break;
    }

    // cout << "min" << min << endl;
    // cout << "max" << max << endl;

    //   Mat thresholded;

    //   result.convertTo(thresholded, CV_8UC1, alpha, beta);

    //   for (int i = 0; i < img.rows; i++) {
    //     for (int j = 0; j < img.cols; j++) {
    //       //cout << int(thresholded.at<unsigned char>(i,j)) << endl;
    //       thresholded.at<unsigned char>(i,j) = thresholded.at<unsigned char>(i,j) > t ? 255 : 0;
    //     }
    //   }

    //   imshow(window3_name, thresholded);
}