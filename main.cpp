#include "MahalanobisDistance.hpp"
#include "PolynomialMahalanobisDistance.hpp"
#include "BhattacharyyaDistance.hpp"
#include "PointCollector.hpp"
#include "imageUtils.hpp"
// #include "mahalaLibWrapper.cpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "images/coelho.jpg";

    bool coordinates = true;
    Mat mat = Mat();
    Mat result;
    PointCollector pc;
    Mat resultCV, resultDiff;
    MahalaDist md;
    PolyMahalaDist pmd;
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
        img = Mat::zeros(size, size, CV_8UC1);
        pc = PointCollector(img);
        mat = pc.collectedCoordinates();
        // double data[6] = {501, 304, 457, 352, 406, 409};
        // mat = Mat(3, 2, CV_64FC1, data);
        average = pc.referenceCoordinate();
    }

    md = MahalaDist(mat, 0.00000000000001, average);
    md.build();
    pmd = PolyMahalaDist(mat, 10, 4e-6, average);
    // pmd.build();
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

    
    // Mat averageT = average.t();
    // cout << averageT << endl;
    // cout << pmd.polynomialProjection(averageT) << endl;
    // cout << averageT.at<double>(0) * averageT.at<double>(0) << endl;
    // cout << averageT.at<double>(1) * averageT.at<double>(1) << endl;
    // cout << averageT.at<double>(0) * averageT.at<double>(1) << endl;
    
    Mat linearized = coordinates ? linearizeImage<double>(img) : linearizeImage<uchar>(img);
    linearized.convertTo(linearized, CV_64FC1);
    // cout << pmd.pointsTo(linearized) << endl;


    // BhattaDist bd = BhattaDist(linearized, 0);
    // bd.build();
    // Mat otherImage = imread("images/building.png", IMREAD_COLOR);
    // Mat linearizedOther = coordinates ? linearizeImage<double>(otherImage) : linearizeImage<uchar>(otherImage);
    // linearized.convertTo(linearized, CV_32FC1);
    // Mat hist, histOther;
    // int r_bins = 256, g_bins = 256, b_bins = 256;
    // int histSize[] = { b_bins, g_bins, r_bins };
    // int channels[] = { 0, 1, 2 };
    // Mat bgrArray[3];
    // split(img, bgrArray);
    // double minHist, maxHist;
    // cv::minMaxLoc(bgrArray[0], &minHist, &maxHist);
    // float b_ranges1[] = { minHist, maxHist };
    // cv::minMaxLoc(bgrArray[1], &minHist, &maxHist);
    // float g_ranges1[] = { minHist, maxHist };
    // cv::minMaxLoc(bgrArray[2], &minHist, &maxHist);
    // float r_ranges1[] = { minHist, maxHist };
    // const float* ranges1[] = { b_ranges1, g_ranges1, r_ranges1 };
    // split(otherImage, bgrArray);
    // cv::minMaxLoc(bgrArray[0], &minHist, &maxHist);
    // float r_ranges2[] = { minHist, maxHist };
    // cv::minMaxLoc(bgrArray[0], &minHist, &maxHist);
    // float g_ranges2[] = { minHist, maxHist };
    // cv::minMaxLoc(bgrArray[0], &minHist, &maxHist);
    // float b_ranges2[] = { minHist, maxHist };
    // const float* ranges2[] = { b_ranges2, g_ranges2, r_ranges2 };
    // calcHist(&img, 1, channels, Mat(), hist, 3, histSize, ranges1);
    // normalize( hist, hist, 0, 1, NORM_MINMAX, -1, Mat() );
    // calcHist(&otherImage, 1, channels, Mat(), histOther, 3, histSize, ranges2);
    // normalize( histOther, histOther, 0, 1, NORM_MINMAX, -1, Mat() );
    // linearizedOther.convertTo(linearizedOther, CV_32FC1);
    // cout << "BhattaDist = " << bd.imageTo<uchar>(otherImage) << endl;
    // cout << "CompareHist = " << compareHist(hist, histOther, HISTCMP_BHATTACHARYYA) << endl;

    Mat icovar = md.c().inv();
    Mat averageCV = md.reference().t();


    for (int i = 0; i < linearized.rows; i++) {
        distMat.at<double>(i) = Mahalanobis(linearized.row(i), averageCV, icovar);
    }
    resultCV = delinearizeImage<double>(distMat, img.rows, img.cols);

    resultDiff = result - resultCV;
    // sqrt(resultDiff, resultDiff);
    // resultDiff = resultDiff.mul(resultDiff);

    normalize(result, result, 255, 0, NORM_MINMAX);
    result.convertTo(result, CV_8UC1);
    double min, max;
    cv::minMaxLoc(result, &min, &max);
    std::cout << "min" << min << endl;
    std::cout << "max" << max << endl;
    // result = (result - min) / (max - min);

    normalize(resultCV, resultCV, 255, 0, NORM_MINMAX);
    resultCV.convertTo(resultCV, CV_8UC1);
    double minCV, maxCV;
    cv::minMaxLoc(resultCV, &minCV, &maxCV);
    std::cout << "minCV" << minCV << endl;
    std::cout << "maxCV" << maxCV << endl;
    // resultCV = (resultCV - minCV) / (maxCV - minCV);

    // normalize(resultPoly, resultPoly, 255, 0, NORM_MINMAX);
    // resultPoly.convertTo(resultPoly, CV_8UC1);
    double minPoly, maxPoly;
    cv::minMaxLoc(resultPoly, &minPoly, &maxPoly);
    std::cout << "minPoly" << minPoly << endl;
    std::cout << "maxPoly" << maxPoly << endl;
    // resultPoly = (resultPoly - minPoly) / (maxPoly - minPoly);

    normalize(resultDiff, resultDiff, 255, 0, NORM_MINMAX);
    resultDiff.convertTo(resultDiff, CV_8UC1);
    double minDiff, maxDiff;
    cv::minMaxLoc(resultDiff, &minDiff, &maxDiff);
    std::cout << "minDiff" << minDiff << endl;
    std::cout << "maxDiff" << maxDiff << endl;
    // resultDiff = (resultDiff - minDiff) / (maxDiff - minDiff);

    if (coordinates) {
        for (int i = 0; i < mat.rows; i++) {
            circle(img, Point(mat.row(i)), 3, 255, -1);
            circle(resultCV, Point(mat.row(i)), 3, 255, -1);
            circle(result, Point(mat.row(i)), 3, 255, -1);
            circle(resultPoly, Point(mat.row(i)), 3, 255, -1);
            circle(resultDiff, Point(mat.row(i)), 3, 255, -1);
        }
        circle(img, Point(md.reference()), 5, 255, -1);
        circle(resultCV, Point(md.reference()), 5, 255, -1);
        circle(result, Point(md.reference()), 5, 255, -1);
        circle(resultPoly, Point(md.reference()), 5, 255, -1);
        circle(resultDiff, Point(md.reference()), 5, 255, -1);
    }

    // cout << resultPoly << endl;

    // for (int i = 0; i < resultPoly.rows; i++) {
    //     for (int j = 0; j < resultPoly.cols; j++) {
    //         cout << resultPoly.at<double>(i,j) << endl;
    //         imshow("kkk", resultPoly(Rect(i,j,1,1)));
    //     }
    // }

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

    // cout << "polysize " << resultPoly.size() << endl;

    //imshow("kkk", result_old);
    while (true){
        // imshow("kkkkkkkkkkk", img);
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