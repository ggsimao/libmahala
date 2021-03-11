#include "MahalanobisDistance.hpp"
#include "PolynomialMahalanobisDistance.hpp"
#include "BhattacharyyaDistance.hpp"
#include "PointCollector.hpp"
#include "CoordinateCollector.hpp"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    const char* filename = argc >= 2 ? argv[1] : "coelho.jpg";

    bool coordinates = true;
    Mat mat = Mat();
    Mat result;
    PointCollector pc;
    CoordinateCollector cc;
    Mat resultCV, resultDiff;

    if (!coordinates) {
        pc = PointCollector(filename, IMREAD_COLOR, mat);
        Mat img = imread(filename, IMREAD_COLOR);
        result = Mat(img.size(), CV_64FC1);
        mat = pc.collectedPoints();

        MahalaDist md = MahalaDist(mat, 0.00000001);
        md.build();

        result = md.imageToReference<uchar>(img);

        {
            resultCV = Mat(img.size(), CV_64FC1);
            int numberOfChannels = img.channels();

            Mat linearized = Mat(img.rows * img.cols, numberOfChannels, CV_64FC1);
            Mat distMat = Mat(img.rows * img.cols, 1, CV_64FC1);
            Mat bgrArray[numberOfChannels];
            split(img, bgrArray);
            
            for (int c = 0; c < numberOfChannels; c++) {
                Mat a = bgrArray[c];
                for (int i = 0; i < img.rows; i++) {
                    for (int j = 0; j < img.cols; j++) {
                        int currentIndex = i*img.cols + j;
                        linearized.at<double>(currentIndex, c) = (double)a.at<uchar>(i,j);
                    }
                }
            }

            Mat icovar = md.c();
            Mat average = md.reference();

            std::vector<double> averageVec;
            for (int i = 0; i < average.rows; i++)
                averageVec.push_back(average.at<double>(i));

            for (int i = 0; i < linearized.rows; i++) {
                Mat pixel = linearized.row(i);
                std::vector<double> pixelVec;
                for (int i = pixel.cols; i > 0; i--)
                    pixelVec.push_back(pixel.at<double>(i-1));
                distMat.at<double>(i) = Mahalanobis(pixelVec, averageVec, icovar);
            }
            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    int currentIndex = i*img.cols + j;
                    for (int c = 0; c < numberOfChannels; c++) {
                        resultCV.at<double>(i, j) = distMat.at<double>(currentIndex, c);
                    }
                }
            }
        }

        resultDiff = result - resultCV;
        sqrt(resultDiff, resultDiff);
        resultDiff = resultDiff.mul(resultDiff);
    } else {
        Mat img = Mat::zeros(256, 256, CV_64FC1);
        Mat average = Mat();
        cc = CoordinateCollector(mat, average);
        mat = cc.collectedPoints();
        average = cc.reference();


        MahalaDist md = MahalaDist(mat, 0.00000001, average);
        cout << average << endl;
        md.build();

        result = md.coordinateToReference();

        {
            resultCV = Mat(256, 256, CV_64FC1);
            resultDiff = Mat(256, 256, CV_64FC1);

            Mat linearized = Mat(256 * 256, 2, CV_64FC1);
            Mat distMat = Mat(1, 256 * 256, CV_64FC1);

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    int currentIndex = i*256 + j;
                    linearized.at<double>(currentIndex, 0) = (double)j;
                    linearized.at<double>(currentIndex, 1) = (double)i;
                }
            }

            Mat icovar = md.c();

            std::vector<double> averageVec;
            for (int i = 0; i < average.rows; i++)
                averageVec.push_back(average.at<double>(i));

            for (int i = 0; i < linearized.rows; i++) {
                Mat pixel = linearized.row(i);
                std::vector<double> pixelVec;
                for (int i = pixel.cols; i > 0; i--)
                    pixelVec.push_back(pixel.at<double>(i-1));
                distMat.at<double>(i) = Mahalanobis(pixelVec, averageVec, icovar);
            }

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    resultCV.at<double>(i,j) = distMat.at<double>(i*256 + j);
                }
            }

            for (int i = 0; i < md.inputMatrix().rows; i++) {
                circle(resultCV, Point(md.inputMatrix().at<double>(i,0), md.inputMatrix().at<double>(i,1)), 3, 1, -1);
            }
            circle(resultCV, Point(md.reference()), 5, 1, -1);

            resultDiff = result - resultCV;
            sqrt(resultDiff, resultDiff);
            resultDiff = resultDiff.mul(resultDiff);
        }
    }

    double min, max;
    cv::minMaxLoc(result, &min, &max);
    std::cout << "min" << min << endl;
    std::cout << "max" << max << endl;

    double minCV, maxCV;
    cv::minMaxLoc(resultCV, &minCV, &maxCV);
    std::cout << "minCV" << minCV << endl;
    std::cout << "maxCV" << maxCV << endl;

    double minDiff, maxDiff;
    cv::minMaxLoc(resultDiff, &minDiff, &maxDiff);
    std::cout << "minDiff" << minDiff << endl;
    std::cout << "maxDiff" << maxDiff << endl;

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

    //imshow("kkk", result_old);
    while (true){
        // imshow("kkkkkkkkkkk", cc.paintedImage());
        if (coordinates) {
            imshow("kkkkkkkkkkk", cc.paintedImage());
        } else {
            imshow("kkkkkkkkkkk", pc.paintedImage());
        }
        imshow("meuresultado", (result - min) / (max));
        imshow("cvresultado", (resultCV - minCV) / (maxCV));
        imshow("diffresultado", (resultDiff - minCV) / (maxCV));
        if (waitKey(0) == 27) break;
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