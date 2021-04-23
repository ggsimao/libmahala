#include "libmahala/MahalanobisDistance.hpp"
#include "libmahala/BhattacharyyaDistance.hpp"
#include "libmahala/PolynomialMahalanobisDistance.hpp"
#include "libmahala/PointCollector.hpp"

#include <iostream>
#include <chrono>
#include <ctime> 

void test(String name, Mat& input, Mat& resultEuc, Mat& resultMaha,
            Mat& resultMaha2, Mat& resultMaha4, Mat& resultMaha6);

int main() {
    
    cv::Mat input_flower, input_church, input_horses, input_place, input_starfish;
    cv::Mat thEuc_flower, thMaha_flower, thMaha2_flower, thMaha4_flower, thMaha6_flower;
    cv::Mat thEuc_church, thMaha_church, thMaha2_church, thMaha4_church, thMaha6_church;
    cv::Mat thEuc_horses, thMaha_horses, thMaha2_horses, thMaha4_horses, thMaha6_horses;
    cv::Mat thEuc_place, thMaha_place, thMaha2_place, thMaha4_place, thMaha6_place;
    cv::Mat thEuc_starfish, thMaha_starfish, thMaha2_starfish, thMaha4_starfish,
            thMaha6_starfish;
    std::vector<String> names = {"flower", "church", "horses", "place", "starfish"};

    test(names[0], input_flower, thEuc_flower, thMaha_flower, thMaha2_flower,
            thMaha4_flower, thMaha6_flower);
    test(names[1], input_church, thEuc_church, thMaha_church, thMaha2_church,
            thMaha4_church, thMaha6_church);
    test(names[2], input_horses, thEuc_horses, thMaha_horses, thMaha2_horses,
            thMaha4_horses, thMaha6_horses);
    test(names[3], input_place, thEuc_place, thMaha_place, thMaha2_place,
            thMaha4_place, thMaha6_place);
    test(names[4], input_starfish, thEuc_starfish, thMaha_starfish, thMaha2_starfish,
            thMaha4_starfish, thMaha6_starfish);

    auto startBhatt = std::chrono::system_clock::now();
    std::vector<int> channels = {0, 1, 2};
    std::vector<int> size = {256, 256, 256};
    std::vector<float> ranges = {0, 256, 0, 256, 0, 256};
    BhattaDist bd = BhattaDist(channels, size, ranges);

    thMaha6_flower.convertTo(thMaha6_flower, CV_8UC1);
    thMaha6_church.convertTo(thMaha6_church, CV_8UC1);
    thMaha6_horses.convertTo(thMaha6_horses, CV_8UC1);
    thMaha6_place.convertTo(thMaha6_place, CV_8UC1);
    thMaha6_starfish.convertTo(thMaha6_starfish, CV_8UC1);

    double dist_flower2church = bd.calcBetweenImg(input_flower, input_church,
                                thMaha6_flower, thMaha6_church);
    double dist_flower2horses = bd.calcBetweenImg(input_flower, input_horses,
                                thMaha6_flower, thMaha6_horses);
    double dist_flower2place = bd.calcBetweenImg(input_flower, input_place,
                                thMaha6_flower, thMaha6_place);
    double dist_flower2starfish = bd.calcBetweenImg(input_flower, input_starfish,
                                thMaha6_flower, thMaha6_starfish);

    double dist_horses2church = bd.calcBetweenImg(input_horses, input_church,
                                thMaha6_horses, thMaha6_church);
    double dist_horses2place = bd.calcBetweenImg(input_horses, input_place,
                                thMaha6_horses, thMaha6_place);
    double dist_horses2starfish = bd.calcBetweenImg(input_horses, input_starfish,
                                thMaha6_horses, thMaha6_starfish);

    auto endBhatt = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsBhatt = startBhatt-endBhatt;
    cout << "elapsed_secondsBhatt: " << elapsed_secondsBhatt.count() << endl;

    cout << "dist_flower2church: " << dist_flower2church << endl;
    cout << "dist_flower2horses: " << dist_flower2horses << endl;
    cout << "dist_flower2place: " << dist_flower2place << endl;
    cout << "dist_flower2starfish: " << dist_flower2starfish << endl;

    cout << "dist_horses2church: " << dist_horses2church << endl;
    cout << "dist_horses2place: " << dist_horses2place << endl;
    cout << "dist_horses2starfish: " << dist_horses2starfish << endl;
}

void test(String name, Mat& input, Mat& resultEuc, Mat& resultMaha,
            Mat& resultMaha2, Mat& resultMaha4, Mat& resultMaha6) {
    cout << name << "_begin" << endl;

    cv::Mat img = cv::imread("images/"+name+".jpg", cv::IMREAD_COLOR);
    input = img.clone();
    PointCollector pc = PointCollector(img);

    cv::Mat cp = pc.collectedPixels();
    cv::Mat rp = pc.referencePixel();

    cv::Mat cc = pc.collectedCoordinates();
    cv::Mat rc = pc.referenceCoordinate();
    cv::Mat painted_img = img.clone();

    // cout << "pontos extraídos" << endl;

    cv::circle(painted_img, Point(rc), 5, Scalar(255, 255, 255), -1);
    // cout << "círculo grande desenhado" << endl;
    for (int i = 0; i < cp.rows; i++) {
        cv::circle(painted_img, Point(cc.row(i)), 3, Scalar(255, 255, 255), -1);
    }
    // cout << "círculos pequenos desenhados" << endl;

    cv::imwrite("images/"+name+"_painted.jpg", painted_img);

    // cout << "imagem salva" << endl;

    auto startEuc = std::chrono::system_clock::now();
    cv::Mat distsEuc = cv::Mat(img.size(), CV_64F);
    // cout << "matriz criada" << endl;
    for (uint i = 0; i < img.rows; i++) {
        for (uint j = 0; j < img.cols; j++) {
            cv::Mat pointSplit = cv::Mat(3, 1, CV_64F);
            // cout << "matrizinha criada" << endl;
            pointSplit.at<double>(0) = double(img.at<Vec3b>(i, j)[0]);
            pointSplit.at<double>(1) = double(img.at<Vec3b>(i, j)[1]);
            pointSplit.at<double>(2) = double(img.at<Vec3b>(i, j)[2]);
            // cout << "valores setados" << endl;
            distsEuc.at<double>(i, j) = norm(pointSplit - rp);
            // cout << "distância calculada" << endl;
        }
    }
    auto endEuc = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsEuc = endEuc-startEuc;
    cout << "elapsed_secondsEuc: " << elapsed_secondsEuc.count() << endl;

    double minEuc, maxEuc;
    cv::minMaxLoc(distsEuc, &minEuc, &maxEuc);
    cv::threshold(distsEuc, resultEuc, maxEuc/2, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("images/"+name+"_thresholdedEuc.jpg", resultEuc);

    auto startMahala = std::chrono::system_clock::now();
    MahalaDist md = MahalaDist(cp, 4e-6, rp);
    md.build();
    cv::Mat resultMahala = md.imageToReference<uchar>(img);
    auto endMahala = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsMahala = endMahala-startMahala;
    cout << "elapsed_secondsMahala: " << elapsed_secondsMahala.count() << endl;

    double minMahala, maxMahala;
    cv::minMaxLoc(resultMahala, &minMahala, &maxMahala);
    cv::threshold(resultMahala, resultMaha, maxMahala/2, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("images/"+name+"_thresholdedMahala.jpg", resultMaha);

    auto startPolyMahala2 = std::chrono::system_clock::now();
    PolyMahalaDist pmd2 = PolyMahalaDist(cp, 2, 4e-6, rp);
    cv::Mat resultPolyMahala2 = pmd2.imageToReference<uchar>(img);
    auto endPolyMahala2 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsPolyMahala2 = endPolyMahala2-startPolyMahala2;
    cout << "elapsed_secondsPolyMahala2: " << elapsed_secondsPolyMahala2.count() << endl;

    double minPolyMahala2, maxPolyMahala2;
    cv::minMaxLoc(resultMahala, &minPolyMahala2, &maxPolyMahala2);
    cv::threshold(resultPolyMahala2, resultMaha2, 1, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("images/"+name+"_thresholdedPolyMahala2.jpg", resultMaha2);

    auto startPolyMahala4 = std::chrono::system_clock::now();
    PolyMahalaDist pmd4 = PolyMahalaDist(cp, 4, 4e-6, rp);
    cv::Mat resultPolyMahala4 = pmd4.imageToReference<uchar>(img);
    auto endPolyMahala4 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsPolyMahala4 = endPolyMahala4-startPolyMahala4;
    cout << "elapsed_secondsPolyMahala4: " << elapsed_secondsPolyMahala4.count() << endl;

    double minPolyMahala4, maxPolyMahala4;
    cv::minMaxLoc(resultMahala, &minPolyMahala4, &maxPolyMahala4);
    cv::threshold(resultPolyMahala4, resultMaha4, 1, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("images/"+name+"_thresholdedPolyMahala4.jpg", resultMaha4);

    auto startPolyMahala6 = std::chrono::system_clock::now();
    PolyMahalaDist pmd6 = PolyMahalaDist(cp, 6, 4e-6, rp);
    cv::Mat resultPolyMahala6 = pmd6.imageToReference<uchar>(img);
    auto endPolyMahala6 = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_secondsPolyMahala6 = endPolyMahala6-startPolyMahala6;
    cout << "elapsed_secondsPolyMahala6: " << elapsed_secondsPolyMahala6.count() << endl;

    double minPolyMahala6, maxPolyMahala6;
    cv::minMaxLoc(resultMahala, &minPolyMahala6, &maxPolyMahala6);
    cv::threshold(resultPolyMahala6, resultMaha6, 1, 255, cv::THRESH_BINARY_INV);
    cv::imwrite("images/"+name+"_thresholdedPolyMahala6.jpg", resultMaha6);
    cout << name << "_end" << endl;
}