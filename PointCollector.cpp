#include "PointCollector.hpp"

PointCollector::PointCollector(Mat& input, Mat& output)
    : _inputImage(input), _collectedPoints(output), _showCollectedPoints(true) {
        _inputImage.copyTo(_paintedImage);

        namedWindow("image");

        bool pressed = false;
        CallbackParams cp = {_inputImage, _paintedImage, _collectedPoints, pressed};

        setMouseCallback("image", onMouse, (void*)&cp);

        while (true) {
            imshow("image", _showCollectedPoints ? _paintedImage : _inputImage);

            char c = (char) waitKey(1);

            if (c == 27) break;

            switch (c) {
                case 's':
                    _showCollectedPoints != _showCollectedPoints;
                    break;
                // case 
            }
        }

        destroyWindow("image");
    }

PointCollector::PointCollector(const char* path, cv::ImreadModes flags, Mat& output)
    : _inputImage(imread(path, flags)), _collectedPoints(output), _showCollectedPoints(true) {
        _inputImage.copyTo(_paintedImage);

        namedWindow("image");

        bool pressed = false;
        CallbackParams cp = {_inputImage, _paintedImage, _collectedPoints, pressed};

        setMouseCallback("image", onMouse, (void*)&cp);

        while (true) {
            imshow("image", _showCollectedPoints ? _paintedImage : _inputImage);

            char c = (char) waitKey(1);

            if (c == 27) break;

            switch (c) {
                case 's':
                    _showCollectedPoints != _showCollectedPoints;
                    break;
                // case 
            }
        }

        destroyWindow("image");
    }

PointCollector::PointCollector() {}
    
PointCollector::~PointCollector() {}



Mat& PointCollector::inputImage() {
    return _inputImage;
}

Mat& PointCollector::paintedImage() {
    return _paintedImage;
}

Mat& PointCollector::collectedPoints() {
    return _collectedPoints;
}

void PointCollector::run() {
    // namedWindow("image");
    // setMouseCallback("image", onMouse);
    // destroyWindow("image");
}

void PointCollector::onMouse(int event, int x, int y, int flags, void* param) {
    CallbackParams* mp = (CallbackParams*) param;
    bool& pressed = mp->pressed;
    // cout << "kkk" << endl;
    if (event == EVENT_LBUTTONDOWN) pressed = true;
    if (event == EVENT_LBUTTONUP) pressed = false;
    if (pressed) {
    // if (first) {
    //   firstPoint = Mat(2,1, CV_64FC1);
    //   firstPoint.at<double>(0) = x;
    //   firstPoint.at<double>(1) = y;
    //   // firstPoint.at<double>(2) = 0.0;
    //   first = 0;

      circle(mp->paintedImg, Point(x,y), 3, Scalar(255, 255, 255), -1);
    // } else {
    //   circle(mp->img, Point(x,y), 3, Scalar(255), -1);
    // }

    int chans = mp->img.channels();

    Mat bgr = Mat(1, chans, CV_64FC1);
    Mat splitmat[chans];
    split(Mat(mp->img, Rect(x,y,1,1)).clone(), splitmat);

    switch (mp->img.type() % 8) {
        case 0:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<uchar>(0,0);    
            }
            break;
        case 1:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<schar>(0,0);    
            }
            break;
        case 2:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<ushort>(0,0);    
            }
            break;
        case 3:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<short>(0,0);    
            }
            break;
        case 4:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<int>(0,0);    
            }
            break;
        case 5:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<float>(0,0);    
            }
            break;
        case 6:
            for (int i = 0; i < chans; i++) {
                bgr.at<double>(0,i) = (double)splitmat[i].at<double>(0,0);    
            }
            break;
        default:
            assert(mp->img.type() % 8 != 7 || mp->img.type() >= 0);
    }
    // bgr.at<double>(0,0) = (double)mp->img.at<Vec3b>(y,x)[0];
    // bgr.at<double>(0,1) = (double)mp->img.at<Vec3b>(y,x)[1];
    // bgr.at<double>(0,2) = (double)mp->img.at<Vec3b>(y,x)[2];

    // imshow("image", mp->img);

    mp->points.push_back(bgr);
    // cout << mp->points << endl;
  }
}