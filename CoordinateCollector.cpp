#include "CoordinateCollector.hpp"

CoordinateCollector::CoordinateCollector(Mat& output, Mat& reference)
    : _collectedPoints(output), _reference(reference), _showCollectedPoints(true) {
        _inputImage = Mat::zeros(256, 256, CV_64FC1);
        _inputImage.copyTo(_paintedImage);

        namedWindow("image");

        bool pressed = false;
        bool pressedRight = false;
        // Mat ref = Mat(0, 2, CV_64FC1);
        CallbackParams cp = {_inputImage, _paintedImage, _collectedPoints, pressed, pressedRight, _reference};

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

CoordinateCollector::CoordinateCollector() {}
    
CoordinateCollector::~CoordinateCollector() {}

Mat& CoordinateCollector::paintedImage() {
    return _paintedImage;
}
Mat& CoordinateCollector::collectedPoints() {
    return _collectedPoints;
}
Mat& CoordinateCollector::reference() {
    return _reference;
}

void CoordinateCollector::run() {
    // namedWindow("image");
    // setMouseCallback("image", onMouse);
    // destroyWindow("image");
}

void CoordinateCollector::onMouse(int event, int x, int y, int flags, void* param) {
    CallbackParams* mp = (CallbackParams*) param;
    bool& pressed = mp->pressed;
    bool& pressedRight = mp->pressedRight;
    // cout << "kkk" << endl;
    if (event == EVENT_LBUTTONDOWN) pressed = true;
    if (event == EVENT_LBUTTONUP) pressed = false;
    if (event == EVENT_RBUTTONDOWN) pressedRight = true;
    if (event == EVENT_RBUTTONUP) pressedRight = false;
    // cout << mp->reference.rows << endl;
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

        Mat bgr = Mat(1, 2, CV_64FC1);
    
        bgr.at<double>(0,0) = (double)x;
        bgr.at<double>(0,1) = (double)y;

        // cout << bgr << endl;
        
        // bgr.at<double>(0,0) = (double)mp->img.at<Vec3b>(y,x)[0];
        // bgr.at<double>(0,1) = (double)mp->img.at<Vec3b>(y,x)[1];
        // bgr.at<double>(0,2) = (double)mp->img.at<Vec3b>(y,x)[2];

        // imshow("image", mp->img);

        mp->points.push_back(bgr);
    // cout << mp->points << endl;
    }
    if (pressedRight && mp->reference.rows <= 0) {
        Mat bgr = Mat(1, 2, CV_64FC1);
        bgr.at<double>(0,0) = (double)x;
        bgr.at<double>(0,1) = (double)y;
        mp->reference = bgr.t();
        // cout << bgr.type() << endl;
        circle(mp->paintedImg, Point(x,y), 5, Scalar(255, 255, 255), -1);
    }
}