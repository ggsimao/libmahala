#include "PointCollector.hpp"

PointCollector::PointCollector(Mat& input)
    : _inputImage(input), _showCollectedPoints(true) {
        _paintedImage = _inputImage.clone();

        namedWindow("image");

        bool pressed = false;
        bool pressedRight = false;
        CallbackParams cp = {_inputImage, _paintedImage, _collectedPixels, 
                             _collectedCoordinates, pressed, pressedRight,
                             _referencePixel, _referenceCoordinate};

        setMouseCallback("image", onMouse, (void*)&cp);

        while (true) {
            imshow("image", _showCollectedPoints ? _paintedImage : _inputImage);

            char c = (char) waitKey(1);

            // cout << c << endl;

            if (c == 27) break;

            switch (c) {
                case 's':
                    _showCollectedPoints = !_showCollectedPoints;
                    break;
                case 'r':
                    _collectedPixels.release();
                    _collectedCoordinates.release();
                    _referencePixel.release();
                    _referenceCoordinate.release();
                    _paintedImage = _inputImage.clone();
                    break;
                // case 
            }
        }

        destroyWindow("image");
    }

PointCollector::PointCollector(const char* path, cv::ImreadModes flags)
    : _inputImage(imread(path, flags)), _showCollectedPoints(true) {
        _inputImage.copyTo(_paintedImage);

        namedWindow("image");

        bool pressed = false;
        bool pressedRight = false;
        CallbackParams cp = {_inputImage, _paintedImage, _collectedPixels, 
                             _collectedCoordinates, pressed, pressedRight, 
                             _referencePixel, _referenceCoordinate};

        setMouseCallback("image", onMouse, (void*)&cp);

        while (true) {
            imshow("image", _showCollectedPoints ? _paintedImage : _inputImage);

            char c = (char) waitKey(1);

            // cout << c << endl;

            if (c == 27) break;

            switch (c) {
                case 's':
                    _showCollectedPoints = !_showCollectedPoints;
                    break;
                case 'r':
                    _collectedPixels.release();
                    _collectedCoordinates.release();
                    _referencePixel.release();
                    _referenceCoordinate.release();
                    _paintedImage = _inputImage.clone();
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

Mat& PointCollector::collectedPixels() {
    return _collectedPixels;
}
Mat& PointCollector::collectedCoordinates() {
    return _collectedCoordinates;
}
Mat& PointCollector::referencePixel() {
    return _referencePixel;
}
Mat& PointCollector::referenceCoordinate() {
    return _referenceCoordinate;
}

void PointCollector::run() {
    // namedWindow("image");
    // setMouseCallback("image", onMouse);
    // destroyWindow("image");
}

void PointCollector::onMouse(int event, int x, int y, int flags, void* param) {
    CallbackParams* mp = (CallbackParams*) param;
    bool& pressed = mp->pressed;
    bool& pressedRight = mp->pressedRight;
    // cout << "kkk" << endl;
    if (event == EVENT_LBUTTONDOWN) pressed = true;
    if (event == EVENT_LBUTTONUP) pressed = false;
    if (event == EVENT_RBUTTONDOWN) pressedRight = true;
    if (event == EVENT_RBUTTONUP) pressedRight = false;
    if (pressed || pressedRight) {
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

        Mat xy = Mat(1, 2, CV_64FC1);
        xy.at<double>(0) = x;
        xy.at<double>(1) = y;

        if (pressedRight) {
            mp->referencePixel = bgr.t();
            mp->referenceCoordinate = xy.t();
            mp->paintedImg = mp->img.clone();
            circle(mp->paintedImg, Point(x,y), 5, Scalar(255, 255, 255), -1);
            for (int i = 0; i < mp->coordinates.rows; i++) {
                Point toDraw = Point(mp->coordinates.at<double>(i, 0), mp->coordinates.at<double>(i, 1));
                circle(mp->paintedImg, toDraw, 3, Scalar(255, 255, 255), -1);
            }
            // cout << mp->coordinates.rows << endl;
        } else {
            mp->pixels.push_back(bgr);
            mp->coordinates.push_back(xy);
            circle(mp->paintedImg, Point(x,y), 3, Scalar(255, 255, 255), -1);
            // cout << mp->coordinates.rows << endl;
        }
        // cout << mp->points << endl;
    }
}