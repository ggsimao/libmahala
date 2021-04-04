#include "PointCollector.hpp"

PointCollector::PointCollector(Mat& input) {
    Mat paintedImage = input.clone();
    bool showCollectedPoints = true;

    Scalar color = Scalar(255, 255, 255);

    namedWindow("image");

    bool pressedLeft = false;
    bool pressedRight = false;
    CallbackParams cp = {input, paintedImage, _collectedPixels, 
                         _collectedCoordinates, pressedLeft, pressedRight,
                         _referencePixel, _referenceCoordinate, color};

    setMouseCallback("image", onMouse, (void*)&cp);

    while (true) {
        imshow("image", showCollectedPoints ? paintedImage : input);

        int c = waitKeyEx(1);

        // cout << (short)c << endl;
        // cout << c << endl;

        if ((short)c == 27) break;

        switch (c) {
            case 1048691: // s key
                showCollectedPoints = !showCollectedPoints;
                break;
            case 1048690: // r key
                _collectedPixels.release();
                _collectedCoordinates.release();
                _referencePixel.release();
                _referenceCoordinate.release();
                paintedImage = input.clone();
                break;
            case 1048625: // 1 key
                color[0] = max(0.0, color[0]-1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048626: // 2 key
                color[0] = min(255.0, color[0]+1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048627: // 3 key
                color[1] = max(0.0, color[1]-1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048628: // 4 key
                color[1] = min(255.0, color[1]+1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048629: // 5 key
                color[2] = max(0.0, color[2]-1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048630: // 6 key
                color[2] = min(255.0, color[2]+1);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114145: // SHIFT+1 key
                color[0] = max(0.0, color[0]-16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114176: // SHIFT+2 key
                color[0] = min(255.0, color[0]+16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114147: // SHIFT+3 key
                color[1] = max(0.0, color[1]-16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114148: // SHIFT+4 key
                color[1] = min(255.0, color[1]+16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114149: // SHIFT+5 key
                color[2] = max(0.0, color[2]-16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1179223: // SHIFT+6 key
                color[2] = min(255.0, color[2]+16);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572913: // ALT+1 key
                color[0] = max(0.0, color[0]-64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572914: // ALT+2 key
                color[0] = min(255.0, color[0]+64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572915: // ALT+3 key
                color[1] = max(0.0, color[1]-64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572916: // ALT+4 key
                color[1] = min(255.0, color[1]+64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572917: // ALT+5 key
                color[2] = max(0.0, color[2]-64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572918: // ALT+6 key
                color[2] = min(255.0, color[2]+64);
                redraw(input, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
        }
    }

    cv::destroyWindow("image");
}

PointCollector::PointCollector(const char* path, int flags) {
    cv::ImreadModes cvFlags = (cv::ImreadModes) flags;
    Mat inputImage = imread(path, cvFlags);
    Mat paintedImage = inputImage.clone();
    bool showCollectedPoints = true;

    Scalar color = Scalar(255, 255, 255);

    namedWindow("image");

    bool pressedLeft = false;
    bool pressedRight = false;
    CallbackParams cp = {inputImage, paintedImage, _collectedPixels, 
                         _collectedCoordinates, pressedLeft, pressedRight, 
                         _referencePixel, _referenceCoordinate, color};

    setMouseCallback("image", onMouse, (void*)&cp);

    while (true) {
        imshow("image", showCollectedPoints ? paintedImage : inputImage);

        int c = waitKeyEx(1);

        // cout << c << endl;

        if ((short)c == 27) break;

        switch (c) {
            case 1048691: // s key
                showCollectedPoints = !showCollectedPoints;
                break;
            case 1048690: // r key
                _collectedPixels.release();
                _collectedCoordinates.release();
                _referencePixel.release();
                _referenceCoordinate.release();
                paintedImage = inputImage.clone();
                break;
            case 1048625: // 1 key
                color[0] = max(0.0, color[0]-1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048626: // 2 key
                color[0] = min(255.0, color[0]+1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048627: // 3 key
                color[1] = max(0.0, color[1]-1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048628: // 4 key
                color[1] = min(255.0, color[1]+1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048629: // 5 key
                color[2] = max(0.0, color[2]-1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1048630: // 6 key
                color[2] = min(255.0, color[2]+1);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114145: // SHIFT+1 key
                color[0] = max(0.0, color[0]-16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114176: // SHIFT+2 key
                color[0] = min(255.0, color[0]+16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114147: // SHIFT+3 key
                color[1] = max(0.0, color[1]-16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114148: // SHIFT+4 key
                color[1] = min(255.0, color[1]+16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1114149: // SHIFT+5 key
                color[2] = max(0.0, color[2]-16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1179223: // SHIFT+6 key
                color[2] = min(255.0, color[2]+16);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572913: // ALT+1 key
                color[0] = max(0.0, color[0]-64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572914: // ALT+2 key
                color[0] = min(255.0, color[0]+64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572915: // ALT+3 key
                color[1] = max(0.0, color[1]-64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572916: // ALT+4 key
                color[1] = min(255.0, color[1]+64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572917: // ALT+5 key
                color[2] = max(0.0, color[2]-64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
            case 1572918: // ALT+6 key
                color[2] = min(255.0, color[2]+64);
                redraw(inputImage, paintedImage, _collectedCoordinates, _referenceCoordinate, color);
                break;
        }
    }

    cv::destroyWindow("image");
}

PointCollector::PointCollector() {}

PointCollector::~PointCollector() {}



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



void PointCollector::redraw(const Mat& originalImage, Mat& imageToPaint, const Mat& points, const Mat& reference, const Scalar& color) {
    // cout << color << endl;
    imageToPaint = originalImage.clone();
    if (reference.data) {
        circle(imageToPaint, Point(reference.at<double>(0),reference.at<double>(1)), 5, color, -1);
    }
    for (int i = 0; i < points.rows; i++) {
        Point toDraw = Point(points.at<double>(i, 0), points.at<double>(i, 1));
        circle(imageToPaint, toDraw, 3, color, -1);
    }
}

void PointCollector::onMouse(int event, int x, int y, int flags, void* param) {
    CallbackParams* mp = (CallbackParams*) param;
    bool& pressedLeft = mp->pressedLeft;
    bool& pressedRight = mp->pressedRight;


    if (event == EVENT_LBUTTONDOWN) pressedLeft = true;
    if (event == EVENT_LBUTTONUP) pressedLeft = false;
    if (event == EVENT_RBUTTONDOWN) pressedRight = true;
    if (event == EVENT_RBUTTONUP) pressedRight = false;
    if (pressedLeft || pressedRight) {
        int chans = mp->img.channels();
        x = max(min(mp->img.cols-1, x), 0);
        y = max(min(mp->img.rows-1, y), 0);

        // cout << "x = " << x << "; y = " << y << endl;

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
            redraw(mp->img, mp->paintedImg, mp->coordinates, mp->referenceCoordinate, mp->color);
            // mp->paintedImg = mp->img.clone();
            // circle(mp->paintedImg, Point(x,y), 5, Scalar(255, 255, 255), -1);
            // for (int i = 0; i < mp->coordinates.rows; i++) {
            //     Point toDraw = Point(mp->coordinates.at<double>(i, 0), mp->coordinates.at<double>(i, 1));
            //     circle(mp->paintedImg, toDraw, 3, Scalar(255, 255, 255), -1);
            // }
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