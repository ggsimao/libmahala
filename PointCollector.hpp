#pragma once

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <vector>
#include <math.h>

using namespace cv;
using namespace std;

// TODO: Assign keys for changing the color of the collected points highlight

/*! \brief Creates an interactible window where the user can collect points from an image.
           While the window is active, the 'R' key resets all attributes, the 'S' toggles
           the collected points highlight, the 'Esc' key closes the window and ends the
           point collection process, the '1' and '2' control the blue value of the highlight
           color, the '3' and '4' keys control the green value, and the '5' and '6' keys
           control the red value. The regular color offset is 1, being 16 when the 'Shift'
           key is being pressed and 64 and the 'Alt' key is being pressed.
    \param _collectedPixels The values of the collected pixels organized in a NxD matrix
                            where N is the number of collected points and D is the number
                            of channels in the image
    \param _collectedCoordinates The row and column indexes of the collected pixels organized
                                 in a Nx2 matrix where N is the number of collected points;
                                 the row index is stored in the second column of the matrix
                                 while the column index is stored in the first column
    \param _referencePixel The pixel values of the single point chosen with the right mouse button
    \param _referenceCoordinate The row and column indexes of the single point chosen with
                                the right mouse button
*/
class PointCollector {
public:
    /*! \brief Constructor that shows and maintains the interactible window.
        \param input The image that will be shown and that the user will gather
                     samples from
    */
    PointCollector(Mat& input);
    /*! \brief Constructor that shows and maintains the interactible window
        \param path The path to the fila that contains the image that will be shown
                    and that the user will gather samples from
        \param flags The flags that OpenCV will use to open the file
    */
    PointCollector(const char* path, cv::ImreadModes flags);
    /*! \brief Default constructor for declaring before initializing
    */
    PointCollector();
    virtual ~PointCollector();

    /*
     * GETTERS
     */
    Mat& collectedPixels();
    Mat& collectedCoordinates();
    Mat& referencePixel();
    Mat& referenceCoordinate();
private:
    Mat _collectedPixels;
    Mat _collectedCoordinates;
    Mat _referencePixel;
    Mat _referenceCoordinate;

    /*! \brief Struct used to pass variables back-and-forth between the constructor
               and the onMouse callback method
        \param img The input image
        \param paintedImg The input image with the selected points highlighted
        \param pixels The values of the collected pixels organized in a NxD matrix
                      where N is the number of collected points and D is the number
                      of channels in the image
        \param coordinates The row and column indexes of the collected pixels organized
                           in a Nx2 matrix where N is the number of collected points;
                           the row index is stored in the second column of the matrix
                           while the column index is stored in the first column
        \param pressedLeft Determines whether the left mouse button is pressed
        \param pressedRight Determines whether the right mouse button is pressed
        \param referencePixel The pixel values of the single point chosen with the right mouse button
        \param referenceCoordinate The row and column indexes of the single point chosen with
                                   the right mouse button
        \param color Color in which the selected points will be painted on paintedImg
    */
    struct CallbackParams {
        Mat& img;
        Mat& paintedImg;
        Mat& pixels;
        Mat& coordinates;
        bool& pressedLeft;
        bool& pressedRight;
        Mat& referencePixel;
        Mat& referenceCoordinate;
        Scalar& color;
    };

    static void redraw(const Mat& originalImage, Mat& imageToPaint, const Mat& points, const Mat& reference, const Scalar& color);

    static void onMouse(int event, int x, int y, int flags, void* param);
};