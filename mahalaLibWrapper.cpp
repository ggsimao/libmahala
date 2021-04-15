#include <Python.h>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "MahalanobisDistance.hpp"
#include "PolynomialMahalanobisDistance.hpp"
#include "BhattacharyyaDistance.hpp"
#include "PointCollector.hpp"

#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(libmahalapy, m) {
    NDArrayConverter::init_numpy();

    cv::Mat emptyMat = cv::Mat();

    py::class_<MahalaDist>(m, "MahalaDist")
        .def(py::init<const cv::Mat&, double, cv::Mat>(),
            py::arg("input"), py::arg("smin") = 4e-6, py::arg("reference") = emptyMat)
        .def_property_readonly("reference", &MahalaDist::reference)
        .def_property_readonly("dimension", &MahalaDist::dimension)
        .def_property_readonly("dirty", &MahalaDist::dirty)
        .def_property_readonly("u", &MahalaDist::u)
        .def_property_readonly("w", &MahalaDist::w)
        .def_property_readonly("c", &MahalaDist::c)
        .def_property_readonly("sigma2", &MahalaDist::sigma2)
        .def_property("smin", 
            static_cast<double (MahalaDist::*)()> (&MahalaDist::smin),
            static_cast<void (MahalaDist::*)(double)> (&MahalaDist::smin))
        .def("build", &MahalaDist::build)
        .def("pointTo", &MahalaDist::pointTo,
            py::arg("point1"), py::arg("point2"))
        .def("pointToReference", &MahalaDist::pointToReference,
            py::arg("point"))
        .def("pointsTo", &MahalaDist::pointsTo,
            py::arg("points"), py::arg("ref"))
        .def("pointsToReference", &MahalaDist::pointsToReference,
            py::arg("points"))
        .def("imageTo_uchar", &MahalaDist::imageTo<uchar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_shcar", &MahalaDist::imageTo<schar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_ushort", &MahalaDist::imageTo<ushort>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_short", &MahalaDist::imageTo<short>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_int", &MahalaDist::imageTo<int>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_float", &MahalaDist::imageTo<float>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_double", &MahalaDist::imageTo<double>,
            py::arg("image"), py::arg("refVector"))
        .def("imageToReference_uchar", &MahalaDist::imageToReference<uchar>,
            py::arg("image"))
        .def("imageToReference_schar", &MahalaDist::imageToReference<schar>,
            py::arg("image"))
        .def("imageToReference_ushort", &MahalaDist::imageToReference<ushort>,
            py::arg("image"))
        .def("imageToReference_short", &MahalaDist::imageToReference<short>,
            py::arg("image"))
        .def("imageToReference_int", &MahalaDist::imageToReference<int>,
            py::arg("image"))
        .def("imageToReference_float", &MahalaDist::imageToReference<float>,
            py::arg("image"))
        .def("imageToReference_double", &MahalaDist::imageToReference<double>,
            py::arg("image"))
        ;
    
    py::class_<PointCollector>(m, "PointCollector")
        .def(py::init<cv::Mat&>(),
            py::arg("input"))
        .def(py::init<const char*, int>(),
            py::arg("path"), py::arg("flags"))
        .def_property_readonly("collectedPixels", &PointCollector::collectedPixels)
        .def_property_readonly("collectedCoordinates", &PointCollector::collectedCoordinates)
        .def_property_readonly("referencePixel", &PointCollector::referencePixel)
        .def_property_readonly("referenceCoordinate", &PointCollector::referenceCoordinate)
        ;

    py::class_<PolyMahalaDist>(m, "PolyMahalaDist")
        .def(py::init<const cv::Mat&, int, double, cv::Mat>(),
            py::arg("input"), py::arg("order"), py::arg("sig_max") = 4e-6, py::arg("reference") = emptyMat)
        .def_property_readonly("reference", &PolyMahalaDist::reference)
        .def_property_readonly("dimension", &PolyMahalaDist::dimension)
        .def_property_readonly("eps_svd", &PolyMahalaDist::eps_svd)
        .def_property_readonly("order", &PolyMahalaDist::order)
        .def("pointTo", &PolyMahalaDist::pointTo,
            py::arg("im_data"), py::arg("refVector"))
        .def("pointToReference", &PolyMahalaDist::pointToReference,
            py::arg("im_data"))
        .def("pointsTo", &PolyMahalaDist::pointsTo,
            py::arg("im_data"), py::arg("refVector"))
        .def("pointsToReference", &PolyMahalaDist::pointsToReference,
            py::arg("im_data"))
        .def("imageTo_uchar", &PolyMahalaDist::imageTo<uchar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_shcar", &PolyMahalaDist::imageTo<schar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_ushort", &PolyMahalaDist::imageTo<ushort>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_short", &PolyMahalaDist::imageTo<short>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_int", &PolyMahalaDist::imageTo<int>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_float", &PolyMahalaDist::imageTo<float>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo_double", &PolyMahalaDist::imageTo<double>,
            py::arg("image"), py::arg("refVector"))
        .def("imageToReference_uchar", &PolyMahalaDist::imageToReference<uchar>,
            py::arg("image"))
        .def("imageToReference_schar", &PolyMahalaDist::imageToReference<schar>,
            py::arg("image"))
        .def("imageToReference_ushort", &PolyMahalaDist::imageToReference<ushort>,
            py::arg("image"))
        .def("imageToReference_short", &PolyMahalaDist::imageToReference<short>,
            py::arg("image"))
        .def("imageToReference_int", &PolyMahalaDist::imageToReference<int>,
            py::arg("image"))
        .def("imageToReference_float", &PolyMahalaDist::imageToReference<float>,
            py::arg("image"))
        .def("imageToReference_double", &PolyMahalaDist::imageToReference<double>,
            py::arg("image"))
        ;

    py::class_<BhattaDist>(m, "BhattaDist")
        .def(py::init<std::vector<int>, std::vector<int>, std::vector<float>>(),
            py::arg("channels"), py::arg("histSize"), py::arg("ranges"))
        .def_property_readonly("channels", &BhattaDist::channels)
        .def_property_readonly("histSize", &BhattaDist::histSize)
        .def_property_readonly("ranges", &BhattaDist::ranges)
        .def_static("calcBetweenHist", &BhattaDist::calcBetweenHist,
            py::arg("hist1"), py::arg("hist2"))
        .def("calcBetweenImg", &BhattaDist::calcBetweenImg,
            py::arg("image1"), py::arg("image2"), py::arg("mask1") = emptyMat, py::arg("mask2") = emptyMat)
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints,
            py::arg("points1"), py::arg("points2"))
        ;
}