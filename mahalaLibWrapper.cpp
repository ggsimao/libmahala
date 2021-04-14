#include <Python.h>
#include "pybind11/pybind11.h"
#include "MahalanobisDistance.hpp"
#include "PolynomialMahalanobisDistance.hpp"
#include "BhattacharyyaDistance.hpp"
#include "imageUtils.hpp"
#include "mathUtils.hpp"
#include "PointCollector.hpp"

#include "ndarray_converter.h"

namespace py = pybind11;

PYBIND11_MODULE(libmahala, m) {
    NDArrayConverter::init_numpy();

    cv::Mat emptyMat = cv::Mat();

    py::class_<MahalaDist>(m, "MahalaDist")
        .def(py::init<const cv::Mat&, double, cv::Mat>(),
            py::arg("input"), py::arg("smin"), py::arg("reference") = emptyMat)
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
        .def("imageTo", &MahalaDist::imageTo<uchar>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<schar>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<ushort>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<short>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<int>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<float>,
            py::arg("image"), py::arg("ref"))
        .def("imageTo", &MahalaDist::imageTo<double>,
            py::arg("image"), py::arg("ref"))
        .def("imageToReference", &MahalaDist::imageToReference<uchar>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<schar>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<ushort>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<short>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<int>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<float>,
            py::arg("image"))
        .def("imageToReference", &MahalaDist::imageToReference<double>,
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
            py::arg("input"), py::arg("order"), py::arg("sig_max"), py::arg("reference") = emptyMat)
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
        .def("imageTo", &PolyMahalaDist::imageTo<uchar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<schar>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<ushort>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<short>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<int>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<float>,
            py::arg("image"), py::arg("refVector"))
        .def("imageTo", &PolyMahalaDist::imageTo<double>,
            py::arg("image"), py::arg("refVector"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<uchar>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<schar>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<ushort>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<short>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<int>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<float>,
            py::arg("image"))
        .def("imageToReference", &PolyMahalaDist::imageToReference<double>,
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
            py::arg("image1"), py::arg("image2"), py::arg("mask1"), py::arg("mask2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<uchar>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<schar>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<ushort>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<short>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<int>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<float>,
            py::arg("points1"), py::arg("points2"))
        .def("calcBetweenPoints", &BhattaDist::calcBetweenPoints<double>,
            py::arg("points1"), py::arg("points2"))
        ;

    m.def("linearizeImage", linearizeImage<uchar>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<schar>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<ushort>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<short>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<int>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<float>,
        py::arg("image"));
    m.def("linearizeImage", linearizeImage<double>,
        py::arg("image"));
    m.def("delinearizeImage", delinearizeImage<uchar>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<schar>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<ushort>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<short>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<int>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<float>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));
    m.def("delinearizeImage", delinearizeImage<double>,
        py::arg("image"), py::arg("rows"), py::arg("cols"));

    m.def("calc_mean", calc_mean,
        py::arg("data"));
    m.def("getMaxValue", getMaxValue,
        py::arg("in"), py::arg("size"));
    m.def("getMaxAbsValue", getMaxAbsValue,
        py::arg("in"), py::arg("size"));
    m.def("polynomialProjection", polynomialProjection,
        py::arg("vec"));
    m.def("find_eq", find_eq,
        py::arg("opt"), py::arg("in"), py::arg("size"));
    m.def("calcVarianceScalar", calcVarianceScalar,
        py::arg("A"), py::arg("column"));
    m.def("calcVarianceVector", calcVarianceVector,
        py::arg("A"));
    m.def("removeNullIndexes", removeNullIndexes,
        py::arg("A"), py::arg("ind_use"));
    m.def("removeNullDimensions", removeNullDimensions,
        py::arg("in"), py::arg("size"));
}