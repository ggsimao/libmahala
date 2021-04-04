#include <Python.h>
#include "pybind11/pybind11.h"
#include "MahalanobisDistance.hpp"
#include "imageUtils.cpp"
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
        .def(py::init<const char*, cv::ImreadModes>(),
            py::arg("path"), py::arg("flags"))
        .def_property_readonly("collectedPixels", &PointCollector::collectedPixels)
        .def_property_readonly("collectedCoordinates", &PointCollector::collectedCoordinates)
        .def_property_readonly("referencePixel", &PointCollector::referencePixel)
        .def_property_readonly("referenceCoordinate", &PointCollector::referenceCoordinate)
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
}