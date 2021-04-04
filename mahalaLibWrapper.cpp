#include <Python.h>
#include "extern/pybind11/include/pybind11.h"
#include "MahalanobisDistance.hpp"
#include "imageUtils.cpp"
#include "PointCollector.hpp"
// using namespace boost::python;

namespace py = pybind11;

// PointCollector 

PYBIND11_MODULE(libmahala, m) {
    py::class_<MahalaDist>(m, "MahalaDist")
        .def(py::init<const cv::Mat&, double, cv::Mat>(),
            "input"_a, "smin"_a, "reference"_a = cv::Mat())
        .def_readonly("reference", &MahalaDist::reference)
        .def_readonly("dimension", &MahalaDist::dimension)
        .def_readonly("dirty", &MahalaDist::dirty)
        .def_readonly("u", &MahalaDist::u)
        .def_readonly("w", &MahalaDist::w)
        .def_readonly("c", &MahalaDist::c)
        .def_readonly("sigma2", &MahalaDist::sigma2)
        .def_property("smin", 
            static_cast<double (MahalaDist::*)()> (&MahalaDist::smin),
            static_cast<void (MahalaDist::*)(double)> (&MahalaDist::smin))
        .def("build", &MahalaDist::build)
        .def("pointTo", &MahalaDist::pointTo,
            "point1"_a, "point2"_a)
        .def("pointToReference", &MahalaDist::pointToReference,
            "point"_a)
        .def("pointsTo", &MahalaDist::pointsTo,
            "points"_a, "ref"_a)
        .def("pointsToReference", &MahalaDist::pointsToReference,
            "points"_a)
        .def("imageTo", &MahalaDist::imageTo<uchar>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<schar>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<ushort>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<short>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<int>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<float>,
            "image"_a, "ref"_a)
        .def("imageTo", &MahalaDist::imageTo<double>,
            "image"_a, "ref"_a)
        .def("imageToReference", &MahalaDist::imageToReference<uchar>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<schar>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<ushort>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<short>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<int>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<float>,
            "image"_a)
        .def("imageToReference", &MahalaDist::imageToReference<double>,
            "image"_a)
        ;
    
    py::class_<PointCollector>(m, "PointCollector")
        .def(init<cv::Mat&>(),
            "input"_a)
        .def(init<const char*, cv::ImreadModes>(),
            "path"_a, "flags"_a)
        .def_readonly("collectedPixels", &PointCollector::collectedPixels)
        .def_readonly("collectedCoordinates", &PointCollector::collectedCoordinates)
        .def_readonly("referencePixel", &PointCollector::referencePixel)
        .def_readonly("referenceCoordinate", &PointCollector::referenceCoordinate)
        ;

    m.def("linearizeImage", linearizeImage<uchar>,
            "image"_a)
    m.def("linearizeImage", linearizeImage<schar>,
        "image"_a)
    m.def("linearizeImage", linearizeImage<ushort>,
        "image"_a)
    m.def("linearizeImage", linearizeImage<short>,
        "image"_a)
    m.def("linearizeImage", linearizeImage<int>,
        "image"_a)
    m.def("linearizeImage", linearizeImage<float>,
        "image"_a)
    m.def("linearizeImage", linearizeImage<double>,
        "image"_a)
    m.def("delinearizeImage", delinearizeImage<uchar>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<schar>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<ushort>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<short>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<int>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<float>,
        "image"_a, "rows"_a, "cols"_a)
    m.def("delinearizeImage", delinearizeImage<double>,
        "image"_a, "rows"_a, "cols"_a)
}

// BOOST_PYTHON_MODULE(libmahala)
// {
//     double(MahalaDist::*getsmin)() = &MahalaDist::smin;
//     void(MahalaDist::*setsmin)(double) = &MahalaDist::smin;
//     class_<MahalaDist>("MahalaDist")
//         .def(init<const cv::Mat&, double, optional<cv::Mat>>())
//         .add_property("reference", &MahalaDist::reference)
//         .add_property("dimension", &MahalaDist::dimension)
//         .add_property("dirty", &MahalaDist::dirty)
//         .add_property("u", &MahalaDist::u)
//         .add_property("w", &MahalaDist::w)
//         .add_property("c", &MahalaDist::c)
//         .add_property("sigma2", &MahalaDist::sigma2)
//         .add_property("smin", getsmin, setsmin)
//         .def("build", &MahalaDist::build)
//         .def("pointTo", &MahalaDist::pointTo)
//         .def("pointToReference", &MahalaDist::pointToReference)
//         .def("pointsTo", &MahalaDist::pointsTo)
//         .def("pointsToReference", &MahalaDist::pointsToReference)
//         ;
    
//     class_<PointCollector>("PointCollector")
//         .def(init<cv::Mat&>())
//         .def(init<const char*, cv::ImreadModes>())
//         .add_property("collectedPixels", &PointCollector::collectedPixels)
//         .add_property("collectedCoordinates", &PointCollector::collectedCoordinates)
//         .add_property("referencePixel", &PointCollector::referencePixel)
//         .add_property("referenceCoordinate", &PointCollector::referenceCoordinate)
//         ;
// }

// static auto the_class_ = boost::python::class_<MahalaDist> ("MahalaDist");
// template <typename T> void addTemplateMethods() {
//     def("linearizeImage", (cv::Mat (T&))(&linearizeImage));
//     def("delinearizeImage", (cv::Mat (T&))(&delinearizeImage));

//     the_class_
//         .def("imageTo", (void(MahalaDist::*)(T&))(&MahalaDist::imageTo))
//         .def("pointsToReference", (void(MahalaDist::*)(T&))(&MahalaDist::pointsToReference))
//         ;
// }