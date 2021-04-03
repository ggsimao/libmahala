#include <Python.h>
#include <boost/python/numpy.hpp>
#include <boost/python.hpp>
#include "MahalanobisDistance.hpp"
#include "imageUtils.cpp"
#include "PointCollector.hpp"
#include <glib-unix.h>

using namespace boost::python;

// PointCollector 

BOOST_PYTHON_MODULE(libmahala)
{
    double(MahalaDist::*getsmin)() = &MahalaDist::smin;
    void(MahalaDist::*setsmin)(double) = &MahalaDist::smin;
    class_<MahalaDist>("MahalaDist")
        .def(init<const cv::Mat&, double, optional<cv::Mat>>())
        .add_property("reference", &MahalaDist::reference)
        .add_property("dimension", &MahalaDist::dimension)
        .add_property("dirty", &MahalaDist::dirty)
        .add_property("u", &MahalaDist::u)
        .add_property("w", &MahalaDist::w)
        .add_property("c", &MahalaDist::c)
        .add_property("sigma2", &MahalaDist::sigma2)
        .add_property("smin", getsmin, setsmin)
        .def("build", &MahalaDist::build)
        .def("pointTo", &MahalaDist::pointTo)
        .def("pointToReference", &MahalaDist::pointToReference)
        .def("pointsTo", &MahalaDist::pointsTo)
        .def("pointsToReference", &MahalaDist::pointsToReference)
        ;
    
    class_<PointCollector>("PointCollector")
        .def(init<cv::Mat&>())
        .def(init<const char*, cv::ImreadModes>())
        .add_property("collectedPixels", &PointCollector::collectedPixels)
        .add_property("collectedCoordinates", &PointCollector::collectedCoordinates)
        .add_property("referencePixel", &PointCollector::referencePixel)
        .add_property("referenceCoordinate", &PointCollector::referenceCoordinate)
        ;
}

static auto the_class_ = boost::python::class_<MahalaDist> ("MahalaDist");
template <typename T> void addTemplateMethods() {
    def("linearizeImage", (cv::Mat (T&))(&linearizeImage));
    def("delinearizeImage", (cv::Mat (T&))(&delinearizeImage));

    the_class_
        .def("imageTo", (void(MahalaDist::*)(T&))(&MahalaDist::imageTo))
        .def("pointsToReference", (void(MahalaDist::*)(T&))(&MahalaDist::pointsToReference))
        ;
}