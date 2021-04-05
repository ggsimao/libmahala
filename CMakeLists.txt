cmake_minimum_required(VERSION 3.10)

# set the project name
project(TCCLibrary CXX)


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_MODULE_PREFIX "")

# find_package(PythonLibs REQUIRED)
# find_package(Boost COMPONENTS python REQUIRED)
find_package(pybind11 CONFIG)
find_package(OpenCV REQUIRED)
find_package(Python COMPONENTS NumPy REQUIRED)
add_subdirectory(extern/pybind11)

add_executable(main main.cpp
                    PointCollector.cpp
                    PolynomialMahalanobisDistance.cpp
                    MahalanobisDistance.cpp
                    BhattacharyyaDistance.cpp
                    imageUtils.cpp)
pybind11_add_module(libmahala SHARED
                        MahalanobisDistance.cpp
                        PointCollector.cpp
                        imageUtils.cpp
                        mahalaLibWrapper.cpp
                        ndarray_converter.cpp)
# add_library(mahala SHARED MahalanobisDistance.cpp
#                           PointCollector.cpp
#                           imageUtils.cpp)

# include_directories(${Boost_INCLUDE_DIRS})
# include_directories(${PythonLibs_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

# SET(ENV{BOOST_ROOT} "/usr/include/boost_1_62_0")
# SET(ENV{PYTHONPATH} "/usr/include/python2.7")

# execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import numpy; print(numpy.get_include())" OUTPUT_VARIABLE NUMPY_INCLUDE OUTPUT_STRIP_TRAILING_WHITESPACE)
include_directories(${NUMPY_INCLUDE})

# set_target_properties(libmahala PROPERTIES NO_SONAME ON)
# set_target_properties(${OpenCV_LIBS} PROPERTIES NO_SONAME ON)
# set_target_properties(${OpenCV_LIBS} PROPERTIES IMPORTED_NO_SONAME ON)
set_target_properties(libmahala PROPERTIES OUTPUT_NAME "libmahala")
set_target_properties(libmahala PROPERTIES SUFFIX ".so")
# message(STATUS "OpenCV_LIBS: " ${OpenCV_LIBS})

target_include_directories(libmahala PRIVATE ${PYTHON_INCLUDE_DIRS})
target_link_libraries(libmahala PRIVATE ${OpenCV_LIBS})
# target_link_options(mahala PRIVATE)
# set_target_properties(mahala PROPERTIES VERSION 0.0.1)
target_link_libraries(main ${OpenCV_LIBS})