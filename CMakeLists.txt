cmake_minimum_required(VERSION 3.10)

project(LibraryMahalanobis VERSION 1.0.2)

set(WITH_PYTHON 1)

if(NO_PYTHON) 
    set(WITH_PYTHON 0)
endif()

set(WITH_CPP 1)

if(NO_CPP) 
    set(WITH_CPP 0)
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_SHARED_MODULE_PREFIX "")

find_package(OpenCV 4.0.0 REQUIRED)

include(GNUInstallDirs)

# set(LIBMAHALA_SOURCE_DIR ${PROJECT_SOURCE_DIR}/src)
# set(LIBMAHALA_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/include)

# add_executable(mahala_test mahala_test.cpp
#                 PointCollector.cpp
#                 PolynomialMahalanobisDistance.cpp
#                 MahalanobisDistance.cpp
#                 BhattacharyyaDistance.cpp)
# target_include_directories(mahala_test PRIVATE ${PYTHON_INCLUDE_DIRS})
# target_link_libraries(mahala_test PRIVATE ${OpenCV_LIBS})
include_directories(${OpenCV_INCLUDE_DIRS})

if(WITH_CPP)
    add_library(libmahala SHARED
                    PointCollector.cpp
                    PolynomialMahalanobisDistance.cpp
                    MahalanobisDistance.cpp
                    BhattacharyyaDistance.cpp)

    set_target_properties(libmahala PROPERTIES OUTPUT_NAME "libmahala")
    set_target_properties(libmahala PROPERTIES PREFIX "")
    # set_target_properties(libmahala PROPERTIES SUFFIX ".so")


    target_include_directories(libmahala PRIVATE ${PYTHON_INCLUDE_DIRS})
    target_link_libraries(libmahala PRIVATE ${OpenCV_LIBS})

    set_target_properties(libmahala PROPERTIES VERSION ${PROJECT_VERSION})

    configure_file(libmahala.pc.in libmahala.pc @ONLY)

    set_target_properties(libmahala PROPERTIES PUBLIC_HEADER 
                            "MahalanobisDistance.hpp;PolynomialMahalanobisDistance.hpp;BhattacharyyaDistance.hpp;PointCollector.hpp")

    install(TARGETS libmahala
                EXPORT LibraryMahalanobisTargets
                LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
                PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libmahala)
    install(EXPORT LibraryMahalanobisTargets
                FILE LibraryMahalanobisTargets.cmake
                # NAMESPACE LibraryMahalanobis::
                DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibraryMahalanobis
    )
    install(FILES ${CMAKE_BINARY_DIR}/libmahala.pc
                DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)
endif()


if(WITH_PYTHON)
    find_package(Python 3.8.0 COMPONENTS Interpreter Development NumPy REQUIRED)
    add_subdirectory(extern/pybind11)

    pybind11_add_module(libmahalapy MODULE
                        MahalanobisDistance.cpp
                        PolynomialMahalanobisDistance.cpp
                        BhattacharyyaDistance.cpp
                        PointCollector.cpp
                        mahalaLibWrapper.cpp
                        ndarray_converter.cpp)

    include_directories(${PYTHON_INCLUDE_DIRS})
    include_directories(${NUMPY_INCLUDE})

    set_target_properties(libmahalapy PROPERTIES OUTPUT_NAME "libmahalapy")
    set_target_properties(libmahalapy PROPERTIES SUFFIX ".so")
    target_link_libraries(libmahalapy PRIVATE ${OpenCV_LIBS})
    set_target_properties(libmahalapy PROPERTIES VERSION ${PROJECT_VERSION})

    # message(STATUS ${Python_STDLIB})
    install(TARGETS libmahalapy
            LIBRARY DESTINATION ${Python_STDLIB})

    # find_program(PYTHON "python")

    # if (PYTHON)
    #     set(SETUP_PY    "${CMAKE_CURRENT_BINARY_DIR}/../install_lib.py")

    #     install(CODE "execute_process(COMMAND ${PYTHON} ${SETUP_PY})")
    # endif()
endif()