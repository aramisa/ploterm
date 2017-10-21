cmake_minimum_required (VERSION 2.8)
add_definitions(-std=c++11)
# set(CMAKE_CXX_STANDARD 11)  # only works with cmake>=3.1
project (ploterm)

# The version number.
set (Tutorial_VERSION_MAJOR 1)
set (Tutorial_VERSION_MINOR 0)

FIND_PACKAGE(SWIG REQUIRED)
INCLUDE(${SWIG_USE_FILE})

SET(Python_ADDITIONAL_VERSIONS 2.7)
FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
SET(CMAKE_SWIG_FLAGS "")

SET_SOURCE_FILES_PROPERTIES(ploterm.i PROPERTIES CPLUSPLUS ON)
SET_SOURCE_FILES_PROPERTIES(ploterm.i PROPERTIES SWIG_FLAGS "-includeall")
SWIG_ADD_MODULE(ploterm python ploterm.i ploterm.cpp)
SWIG_LINK_LIBRARIES(ploterm ${PYTHON_LIBRARIES})

execute_process(COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
install(TARGETS _ploterm DESTINATION ${PYTHON_SITE_PACKAGES})
install(FILES ${CMAKE_BINARY_DIR}/ploterm.py DESTINATION ${PYTHON_SITE_PACKAGES})
# add_executable(ploterm ploterm.cpp)
