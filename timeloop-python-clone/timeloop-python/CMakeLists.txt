cmake_minimum_required(VERSION 3.4)
project(timeloop_python)

if(DEFINED ENV{LIBTIMELOOP_PATH})
    set(LIBTIMELOOP_PATH $ENV{LIBTIMELOOP_PATH})
else()
    set(LIBTIMELOOP_PATH ${PROJECT_SOURCE_DIR}/lib/timeloop)
endif()
message(STATUS "Using Timeloop library from: ${LIBTIMELOOP_PATH}")

find_library(LIBTIMELOOP timeloop-model HINTS ${LIBTIMELOOP_PATH}/lib REQUIRED)

add_library(define_gterminate SHARED bindings/gterminate.cpp)

set(BINDINGS_SRC 
    bindings/accelergy_interface.cpp
    bindings/bindings.cpp
    bindings/config.cpp
    bindings/mapping.cpp
    bindings/model/engine.cpp
    bindings/model/sparse_optimization_info.cpp
    bindings/problem.cpp)
add_subdirectory(lib/pybind11)
pybind11_add_module(bindings ${BINDINGS_SRC})
set_property(TARGET bindings PROPERTY CXX_STANDARD 17)
target_include_directories(bindings PRIVATE
    include
    ${LIBTIMELOOP_PATH}/include
    lib/pybind11/include)
target_link_libraries(bindings PRIVATE config++ yaml-cpp ${LIBTIMELOOP} define_gterminate)

# VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(bindings PRIVATE VERSION_INFO=${VERSION_INFO})
