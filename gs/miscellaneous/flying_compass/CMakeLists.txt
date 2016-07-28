#====================================================
# CMakeLists configuration file for Flying Compass
# this file aims to build a test routine
# 
# Author: Roice(LUO Bing)
# Date:   
#         2016-06-17 Create this file
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

#====================================================
# ================ User can Access ==================
#====================================================
# install path
# Project name
set(PRJ_NAME Flying_Odor_Compass)
# Debug version
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
# ===================================================

#====================================================
# ============= Don't touch the following ===========
#====================================================
project(${PRJ_NAME})
#======== Detect which platform in use ========
# if compile on Mac OSX, the "-framework Cocoa" flag should be added
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -framework Cocoa")
    message("Compile on APPLE MACOSX")
elseif(UNIX AND NOT APPLE)
    message("Compile on LINUX")
endif()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#============== Find Dependencies ================
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#---- External CUDA related settings ----
#find_package(CUDA REQUIRED)
# nvcc flags
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-use_fast_math)

#---- External HDF5 related, for data recording
find_package(HDF5 COMPONENTS C CXX REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})

#---- External Eigen library
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/scripts)
find_package(Eigen3 REQUIRED)

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#============== Build the project ================
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#---- source files include path settings ----
# add source directory into include path
#   add root dir to search for header files of RAOS projects
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/src/foc/kf)
#   add liquid-dsp dir to search for liquid.h header for dsp
include_directories(${PROJECT_SOURCE_DIR}/3rd-party/liquid-dsp/include)
#   add cblas dir to search for cblas.h header for linear algebra
include_directories(${PROJECT_SOURCE_DIR}/../../../3rdparty/blas/CBLAS/include)
#   add ziggurat source file to generate normal distribution random numbers
include_directories(${PROJECT_SOURCE_DIR}/../../../3rdparty/ziggurat)
#   add eigen header dir
include_directories(${EIGEN3_INCLUDE_DIR})

#---- library files path settings ----
# add library dir into lib path
#   add liquid-dsp lib dir
link_directories(${PROJECT_SOURCE_DIR}/3rd-party/liquid-dsp)
#   add blas and cblas lib dir
link_directories(${PROJECT_SOURCE_DIR}/../../../3rdparty/blas/BLAS-3.6.0)
link_directories(${PROJECT_SOURCE_DIR}/../../../3rdparty/blas/CBLAS/lib)
#   libgfortran
if(APPLE)
    link_directories(/usr/local/lib/gcc/6) # changes with gcc version
endif(APPLE)

#---- start compiling ----
set(LIB_FOC_NAME foc) # flying odor compass
# make a library from ui files
add_library(${LIB_FOC_NAME} 
    src/foc/flying_odor_compass.cxx
    src/foc/foc_noise_reduction.cxx
    src/foc/foc_interp.cxx
    src/foc/foc_smooth.cxx
    src/foc/foc_diff.cxx
    src/foc/foc_delta.cxx
    )
target_compile_features(foc PRIVATE cxx_constexpr)
# link external liquid-dsp library
target_link_libraries(${LIB_FOC_NAME} liquid)
# compile main file
add_executable(${PRJ_NAME} src/main.cxx src/record.cxx)
add_dependencies(${PRJ_NAME} ${LIB_FOC_NAME})

#---- start linking ----
# Note: the former line depends on the next line
# link GUI library created above
target_link_libraries(${PRJ_NAME} ${LIB_FOC_NAME})
# link external blas library
target_link_libraries(${PRJ_NAME} cblas)
target_link_libraries(${PRJ_NAME} blas)
target_link_libraries(${PRJ_NAME} gfortran)
target_link_libraries(${PRJ_NAME} pthread)
# link extern hdf5 library
target_link_libraries(${PRJ_NAME} ${HDF5_LIBRARIES})
