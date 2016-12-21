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
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)
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
find_package(CUDA REQUIRED)
# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-use_fast_math;)
#list(APPEND CUDA_DEV_LIBRARIES ${CUDA_cufft_LIBRARY})
link_directories(${CUDA_LIBRARY_DIRS})

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
include_directories(${PROJECT_SOURCE_DIR}/../../../player/src/)
include_directories(${PROJECT_SOURCE_DIR}/../../../player/src/foc/kf)
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
#   add wavelib dir
link_directories(${PROJECT_SOURCE_DIR}/../../../3rdparty/wavelib/linuxstatic)
#   libgfortran
if(APPLE)
    link_directories(/usr/local/lib/gcc/6) # changes with gcc version
endif(APPLE)

#---- start compiling ----
set(LIB_FOC_NAME foc) # flying odor compass
# make a library from ui files
add_library(${LIB_FOC_NAME} 
    ../../../player/src/foc/flying_odor_compass.cxx
    #../../../player/src/foc/foc_noise_reduction.cxx
    ../../../player/src/foc/foc_interp.cxx
    #../../../player/src/foc/foc_wt.cxx
    #../../../player/src/foc/foc_smooth.cxx
    #../../../player/src/foc/foc_diff.cxx
    #../../../player/src/foc/foc_edge.cxx 
    #../../../player/src/foc/foc_tdoa.cxx
    #../../../player/src/foc/foc_std.cxx
    #../../../player/src/foc/foc_estimate.cxx
    ../../../player/src/foc/foc_wind.cxx
    #../../../player/src/foc/wake_qr.cxx
    #../../../player/src/foc/virtual_plume.cxx
    ../../../player/src/foc/vector_rotation.cxx
    ../../../3rdparty/ziggurat/ziggurat.cpp
    )
cuda_add_library(foc_wt_gpu ../../../player/src/foc/foc_wavelet.cu)
target_compile_features(foc PRIVATE cxx_constexpr)
# link foc_wt_gpu
target_link_libraries(${LIB_FOC_NAME} foc_wt_gpu)
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
# link CUDA library
#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${CUDA_DEV_LIBRARIES})
# link extern hdf5 library
target_link_libraries(${PRJ_NAME} ${HDF5_LIBRARIES})
