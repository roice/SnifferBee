#====================================================
# CMakeLists configuration file for GSRAO project
# this file aims to build the GSRAO project
# 
# Author: Roice(LUO Bing)
# Date:   
#         2016-04-16 Create this file
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

#====================================================
# ================ User can Access ==================
#====================================================
# install path
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# Project name
set(PRJ_NAME GSRAO)
# Debug version
set(CMAKE_BUILD_TYPE Debug)
# ===================================================

#====================================================
# ============= Don't touch the following ===========
#====================================================
project(${PRJ_NAME})
#======== Detect which platform in use ========
# if compile on Mac OSX, the "-framework Cocoa" flag should be added
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa -framework OpenGL -framework GLUT")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -framework Cocoa -framework OpenGL -framework GLUT")
    message("Compile on APPLE MACOSX")
elseif(UNIX)
    message("Compile on UNIX")
endif()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#============== Find Dependencies ================
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#---- External FLTK related settings ----
# Set FLTK_DIR to the directory which contains FLTKConfig.cmake file
set(FLTK_DIR "${PROJECT_SOURCE_DIR}/../../3rdparty/fltk-1.3.x-r11608/build")
# print out paths in order to check if goes wrong
message("Project Source Dir: \"${PROJECT_SOURCE_DIR}\"")
message("Project Binary Dir: \"${PROJECT_BINARY_DIR}\"")
message("FLTK_DIR (where the FLTKConfig.cmake file is): \"${FLTK_DIR}\"")
# find the FLTK package
# "NO_MODULE" tells CMake to search only for the FLTKConfig file, not using
# the FindFLTK.cmake supplied with CMake, which doesn't work with this version
# of FLTK
find_package(FLTK REQUIRED NO_MODULE)
find_package(OpenGL REQUIRED)
# this file is in the same folder as FLTKConfig.cmake file
include(${FLTK_USE_FILE})

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
#   add root dir to search for header files of GSRAO projects
include_directories(${PROJECT_SOURCE_DIR}/src)
#   add kalman filter headers
include_directories(${PROJECT_SOURCE_DIR}/src/mocap/kf)
#   add cblas dir to search for cblas.h header for linear algebra
include_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/blas/CBLAS/include)
#   add ziggurat source file to generate normal distribution random numbers
include_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/ziggurat)
#   add liquid-dsp dir to search for liquid.h header for dsp
include_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/liquid-dsp/include)
#   add eigen header dir
include_directories(${EIGEN3_INCLUDE_DIR})

#---- library files path settings ----
# add library dir into lib path
#   add fltk lib directory
link_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/fltk-1.3.x-r11608/build/lib)
#   add blas and cblas lib dir
link_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/blas/BLAS-3.6.0)
link_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/blas/CBLAS/lib)
#   add liquid-dsp lib dir
link_directories(${PROJECT_SOURCE_DIR}/../../3rdparty/liquid-dsp)

#---- start compiling ----
set(LIB_UI_NAME ui)
# make a library from ui files
add_library(${LIB_UI_NAME} src/ui/UI.cxx src/ui/View.cxx src/ui/agv.cxx
    src/ui/draw/DrawScene.cxx src/ui/draw/materials.cxx
    src/ui/draw/draw_arena.cxx src/ui/draw/draw_robots.cxx
    src/ui/draw/draw_qr.cxx src/ui/draw/draw_wave.cxx
    src/ui/draw/draw_arrow.cxx src/ui/draw/draw_wind.cxx
    # 3rdparty fltk widgets
    src/ui/widgets/Fl_LED_Button/Fl_LED_Button.cxx)
# compile main file
add_executable(${PRJ_NAME} src/main.cxx src/GSRAO_Config.cxx
    src/io/serial.cxx src/io/serial_spp.cxx src/io/serial_mbsp.cxx
    src/io/serial_anemometer.cxx src/io/record.cxx
    src/mocap/packet_client.cxx src/mocap/mocap_noise_reduction.cxx
    src/robot/robot.cxx src/robot/microbee.cxx
    src/method/method.cxx src/method/hover_measure.cxx src/method/back_forth_measure.cxx
    src/common/vector_rotation.cxx)
target_compile_features(${PRJ_NAME} PRIVATE cxx_constexpr)
add_dependencies(${PRJ_NAME} ${LIB_UI_NAME})

#---- start linking ----
# Note: the former line depends on the next line
# link GUI library created above
target_link_libraries(${PRJ_NAME} ${LIB_UI_NAME})
# link external FLTK library
target_link_libraries(${PRJ_NAME} fltk)
# link external OpenGL library
target_link_libraries(${PRJ_NAME} fltk_gl)
# link external image library
target_link_libraries(${PRJ_NAME} fltk_images)
# link external blas library
target_link_libraries(${PRJ_NAME} cblas)
target_link_libraries(${PRJ_NAME} blas)
target_link_libraries(${PRJ_NAME} gfortran)
target_link_libraries(${PRJ_NAME} pthread)
# link extern hdf5 library
target_link_libraries(${PRJ_NAME} ${HDF5_LIBRARIES})
# link external liquid-dsp library
target_link_libraries(${PRJ_NAME} liquid)

# install
#install(TARGETS ${PRJ_NAME}
#    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../bin
#    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../lib
#    )
