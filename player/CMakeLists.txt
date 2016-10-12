#====================================================
# CMakeLists configuration file for RAO player project
# this file aims to build the RAO player project
# 
# Author: Roice(LUO Bing)
# Date:   
#         2016-08-07 Create this file
cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

#====================================================
# ================ User can Access ==================
#====================================================
# install path
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
# Project name
set(PRJ_NAME RAO_Player)
# Debug version
#set(CMAKE_BUILD_TYPE Debug)
# Release version
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
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -framework Cocoa  -framework OpenGL -framework GLUT")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -framework Cocoa  -framework OpenGL -framework GLUT")
    message("Compile on APPLE MACOSX")
elseif(UNIX)
    message("Compile on UNIX")
endif()

#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#============== Find Dependencies ================
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#---- External FLTK related settings ----
# Set FLTK_DIR to the directory which contains FLTKConfig.cmake file
if(UNIX AND NOT APPLE) # MacOSX use brew fltk
    set(FLTK_DIR "${PROJECT_SOURCE_DIR}/../3rdparty/fltk-1.3.x-r11608/build")
endif()
# print out paths in order to check if goes wrong
message("Project Source Dir: \"${PROJECT_SOURCE_DIR}\"")
message("Project Binary Dir: \"${PROJECT_BINARY_DIR}\"")
if(UNIX AND NOT APPLE) # MacOSX use brew fltk
    message("FLTK_DIR (where the FLTKConfig.cmake file is): \"${FLTK_DIR}\"")
endif()
# find the FLTK package
# "NO_MODULE" tells CMake to search only for the FLTKConfig file, not using
# the FindFLTK.cmake supplied with CMake, which doesn't work with this version
# of FLTK
if(UNIX AND NOT APPLE)
    find_package(FLTK REQUIRED NO_MODULE)
elseif(APPLE)
    find_package(FLTK REQUIRED)
endif()
find_package(OpenGL REQUIRED)
# this file is in the same folder as FLTKConfig.cmake file
if(UNIX AND NOT APPLE)
    include(${FLTK_USE_FILE})
endif()
# gl.h and glu.h
if(UNIX AND NOT APPLE)
    add_definitions(-DGL_HEADER="FL/gl.h" -DGLU_HEADER="FL/glu.h" -DGLUT_HEADER="FL/glut.H")
elseif(APPLE)
    #add_definitions(-DGL_HEADER=<OpenGL/gl.h> -DGLU_HEADER=<OpenGL/glu.h> -DGLUT_HEADER=<GLUT/glut.h>)
    add_definitions(-DGL_HEADER="FL/gl.h" -DGLU_HEADER="FL/glu.h" -DGLUT_HEADER="FL/glut.H")
endif()

#---- External CUDA related settings ----
if(UNIX AND NOT APPLE)
    find_package(CUDA REQUIRED)
    # nvcc flags
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=sm_50;-use_fast_math)
endif()

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
#   add cblas dir to search for cblas.h header for linear algebra
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/blas/CBLAS/include)
#   add ziggurat source file to generate normal distribution random numbers
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/ziggurat)
#   add liquid-dsp dir to search for liquid.h header for dsp
include_directories(${PROJECT_SOURCE_DIR}/../3rdparty/liquid-dsp/include)
#   add eigen header dir
include_directories(${EIGEN3_INCLUDE_DIR})

#---- library files path settings ----
# add library dir into lib path
#   add fltk lib directory
if(UNIX AND NOT APPLE)
    link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/fltk-1.3.x-r11608/build/lib)
endif()
#   add blas and cblas lib dir
link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/blas/BLAS-3.6.0)
link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/blas/CBLAS/lib)
#   add liquid-dsp lib dir
link_directories(${PROJECT_SOURCE_DIR}/../3rdparty/liquid-dsp)
#   libgfortran
if(APPLE)
    link_directories(/usr/local/lib/gcc/6/)
endif()

#---- start compiling ----
set(LIB_UI_NAME ui)
# make a library from ui files
add_library(${LIB_UI_NAME} src/ui/UI.cxx src/ui/View.cxx src/ui/agv.cxx
    src/ui/draw/DrawScene.cxx src/ui/draw/materials.cxx
    src/ui/draw/draw_arena.cxx src/ui/draw/draw_robots.cxx
    src/ui/draw/draw_wave.cxx src/ui/draw/draw_qr.cxx
    src/robot/robot.cxx src/ui/draw/draw_arrow.cxx
    src/ui/draw/draw_foc.cxx
    # 3rdparty fltk widgets
    src/ui/widgets/Fl_LED_Button/Fl_LED_Button.cxx)
set(LIB_FOC_NAME foc) # flying odor compass
# make a library from ui files
cuda_add_library(${LIB_FOC_NAME} 
    src/foc/flying_odor_compass.cxx
    src/foc/foc_noise_reduction.cxx
    src/foc/foc_interp.cxx
    src/foc/foc_smooth.cxx
    src/foc/foc_diff.cxx
    src/foc/foc_edge.cxx
    src/foc/foc_std.cxx
    src/foc/foc_tdoa.cxx
    src/foc/foc_estimate.cxx
    src/foc/foc_wind.cxx
    src/foc/wake_qr.cxx
    src/foc/virtual_plume.cu
    src/foc/vector_rotation.cxx
    ../3rdparty/ziggurat/ziggurat.cpp
    )
target_compile_features(${LIB_FOC_NAME} PRIVATE cxx_constexpr)
# link external liquid-dsp library
target_link_libraries(${LIB_FOC_NAME} liquid)
# compile main file
add_executable(${PRJ_NAME} src/main.cxx src/Player_Config.cxx src/io/play_thread.cxx)
add_dependencies(${PRJ_NAME} ${LIB_UI_NAME} ${LIB_FOC_NAME})

#---- start linking ----
# Note: the former line depends on the next line
# link GUI library created above
target_link_libraries(${PRJ_NAME} ${LIB_UI_NAME})
# link FOC library created above
target_link_libraries(${PRJ_NAME} ${LIB_FOC_NAME})
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

# install
#install(TARGETS ${PRJ_NAME}
#    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/../bin
#    LIBRARY DESTINATION ${PROJECT_SOURCE_DIR}/../lib
#    )
