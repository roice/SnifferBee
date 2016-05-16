####==================================================
#### Compile, Setup and install script of GSRAO
####                For Unix
####==================================================

#!/bin/sh

# Get the absolute TOP path of this project
prjtop=$(cd "$(dirname "$0")"; pwd)
echo "Absolute path of project top directory is: "$prjtop
sleep 1

##======== Compile 3rd Party softwares ========
echo "Start Compiling 3d party soft ..."
# Compile fltk
echo "Start Compiling FLTK..."
sleep 1
cd $prjtop/3rdparty
tar -xjf fltk-1.3.x-r11608.tar.bz2
cd fltk-1.3.x-r11608
mkdir -p build/install
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install ..
make
make install
# Compile blas
echo "Start Compiling BLAS..."
sleep 1
cd $prjtop/3rdparty/blas/BLAS-3.6.0
make
mv blas_*.a libblas.a
# Compile cblas
echo "Start Compiling CBLAS..."
sleep 1
cd $prjtop/3rdparty/blas/CBLAS
make
mv lib/cblas_*.a lib/libcblas.a


##======== Compile GSRAO ========
#cd $prjtop/src
#mkdir build
#cd build
#cmake ..
#make
#make install
