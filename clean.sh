####=================================
#### Uninstall all build files
####=================================

#!/bin/sh

# Get project top directory abs path
prjtop=$(cd "$(dirname "$0")"; pwd)
echo "Absolute path of project top directory is: "$prjtop
sleep 2

echo "Start cleaning ..."
sleep 1
# clear 3rd party software builds
cd $prjtop/3rdparty
rm -rf fltk-1.3.x-r11608
rm blas/BLAS-3.6.0/*.o
rm blas/CBLAS/testing/x*cblat*
rm blas/CBLAS/src/*.o

# clear GSRAO builds
#cd $prjtop/src
#rm -rf build

echo "Uninstall done"
