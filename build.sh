#!/bin/bash

export TORCH_NVCC_FLAGS="-D__CUDA_NO_HALF_OPERATORS__"

BUILD_DIR=build
INSTALL_PREFIX=install

rm -rf $INSTALL_PREFIX
mkdir $INSTALL_PREFIX

rm -rf $BUILD_DIR
mkdir $BUILD_DIR && cd $BUILD_DIR

/home/licheng/3rdparty/cmake-3.14/bin/cmake -DCMAKE_BUILD_TYPE=Release -DTIME_LOG=ON ..
make -j 128
make install
#make clean