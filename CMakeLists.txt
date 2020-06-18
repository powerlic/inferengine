# CMakeList.txt: TuniGoods 的 CMake 项目
cmake_minimum_required (VERSION 3.10)

project(InferEngine LANGUAGES CXX C)


set(CMAKE_BUILD_TYPE "Release")

option(Timer "add time log information" ON)


if(TIME_LOG)
    message("TIME_LOG")
    add_definitions(-DTIME_LOG)
endif()

# set cuda path
set(LIB_PATH "/home/licheng/3rdparty/")

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(CUDA REQUIRED)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")

set(Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")
message("install prefix: ${CMAKE_INSTALL_PREFIX}")

add_subdirectory(src/infer)
add_subdirectory(src/algo)
add_subdirectory(examples/infer)
add_subdirectory(examples/algo)