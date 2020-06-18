cmake_minimum_required (VERSION 3.10)
project(TorchInferEngine LANGUAGES CXX C CUDA)


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g3 -std=c++11 -w -Wno-error=deprecated-declarations -Wno-deprecated-declarations")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -D__CUDA_NO_HALF_OPERATORS__")
# CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
#message("CUDA_INCLUDE_DIRS" ${CUDA_INCLUDE_DIRS})

# OpenCV
set(OpenCV_DIR "${LIB_PATH}/opencv-3.3.0-cuda/share/OpenCV")
#message(${OpenCV_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 项目头文件目录
set(PROJECT_INCLUDES
    ./
    libtorch
    ../
)

foreach(PROJECT_INCLUDE ${PROJECT_INCLUDES})
    include_directories(${PROJECT_SOURCE_DIR}/${PROJECT_INCLUDE})
endforeach(PROJECT_INCLUDE)

# 项目源文件
#file(GLOB_RECURSE INFERENCE_SOURCES ./ *.cpp)
set(SRC
    ./infer.cpp
    ./infer_creator.cpp
    ./libtorch/classifier.cpp
    ./libtorch/fasterrcnn.cpp
    ./libtorch/fasterrcnnfpn.cpp
    ./libtorch/maskrcnn.cpp
    ./libtorch/nms_kernel.cu
    ./libtorch/nms_layer_cuda.cu
    ./libtorch/nms_layer.cpp
    ./libtorch/roi_align_kernel.cu
    ./libtorch/roi_align_kernel_v2.cu
    ./libtorch/roi_align_cuda.cpp
    ./libtorch/roi_align.h
    ./libtorch/nms_cpu.cpp
    ./libtorch/anchors.cpp
    ./libtorch/ssd.cpp
    ../utils/timer.hpp
    ../utils/logging.hpp
    ../utils/jsoncpp.cpp
    )

#cuda_add_library(ROIAlign STATIC ./libtorch/ROIAlign_cuda.h ./libtorch/ROIAlign_cuda.cu)
#message("src" ${SRC})
#message("CMAKE_CUDA_FLAGS" ${CMAKE_CUDA_FLAGS})

set(DEP_LIBS  ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})


# 编译 "tuniinfe" 库, 输出到 ${CMAKE_BINARY_DIR}/lib
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
add_library(TorchInferEngine SHARED ${SRC})
target_link_libraries(TorchInferEngine ${DEP_LIBS})
set_target_properties(TorchInferEngine PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

################################### 安装配置 ########################################
install (TARGETS TorchInferEngine
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
