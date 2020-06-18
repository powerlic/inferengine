# 编译 Demo
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(LIB_PATH "/home/licheng/3rdparty")


set(OpenCV_DIR "${LIB_PATH}/opencv-3.3.0-cuda/share/OpenCV")
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src/algo)
include_directories(${PROJECT_SOURCE_DIR}/examples/algo)

#set(DEP_LIBS ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} TorchInferEngine ultron_ai)
set(DEP_LIBS TorchInferEngine ultron_ai)

set(EXAMPLE_BIN_PATH ${CMAKE_BINARY_DIR})


add_executable(test_video_detector test_video_detector.cpp)
set_target_properties(test_video_detector PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_video_detector ${DEP_LIBS})