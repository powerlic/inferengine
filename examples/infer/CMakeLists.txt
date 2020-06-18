set(LIB_PATH "/home/licheng/3rdparty")

find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
message(${CUDA_INCLUDE_DIRS})

#find_package(Boost COMPONENTS system filesystem regex REQUIRED)

# OpenCV
set(OpenCV_DIR "${LIB_PATH}/opencv-3.3.0-cuda/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/src/infer)
include_directories(${PROJECT_SOURCE_DIR}/src/infer/libtorch)
set(EXAMPLE_BIN_PATH ${CMAKE_BINARY_DIR}/)

#set(DEP_LIBS  ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${Boost_LIBRARIES} TorchInferEngine)

set(DEP_LIBS TorchInferEngine)
#set(DEP_LIBS ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS})

add_executable(test_ssd test_ssd.cpp)
set_target_properties(test_ssd PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_ssd ${DEP_LIBS})

add_executable(test_classifier test_classifier.cpp)
set_target_properties(test_classifier PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_classifier ${DEP_LIBS})

add_executable(test_fasterrcnn test_fasterrcnn.cpp)
set_target_properties(test_fasterrcnn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_fasterrcnn ${DEP_LIBS})


add_executable(test_maskrcnn test_maskrcnn.cpp)
set_target_properties(test_maskrcnn PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${EXAMPLE_BIN_PATH})
target_link_libraries(test_maskrcnn ${DEP_LIBS})