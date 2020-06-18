set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -g3 -std=c++11 -fopenmp")

set(CMAKE_BUILD_TYPE "Release")
#输出包名字
set(OUTPUT_LIBRARY_NAME ultron_ai)

if(NOT DEFINED LIB_PATH)
    set(LIB_PATH "/home/licheng/3rdparty")
endif()


# set(Torch_DIR "${LIB_PATH}/libtorch/share/cmake/Torch")
# find_package(Torch REQUIRED)

# CUDA
# set(LIB_PATH "/home/chengli12/3rdparty/")
# set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
# find_package(CUDA REQUIRED)
#message("CUDA_INCLUDE_DIRS" ${CUDA_INCLUDE_DIRS})

# OpenCV
set(OpenCV_DIR "${LIB_PATH}/opencv-3.3.0-cuda/share/OpenCV")
#message(${OpenCV_DIR})
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

include_directories("${LIB_PATH}/eigen3")
set(TBB_DIR "${LIB_PATH}/tbb2018_20171205oss/cmake")
find_package(TBB COMPONENTS tbb tbbmalloc)


#src
set(ALGO_SRC
    #video_struct
    #detector
    detector/detector_types.cpp
    detector/detector_impl.cpp
    detector/detector.cpp
    tracker/video_tracker.cpp
    tracker/tracker.cpp
    tracker/kcf.cpp
    tracker/Hungarian.cpp
)

#other include
set(ALGO_INCLUDE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/algo
    ${PROJECT_SOURCE_DIR}/src/algo/detector
    ${PROJECT_SOURCE_DIR}/src/
    )

include_directories(${ALGO_INCLUDE})
SET(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/lib)


add_library(${OUTPUT_LIBRARY_NAME} SHARED ${ALGO_SRC})
set(DEP_LIBS  ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS} ${TBB_IMPORTED_TARGETS} TorchInferEngine)
target_link_libraries(${OUTPUT_LIBRARY_NAME} ${DEP_LIBS})


#SET_TARGET_PROPERTIES(${OUTPUT_LIBRARY_NAME} PROPERTIES VERSION ${PROJECT_VERSION})


install (TARGETS ${OUTPUT_LIBRARY_NAME}
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    )

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include
    FILES_MATCHING
    REGEX "\\.hpp$|\\.h$"
    )