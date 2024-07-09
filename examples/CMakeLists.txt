cmake_minimum_required(VERSION 3.15)

project(microgradpp_examples)
set(CMAKE_CXX_STANDARD 17)

# Define the project root directory
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add OpenCV submodule
find_package(OpenCV REQUIRED)

#include(FetchContent)
#
#FetchContent_Declare(matplotplusplus
#        GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
#        GIT_TAG origin/master) # or whatever tag you want
#
#FetchContent_GetProperties(matplotplusplus)
#if(NOT matplotplusplus_POPULATED)
#    FetchContent_Populate(matplotplusplus)
#    add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
#endif()

# Include directories for examples
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})

## Add example_mlp
add_executable(example_mlp mlp.cpp)
target_link_libraries(example_mlp PUBLIC microgradpp)

# Add example_images
add_executable(example_images images.cpp)
target_link_libraries(example_images PUBLIC microgradpp ${OpenCV_LIBS})
