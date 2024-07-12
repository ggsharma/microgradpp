cmake_minimum_required(VERSION 3.15)

project(microgradpp_examples)
set(CMAKE_CXX_STANDARD 17)

# Define the project root directory
set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Add OpenCV submodule
find_package(OpenCV REQUIRED)

# Check if OpenCV was found
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV: ${OpenCV_VERSION}")
    add_definitions(-DOPENCV_FOUND)
    # Include directories for examples
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include ${OpenCV_INCLUDE_DIRS})
    # Add example_images
    add_executable(example_images images.cpp)
    target_link_libraries(example_images PUBLIC microgradpp ${OpenCV_LIBS})
else()
    # Include directories for examples
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include)
    message(WARNING "OpenCV not found")
endif()



## Add example_mlp
add_executable(example_mlp mlp.cpp)
target_link_libraries(example_mlp PUBLIC microgradpp)


