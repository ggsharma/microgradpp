cmake_minimum_required(VERSION 3.15)

project(microgradpp)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# For visualization
#add_subdirectory(3p/matplotplusplus)

# Set the path to your local OpenCV directory
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3p/opencv/build)

# Find OpenCV package with hints
find_package(OpenCV REQUIRED PATHS ${OpenCV_DIR} NO_DEFAULT_PATH)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Define your library (microgradpp)
add_library(microgradpp
        main.cpp
        # Add other source files if needed
)

# Set include directories for your library
target_include_directories(microgradpp PUBLIC include)

add_executable(m++ main.cpp)

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set compiler flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Apply the same compiler flags to your targets
set_target_properties(microgradpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
set_target_properties(m++ PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")

option(BUILD_EXAMPLES "Build examples" ON)

if(BUILD_EXAMPLES)
#    add_executable(example_mlp examples/mlp.cpp)
#    target_link_libraries(example_mlp PUBLIC matplot)

    add_executable(example_images examples/images.cpp)
    target_link_libraries(example_images PUBLIC  ${OpenCV_LIBS})

    set_target_properties(m++ PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    target_link_libraries(m++ microgradpp)
endif()




# Link the executable with the library
target_link_libraries(m++ microgradpp)

#######################################################
## BUILDING TESTS
#######################################################
#
## Set path to include directory for third-party headers (json.hpp)
#set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/3p/json)
#
## Include directories for tests
#set(TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)
#
## Add the third-party headers to include directory
#include_directories(${JSON_INCLUDE_DIR})
#include_directories(${TESTS_INCLUDE_DIR})
#
#
## Add a custom command to run the Python script to prepare expected values
#add_custom_command(
#        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test1_output.json
#        COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
#        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_commands.json
#        COMMENT "Running Python script to prepare expected values"
#)
#
## Add a custom target to run the tests
#add_custom_target(run_tests ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test1_output.json)
#
#
## Function to find all test executables starting with "test"
#function(find_test_executables)
#    file(GLOB TEST_EXECUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/tests/test*.cpp")
#    foreach(test_executable ${TEST_EXECUTABLES})
#        get_filename_component(test_name ${test_executable} NAME_WE)
#        add_executable(${test_name} ${test_executable})
#        target_link_libraries(${test_name} PRIVATE microgradpp)
#        add_dependencies(${test_name} run_tests)
#        add_custom_command(TARGET ${test_name} POST_BUILD
#                COMMAND ${CMAKE_COMMAND} -E echo "Running ${test_name}..."
#                COMMAND ./${test_name}
#        )
#    endforeach()
#endfunction()
#
## Call the function to find and configure test executables
#find_test_executables()