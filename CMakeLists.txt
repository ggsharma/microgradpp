cmake_minimum_required(VERSION 3.15)

project(microgradpp)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Define your library (microgradpp)
add_library(microgradpp
        main.cpp
        # Add other source files if needed
)

# Set include directories for your library
target_include_directories(microgradpp PUBLIC include)

add_executable(m++ main.cpp)

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