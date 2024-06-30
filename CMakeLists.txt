cmake_minimum_required(VERSION 3.15)

project(microgradpp)
set(CMAKE_CXX_STANDARD 17)

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

# Set path to include directory for third-party headers (json.hpp)
set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests/3p/json)

# Include directories for tests
set(TESTS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tests)

# Add the third-party headers to include directory
include_directories(${JSON_INCLUDE_DIR})
include_directories(${TESTS_INCLUDE_DIR})

# Add a custom command to run the Python script to prepare expected values
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test1_output.json
        COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_tests.py ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_commands.json
        COMMENT "Running Python script to prepare expected values"
)

# Add a custom target to run the tests
add_custom_target(run_tests ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/test1_output.json)


# Build the tests executable
add_executable(tests
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/test1.cpp
        # Add other test source files here
)

# Link the microgradpp library to the tests executable
target_link_libraries(tests PRIVATE microgradpp)

# Ensure the tests are run after generating expected values
add_dependencies(tests run_tests)
