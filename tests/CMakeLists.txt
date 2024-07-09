cmake_minimum_required(VERSION 3.15)

project(microgradpp_tests)
set(CMAKE_CXX_STANDARD 17)

set(PROJECT_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(TESTS_INCLUDE_DIR ${PROJECT_ROOT}/tests)
set(JSON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3p/json)

# Include directories for tests
include_directories(${TESTS_INCLUDE_DIR} ${JSON_INCLUDE_DIR} ${PROJECT_ROOT}/include)

# Add the third-party headers to include directory
include_directories(${JSON_INCLUDE_DIR})
include_directories(${TESTS_INCLUDE_DIR})

# Set the output directory for the test executables
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_ROOT}/build)

# Add a custom target to run the Python script to prepare expected values
add_custom_target(generate_baseline
        COMMAND ${CMAKE_COMMAND} -E env python3 ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py ${CMAKE_CURRENT_SOURCE_DIR}/test_commands.json ${CMAKE_CURRENT_BINARY_DIR}/baseline.json
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/run_tests.py ${CMAKE_CURRENT_SOURCE_DIR}/test_commands.json
        COMMENT "Running Python script to prepare expected values"
        BYPRODUCTS ${CMAKE_CURRENT_BINARY_DIR}/baseline.json
)

# Function to find all test executables starting with "test"
file(GLOB TEST_EXECUTABLES "${CMAKE_CURRENT_SOURCE_DIR}/test*.cpp")
foreach(test_executable ${TEST_EXECUTABLES})
    get_filename_component(test_name ${test_executable} NAME_WE)
    add_executable(${test_name} ${test_executable})
    target_link_libraries(${test_name} PRIVATE microgradpp)
    add_dependencies(${test_name} generate_baseline)
    add_custom_command(TARGET ${test_name} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E echo "Running ${test_name}..."
            COMMAND ${EXECUTABLE_OUTPUT_PATH}/${test_name}
    )
endforeach()