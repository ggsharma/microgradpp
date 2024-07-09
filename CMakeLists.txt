cmake_minimum_required(VERSION 3.17)

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

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Set compiler flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Apply the same compiler flags to your targets
set_target_properties(microgradpp PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
set_target_properties(m++ PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")

if(BUILD_EXAMPLES)
    message(STATUS "Examples will be built")
add_subdirectory(examples)
endif ()
if(BUILD_TESTS)
    message(STATUS "Tests will be built")
add_subdirectory(tests)
endif ()
# Print a message to the console
message(STATUS "Finished building the microgradpp library")
