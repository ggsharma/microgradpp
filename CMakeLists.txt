cmake_minimum_required(VERSION 3.17)

project(microgradpp)
set(CMAKE_CXX_STANDARD 17)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Add the microgradpp source files, excluding main.cpp (header-only, so no .cpp files)
# Define the microgradpp library
add_library(microgradpp INTERFACE)
target_include_directories(microgradpp INTERFACE include)

find_package(TBB REQUIRED)
target_link_libraries(microgradpp INTERFACE TBB::tbb)

# Set compiler flags for Release build
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Set build type to Debug by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

# Option to build the main executable
option(BUILD_MAIN_EXECUTABLE "Build the main executable" ON)

if(BUILD_MAIN_EXECUTABLE)
    add_executable(m++ main.cpp)
    set_target_properties(m++ PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS_RELEASE}")
    message(STATUS "Main executable will be built")
endif()

# Optionally build examples
if(BUILD_EXAMPLES)
    message(STATUS "Examples will be built")
    add_subdirectory(examples)
endif()

# Optionally build tests
if(BUILD_TESTS)
    message(STATUS "Tests will be built")
    add_subdirectory(tests)
endif()

# Print a message to the console
message(STATUS "Finished building the microgradpp library")
