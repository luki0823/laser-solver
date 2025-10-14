cmake_minimum_required(VERSION 3.16)
project(laser_solver LANGUAGES CXX)

# --- compiler settings ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# optional: add warnings for safer dev builds
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -O2)
endif()

# --- source layout ---
# your includes are in include/ls/
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# --- executables ---
# you can add more apps later (like shocktube, weno_test, etc.)
add_executable(laser1d apps/laser1d.cpp)

# link directories and include paths (useful for future libraries)
target_include_directories(laser1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# optional output path organization
set_target_properties(laser1d PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

message(STATUS "✅ laser_solver build configured successfully")
