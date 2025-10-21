cmake_minimum_required(VERSION 3.15)
project(laser_solver LANGUAGES CXX)

# -----------------------------
# 🔧 Compiler setup
# -----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set a nice output directory for binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Include headers
include_directories(${CMAKE_SOURCE_DIR}/include)

# -----------------------------
# 📦 External dependency: toml11 (header-only)
# -----------------------------
set(TOML_HEADER "${CMAKE_SOURCE_DIR}/include/ls/external/toml.hpp")

if(NOT EXISTS ${TOML_HEADER})
    message(STATUS "Fetching toml11 header...")
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/include/ls/external)
    file(DOWNLOAD
        "https://raw.githubusercontent.com/ToruNiina/toml11/main/toml.hpp"
        ${TOML_HEADER}
        SHOW_PROGRESS
    )
endif()

# -----------------------------
# 🧩 Library target (header-only for now)
# -----------------------------
add_library(ls INTERFACE)
target_include_directories(ls INTERFACE include)

# -----------------------------
# 🚀 Applications
# -----------------------------
# Each app is a standalone executable that links against the 'ls' interface lib.

#add_executable(laser1d apps/laser1d.cpp)
#target_link_libraries(laser1d PRIVATE ls)

add_executable(shocktube apps/shocktube.cpp)
target_link_libraries(shocktube PRIVATE ls)


# (future)
# add_executable(shocktube1d apps/shocktube1d.cpp)
# target_link_libraries(shocktube1d PRIVATE ls)
# add_executable(laser2d apps/laser2d.cpp)
# target_link_libraries(laser2d PRIVATE ls)

# -----------------------------
# ✅ Status message
# -----------------------------
message(STATUS "✅ laser_solver build configured successfully")
message(STATUS "   Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
