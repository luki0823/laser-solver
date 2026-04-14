cmake_minimum_required(VERSION 3.15)
project(laser_solver LANGUAGES CXX)

# -----------------------------
# 🔧 Compiler setup
# -----------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Output directory for binaries
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
target_include_directories(ls INTERFACE ${CMAKE_SOURCE_DIR}/include)

# -----------------------------
# 🧵 OpenMP (shared-memory parallelism)
# -----------------------------
find_package(OpenMP)

# -----------------------------
# 🚀 Applications
# -----------------------------
# add_executable(laser1d apps/laser1d.cpp)
# target_link_libraries(laser1d PRIVATE ls)

# add_executable(shocktube apps/shocktube.cpp)
# target_link_libraries(shocktube PRIVATE ls)

# add_executable(shocktube2d apps/shocktube2d.cpp)
# target_link_libraries(shocktube2d PRIVATE ls)

add_executable(laser2d apps/laser2d.cpp)
target_link_libraries(laser2d PRIVATE ls)

if(OpenMP_CXX_FOUND)
    target_link_libraries(laser2d PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(laser2d PRIVATE LS_USE_OPENMP=1)
    message(STATUS "✅ OpenMP enabled for laser2d")
else()
    message(STATUS "ℹ️ OpenMP not found; building single-threaded")
endif()

# Add below laser2d target
add_executable(convergence2d apps/convergence2d.cpp)
target_link_libraries(convergence2d PRIVATE ls)

if(OpenMP_CXX_FOUND)
    target_link_libraries(convergence2d PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(convergence2d PRIVATE LS_USE_OPENMP=1)
endif()


# -----------------------------
# ✅ Status message
# -----------------------------
message(STATUS "✅ laser_solver build configured successfully")
message(STATUS "   Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "   Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
