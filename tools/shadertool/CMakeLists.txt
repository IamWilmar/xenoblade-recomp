# shadertool (Mac): traduce shaders NVN volcados a SPIR-V con el recompilador de Eden.
# cmake -S tools/shadertool -B build/shadertool && cmake --build build/shadertool -j8
cmake_minimum_required(VERSION 3.22)
project(shadertool CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release CACHE STRING "")

add_subdirectory(../../runtime/recompiler ${CMAKE_BINARY_DIR}/recompiler)

add_executable(shadertool main.cpp)
target_link_libraries(shadertool PRIVATE eden_shader)
