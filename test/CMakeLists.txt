cmake_minimum_required(VERSION 3.18)
project(
   CudaVectorTest
   VERSION 1.0
   LANGUAGES CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(CudaVectorTest Test.cu)
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT CudaVectorTest)

include_directories("../src")





