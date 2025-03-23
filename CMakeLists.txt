cmake_minimum_required(VERSION 3.25)
project(BVH2)

include_directories(include)

set(CMAKE_CXX_STANDARD 17)

add_executable(BVH2 test/main.cpp)