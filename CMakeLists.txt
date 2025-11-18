cmake_minimum_required(VERSION 3.15)
project(BVH2 LANGUAGES CXX)

option(BVH2_ENABLE_TESTS "Build BVH2 tests" ON)
option(BVH2_ENABLE_OPENMP "Enable OpenMP if available" ON)
option(BVH2_ENABLE_AVX2 "Enable AVX2 if the compiler supports it" ON)
option(BVH2_USE_LOCAL_TASKFLOW "Use bundled taskflow headers from this repository" ON)

set(BVH2_TASKFLOW_INCLUDE_DIR "" CACHE PATH "Optional external Taskflow include directory")

add_library(BVH2 INTERFACE)

target_compile_features(BVH2 INTERFACE cxx_std_17)

target_include_directories(BVH2 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(BVH2_TASKFLOW_INCLUDE_DIR)
    target_include_directories(BVH2 INTERFACE "${BVH2_TASKFLOW_INCLUDE_DIR}")
elseif(BVH2_USE_LOCAL_TASKFLOW)
    target_include_directories(BVH2 INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/taskflow
    )
endif()

# Warnings and optional CPU features
if(MSVC)
    target_compile_options(BVH2 INTERFACE /W4)
else()
    target_compile_options(BVH2 INTERFACE -Wall -Wextra)
endif()

if(BVH2_ENABLE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(BVH2 INTERFACE OpenMP::OpenMP_CXX)
    endif()
endif()

if(BVH2_ENABLE_AVX2)
    include(CheckCXXCompilerFlag)
    if(MSVC)
        check_cxx_compiler_flag("/arch:AVX2" BVH2_HAS_ARCH_AVX2)
        if(BVH2_HAS_ARCH_AVX2)
            target_compile_options(BVH2 INTERFACE /arch:AVX2)
        endif()
    else()
        check_cxx_compiler_flag("-mavx2" BVH2_HAS_MAVX2)
        if(BVH2_HAS_MAVX2)
            target_compile_options(BVH2 INTERFACE -mavx2)
        endif()
    endif()
endif()

if(BVH2_ENABLE_TESTS)
    include(FetchContent)
    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/release-1.12.1.zip
    )

    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_subdirectory(test)
endif()
