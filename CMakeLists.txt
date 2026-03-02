cmake_minimum_required(VERSION 3.15)
project(BVH2 LANGUAGES CXX)

option(BVH2_ENABLE_TESTS "Build BVH2 tests" ON)
option(BVH2_ENABLE_OPENMP "Enable OpenMP for test/perf targets if available" OFF)
option(BVH2_ENABLE_AVX2 "Enable AVX2 for test/perf targets if the compiler supports it" OFF)
option(BVH2_USE_LOCAL_TASKFLOW "Use bundled taskflow headers from this repository" ON)

set(BVH2_TASKFLOW_INCLUDE_DIR "" CACHE PATH "Optional external Taskflow include directory")

add_library(BVH2 INTERFACE)

target_compile_features(BVH2 INTERFACE cxx_std_17)

target_include_directories(BVH2 INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR "")

if(BVH2_TASKFLOW_INCLUDE_DIR)
    if(EXISTS "${BVH2_TASKFLOW_INCLUDE_DIR}/taskflow/taskflow.hpp")
        set(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR "${BVH2_TASKFLOW_INCLUDE_DIR}")
    elseif(EXISTS "${BVH2_TASKFLOW_INCLUDE_DIR}/taskflow.hpp")
        get_filename_component(BVH2_TASKFLOW_INCLUDE_PARENT "${BVH2_TASKFLOW_INCLUDE_DIR}" DIRECTORY)
        if(EXISTS "${BVH2_TASKFLOW_INCLUDE_PARENT}/taskflow/taskflow.hpp")
            set(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR "${BVH2_TASKFLOW_INCLUDE_PARENT}")
        else()
            message(FATAL_ERROR
                    "BVH2_TASKFLOW_INCLUDE_DIR='${BVH2_TASKFLOW_INCLUDE_DIR}' is not compatible with <taskflow/taskflow.hpp> includes.")
        endif()
    else()
        message(FATAL_ERROR
                "BVH2_TASKFLOW_INCLUDE_DIR='${BVH2_TASKFLOW_INCLUDE_DIR}' does not contain taskflow headers.")
    endif()
endif()

if(NOT BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR AND BVH2_USE_LOCAL_TASKFLOW)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/taskflow/taskflow/taskflow.hpp")
        set(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/taskflow")
    endif()
endif()

if(NOT BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR)
    find_path(BVH2_TASKFLOW_SYSTEM_INCLUDE_DIR taskflow/taskflow.hpp)
    if(BVH2_TASKFLOW_SYSTEM_INCLUDE_DIR)
        set(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR "${BVH2_TASKFLOW_SYSTEM_INCLUDE_DIR}")
    endif()
endif()

if(BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR)
    target_include_directories(BVH2 INTERFACE "${BVH2_TASKFLOW_RESOLVED_INCLUDE_DIR}")
else()
    message(FATAL_ERROR
            "Taskflow headers were not found. Set BVH2_TASKFLOW_INCLUDE_DIR or enable BVH2_USE_LOCAL_TASKFLOW.")
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
