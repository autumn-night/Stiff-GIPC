cmake_minimum_required(VERSION 3.18)    
# support relative path, cuda_std_17, archive manipulation

# project meta
# ============

set(CMAKE_BUILD_TYPE "Release")
if(UNIX)
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose Release or Debug")
    endif()
    set(CMAKE_CXX_FLAGS "-O3")
endif()

project(StiffGIPC LANGUAGES CXX CUDA)

option(BUILD_GIPC_VIEWER "Build the OpenGL viewer target" ON)
option(BUILD_GIPC_BENCH "Build the headless benchmark target" ON)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# find packages
enable_language(CUDA)
find_package(Eigen3 REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(CUDAToolkit REQUIRED)

if(BUILD_GIPC_VIEWER)
    find_package(GLEW REQUIRED)
    find_package(GLUT REQUIRED)
    find_package(OpenGL REQUIRED)
endif()

execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE GIPC_GIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT GIPC_GIT_HASH)
    set(GIPC_GIT_HASH "unknown")
endif()

# add external libraries
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native")
endif()
#add_subdirectory("External/muda")


file(GLOB_RECURSE STIFFGIPC_CORE_SOURCE CONFIGURE_DEPENDS
    "StiffGIPC/*.cu" "StiffGIPC/*.cpp" "app/common/*.cpp")
list(REMOVE_ITEM STIFFGIPC_CORE_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/StiffGIPC/gl_main.cu")
file(GLOB_RECURSE STIFFGIPC_CORE_HEADER CONFIGURE_DEPENDS
    "StiffGIPC/*.h" "StiffGIPC/*.cuh" "StiffGIPC/*.hpp" "StiffGIPC/*.inl" "StiffGIPC/*.inc"
    "app/common/*.h")

set_source_files_properties(
    "app/common/scene_registry.cpp"
    "app/common/sim_settings.cpp"
    "app/common/sim_bootstrap.cpp"
    "app/bench/benchmark_runner.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/common/sim_bootstrap.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/common/scene_registry.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/common/sim_settings.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/app/bench/benchmark_runner.cpp"
    PROPERTIES LANGUAGE CUDA)

add_library(stiffgipc_core)
target_compile_features(stiffgipc_core PRIVATE cuda_std_17 cxx_std_17)
target_include_directories(stiffgipc_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(stiffgipc_core PUBLIC "StiffGIPC/")
target_include_directories(stiffgipc_core PUBLIC "StiffGIPC/muda/src/")
target_sources(stiffgipc_core PRIVATE ${STIFFGIPC_CORE_SOURCE} ${STIFFGIPC_CORE_HEADER})

if(BUILD_GIPC_VIEWER)
    add_executable(gipc_viewer app/viewer/main.cpp StiffGIPC/gl_main.cu)
    target_compile_features(gipc_viewer PRIVATE cuda_std_17 cxx_std_17)
endif()

file(GLOB_RECURSE GIPC_BENCH_SOURCE CONFIGURE_DEPENDS "app/bench/*.cpp")
file(GLOB_RECURSE GIPC_BENCH_HEADER CONFIGURE_DEPENDS "app/bench/*.h")
if(BUILD_GIPC_BENCH)
    add_executable(gipc_bench)
    target_compile_features(gipc_bench PRIVATE cuda_std_17 cxx_std_17)
    target_sources(gipc_bench PRIVATE ${GIPC_BENCH_SOURCE} ${GIPC_BENCH_HEADER})
endif()

# add assets and output directory, that we can easily access them in the code
target_compile_definitions(stiffgipc_core PUBLIC GIPC_ASSETS_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Assets/")
target_compile_definitions(stiffgipc_core PUBLIC GIPC_OUTPUT_DIR="${CMAKE_CURRENT_SOURCE_DIR}/Output/")
target_compile_definitions(stiffgipc_core PUBLIC GIPC_GIT_HASH="${GIPC_GIT_HASH}")
target_compile_definitions(stiffgipc_core PUBLIC GIPC_BUILD_TYPE="${CMAKE_BUILD_TYPE}")

target_compile_definitions(stiffgipc_core PRIVATE
USE_SNK1 
ADAPTIVE_KAPPA 
USE_FRICTION 
USE_QUADRATIC_BENDING
)


find_package(CUDAToolkit REQUIRED)

# link cuda libraries
#include cuda directories
message(STATUS "CUDA library includes: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

target_include_directories(stiffgipc_core PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

if(CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 13)
    target_include_directories(stiffgipc_core PRIVATE ${CUDAToolkit_INCLUDE_DIRS}/cccl)
endif()

target_link_libraries(stiffgipc_core PUBLIC cusparse cublas cusolver cudart_static)
target_link_libraries(stiffgipc_core PUBLIC metis_partition CUDA::cudart Eigen3::Eigen)
target_link_libraries(stiffgipc_core PUBLIC nlohmann_json::nlohmann_json)

if(BUILD_GIPC_VIEWER)
    target_link_libraries(gipc_viewer PRIVATE stiffgipc_core GLUT::GLUT GLEW::GLEW OpenGL::GLU)
endif()
if(BUILD_GIPC_BENCH)
    target_link_libraries(gipc_bench PRIVATE stiffgipc_core)
    set_target_properties(gipc_bench PROPERTIES CUDA_STANDARD 17 CUDA_ARCHITECTURES native)
endif()

set_target_properties(stiffgipc_core PROPERTIES
    CUDA_EXTENSIONS ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_STANDARD 17
    CUDA_ARCHITECTURES native)

if(BUILD_GIPC_VIEWER)
    set_target_properties(gipc_viewer PROPERTIES
        CUDA_EXTENSIONS ON
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_STANDARD 17
        CUDA_ARCHITECTURES native)
endif()

target_compile_options(stiffgipc_core
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        ${CMAKE_CUDA_FLAGS}
        --use_fast_math
        -lineinfo
        --ptxas-options=-allow-expensive-optimizations=true
        -Xcudafe
        "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
    >
    PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        --expt-relaxed-constexpr
        --default-stream=per-thread
    >
)

if(BUILD_GIPC_VIEWER)
    target_compile_options(gipc_viewer
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            ${CMAKE_CUDA_FLAGS}
            --use_fast_math
            -lineinfo
            --ptxas-options=-allow-expensive-optimizations=true
            -Xcudafe
            "--display_error_number --diag_suppress=implicit_return_from_non_void_function --diag_suppress=2642"
        >
        PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:
            --extended-lambda
            --expt-relaxed-constexpr
            --default-stream=per-thread
        >
    )
endif()

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${STIFFGIPC_CORE_SOURCE} ${STIFFGIPC_CORE_HEADER} ${GIPC_BENCH_SOURCE} ${GIPC_BENCH_HEADER})

add_subdirectory("MeshProcess")
