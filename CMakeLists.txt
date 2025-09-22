cmake_minimum_required(VERSION 3.15...3.27)
project(datastates-llm LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
option(BUILD_PYTHON_BINDINGS "Build Python bindings using nanobind" ON)

find_package(CUDAToolkit REQUIRED)
# Locate liburing: E.g., `export LIBURING_PATH=$HOME/softwares/liburing`
find_library(LIBURING_LIB uring PATHS $ENV{LIBURING_PATH}/lib REQUIRED)
find_path(LIBURING_INCLUDE liburing.h PATHS $ENV{LIBURING_PATH}/include REQUIRED)

if(BUILD_PYTHON_BINDINGS)
    find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
    find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB}/nanobind)
endif()

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/
)

file(GLOB_RECURSE CORE_SOURCES
    src/pool/*.cpp
    src/tiers/*.cpp
    src/serializers/*.cpp
    src/file_handlers/*.cpp
    src/core_impl.cpp
    src/datastates.cpp
)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(datastates_core STATIC ${CORE_SOURCES})

target_include_directories(datastates_core PUBLIC ${PROJECT_SOURCE_DIR}/include ${LIBURING_INCLUDE})
target_link_libraries(datastates_core PUBLIC CUDA::cudart ${LIBURING_LIB})

install(DIRECTORY include/
    DESTINATION datastates/include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS datastates_core
    ARCHIVE DESTINATION datastates/lib
    LIBRARY DESTINATION datastates/lib
)

if(BUILD_PYTHON_BINDINGS)

    file(GLOB_RECURSE NB_SOURCES
        src/nb_datastates_core.cpp
        src/state_io_engine.cpp
        src/state_io_engine_impl.cpp
        src/state_provider/*.cpp
    )

    nanobind_add_module(datastates_core_py MODULE ${NB_SOURCES})
    target_link_libraries(datastates_core_py PRIVATE datastates_core)
    target_include_directories(datastates_core_py PRIVATE ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/src/state_provider)
    set_target_properties(datastates_core_py PROPERTIES OUTPUT_NAME "datastates_core")

    install(TARGETS datastates_core_py
        LIBRARY DESTINATION datastates
    )
endif()

add_subdirectory(tests)
