cmake_minimum_required(VERSION 3.15...3.27)
project(vlcc_states LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB}/nanobind)

set(NANOBIND_INCLUDE_DIR "${Python_SITELIB}/nanobind/include")
message(STATUS "Using nanobind include directory: ${NANOBIND_INCLUDE_DIR}")


set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 17) # nanobind requires C++17 or later
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE VLCC_SOURCES
    src/*.cpp
    src/serializers/*.cpp
)

# add_library(vlcc_states SHARED ${VLCC_SOURCES})
nanobind_add_module(vlcc_states MODULE NB_STATIC ${VLCC_SOURCES})

target_link_libraries(vlcc_states PRIVATE CUDA::cudart)
target_include_directories(vlcc_states PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
    ${NANOBIND_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

install(DIRECTORY include/
    DESTINATION vlcc_states/include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS vlcc_states
    EXPORT vlcc_statesTargets
    LIBRARY DESTINATION vlcc_states
    ARCHIVE DESTINATION vlcc_states
)

install(EXPORT vlcc_statesTargets
    FILE vlcc_statesTargets.cmake
    NAMESPACE vlcc_states::
    DESTINATION vlcc_states/lib/cmake/
)

# Lots of fixes required for Python bindings (e.g. iterpreting the state_region pointer as a Python object)
install(FILES src/bindings/python/__init__.py DESTINATION vlcc_states)

# Add tests
add_subdirectory(tests)


