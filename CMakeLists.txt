cmake_minimum_required(VERSION 3.15...3.27)
project(datastates-llm LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB}/nanobind)

include_directories(
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src/
)

file(GLOB_RECURSE CORE_SOURCES
    src/pool/*.cpp
    src/tiers/*.cpp
    src/serializers/*.cpp
    src/state_provider/*.cpp
    src/core_impl.cpp
    src/datastates.cpp
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_library(datastates_core STATIC ${CORE_SOURCES})
# target_compile_definitions(datastates_core PUBLIC __PROFILE) # Uncomment to enable profiling and debugging


target_include_directories(datastates_core PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(datastates_core PUBLIC CUDA::cudart)


set(NB_PYTHON_MODULE_SRC src/nb_datastates_core.cpp)

nanobind_add_module(datastates_core_py MODULE ${NB_PYTHON_MODULE_SRC})
target_link_libraries(datastates_core_py PRIVATE datastates_core)
target_include_directories(datastates_core_py PRIVATE ${PROJECT_SOURCE_DIR}/include)
set_target_properties(datastates_core_py PROPERTIES OUTPUT_NAME "datastates_core")

install(TARGETS datastates_core_py
    LIBRARY DESTINATION datastates
)

install(DIRECTORY include/
    DESTINATION datastates/include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS datastates_core
    ARCHIVE DESTINATION datastates/lib
    LIBRARY DESTINATION datastates/lib
)
