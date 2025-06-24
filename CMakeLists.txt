# datastates-llm/CMakeLists.txt
cmake_minimum_required(VERSION 3.15...3.27)
project(datastates-llm LANGUAGES CXX CUDA)

# Find Python for pybind11 and setting install paths
find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Python interpreter: ${PYTHON_EXECUTABLE}")
message(STATUS "Python includes: ${PYTHON_INCLUDE_DIRS}")
message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")

add_subdirectory(datastates/core)