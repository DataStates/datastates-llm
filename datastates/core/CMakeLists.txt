# Find Dependencies
find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB}/nanobind)
find_package(Torch REQUIRED HINTS ${Python_SITELIB}/torch/share/cmake/Torch)
find_package(CUDAToolkit REQUIRED)
find_library(TORCH_PYTHON_LIBRARY
    NAMES torch_python
    PATHS ${TORCH_LIB_PATH} ${Python_SITELIB}/torch/lib
    REQUIRED
)
message(STATUS "Found Torch: ${TORCH_FOUND}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "Torch include dirs: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Python Sitelib is ${Python_SITELIB}")
# ----------------------------------------
# 1. Build C++ core
file(GLOB_RECURSE CORE_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/common/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/pool/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tiers/*.cpp"
)

nanobind_add_module(datastates_core MODULE ${CORE_SOURCES})
set_target_properties(datastates_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(datastates_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${TORCH_INCLUDE_DIRS} 
    ${TORCH_INCLUDE_DIRS}/torch/csrc/api/include
)

target_link_libraries(datastates_core PUBLIC
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
    CUDA::cudart
)

# set(PY_SITE_PACKAGES "${Python_SITELIB}/datastates/core")
install(TARGETS datastates_core
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    # LIBRARY DESTINATION ${PY_SITE_PACKAGES}
)

# (Optional) Install headers for C++ usage
install(DIRECTORY src/
    DESTINATION include/datastates
    FILES_MATCHING PATTERN "*.hpp"
)