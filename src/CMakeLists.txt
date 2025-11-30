find_package(CUDAToolkit REQUIRED)

# Find liburing
set(LIBURING_ROOT "" CACHE PATH "Optional root directory of the liburing installation (contains include/ and lib/).")

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(LIBURING_PKG liburing)
endif()

set(_liburing_hint_paths)
if(LIBURING_ROOT)
    list(APPEND _liburing_hint_paths "${LIBURING_ROOT}")
endif()

if(NOT LIBURING_ROOT AND DEFINED ENV{LIBURING_ROOT})
    set(LIBURING_ROOT "$ENV{LIBURING_ROOT}")
endif()
if(LIBURING_ROOT)
    list(APPEND _liburing_hint_paths "${LIBURING_ROOT}")
endif()

if(LIBURING_PKG_FOUND)
    list(APPEND _liburing_hint_paths
        ${LIBURING_PKG_LIBRARY_DIRS}
        ${LIBURING_PKG_INCLUDE_DIRS}
    )
endif()

find_path(LIBURING_INCLUDE_DIR
    NAMES liburing.h
    HINTS ${_liburing_hint_paths}
    PATH_SUFFIXES include include/liburing
)
find_library(LIBURING_LIBRARY
    NAMES uring
    HINTS ${_liburing_hint_paths}
    PATH_SUFFIXES lib lib64
)

if(NOT LIBURING_INCLUDE_DIR OR NOT LIBURING_LIBRARY)
    message(FATAL_ERROR
        "Could not locate liburing. "
        "Set env variable LIBURING_ROOT to the liburing prefix (containing include/ and lib/)."
    )
endif()

add_library(liburing::liburing UNKNOWN IMPORTED)
set_target_properties(liburing::liburing PROPERTIES
    IMPORTED_LOCATION             "${LIBURING_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${LIBURING_INCLUDE_DIR}"
)

mark_as_advanced(LIBURING_ROOT LIBURING_INCLUDE_DIR LIBURING_LIBRARY)

# Build datastates_core library
set(CORE_SOURCES
    ${PROJECT_SOURCE_DIR}/src/pool/mem_pool.cpp
    ${PROJECT_SOURCE_DIR}/src/tiers/gpu_tier.cpp
    ${PROJECT_SOURCE_DIR}/src/tiers/host_tier.cpp
    ${PROJECT_SOURCE_DIR}/src/tiers/file_tier.cpp
    ${PROJECT_SOURCE_DIR}/src/file_handlers/pwrite_handler.cpp
    ${PROJECT_SOURCE_DIR}/src/file_handlers/io_uring_handler.cpp
    ${PROJECT_SOURCE_DIR}/src/file_handlers/base_file_handler.cpp
    ${PROJECT_SOURCE_DIR}/src/core_impl.cpp
    ${PROJECT_SOURCE_DIR}/src/datastates.cpp
)

add_library(datastates_core SHARED ${CORE_SOURCES})

target_include_directories(datastates_core PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(datastates_core PUBLIC CUDA::cudart liburing::liburing)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
    DESTINATION datastates/include
    FILES_MATCHING PATTERN "*.hpp"
)

install(TARGETS datastates_core
    ARCHIVE DESTINATION datastates/lib
    LIBRARY DESTINATION datastates/lib
)