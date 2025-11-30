find_package(Python 3.8 COMPONENTS Interpreter Development REQUIRED)
find_package(nanobind CONFIG REQUIRED HINTS ${Python_SITELIB}/nanobind)

file(GLOB_RECURSE NB_SOURCES
    ${PROJECT_SOURCE_DIR}/python/nb_datastates_core.cpp
    ${PROJECT_SOURCE_DIR}/python/state_io_engine.cpp
    ${PROJECT_SOURCE_DIR}/python/state_io_engine_impl.cpp
    ${PROJECT_SOURCE_DIR}/python/state_provider/*.cpp
)

nanobind_add_module(datastates_core_py MODULE ${NB_SOURCES})

target_link_libraries(datastates_core_py PRIVATE datastates_core)
target_include_directories(datastates_core_py PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/python/state_provider
)
set_target_properties(datastates_core_py PROPERTIES OUTPUT_NAME "datastates_core")
set_target_properties(datastates_core_py PROPERTIES
    BUILD_RPATH "$ORIGIN/lib"
    INSTALL_RPATH "$ORIGIN/lib"
)

install(TARGETS datastates_core_py
    LIBRARY DESTINATION datastates
)

add_custom_target(datastates_python_install ALL
    COMMAND ${Python_EXECUTABLE} -m pip install --upgrade nanobind fasteners
    COMMAND ${Python_EXECUTABLE} -m pip install .
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/llm
    DEPENDS datastates_core_py
    COMMENT "Installing Python package and dependencies via pip"
)