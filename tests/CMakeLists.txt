# Build C++ test for datastates_core

add_executable(test_core_engine cpp/test_core_engine.cpp)
target_include_directories(test_core_engine PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)
target_link_libraries(test_core_engine PRIVATE datastates_core)

# Ensure correct CUDA runtime linking if needed
target_link_libraries(test_core_engine PRIVATE CUDA::cudart)

# Install the test binary into datastates/test/
install(TARGETS test_core_engine
    RUNTIME DESTINATION datastates/test
)
