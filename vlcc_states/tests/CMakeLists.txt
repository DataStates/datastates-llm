# # Build C++ test for vlcc_states
add_executable(test_composite_rawptr cpp/test_composite_rawptr.cpp)
target_link_libraries(test_composite_rawptr PRIVATE vlcc_states cuda)

# add_executable(test_vlcc_states cpp/test.cpp)
# target_include_directories(test_vlcc_states PRIVATE
#     ${PROJECT_SOURCE_DIR}/include
# )
# target_link_libraries(test_vlcc_states PRIVATE vlcc_states)

# # Ensure correct CUDA runtime linking if needed
# target_link_libraries(test_vlcc_states PRIVATE CUDA::cudart)

# # Install the test binary into vlcc_states/test/
# install(TARGETS test_vlcc_states
#     RUNTIME DESTINATION vlcc_states/test
# )
