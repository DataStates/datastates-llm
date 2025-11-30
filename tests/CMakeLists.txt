enable_testing()
add_executable(test_core_engine cpp/test_core_engine.cpp)
target_link_libraries(test_core_engine PRIVATE datastates_core)

add_test(NAME cpp_test_core_engine
         COMMAND test_core_engine)
