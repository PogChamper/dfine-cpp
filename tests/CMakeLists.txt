# dfine tests. CPU-only tests run everywhere (CI included); GPU tests exit 77
# (SKIP_RETURN_CODE) when the prerequisites — a CUDA device and DFINE_TEST_ENGINE
# pointing at a dynamic-batch engine — are missing.

# CPU-only: header-only validation logic, no CUDA/TensorRT link needed.
add_executable(dfine_test_image_layout test_image_layout.cpp)
target_include_directories(dfine_test_image_layout
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_image_layout PRIVATE dfine_warnings)
add_test(NAME image_layout COMMAND dfine_test_image_layout)

# GPU: TrtSession shape-transition recovery (frozen rejects, failpoint faults).
add_executable(dfine_test_shape_transitions test_shape_transitions.cpp)
target_include_directories(dfine_test_shape_transitions
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_shape_transitions PRIVATE dfine::dfine dfine_warnings)
add_test(NAME shape_transitions COMMAND dfine_test_shape_transitions)
set_tests_properties(shape_transitions PROPERTIES SKIP_RETURN_CODE 77)
