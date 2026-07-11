# dfine tests. CPU-only tests run everywhere (CI included); GPU tests exit 77
# (SKIP_RETURN_CODE) when the prerequisites — a CUDA device and DFINE_TEST_ENGINE
# pointing at a dynamic-batch engine — are missing.

# CPU-only: header-only validation logic, no CUDA/TensorRT link needed.
add_executable(dfine_test_image_layout test_image_layout.cpp)
target_include_directories(dfine_test_image_layout
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_image_layout PRIVATE dfine_warnings)
add_test(NAME image_layout COMMAND dfine_test_image_layout)

# CPU-only: the public decode limit remains independent of query count.
add_executable(dfine_test_postprocess
  test_postprocess.cpp ${CMAKE_SOURCE_DIR}/src/core/postprocess.cpp)
target_include_directories(dfine_test_postprocess PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(dfine_test_postprocess PRIVATE dfine_warnings)
add_test(NAME postprocess COMMAND dfine_test_postprocess)

# CPU-only: sidecar value validation (links libdfine for EngineMeta only).
add_executable(dfine_test_engine_meta test_engine_meta.cpp)
target_include_directories(dfine_test_engine_meta
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_engine_meta
  PRIVATE dfine::dfine dfine_warnings TensorRT::Runtime CUDA::cudart)
add_test(NAME engine_meta COMMAND dfine_test_engine_meta)

# GPU: TrtSession shape-transition recovery (frozen rejects, failpoint faults).
add_executable(dfine_test_shape_transitions test_shape_transitions.cpp)
target_include_directories(dfine_test_shape_transitions
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_shape_transitions
  PRIVATE dfine::dfine dfine_warnings TensorRT::Runtime CUDA::cudart)
add_test(NAME shape_transitions COMMAND dfine_test_shape_transitions)
set_tests_properties(shape_transitions PROPERTIES SKIP_RETURN_CODE 77)

# GPU: public-API recovery — frozen rejects, failpoint faults, stride parity,
# full-graph replay gating (needs DFINE_TEST_ENGINE_G0 for the graph section).
add_executable(dfine_test_detector_recovery test_detector_recovery.cpp)
target_include_directories(dfine_test_detector_recovery
  PRIVATE ${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(dfine_test_detector_recovery
  PRIVATE dfine::dfine dfine_warnings TensorRT::Runtime CUDA::cudart)
add_test(NAME detector_recovery COMMAND dfine_test_detector_recovery)
set_tests_properties(detector_recovery PROPERTIES SKIP_RETURN_CODE 77)
