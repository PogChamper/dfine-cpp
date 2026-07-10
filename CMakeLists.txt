cmake_minimum_required(VERSION 3.20)
project(dfine VERSION 0.3.1 LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # compile_commands.json for clang-tidy/clangd

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Sanitizer build types (host C++ only; CUDA device code is not instrumented):
#   -DCMAKE_BUILD_TYPE=ASAN   AddressSanitizer + UBSan   (use for host-isolated runs)
#   -DCMAKE_BUILD_TYPE=UBSAN  UndefinedBehaviorSanitizer (safe for the full pipeline)
set(CMAKE_CXX_FLAGS_ASAN  "-O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined"
    CACHE STRING "" FORCE)
set(CMAKE_CXX_FLAGS_UBSAN "-O1 -g -fno-omit-frame-pointer -fsanitize=undefined"
    CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_ASAN     "-fsanitize=address,undefined" CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_ASAN  "-fsanitize=address,undefined" CACHE STRING "" FORCE)
set(CMAKE_EXE_LINKER_FLAGS_UBSAN    "-fsanitize=undefined"         CACHE STRING "" FORCE)
set(CMAKE_SHARED_LINKER_FLAGS_UBSAN "-fsanitize=undefined"         CACHE STRING "" FORCE)

# RTX 4070 Ti SUPER (Ada). Override with -DCMAKE_CUDA_ARCHITECTURES=... for other GPUs.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

option(DFINE_BUILD_APPS "Build CLI apps" ON)
option(DFINE_BUILD_C_API "Build the stable C ABI (c_api.h/.cpp) into libdfine" ON)
option(DFINE_BUILD_TESTS "Build test binaries (ctest)" ON)
option(DFINE_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# Warning flags carried as an INTERFACE target so they apply to our C++ sources
# only (guarded by COMPILE_LANGUAGE:CXX so nvcc's generated code and third-party
# headers like stb_image are unaffected).
add_library(dfine_warnings INTERFACE)
target_compile_options(dfine_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;$<$<BOOL:${DFINE_WARNINGS_AS_ERRORS}>:-Werror>>)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED)

# nlohmann/json (header-only). Prefer an installed copy (config package or a stray
# header next to the CUDA toolkit / conda env); fall back to FetchContent.
find_package(nlohmann_json 3.9 QUIET CONFIG)
if(NOT nlohmann_json_FOUND)
  find_path(DFINE_JSON_INCLUDE nlohmann/json.hpp
    HINTS ${CUDAToolkit_ROOT}/include $ENV{CONDA_PREFIX}/include /usr/include)
  if(DFINE_JSON_INCLUDE)
    add_library(dfine_nlohmann_json INTERFACE)
    add_library(nlohmann_json::nlohmann_json ALIAS dfine_nlohmann_json)
    target_include_directories(dfine_nlohmann_json INTERFACE ${DFINE_JSON_INCLUDE})
    message(STATUS "dfine: nlohmann/json header at ${DFINE_JSON_INCLUDE}")
  else()
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
      GIT_REPOSITORY https://github.com/nlohmann/json.git
      GIT_TAG v3.11.3 GIT_SHALLOW TRUE)
    FetchContent_MakeAvailable(nlohmann_json)
  endif()
endif()

# ---------------------------------------------------------------------------
# libdfine — preprocess (CUDA) + engine orchestration + detection decode.
# ---------------------------------------------------------------------------
add_library(dfine SHARED
  src/core/log.cpp
  src/core/engine_meta.cpp
  src/core/failpoint.cpp
  src/core/trt_session.cpp
  src/core/postprocess.cpp
  src/core/cuda_preprocess.cu
  src/core/decode_gpu.cu
  src/tasks/detector.cpp)
add_library(dfine::dfine ALIAS dfine)

# Stable C ABI (extern "C") compiled into libdfine. DFINE_BUILDING_LIB drives the
# dllexport side of DFINE_API on Windows; harmless on ELF (visibility default).
if(DFINE_BUILD_C_API)
  target_sources(dfine PRIVATE src/c_api/c_api.cpp)
  target_compile_definitions(dfine PRIVATE DFINE_BUILDING_LIB)
endif()

target_include_directories(dfine
  PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(dfine
  PUBLIC  TensorRT::TensorRT CUDA::cudart
  PRIVATE nlohmann_json::nlohmann_json)
target_compile_features(dfine PUBLIC cxx_std_17)
# ABI version, decoupled from the project version: bumped 0 -> 1 with the C ABI
# v2 break (dfine_options_t gained struct_size), so binaries linked against
# libdfine.so.0 fail at LOAD instead of misreading the new struct layout.
set(DFINE_ABI_VERSION 1)
set_target_properties(dfine PROPERTIES
  VERSION ${PROJECT_VERSION} SOVERSION ${DFINE_ABI_VERSION}
  CUDA_SEPARABLE_COMPILATION OFF)

# ---------------------------------------------------------------------------
# Apps.
# ---------------------------------------------------------------------------
if(DFINE_BUILD_APPS)
  # Engine introspection — TensorRT only, no libdfine dependency.
  add_executable(dfine_inspect apps/dfine_inspect.cpp)
  target_include_directories(dfine_inspect PRIVATE include)
  target_link_libraries(dfine_inspect PRIVATE TensorRT::TensorRT CUDA::cudart)

  add_executable(dfine_smoke apps/dfine_smoke.cpp)
  target_include_directories(dfine_smoke PRIVATE include src)
  target_link_libraries(dfine_smoke PRIVATE dfine::dfine)

  add_executable(dfine_build apps/dfine_build.cpp)
  target_include_directories(dfine_build PRIVATE include src)
  target_link_libraries(dfine_build PRIVATE dfine::dfine nlohmann_json::nlohmann_json)

  # Image decode helper (vendored stb_image) for the image-consuming apps.
  add_library(dfine_image_io STATIC apps/image_io.cpp)
  target_include_directories(dfine_image_io
    PUBLIC include PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

  add_executable(dfine_detect apps/dfine_detect.cpp)
  target_include_directories(dfine_detect PRIVATE include)
  target_link_libraries(dfine_detect PRIVATE dfine::dfine dfine_image_io)

  add_executable(dfine_coco_eval apps/dfine_coco_eval.cpp)
  target_include_directories(dfine_coco_eval PRIVATE include)
  target_link_libraries(dfine_coco_eval PRIVATE dfine::dfine dfine_image_io)

  add_executable(dfine_bench apps/dfine_bench.cpp)
  target_include_directories(dfine_bench PRIVATE include src)
  target_link_libraries(dfine_bench PRIVATE dfine::dfine dfine_image_io)

  # C-ABI test harnesses (only when the C ABI is compiled in).
  if(DFINE_BUILD_C_API)
    # Pure-C smoke: proves c_api.h is valid C and no exception escapes the ABI.
    add_executable(dfine_capi_smoke apps/dfine_capi_smoke.c)
    target_include_directories(dfine_capi_smoke PRIVATE include)
    target_link_libraries(dfine_capi_smoke PRIVATE dfine::dfine)
    target_compile_options(dfine_capi_smoke PRIVATE
      -Wall -Wextra -Wpedantic $<$<BOOL:${DFINE_WARNINGS_AS_ERRORS}>:-Werror>)

    # C++ parity check: C ABI detections must equal DFineDetector::detect exactly.
    add_executable(dfine_capi_parity apps/dfine_capi_parity.cpp)
    target_include_directories(dfine_capi_parity PRIVATE include)
    target_link_libraries(dfine_capi_parity PRIVATE dfine::dfine dfine_image_io)
  endif()
endif()

# Apply warning flags to our own targets (dfine_image_io compiles vendored stb, skip it;
# dfine_capi_smoke is C and carries its own flags above).
set(_dfine_warned dfine)
if(DFINE_BUILD_APPS)
  list(APPEND _dfine_warned dfine_inspect dfine_smoke dfine_build dfine_detect dfine_coco_eval dfine_bench)
  if(DFINE_BUILD_C_API)
    list(APPEND _dfine_warned dfine_capi_parity)
  endif()
endif()
foreach(_t IN LISTS _dfine_warned)
  target_link_libraries(${_t} PRIVATE dfine_warnings)
endforeach()

# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------
if(DFINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

message(STATUS "dfine ${PROJECT_VERSION}: TensorRT ${TensorRT_VERSION} (${TENSORRT_INCLUDE_DIR}), "
               "CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
