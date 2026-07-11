# FindTensorRT.cmake
#
# Locates TensorRT headers and requested libraries. Components:
#   Runtime     TensorRT::Runtime (nvinfer)
#   OnnxParser  TensorRT::OnnxParser (nvonnxparser, implies Runtime)
#
# Hints (in order):
#   -DTENSORRT_DIR=<path>    CMake variable
#   $TENSORRT_DIR            environment variable
#   /usr/local/TensorRT
#   /opt/tensorrt
#   /usr (system install / Jetson default)

set(_TRT_HINTS
    ${TENSORRT_DIR}
    $ENV{TENSORRT_DIR}
    /usr/local/TensorRT
    /opt/tensorrt
    /usr
)

find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS ${_TRT_HINTS}
    PATH_SUFFIXES
        include
        include/aarch64-linux-gnu
        include/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_LIB
    NAMES nvinfer
    HINTS ${_TRT_HINTS}
    PATH_SUFFIXES
        lib lib64
        lib/aarch64-linux-gnu
        lib/x86_64-linux-gnu
)

if("OnnxParser" IN_LIST TensorRT_FIND_COMPONENTS)
    find_library(TENSORRT_NVONNXPARSER_LIB
        NAMES nvonnxparser
        HINTS ${_TRT_HINTS}
        PATH_SUFFIXES
            lib lib64
            lib/aarch64-linux-gnu
            lib/x86_64-linux-gnu
    )
endif()

# Try to read the version from NvInferVersion.h if available.
if(TENSORRT_INCLUDE_DIR AND EXISTS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h")
    file(STRINGS "${TENSORRT_INCLUDE_DIR}/NvInferVersion.h" _trt_version_lines
        REGEX "^#define[ \t]+NV_TENSORRT_(MAJOR|MINOR|PATCH)[ \t]+[0-9]+")
    foreach(_line IN LISTS _trt_version_lines)
        if(_line MATCHES "NV_TENSORRT_MAJOR[ \t]+([0-9]+)")
            set(TENSORRT_VERSION_MAJOR ${CMAKE_MATCH_1})
        elseif(_line MATCHES "NV_TENSORRT_MINOR[ \t]+([0-9]+)")
            set(TENSORRT_VERSION_MINOR ${CMAKE_MATCH_1})
        elseif(_line MATCHES "NV_TENSORRT_PATCH[ \t]+([0-9]+)")
            set(TENSORRT_VERSION_PATCH ${CMAKE_MATCH_1})
        endif()
    endforeach()
    if(DEFINED TENSORRT_VERSION_MAJOR)
        set(TensorRT_VERSION
            "${TENSORRT_VERSION_MAJOR}.${TENSORRT_VERSION_MINOR}.${TENSORRT_VERSION_PATCH}")
    endif()
endif()

include(FindPackageHandleStandardArgs)
set(TensorRT_Runtime_FOUND FALSE)
if(TENSORRT_INCLUDE_DIR AND TENSORRT_NVINFER_LIB)
    set(TensorRT_Runtime_FOUND TRUE)
endif()
set(TensorRT_OnnxParser_FOUND FALSE)
if(TensorRT_Runtime_FOUND AND TENSORRT_NVONNXPARSER_LIB)
    set(TensorRT_OnnxParser_FOUND TRUE)
endif()
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TENSORRT_INCLUDE_DIR
        TENSORRT_NVINFER_LIB
    VERSION_VAR TensorRT_VERSION
    HANDLE_COMPONENTS
)

if(TensorRT_Runtime_FOUND AND NOT TARGET TensorRT::Runtime)
    add_library(TensorRT::Runtime UNKNOWN IMPORTED)
    set_target_properties(TensorRT::Runtime PROPERTIES
        IMPORTED_LOCATION ${TENSORRT_NVINFER_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
    )
endif()

if(TensorRT_OnnxParser_FOUND AND NOT TARGET TensorRT::OnnxParser)
    add_library(TensorRT::OnnxParser UNKNOWN IMPORTED)
    set_target_properties(TensorRT::OnnxParser PROPERTIES
        IMPORTED_LOCATION ${TENSORRT_NVONNXPARSER_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
        INTERFACE_LINK_LIBRARIES TensorRT::Runtime
    )
endif()

mark_as_advanced(
    TENSORRT_INCLUDE_DIR
    TENSORRT_NVINFER_LIB
    TENSORRT_NVONNXPARSER_LIB
)
