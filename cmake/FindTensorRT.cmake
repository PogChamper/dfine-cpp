# FindTensorRT.cmake
#
# Locates TensorRT headers and libraries, exposing the imported target
# TensorRT::TensorRT (interface includes + nvinfer + nvinfer_plugin + nvonnxparser).
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

find_library(TENSORRT_NVONNXPARSER_LIB
    NAMES nvonnxparser
    HINTS ${_TRT_HINTS}
    PATH_SUFFIXES
        lib lib64
        lib/aarch64-linux-gnu
        lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVINFER_PLUGIN_LIB
    NAMES nvinfer_plugin
    HINTS ${_TRT_HINTS}
    PATH_SUFFIXES
        lib lib64
        lib/aarch64-linux-gnu
        lib/x86_64-linux-gnu
)

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
find_package_handle_standard_args(TensorRT
    REQUIRED_VARS
        TENSORRT_INCLUDE_DIR
        TENSORRT_NVINFER_LIB
        TENSORRT_NVONNXPARSER_LIB
    VERSION_VAR TensorRT_VERSION
)

if(TensorRT_FOUND AND NOT TARGET TensorRT::TensorRT)
    add_library(TensorRT::TensorRT UNKNOWN IMPORTED)
    set_target_properties(TensorRT::TensorRT PROPERTIES
        IMPORTED_LOCATION ${TENSORRT_NVINFER_LIB}
        INTERFACE_INCLUDE_DIRECTORIES ${TENSORRT_INCLUDE_DIR}
    )
    set(_trt_extra_libs ${TENSORRT_NVONNXPARSER_LIB})
    if(TENSORRT_NVINFER_PLUGIN_LIB)
        list(APPEND _trt_extra_libs ${TENSORRT_NVINFER_PLUGIN_LIB})
    endif()
    set_property(TARGET TensorRT::TensorRT APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${_trt_extra_libs})
endif()

mark_as_advanced(
    TENSORRT_INCLUDE_DIR
    TENSORRT_NVINFER_LIB
    TENSORRT_NVONNXPARSER_LIB
    TENSORRT_NVINFER_PLUGIN_LIB
)
