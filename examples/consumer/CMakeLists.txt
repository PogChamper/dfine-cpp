# Out-of-tree consumer of an INSTALLED dfine — proves `cmake --install` ships a
# complete, relocatable package. Build it against a prefix, from anywhere:
#
#   cmake --install build --prefix /tmp/dfine-prefix
#   cmake -B build-consumer -S examples/consumer \
#         -DCMAKE_PREFIX_PATH=/tmp/dfine-prefix
#   cmake --build build-consumer
#
# CI copies the examples/ directory outside the checkout first, so nothing here
# may reference the repo other than the example source one level up.
cmake_minimum_required(VERSION 3.20)
project(dfine_consumer CXX)

find_package(dfine CONFIG REQUIRED)

add_executable(dfine_consumer ../cpp_detector_example.cpp)
target_link_libraries(dfine_consumer PRIVATE dfine::dfine)
