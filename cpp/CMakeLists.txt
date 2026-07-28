cmake_minimum_required(VERSION 3.20)
project(fingernet_postproc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Header-only postproc library (blocks A-E + .min serialization).
add_library(fingernet_postproc INTERFACE)
add_library(fingernet::postproc ALIAS fingernet_postproc)
target_include_directories(fingernet_postproc INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(fingernet_postproc INTERFACE cxx_std_20)

# Parity harness: C++ blocks vs the Python reference dump.
add_executable(fnet_parity test/parity.cpp)
target_link_libraries(fnet_parity PRIVATE fingernet_postproc)

# Serializer: writes production artifacts (PNG via libpng + .min).
find_package(PNG)
if(PNG_FOUND)
  add_executable(serialize_demo examples/serialize_demo.cpp)
  target_link_libraries(serialize_demo PRIVATE fingernet_postproc PNG::PNG)
else()
  message(STATUS "libpng not found; skipping serialize_demo")
endif()

# arandu integration demos: postproc blocks as arandu graph nodes.
# Needs the arandu headers (header-only core). Point ARANDU_ROOT at the repo.
set(ARANDU_ROOT "/home/jcontreras/work/arandu/cpp" CACHE PATH "arandu repo cpp/ dir")
find_package(Threads REQUIRED)
if(EXISTS "${ARANDU_ROOT}/include/arandu/graph.hpp")
  add_executable(arandu_demo examples/arandu_demo.cpp)
  target_include_directories(arandu_demo PRIVATE ${ARANDU_ROOT}/include)
  target_link_libraries(arandu_demo PRIVATE fingernet_postproc Threads::Threads)
else()
  message(STATUS "arandu headers not found at ${ARANDU_ROOT}; skipping arandu_demo")
endif()

# Complete end-to-end graph: load -> ONNX(fingernet) -> postproc -> serialize.
# Needs ONNX Runtime (GPU capi ok). Provide -DORT_LIBFILE, -DORT_INCLUDE, -DORT_LIBDIR.
option(FINGERNET_WITH_ONNX "Build the complete ONNX end-to-end graph" OFF)
if(FINGERNET_WITH_ONNX)
  if(NOT ORT_LIBFILE OR NOT ORT_INCLUDE)
    message(FATAL_ERROR "FINGERNET_WITH_ONNX needs -DORT_LIBFILE=<lib> -DORT_INCLUDE=<inc>")
  endif()
  if(EXISTS "${ARANDU_ROOT}/include/arandu/graph.hpp" AND PNG_FOUND)
    foreach(tgt full_graph bench_stages)
      add_executable(${tgt} examples/${tgt}.cpp)
      target_include_directories(${tgt} PRIVATE ${ARANDU_ROOT}/include ${ORT_INCLUDE})
      target_link_libraries(${tgt} PRIVATE fingernet_postproc PNG::PNG ${ORT_LIBFILE} Threads::Threads)
      target_compile_definitions(${tgt} PRIVATE FINGERNET_WITH_ONNX)
      if(ORT_LIBDIR)
        set_target_properties(${tgt} PROPERTIES BUILD_RPATH ${ORT_LIBDIR} INSTALL_RPATH ${ORT_LIBDIR})
      endif()
    endforeach()
  else()
    message(STATUS "arandu or PNG missing; skipping full_graph")
  endif()
endif()
