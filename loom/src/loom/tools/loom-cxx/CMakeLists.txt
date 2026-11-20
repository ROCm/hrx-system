# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# loom-cxx links cxx's MLIR/LLVM host back end (iree::third_party::cxx_codegen,
# from cxx's own CMake project) with Loom's C++ importer and compiler. Bazel
# does not build cxx's back end, so this tool is CMake-only and enabled by
# LOOM_CXX_HOST_CODEGEN. The generated tools package adds this directory.
if(NOT LOOM_CXX_HOST_CODEGEN)
  return()
endif()

add_executable(loom-cxx loom-cxx.cc)
set_target_properties(loom-cxx PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_options(loom-cxx PRIVATE
  $<$<CXX_COMPILER_ID:Clang,AppleClang,GNU>:-fexceptions>)
set(_loom_cxx_defs)
foreach(_target IN ITEMS iree::defines loom_defs)
  if(TARGET ${_target})
    list(APPEND _loom_cxx_defs ${_target})
  endif()
endforeach()
target_link_libraries(loom-cxx PRIVATE
  iree::third_party::cxx_codegen
  loom::binding::c::import::cxx
  loom::binding::c::loomc
  loom::binding::c::target::configured
  loom::binding::c::target::configured::ir
  loom::import::cxx
  loom::tooling::cli::loomc_result
  iree::base
  ${_loom_cxx_defs}
  ${IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES})

# loom-cxx searches Loom's loomcxx/ headers implicitly, as clang searches its
# resource directory (find_loomcxx_include_dir in loom-cxx.cc): include/ next
# to the binary in the build tree, a link to the source headers, and
# lib/loom-cxx/include in an installation.
set(_loom_cxx_headers "${IREE_SOURCE_DIR}/loom/src/loom/import/cxx/include/loomcxx")
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/include")
file(CREATE_LINK "${_loom_cxx_headers}" "${CMAKE_CURRENT_BINARY_DIR}/include/loomcxx"
     SYMBOLIC)
install(TARGETS loom-cxx RUNTIME DESTINATION bin COMPONENT loom-cxx)
install(DIRECTORY "${_loom_cxx_headers}" DESTINATION lib/loom-cxx/include
        COMPONENT loom-cxx)

if(IREE_BUILD_TESTS AND Python3_EXECUTABLE)
  # Compiles tiny translation units for the AMDGPU target (no device needed)
  # and inspects the objects: host code, an embedded code object, its
  # load record and the kernel descriptors, from one parse.
  add_test(
    NAME loom/tools/loom-cxx/loom-cxx_test
    COMMAND "${Python3_EXECUTABLE}" -B
      "${CMAKE_CURRENT_SOURCE_DIR}/loom_cxx_test.py"
      --loom-cxx "$<TARGET_FILE:loom-cxx>"
      --target amdgpu:gfx90a
  )
  set_tests_properties(loom/tools/loom-cxx/loom-cxx_test PROPERTIES
    LABELS "hostonly")
  iree_register_test_build_targets(loom/tools/loom-cxx/loom-cxx_test
    TARGETS loom-cxx)
endif()
