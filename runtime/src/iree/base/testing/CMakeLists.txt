# Copyright 2019 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# TODO(scotttodd): clean up bazel_to_cmake handling here
#   * this is a cc_binary in Bazel, but `linkshared` fits iree_cc_library better
#   * the output file name is platform-specific, get it with $<TARGET_FILE:>
iree_cc_library(
  NAME
    dynamic_library_test_library.so
  SRCS
    "dynamic_library_test_library.cc"
  TESTONLY
  SHARED
)

set(_TEST_LIBRARY_FILES
  "$<TARGET_FILE:iree::base::testing::dynamic_library_test_library.so>"
)
if(IREE_BUILD_TESTS AND WIN32)
  set(_TEST_LIBRARY_PDB_DIR "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>")
  set_target_properties(iree_base_testing_dynamic_library_test_library.so
    PROPERTIES
      MSVC_DEBUG_INFORMATION_FORMAT Embedded
      PDB_NAME dynamic_library_test_library
      PDB_OUTPUT_DIRECTORY "${_TEST_LIBRARY_PDB_DIR}"
  )
  target_link_options(iree_base_testing_dynamic_library_test_library.so PRIVATE
    /DEBUG
    /INCREMENTAL:NO
    # Only explicit attachment can locate the matching embedded PDB.
    /PDBALTPATH:dynamic_library_test_library.embedded.pdb
  )
  # CMake does not declare the linker PDB as an output. The empty build event
  # attaches its producing edge to the link without running another command.
  add_custom_command(TARGET iree_base_testing_dynamic_library_test_library.so
    POST_BUILD
    BYPRODUCTS "${_TEST_LIBRARY_PDB_DIR}/dynamic_library_test_library.pdb"
    COMMAND "$<0:>"
  )
  list(APPEND _TEST_LIBRARY_FILES
    "$<TARGET_PDB_FILE:iree::base::testing::dynamic_library_test_library.so>"
  )
endif()

iree_c_embed_data(
  NAME
    dynamic_library_test_library
  SRCS
    ${_TEST_LIBRARY_FILES}
  C_FILE_OUTPUT
    "dynamic_library_test_library_embed.c"
  H_FILE_OUTPUT
    "dynamic_library_test_library_embed.h"
  TESTONLY
  FLATTEN
  PUBLIC
)

iree_cc_test(
  NAME
    dynamic_library_test
  SRCS
    "dynamic_library_test.cc"
  DEPS
    ::dynamic_library_test_library
    iree::base
    iree::base::internal::dynamic_library
    iree::base::internal::symbol_resolver_win32
    iree::io::file_handle
    iree::testing::coordinated_test
    iree::testing::coordinated_test_main
    iree::testing::gtest
    iree::testing::temp_file
  LABELS
    "requires-filesystem"
)
