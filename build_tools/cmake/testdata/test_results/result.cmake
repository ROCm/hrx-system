# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Exercise the output-file contract independently of any test serializer.
if(NOT DEFINED ENV{XML_OUTPUT_FILE} OR NOT IS_DIRECTORY "$ENV{TEST_TMPDIR}")
  message(FATAL_ERROR "Test result environment or scratch directory is missing")
endif()
if(EXISTS "$ENV{XML_OUTPUT_FILE}")
  message(FATAL_ERROR "Test result was not cleared before this invocation")
endif()
file(WRITE "$ENV{XML_OUTPUT_FILE}" "${RESULT_NAME}\n")
if(RESULT_NAME STREQUAL "suite/a-b")
  message(FATAL_ERROR "Intentional fixture failure after writing its result")
endif()
