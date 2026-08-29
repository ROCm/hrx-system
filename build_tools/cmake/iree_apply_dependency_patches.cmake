# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT DEFINED IREE_PATCH_GIT_EXECUTABLE OR
   NOT DEFINED IREE_PATCH_SOURCE_DIR OR
   NOT DEFINED IREE_PATCH_FILES)
  message(FATAL_ERROR
    "The dependency patch driver requires a Git executable, source directory, "
    "and patch files")
endif()

set(_patch_command "${IREE_PATCH_GIT_EXECUTABLE}" apply)
set(_patch_options --whitespace=nowarn ${IREE_PATCH_ARGS})

execute_process(
  COMMAND ${_patch_command} --check ${_patch_options} ${IREE_PATCH_FILES}
  WORKING_DIRECTORY "${IREE_PATCH_SOURCE_DIR}"
  RESULT_VARIABLE _apply_check_result
  OUTPUT_VARIABLE _apply_check_output
  ERROR_VARIABLE _apply_check_error)
if(_apply_check_result EQUAL 0)
  execute_process(
    COMMAND ${_patch_command} ${_patch_options} ${IREE_PATCH_FILES}
    WORKING_DIRECTORY "${IREE_PATCH_SOURCE_DIR}"
    RESULT_VARIABLE _apply_result
    OUTPUT_VARIABLE _apply_output
    ERROR_VARIABLE _apply_error)
  if(NOT _apply_result EQUAL 0)
    message(FATAL_ERROR
      "Dependency patch application failed after its check succeeded:\n"
      "${_apply_output}${_apply_error}")
  endif()
  return()
endif()

# FetchContent may rerun a patch step against an existing source population
# when the command fingerprint changes. Accept the exact already-applied state
# so a reconfigure is idempotent.
execute_process(
  COMMAND ${_patch_command} --reverse --check ${_patch_options}
    ${IREE_PATCH_FILES}
  WORKING_DIRECTORY "${IREE_PATCH_SOURCE_DIR}"
  RESULT_VARIABLE _reverse_check_result
  OUTPUT_VARIABLE _reverse_check_output
  ERROR_VARIABLE _reverse_check_error)
if(_reverse_check_result EQUAL 0)
  return()
endif()

message(FATAL_ERROR
  "The dependency patch set neither applies to the source nor matches its "
  "already-applied state. Remove '${IREE_PATCH_SOURCE_DIR}' so FetchContent "
  "can populate a clean source tree.\n"
  "Apply check:\n${_apply_check_output}${_apply_check_error}\n"
  "Reverse check:\n${_reverse_check_output}${_reverse_check_error}")
