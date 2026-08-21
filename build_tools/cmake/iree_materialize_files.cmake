# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Materializes files for build actions.

function(_iree_validate_source_file SOURCE_FILE)
  if(SOURCE_FILE STREQUAL "")
    message(FATAL_ERROR "source file is required")
  endif()
  if(NOT EXISTS "${SOURCE_FILE}" OR IS_DIRECTORY "${SOURCE_FILE}")
    message(FATAL_ERROR "source file does not exist: ${SOURCE_FILE}")
  endif()
endfunction()

function(_iree_prepare_destination_file DESTINATION_FILE)
  if(DESTINATION_FILE STREQUAL "")
    message(FATAL_ERROR "destination file is required")
  endif()
  get_filename_component(_DESTINATION_DIRECTORY "${DESTINATION_FILE}" DIRECTORY)
  file(MAKE_DIRECTORY "${_DESTINATION_DIRECTORY}")
  # Replace any prior materialization so a link cannot redirect writes into the
  # source and a rebuilt executable cannot leave an alias on the old file.
  file(REMOVE "${DESTINATION_FILE}")
  if(EXISTS "${DESTINATION_FILE}" OR IS_SYMLINK "${DESTINATION_FILE}")
    message(FATAL_ERROR
      "could not remove destination file: ${DESTINATION_FILE}"
    )
  endif()
endfunction()

function(_iree_materialize_copy SOURCE_FILE DESTINATION_FILE)
  _iree_validate_source_file("${SOURCE_FILE}")
  _iree_prepare_destination_file("${DESTINATION_FILE}")
  file(COPY_FILE
    "${SOURCE_FILE}"
    "${DESTINATION_FILE}"
    INPUT_MAY_BE_RECENT
  )
  # Ninja compares the oldest output with the newest input for a multi-output
  # edge, so every destination must advance when the batch runs.
  file(TOUCH "${DESTINATION_FILE}")
endfunction()

function(_iree_materialize_hard_link SOURCE_FILE DESTINATION_FILE)
  _iree_validate_source_file("${SOURCE_FILE}")
  _iree_prepare_destination_file("${DESTINATION_FILE}")
  # Hard links avoid privileged Windows file symlinks and duplicate file data.
  # COPY_ON_ERROR supports filesystems without hard-link support.
  file(CREATE_LINK
    "${SOURCE_FILE}"
    "${DESTINATION_FILE}"
    COPY_ON_ERROR
  )
endfunction()

if(DEFINED IREE_MATERIALIZATION_MANIFEST)
  if(NOT EXISTS "${IREE_MATERIALIZATION_MANIFEST}" OR
     IS_DIRECTORY "${IREE_MATERIALIZATION_MANIFEST}")
    message(FATAL_ERROR
      "materialization manifest does not exist: "
      "${IREE_MATERIALIZATION_MANIFEST}"
    )
  endif()
  if(NOT DEFINED IREE_MATERIALIZATION_STAMP OR
     IREE_MATERIALIZATION_STAMP STREQUAL "")
    message(FATAL_ERROR "IREE_MATERIALIZATION_STAMP is required")
  endif()

  file(STRINGS
    "${IREE_MATERIALIZATION_MANIFEST}"
    _MATERIALIZATION_ENTRIES
    ENCODING UTF-8
  )
  if(NOT _MATERIALIZATION_ENTRIES)
    message(FATAL_ERROR "materialization manifest is empty")
  endif()
  foreach(_ENTRY IN LISTS _MATERIALIZATION_ENTRIES)
    string(FIND "${_ENTRY}" "|" _SEPARATOR_POSITION)
    if(_SEPARATOR_POSITION LESS 1)
      message(FATAL_ERROR "invalid materialization entry: ${_ENTRY}")
    endif()
    math(EXPR _DESTINATION_POSITION "${_SEPARATOR_POSITION} + 1")
    string(SUBSTRING
      "${_ENTRY}" 0 ${_SEPARATOR_POSITION} _SOURCE_FILE
    )
    string(SUBSTRING
      "${_ENTRY}" ${_DESTINATION_POSITION} -1 _DESTINATION_FILE
    )
    _iree_materialize_copy("${_SOURCE_FILE}" "${_DESTINATION_FILE}")
  endforeach()
  file(TOUCH "${IREE_MATERIALIZATION_STAMP}")
elseif(DEFINED IREE_SOURCE_FILE OR DEFINED IREE_DESTINATION_FILE)
  if(NOT DEFINED IREE_SOURCE_FILE OR IREE_SOURCE_FILE STREQUAL "")
    message(FATAL_ERROR "IREE_SOURCE_FILE is required")
  endif()
  if(NOT DEFINED IREE_DESTINATION_FILE OR IREE_DESTINATION_FILE STREQUAL "")
    message(FATAL_ERROR "IREE_DESTINATION_FILE is required")
  endif()
  if(IREE_MATERIALIZE_COPY)
    _iree_materialize_copy(
      "${IREE_SOURCE_FILE}" "${IREE_DESTINATION_FILE}"
    )
  else()
    _iree_materialize_hard_link(
      "${IREE_SOURCE_FILE}" "${IREE_DESTINATION_FILE}"
    )
  endif()
else()
  message(FATAL_ERROR
    "a materialization manifest or source/destination pair is required"
  )
endif()
