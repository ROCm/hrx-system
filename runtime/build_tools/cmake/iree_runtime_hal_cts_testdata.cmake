# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Composes a backend-owned compiled corpus with the shared CTS registry.
# SRCS basenames are the logical names consumed by executable-data lookups.
function(iree_runtime_hal_cts_testdata)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME;TARGET_NAME;TARGET_FAMILY;TARGET_KEY"
    "SRCS;BACKEND_NAMES" ${ARGN})
  if(_RULE_UNPARSED_ARGUMENTS OR NOT _RULE_NAME OR NOT _RULE_SRCS OR
     NOT _RULE_BACKEND_NAMES OR NOT _RULE_TARGET_NAME OR
     NOT _RULE_TARGET_FAMILY OR NOT _RULE_TARGET_KEY)
    message(FATAL_ERROR "Incomplete HAL CTS executable-data declaration")
  endif()

  string(REPLACE "-" "_" _IDENTIFIER "${_RULE_NAME}")
  string(REPLACE "." "_" _IDENTIFIER "${_IDENTIFIER}")
  # CMake repeats the qualified target name in each object directory. Keep the
  # private source basenames compact so MSVC can open the resulting link inputs.
  string(SHA256 _FILE_KEY "${_RULE_NAME}")
  string(SUBSTRING "${_FILE_KEY}" 0 16 _FILE_KEY)
  set(_FILE_NAME "cts_${_FILE_KEY}")
  set(_DATA_NAME "${_RULE_NAME}_data")
  iree_c_embed_data(
    NAME "${_DATA_NAME}"
    SRCS ${_RULE_SRCS}
    C_FILE_OUTPUT "${_FILE_NAME}.c"
    H_FILE_OUTPUT "${_FILE_NAME}.h"
    IDENTIFIER "${_IDENTIFIER}"
    FLATTEN
    TESTONLY
    PUBLIC
  )

  set(_TEMPLATE
    "${IREE_SOURCE_DIR}/runtime/src/iree/hal/cts/util/testdata_target.cc.tpl")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_TEMPLATE}")
  file(READ "${_TEMPLATE}" _CONTENT)
  foreach(_FIELD HEADER_PATH IDENTIFIER TARGET_FAMILY TARGET_FUNC_NAME
      TARGET_KEY TARGET_NAME TARGET_VAR_NAME)
    if(_FIELD STREQUAL "HEADER_PATH")
      set(_VALUE "${_FILE_NAME}.h")
    elseif(_FIELD STREQUAL "TARGET_FAMILY")
      set(_VALUE "${_RULE_TARGET_FAMILY}")
    elseif(_FIELD STREQUAL "TARGET_KEY")
      set(_VALUE "${_RULE_TARGET_KEY}")
    elseif(_FIELD STREQUAL "TARGET_NAME")
      set(_VALUE "${_RULE_TARGET_NAME}")
    else()
      set(_VALUE "${_IDENTIFIER}")
    endif()
    string(REPLACE "{${_FIELD}}" "${_VALUE}" _CONTENT "${_CONTENT}")
  endforeach()
  set(_REGISTRATIONS)
  set(_ORDINAL 0)
  foreach(_BACKEND IN LISTS _RULE_BACKEND_NAMES)
    set(_REGISTRATION "${_FILE_NAME}_${_ORDINAL}.cc")
    math(EXPR _ORDINAL "${_ORDINAL} + 1")
    string(REPLACE "{BACKEND_NAME}" "${_BACKEND}" _REGISTRATION_CONTENT "${_CONTENT}")
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/${_REGISTRATION}" "${_REGISTRATION_CONTENT}")
    list(APPEND _REGISTRATIONS "${_REGISTRATION}")
  endforeach()
  iree_cc_library(
    NAME "${_RULE_NAME}"
    SRCS ${_REGISTRATIONS}
    DEPS "::${_DATA_NAME}" iree::hal::cts::util::registry
    ALWAYSLINK
    TESTONLY
    PUBLIC
  )
endfunction()
