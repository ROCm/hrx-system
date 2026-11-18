# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 WITH LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Loom module linking helpers.

include_guard(GLOBAL)

# Authoring tools execute on the build host, including when their output will
# be embedded in a library compiled for another platform.
function(_loom_host_tool OUTPUT_TOOL NAME)
  if(IREE_HOST_BIN_DIR)
    set(_HOST_SUFFIX "")
    if(CMAKE_HOST_WIN32)
      set(_HOST_SUFFIX ".exe")
    endif()
    file(REAL_PATH "${IREE_HOST_BIN_DIR}" _HOST_BIN_DIR
      BASE_DIRECTORY "${PROJECT_SOURCE_DIR}" EXPAND_TILDE)
    set(_PATH "${_HOST_BIN_DIR}/${NAME}${_HOST_SUFFIX}")
    if(NOT EXISTS "${_PATH}")
      message(FATAL_ERROR
        "Loom generation requires host tool ${_PATH}; "
        "provide ${NAME} in IREE_HOST_BIN_DIR.")
    endif()
    set(${OUTPUT_TOOL} "${_PATH}" PARENT_SCOPE)
  elseif(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR
      "Cross-compiling Loom products requires IREE_HOST_BIN_DIR "
      "with a build-host ${NAME} executable.")
  else()
    set(${OUTPUT_TOOL} "$<TARGET_FILE:loom::tools::${NAME}>" PARENT_SCOPE)
  endif()
endfunction()

# Resolve the declared library graph once every package has registered its
# targets. Direct dependencies remain distinct from the transitive audit
# universe, matching the public linker's strict dependency contract.
function(_loom_module_resolve_libraries TARGET_NAME)
  get_property(_RESOLVED TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_LIBRARIES SET)
  if(_RESOLVED)
    return()
  endif()
  get_property(_RESOLVING TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_RESOLVING)
  if(_RESOLVING)
    message(FATAL_ERROR "Cyclic Loom library dependency: ${TARGET_NAME}")
  endif()
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_RESOLVING TRUE)
  get_property(_DIRECT TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_LIBRARY_FILES)
  get_property(_TARGETS TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_LIBRARY_TARGETS)
  get_property(_SOURCE_TARGETS TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_SOURCE_TARGETS)
  get_property(_DATA TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_DIRECT_DATA)
  get_property(_CXX_INTERFACE TARGET "${TARGET_NAME}" PROPERTY LOOM_CXX_DIRECT_HEADERS SET)
  foreach(_FIELD HEADERS INCLUDE_OPTIONS DEFINE_OPTIONS)
    get_property(_${_FIELD} TARGET "${TARGET_NAME}" PROPERTY LOOM_CXX_DIRECT_${_FIELD})
  endforeach()
  set(_TRANSITIVE)
  foreach(_DEPENDENCY IN LISTS _TARGETS)
    _loom_module_resolve_libraries("${_DEPENDENCY}")
    get_property(_LINK_FILES TARGET "${_DEPENDENCY}" PROPERTY LOOM_MODULE_LINK_FILES)
    list(APPEND _DIRECT ${_LINK_FILES})
    if(_CXX_INTERFACE)
      foreach(_FIELD HEADERS INCLUDE_OPTIONS DEFINE_OPTIONS)
        get_property(_INTERFACE TARGET "${_DEPENDENCY}" PROPERTY LOOM_CXX_${_FIELD})
        list(APPEND _${_FIELD} ${_INTERFACE})
      endforeach()
    endif()
  endforeach()
  foreach(_DEPENDENCY IN LISTS _TARGETS _SOURCE_TARGETS)
    _loom_module_resolve_libraries("${_DEPENDENCY}")
    get_property(_CLOSURE TARGET "${_DEPENDENCY}" PROPERTY LOOM_MODULE_LIBRARIES)
    list(APPEND _TRANSITIVE ${_CLOSURE})
    get_property(_RUNTIME_DATA TARGET "${_DEPENDENCY}" PROPERTY LOOM_MODULE_DATA)
    list(APPEND _DATA ${_RUNTIME_DATA})
  endforeach()
  list(REMOVE_DUPLICATES _DIRECT)
  list(REMOVE_DUPLICATES _TRANSITIVE)
  list(REMOVE_DUPLICATES _DATA)
  if(_DIRECT)
    list(REMOVE_ITEM _TRANSITIVE ${_DIRECT})
  endif()
  get_property(_MODULE TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_FILE)
  if(_MODULE)
    set(_LINK_FILES "${_MODULE}")
  else()
    set(_LINK_FILES "${_DIRECT}")
  endif()
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_LINK_FILES "${_LINK_FILES}")
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_DIRECT_LIBRARIES "${_DIRECT}")
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_TRANSITIVE_LIBRARIES "${_TRANSITIVE}")
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_LIBRARIES "${_DIRECT};${_TRANSITIVE}")
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_DATA "${_DATA}")
  if(_CXX_INTERFACE)
    foreach(_FIELD HEADERS INCLUDE_OPTIONS DEFINE_OPTIONS)
      list(REMOVE_DUPLICATES _${_FIELD})
      set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_CXX_${_FIELD} "${_${_FIELD}}")
    endforeach()
  endif()
  set_property(TARGET "${TARGET_NAME}" PROPERTY LOOM_MODULE_RESOLVING FALSE)
endfunction()

function(loom_finalize_module_libraries)
  get_property(_TARGETS GLOBAL PROPERTY LOOM_MODULE_TARGETS)
  foreach(_TARGET IN LISTS _TARGETS)
    _loom_module_resolve_libraries("${_TARGET}")
  endforeach()
endfunction()

function(_loom_link_input_paths OUTPUT_PATHS OUTPUT_TARGETS OUTPUT_FILES)
  iree_package_source_dir(_IREE_PACKAGE_SOURCE_DIR)
  set(_PATHS)
  set(_TARGETS)
  set(_FILES)
  foreach(_INPUT IN LISTS ARGN)
    if("${_INPUT}" MATCHES "::")
      iree_package_target_name(_TARGET "${_INPUT}")
      list(APPEND _PATHS "$<TARGET_PROPERTY:${_TARGET},LOOM_MODULE_FILE>")
      list(APPEND _TARGETS "${_TARGET}")
    elseif(IS_ABSOLUTE "${_INPUT}" OR "${_INPUT}" MATCHES "^\\$<")
      list(APPEND _PATHS "${_INPUT}")
      list(APPEND _FILES "${_INPUT}")
    else()
      list(APPEND _PATHS "${_IREE_PACKAGE_SOURCE_DIR}/${_INPUT}")
      list(APPEND _FILES "${_IREE_PACKAGE_SOURCE_DIR}/${_INPUT}")
    endif()
  endforeach()
  set(${OUTPUT_PATHS} ${_PATHS} PARENT_SCOPE)
  set(${OUTPUT_TARGETS} ${_TARGETS} PARENT_SCOPE)
  set(${OUTPUT_FILES} ${_FILES} PARENT_SCOPE)
endfunction()

function(loom_module)
  iree_package_binary_dir(_IREE_PACKAGE_BINARY_DIR)
  # Tools, sources, and direct/transitive libraries are explicit inputs. Avoid
  # importing every consuming target's dependency closure into these actions.
  set(CMAKE_ADD_CUSTOM_COMMAND_DEPENDS_EXPLICIT_ONLY ON)
  cmake_parse_arguments(
    _RULE
    "INCLUDE_INPUT_EXPORTS;INCLUDE_INPUT_TESTS;STRIP_CHECK;REQUIRE_RESOLVED_CONFIG;STRICT_DEPS"
    "NAME;COMPONENT;MODE;OUTPUT;OUTPUT_FORMAT"
    "SRCS;LIBRARIES;ROOTS;CONFIGS;DATA"
    ${ARGN}
  )

  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown loom_module arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_module requires NAME")
  endif()
  if(NOT _RULE_SRCS AND NOT _RULE_LIBRARIES)
    message(FATAL_ERROR "loom_module requires SRCS or LIBRARIES")
  endif()
  if(NOT _RULE_MODE)
    set(_RULE_MODE "merge")
  endif()
  if(NOT _RULE_MODE STREQUAL "merge" AND NOT _RULE_MODE STREQUAL "link")
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} has unsupported MODE ${_RULE_MODE}"
    )
  endif()
  if(_RULE_MODE STREQUAL "merge" AND
     (_RULE_ROOTS OR _RULE_INCLUDE_INPUT_EXPORTS OR _RULE_INCLUDE_INPUT_TESTS))
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} merge mode does not accept roots"
    )
  endif()
  if(_RULE_MODE STREQUAL "link" AND
     NOT _RULE_ROOTS AND NOT _RULE_INCLUDE_INPUT_EXPORTS AND NOT _RULE_INCLUDE_INPUT_TESTS)
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} link mode requires ROOTS, INCLUDE_INPUT_EXPORTS, or INCLUDE_INPUT_TESTS"
    )
  endif()
  if(NOT _RULE_OUTPUT_FORMAT)
    set(_RULE_OUTPUT_FORMAT "text")
  endif()
  if(NOT _RULE_OUTPUT_FORMAT STREQUAL "text" AND
     NOT _RULE_OUTPUT_FORMAT STREQUAL "bc")
    message(FATAL_ERROR
      "loom_module ${_RULE_NAME} has unsupported OUTPUT_FORMAT ${_RULE_OUTPUT_FORMAT}"
    )
  endif()
  if(NOT _RULE_OUTPUT)
    if(_RULE_OUTPUT_FORMAT STREQUAL "bc")
      set(_RULE_OUTPUT "${_RULE_NAME}.loombc")
    else()
      set(_RULE_OUTPUT "${_RULE_NAME}.loom")
    endif()
  endif()

  _loom_link_input_paths(_SOURCES _SOURCE_TARGETS _SOURCE_FILES ${_RULE_SRCS})
  _loom_link_input_paths(_LIBRARY_PATHS _LIBRARY_TARGETS _LIBRARY_FILES ${_RULE_LIBRARIES})
  iree_package_name(_PACKAGE_NAME)
  set(_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_LIBRARIES "$<TARGET_PROPERTY:${_TARGET},LOOM_MODULE_DIRECT_LIBRARIES>")
  set(_TRANSITIVE_LIBRARIES "$<TARGET_GENEX_EVAL:${_TARGET},$<TARGET_PROPERTY:${_TARGET},LOOM_MODULE_TRANSITIVE_LIBRARIES>>")
  set(_OUTPUT "${_IREE_PACKAGE_BINARY_DIR}/${_RULE_OUTPUT}")
  set(_ARGS
    "--mode=${_RULE_MODE}"
    "--to=${_RULE_OUTPUT_FORMAT}"
    ${_SOURCES}
  )
  if(_RULE_COMPONENT)
    list(APPEND _ARGS "--module-name=${_RULE_COMPONENT}" "$<$<NOT:$<CONFIG:Debug>>:--strip-debug>")
  endif()
  list(APPEND _ARGS "$<$<BOOL:${_LIBRARIES}>:--library=$<JOIN:${_LIBRARIES},$<SEMICOLON>--library=>>")
  list(APPEND _ARGS
    "$<$<BOOL:${_TRANSITIVE_LIBRARIES}>:--transitive-library=$<JOIN:${_TRANSITIVE_LIBRARIES},$<SEMICOLON>--transitive-library=>>")
  foreach(_ROOT IN LISTS _RULE_ROOTS)
    list(APPEND _ARGS "--root=${_ROOT}")
  endforeach()
  foreach(_CONFIG IN LISTS _RULE_CONFIGS)
    list(APPEND _ARGS "--config=${_CONFIG}")
  endforeach()
  if(_RULE_INCLUDE_INPUT_EXPORTS)
    list(APPEND _ARGS "--include-input-exports=true")
  endif()
  if(_RULE_INCLUDE_INPUT_TESTS)
    list(APPEND _ARGS "--include-input-tests")
  endif()
  if(_RULE_STRIP_CHECK)
    list(APPEND _ARGS "--strip-check=true")
  endif()
  if(_RULE_REQUIRE_RESOLVED_CONFIG)
    list(APPEND _ARGS "--require-resolved-config=true")
  endif()
  if(_RULE_STRICT_DEPS)
    list(APPEND _ARGS "--strict-deps")
  endif()
  list(APPEND _ARGS "--output=${_OUTPUT}")
  _loom_host_tool(_LINK_TOOL loom-link)

  add_custom_command(
    OUTPUT
      "${_OUTPUT}"
    COMMAND
      "${_LINK_TOOL}" "${_ARGS}"
    DEPENDS
      "${_LINK_TOOL}"
      ${_SOURCES}
      "${_LIBRARIES}"
      "${_TRANSITIVE_LIBRARIES}"
    WORKING_DIRECTORY
      "${PROJECT_SOURCE_DIR}"
    COMMENT
      "Linking Loom module ${_RULE_OUTPUT}"
    VERBATIM
    COMMAND_EXPAND_LISTS
  )
  set_source_files_properties(
    "${_RULE_OUTPUT}"
    "${_OUTPUT}"
    PROPERTIES GENERATED TRUE
  )

  # Attach the producer directly; a DEPENDS-only wrapper adds a second custom
  # command that repeats the host tool's transitive dependency closure.
  add_custom_target("${_TARGET}" SOURCES "${_OUTPUT}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_FILE "${_OUTPUT}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_LIBRARY_FILES "${_LIBRARY_FILES}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_LIBRARY_TARGETS "${_LIBRARY_TARGETS}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_SOURCE_TARGETS "${_SOURCE_TARGETS}")
  set_property(TARGET "${_TARGET}" PROPERTY LOOM_MODULE_DIRECT_DATA "${_RULE_DATA}")
  set_property(GLOBAL APPEND PROPERTY LOOM_MODULE_TARGETS "${_TARGET}")
  foreach(_INPUT_TARGET IN LISTS _SOURCE_TARGETS _LIBRARY_TARGETS)
    iree_register_target_dependency(
      TARGET "${_TARGET}"
      DEPENDENCY "${_INPUT_TARGET}"
    )
  endforeach()
  foreach(_INPUT IN LISTS _SOURCES _LIBRARY_PATHS _RULE_DATA)
    iree_generated_output_add_consumer("${_INPUT}" "${_TARGET}")
  endforeach()
  iree_register_generated_output_producer("${_TARGET}"
    OUTPUTS "${_OUTPUT}"
  )
endfunction()
