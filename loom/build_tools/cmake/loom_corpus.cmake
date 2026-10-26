# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/licenses/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Produces the stable target stem shared by source owners and consumers.
function(_loom_corpus_program_stem OUTPUT SOURCE_ID)
  string(REGEX REPLACE "\\.loom$" "" _PROGRAM_STEM "${SOURCE_ID}")
  string(REGEX REPLACE "[/\\.+-]" "_" _PROGRAM_STEM "${_PROGRAM_STEM}")
  set(${OUTPUT} "${_PROGRAM_STEM}" PARENT_SCOPE)
endfunction()

# Owns one reusable test module for each semantic corpus source.
function(loom_corpus_sources)
  cmake_parse_arguments(_RULE "" "NAME;MANIFEST" "SRCS" ${ARGN})
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_corpus_sources requires NAME")
  endif()
  if(NOT _RULE_MANIFEST)
    message(FATAL_ERROR
      "loom_corpus_sources(${_RULE_NAME}) requires MANIFEST")
  endif()
  if(NOT _RULE_SRCS)
    message(FATAL_ERROR
      "loom_corpus_sources(${_RULE_NAME}) requires SRCS")
  endif()
  file(GLOB_RECURSE _INVENTORY
    CONFIGURE_DEPENDS
    RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.loom"
  )
  set(_DECLARED_SRCS ${_RULE_SRCS})
  list(SORT _INVENTORY)
  list(SORT _DECLARED_SRCS)
  if(NOT "${_INVENTORY}" STREQUAL "${_DECLARED_SRCS}")
    message(FATAL_ERROR
      "loom_corpus_sources(${_RULE_NAME}) sources do not match the package inventory; "
      "declared '${_DECLARED_SRCS}', found '${_INVENTORY}'")
  endif()

  set(_MANIFEST_PROPERTY "LOOM_CORPUS_MANIFEST_${_RULE_MANIFEST}")
  get_property(_REGISTERED GLOBAL PROPERTY "${_MANIFEST_PROPERTY}_SOURCE_IDS" SET)
  if(_REGISTERED)
    message(FATAL_ERROR
      "loom_corpus_sources repeats manifest ${_RULE_MANIFEST}")
  endif()
  set(_SOURCE_IDS)
  set(_SOURCE_PATHS)
  set(_SOURCE_MODULE_TARGETS)
  foreach(_SOURCE IN LISTS _RULE_SRCS)
    set(_SOURCE_ID "${_RULE_MANIFEST}/${_SOURCE}")
    list(APPEND _SOURCE_IDS "${_SOURCE_ID}")
    list(APPEND _SOURCE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/${_SOURCE}")
    _loom_corpus_program_stem(_PROGRAM_STEM "${_SOURCE_ID}")
    loom_test_module(
      NAME "${_PROGRAM_STEM}"
      SRCS "${_SOURCE}"
    )
    iree_package_target_name(_MODULE_TARGET "::${_PROGRAM_STEM}")
    list(APPEND _SOURCE_MODULE_TARGETS "${_MODULE_TARGET}")
  endforeach()
  set_property(GLOBAL PROPERTY
    "${_MANIFEST_PROPERTY}_SOURCE_IDS" "${_SOURCE_IDS}")
  set_property(GLOBAL PROPERTY
    "${_MANIFEST_PROPERTY}_SOURCE_PATHS" "${_SOURCE_PATHS}")
  set_property(GLOBAL PROPERTY
    "${_MANIFEST_PROPERTY}_MODULE_TARGETS" "${_SOURCE_MODULE_TARGETS}")
endfunction()

# Resolves registered semantic manifests after the source tree is available.
function(_loom_corpus_resolve_manifests
    OUTPUT_SOURCE_IDS OUTPUT_SOURCE_PATHS OUTPUT_MODULE_TARGETS)
  set(_SOURCE_IDS)
  set(_SOURCE_PATHS)
  set(_MODULE_TARGETS)
  foreach(_MANIFEST IN LISTS ARGN)
    set(_MANIFEST_PROPERTY "LOOM_CORPUS_MANIFEST_${_MANIFEST}")
    get_property(_REGISTERED GLOBAL
      PROPERTY "${_MANIFEST_PROPERTY}_SOURCE_IDS" SET)
    if(NOT _REGISTERED)
      message(FATAL_ERROR "unknown Loom corpus manifest ${_MANIFEST}")
    endif()
    get_property(_MANIFEST_SOURCE_IDS GLOBAL
      PROPERTY "${_MANIFEST_PROPERTY}_SOURCE_IDS")
    get_property(_MANIFEST_SOURCE_PATHS GLOBAL
      PROPERTY "${_MANIFEST_PROPERTY}_SOURCE_PATHS")
    get_property(_MANIFEST_MODULE_TARGETS GLOBAL
      PROPERTY "${_MANIFEST_PROPERTY}_MODULE_TARGETS")
    list(APPEND _SOURCE_IDS ${_MANIFEST_SOURCE_IDS})
    list(APPEND _SOURCE_PATHS ${_MANIFEST_SOURCE_PATHS})
    list(APPEND _MODULE_TARGETS ${_MANIFEST_MODULE_TARGETS})
  endforeach()
  set(_UNIQUE_SOURCE_IDS ${_SOURCE_IDS})
  list(REMOVE_DUPLICATES _UNIQUE_SOURCE_IDS)
  list(LENGTH _SOURCE_IDS _SOURCE_ID_COUNT)
  list(LENGTH _UNIQUE_SOURCE_IDS _UNIQUE_SOURCE_ID_COUNT)
  if(NOT _SOURCE_ID_COUNT EQUAL _UNIQUE_SOURCE_ID_COUNT)
    message(FATAL_ERROR "Loom corpus manifests contain duplicate source identities")
  endif()
  set(${OUTPUT_SOURCE_IDS} ${_SOURCE_IDS} PARENT_SCOPE)
  set(${OUTPUT_SOURCE_PATHS} ${_SOURCE_PATHS} PARENT_SCOPE)
  set(${OUTPUT_MODULE_TARGETS} ${_MODULE_TARGETS} PARENT_SCOPE)
endfunction()

# Registers target-owned corpus declarations until every target profile exists.
function(loom_corpus_build)
  cmake_parse_arguments(_RULE "" "NAME" "" ${ARGN})
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_corpus_build requires NAME")
  endif()
  iree_package_name(_PACKAGE_NAME)
  set(_ID "${_PACKAGE_NAME}_${_RULE_NAME}")
  set_property(GLOBAL APPEND PROPERTY LOOM_CORPUS_BUILDS "${_ID}")
  set_property(GLOBAL PROPERTY "${_ID}_ARGUMENTS" "${ARGN}")
  iree_package_path(_PACKAGE_PATH)
  set_property(GLOBAL PROPERTY "${_ID}_PACKAGE_PATH" "${_PACKAGE_PATH}")
  foreach(_VARIABLE
      CMAKE_CURRENT_LIST_DIR CMAKE_CURRENT_SOURCE_DIR CMAKE_CURRENT_BINARY_DIR
      IREE_IDE_FOLDER)
    set_property(GLOBAL PROPERTY "${_ID}_${_VARIABLE}" "${${_VARIABLE}}")
  endforeach()
endfunction()

# Registers target-owned correctness declarations until all sources exist.
function(loom_corpus_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()
  cmake_parse_arguments(_RULE "" "NAME;PROFILE" "" ${ARGN})
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_corpus_test requires NAME")
  endif()
  if(NOT _RULE_PROFILE)
    message(FATAL_ERROR
      "loom_corpus_test(${_RULE_NAME}) requires PROFILE")
  endif()
  set(_PROFILE_STEM "${_RULE_PROFILE}")
  if(_PROFILE_STEM MATCHES ":")
    string(REGEX REPLACE "^.*:" "" _PROFILE_STEM "${_PROFILE_STEM}")
  elseif(_PROFILE_STEM MATCHES "/")
    string(REGEX REPLACE "^.*/" "" _PROFILE_STEM "${_PROFILE_STEM}")
  endif()
  string(REGEX REPLACE "[-.+]" "_" _PROFILE_STEM "${_PROFILE_STEM}")
  iree_package_name(_PACKAGE_NAME)
  set(_ID "${_PACKAGE_NAME}_${_RULE_NAME}_${_PROFILE_STEM}")
  get_property(_REGISTERED GLOBAL PROPERTY "${_ID}_ARGUMENTS" SET)
  if(_REGISTERED)
    message(FATAL_ERROR
      "loom_corpus_test(${_RULE_NAME}) repeats profile ${_RULE_PROFILE}")
  endif()
  set_property(GLOBAL APPEND PROPERTY LOOM_CORPUS_TESTS "${_ID}")
  set_property(GLOBAL PROPERTY "${_ID}_ARGUMENTS" "${ARGN}")
  iree_package_path(_PACKAGE_PATH)
  set_property(GLOBAL PROPERTY "${_ID}_PACKAGE_PATH" "${_PACKAGE_PATH}")
  foreach(_VARIABLE
      CMAKE_CURRENT_LIST_DIR CMAKE_CURRENT_SOURCE_DIR CMAKE_CURRENT_BINARY_DIR
      IREE_IDE_FOLDER)
    set_property(GLOBAL PROPERTY "${_ID}_${_VARIABLE}" "${${_VARIABLE}}")
  endforeach()
endfunction()

# Declares one registered corpus after the complete target tree is available.
function(_loom_declare_corpus_build ID)
  foreach(_VARIABLE
      CMAKE_CURRENT_LIST_DIR CMAKE_CURRENT_SOURCE_DIR CMAKE_CURRENT_BINARY_DIR
      IREE_IDE_FOLDER)
    get_property("${_VARIABLE}" GLOBAL PROPERTY "${ID}_${_VARIABLE}")
  endforeach()
  set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
  get_property(IREE_PACKAGE_ROOT_PREFIX GLOBAL PROPERTY "${ID}_PACKAGE_PATH")
  get_property(_ARGUMENTS GLOBAL PROPERTY "${ID}_ARGUMENTS")
  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "MANIFESTS;PROFILES;XFAILS;ALL_ROOTS_XFAIL;EXCLUDES"
    ${_ARGUMENTS}
  )
  if(NOT _RULE_NAME)
    message(FATAL_ERROR "loom_corpus_build requires NAME")
  endif()
  if(NOT _RULE_MANIFESTS)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) requires MANIFESTS")
  endif()
  if(NOT _RULE_PROFILES)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) requires PROFILES")
  endif()
  _loom_corpus_resolve_manifests(
    _SOURCE_IDS _SOURCES _SOURCE_MODULE_TARGETS ${_RULE_MANIFESTS})
  list(LENGTH _SOURCE_IDS _SOURCE_ID_COUNT)
  list(LENGTH _SOURCES _SOURCE_COUNT)
  if(NOT _SOURCE_ID_COUNT EQUAL _SOURCE_COUNT)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) SOURCE_IDS and SRCS must correspond")
  endif()

  set(_PROFILES)
  foreach(_PROFILE_REF IN LISTS _RULE_PROFILES)
    iree_package_target_name(_PROFILE_TARGET "${_PROFILE_REF}")
    if(NOT TARGET "${_PROFILE_TARGET}")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) names unknown profile ${_PROFILE_REF}")
    endif()
    get_property(_IS_PROFILE_SET
      TARGET "${_PROFILE_TARGET}" PROPERTY LOOM_TARGET_PROFILES SET)
    if(NOT _IS_PROFILE_SET)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) profile ${_PROFILE_REF} is not a Loom profile or set")
    endif()
    get_target_property(_TARGET_PROFILES
      "${_PROFILE_TARGET}" LOOM_TARGET_PROFILES)
    list(APPEND _PROFILES ${_TARGET_PROFILES})
  endforeach()
  list(REMOVE_DUPLICATES _PROFILES)
  set(_PROFILE_STEMS)
  foreach(_PROFILE IN LISTS _PROFILES)
    get_target_property(_COMPILER_TARGET "${_PROFILE}" LOOM_COMPILER_TARGET)
    string(REGEX REPLACE "[:/+.]" "-" _PROFILE_STEM "${_COMPILER_TARGET}")
    if(_PROFILE_STEM IN_LIST _PROFILE_STEMS)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) target profiles collide at output identity "
        "${_PROFILE_STEM}")
    endif()
    list(APPEND _PROFILE_STEMS "${_PROFILE_STEM}")
  endforeach()

  list(LENGTH _RULE_XFAILS _RULE_XFAIL_VALUE_COUNT)
  math(EXPR _XFAIL_REMAINDER "${_RULE_XFAIL_VALUE_COUNT} % 4")
  if(NOT _XFAIL_REMAINDER EQUAL 0)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) XFAILS must contain target/source/root/diagnostic groups")
  endif()
  # Target sets expand qualification entries across every selected profile.
  # Index entries by profile and source so generating the source/profile product
  # does not rescan the expanded exception lists for every output.
  set(_INDEX 0)
  while(_INDEX LESS _RULE_XFAIL_VALUE_COUNT)
    list(GET _RULE_XFAILS ${_INDEX} _TARGET_REF)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_XFAILS ${_INDEX} _SOURCE_ID)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_XFAILS ${_INDEX} _ROOT)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_XFAILS ${_INDEX} _DIAGNOSTIC)
    math(EXPR _INDEX "${_INDEX} + 1")
    iree_package_target_name(_XFAIL_TARGET "${_TARGET_REF}")
    if(NOT TARGET "${_XFAIL_TARGET}")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) names unknown xfail target ${_TARGET_REF}")
    endif()
    get_property(_IS_PROFILE_SET
      TARGET "${_XFAIL_TARGET}" PROPERTY LOOM_TARGET_PROFILES SET)
    if(NOT _IS_PROFILE_SET)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) xfail key ${_TARGET_REF} is not a Loom profile or set")
    endif()
    if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) xfail names unknown source ${_SOURCE_ID}")
    endif()
    if(NOT _ROOT MATCHES "^@")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) xfail root ${_ROOT} must begin with '@'")
    endif()
    if(NOT _DIAGNOSTIC)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) xfail for ${_ROOT} requires a diagnostic")
    endif()
    get_target_property(_XFAIL_PROFILES
      "${_XFAIL_TARGET}" LOOM_TARGET_PROFILES)
    foreach(_PROFILE IN LISTS _XFAIL_PROFILES)
      get_target_property(_PROFILE_AVAILABLE
        "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
      if(NOT _PROFILE_AVAILABLE)
        continue()
      endif()
      if(NOT _PROFILE IN_LIST _PROFILES)
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) has an xfail for unselected profile ${_PROFILE}")
      endif()
      string(SHA256 _PROFILE_SOURCE_KEY "${_PROFILE}|${_SOURCE_ID}")
      string(SHA256 _XFAIL_IDENTITY_KEY
        "${_PROFILE}|${_SOURCE_ID}|${_ROOT}")
      set(_XFAIL_IDENTITY_VARIABLE
        "_LOOM_CORPUS_XFAIL_IDENTITY_${_XFAIL_IDENTITY_KEY}")
      if(DEFINED ${_XFAIL_IDENTITY_VARIABLE})
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) repeats xfail ${_ROOT} for ${_PROFILE}")
      endif()
      set(${_XFAIL_IDENTITY_VARIABLE} TRUE)
      set(_XFAIL_ROOTS_VARIABLE
        "_LOOM_CORPUS_XFAIL_ROOTS_${_PROFILE_SOURCE_KEY}")
      set(_XFAIL_DIAGNOSTICS_VARIABLE
        "_LOOM_CORPUS_XFAIL_DIAGNOSTICS_${_PROFILE_SOURCE_KEY}")
      list(APPEND ${_XFAIL_ROOTS_VARIABLE} "${_ROOT}")
      list(APPEND ${_XFAIL_DIAGNOSTICS_VARIABLE} "${_DIAGNOSTIC}")
    endforeach()
  endwhile()

  list(LENGTH _RULE_ALL_ROOTS_XFAIL _RULE_ALL_ROOTS_XFAIL_VALUE_COUNT)
  math(EXPR _ALL_ROOTS_XFAIL_REMAINDER
    "${_RULE_ALL_ROOTS_XFAIL_VALUE_COUNT} % 2")
  if(NOT _ALL_ROOTS_XFAIL_REMAINDER EQUAL 0)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) ALL_ROOTS_XFAIL must contain target/source pairs")
  endif()
  set(_INDEX 0)
  while(_INDEX LESS _RULE_ALL_ROOTS_XFAIL_VALUE_COUNT)
    list(GET _RULE_ALL_ROOTS_XFAIL ${_INDEX} _TARGET_REF)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_ALL_ROOTS_XFAIL ${_INDEX} _SOURCE_ID)
    math(EXPR _INDEX "${_INDEX} + 1")
    iree_package_target_name(_ALL_ROOTS_XFAIL_TARGET "${_TARGET_REF}")
    if(NOT TARGET "${_ALL_ROOTS_XFAIL_TARGET}")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) names unknown all-roots xfail target ${_TARGET_REF}")
    endif()
    get_property(_IS_PROFILE_SET
      TARGET "${_ALL_ROOTS_XFAIL_TARGET}" PROPERTY LOOM_TARGET_PROFILES SET)
    if(NOT _IS_PROFILE_SET)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) all-roots xfail key ${_TARGET_REF} is not a Loom profile or set")
    endif()
    if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) all-roots xfail names unknown source ${_SOURCE_ID}")
    endif()
    get_target_property(_ALL_ROOTS_XFAIL_PROFILES
      "${_ALL_ROOTS_XFAIL_TARGET}" LOOM_TARGET_PROFILES)
    foreach(_PROFILE IN LISTS _ALL_ROOTS_XFAIL_PROFILES)
      get_target_property(_PROFILE_AVAILABLE
        "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
      if(NOT _PROFILE_AVAILABLE)
        continue()
      endif()
      if(NOT _PROFILE IN_LIST _PROFILES)
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) marks all roots xfail for unselected profile ${_PROFILE}")
      endif()
      string(SHA256 _PROFILE_SOURCE_KEY "${_PROFILE}|${_SOURCE_ID}")
      set(_ALL_ROOTS_XFAIL_VARIABLE
        "_LOOM_CORPUS_ALL_ROOTS_XFAIL_${_PROFILE_SOURCE_KEY}")
      if(DEFINED ${_ALL_ROOTS_XFAIL_VARIABLE})
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) repeats all-roots xfail for ${_SOURCE_ID} on ${_PROFILE}")
      endif()
      set(_XFAIL_ROOTS_VARIABLE
        "_LOOM_CORPUS_XFAIL_ROOTS_${_PROFILE_SOURCE_KEY}")
      if(NOT DEFINED ${_XFAIL_ROOTS_VARIABLE})
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) marks all roots xfail for ${_SOURCE_ID} on ${_PROFILE} without diagnostic xfails")
      endif()
      set(${_ALL_ROOTS_XFAIL_VARIABLE} TRUE)
    endforeach()
  endwhile()

  list(LENGTH _RULE_EXCLUDES _RULE_EXCLUDE_VALUE_COUNT)
  math(EXPR _EXCLUDE_REMAINDER "${_RULE_EXCLUDE_VALUE_COUNT} % 3")
  if(NOT _EXCLUDE_REMAINDER EQUAL 0)
    message(FATAL_ERROR
      "loom_corpus_build(${_RULE_NAME}) EXCLUDES must contain target/source/reason groups")
  endif()
  set(_INDEX 0)
  while(_INDEX LESS _RULE_EXCLUDE_VALUE_COUNT)
    list(GET _RULE_EXCLUDES ${_INDEX} _TARGET_REF)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_EXCLUDES ${_INDEX} _SOURCE_ID)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_EXCLUDES ${_INDEX} _REASON)
    math(EXPR _INDEX "${_INDEX} + 1")
    iree_package_target_name(_EXCLUDE_TARGET "${_TARGET_REF}")
    if(NOT TARGET "${_EXCLUDE_TARGET}")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) names unknown exclusion target ${_TARGET_REF}")
    endif()
    get_property(_IS_PROFILE_SET
      TARGET "${_EXCLUDE_TARGET}" PROPERTY LOOM_TARGET_PROFILES SET)
    if(NOT _IS_PROFILE_SET)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) exclusion key ${_TARGET_REF} is not a Loom profile or set")
    endif()
    if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) exclusion names unknown source ${_SOURCE_ID}")
    endif()
    if(NOT _REASON)
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) exclusion for ${_SOURCE_ID} requires a reason")
    endif()
    get_target_property(_EXCLUDE_PROFILES
      "${_EXCLUDE_TARGET}" LOOM_TARGET_PROFILES)
    foreach(_PROFILE IN LISTS _EXCLUDE_PROFILES)
      get_target_property(_PROFILE_AVAILABLE
        "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
      if(NOT _PROFILE_AVAILABLE)
        continue()
      endif()
      if(NOT _PROFILE IN_LIST _PROFILES)
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) excludes unselected profile ${_PROFILE}")
      endif()
      string(SHA256 _PROFILE_SOURCE_KEY "${_PROFILE}|${_SOURCE_ID}")
      set(_EXCLUDE_VARIABLE
        "_LOOM_CORPUS_EXCLUDE_${_PROFILE_SOURCE_KEY}")
      if(DEFINED ${_EXCLUDE_VARIABLE})
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) repeats exclusion for ${_SOURCE_ID} on ${_PROFILE}")
      endif()
      set(_XFAIL_ROOTS_VARIABLE
        "_LOOM_CORPUS_XFAIL_ROOTS_${_PROFILE_SOURCE_KEY}")
      if(DEFINED ${_XFAIL_ROOTS_VARIABLE})
        message(FATAL_ERROR
          "loom_corpus_build(${_RULE_NAME}) cannot both exclude and xfail "
          "${_SOURCE_ID} for ${_PROFILE}")
      endif()
      set(${_EXCLUDE_VARIABLE} TRUE)
    endforeach()
  endwhile()

  iree_package_name(_PACKAGE_NAME)
  set(_PROGRAM_TARGETS)
  set(_SEMANTIC_NAMES)
  set(_ALL_ARTIFACTS)
  set(_ALL_REPORTS)
  set(_ALL_XFAIL_RESULTS)
  math(EXPR _LAST_SOURCE_INDEX "${_SOURCE_COUNT} - 1")
  foreach(_SOURCE_INDEX RANGE ${_LAST_SOURCE_INDEX})
    list(GET _SOURCE_IDS ${_SOURCE_INDEX} _SOURCE_ID)
    list(GET _SOURCES ${_SOURCE_INDEX} _SOURCE)
    if(NOT _SOURCE_ID MATCHES "^([^/]+)/.+\\.loom$")
      message(FATAL_ERROR
        "loom_corpus_build(${_RULE_NAME}) has invalid source identity ${_SOURCE_ID}")
    endif()
    set(_SEMANTIC_NAME "${CMAKE_MATCH_1}")
    list(APPEND _SEMANTIC_NAMES "${_SEMANTIC_NAME}")
    _loom_corpus_program_stem(_PROGRAM_STEM "${_SOURCE_ID}")
    set(_PROGRAM_TARGET "${_PACKAGE_NAME}_${_PROGRAM_STEM}")
    set(_PROGRAM_OUTPUTS)
    set(_PROGRAM_ARTIFACTS)
    set(_PROGRAM_REPORTS)
    set(_PROGRAM_XFAIL_RESULTS)
    set(_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_PROGRAM_STEM}")

    set(_ACTIVE_PROFILES)
    foreach(_PROFILE IN LISTS _PROFILES)
      get_target_property(_AVAILABLE "${_PROFILE}" LOOM_PROFILE_AVAILABLE)
      if(NOT _AVAILABLE)
        continue()
      endif()
      string(SHA256 _PROFILE_SOURCE_KEY "${_PROFILE}|${_SOURCE_ID}")
      set(_EXCLUDE_VARIABLE
        "_LOOM_CORPUS_EXCLUDE_${_PROFILE_SOURCE_KEY}")
      if(NOT DEFINED ${_EXCLUDE_VARIABLE})
        list(APPEND _ACTIVE_PROFILES "${_PROFILE}")
      endif()
    endforeach()

    if(_ACTIVE_PROFILES)
      set(_SUBJECT_MODULE_NAME "${_PROGRAM_STEM}_subjects")
      loom_module(
        NAME "${_SUBJECT_MODULE_NAME}"
        SRCS "${_SOURCE}"
        MODE link
        OUTPUT_FORMAT bc
        INCLUDE_INPUT_TESTS
        STRIP_CHECK
      )
      iree_package_target_name(_SUBJECT_MODULE_TARGET "::${_SUBJECT_MODULE_NAME}")
      get_target_property(_SUBJECT_MODULE
        "${_SUBJECT_MODULE_TARGET}" LOOM_MODULE_FILE)
    endif()

    foreach(_PROFILE IN LISTS _ACTIVE_PROFILES)
      get_target_property(_COMPILER_TARGET "${_PROFILE}" LOOM_COMPILER_TARGET)
      string(REGEX REPLACE "[:/+.]" "-" _PROFILE_STEM "${_COMPILER_TARGET}")
      string(SHA256 _PROFILE_SOURCE_KEY "${_PROFILE}|${_SOURCE_ID}")
      set(_XFAIL_ROOTS_VARIABLE
        "_LOOM_CORPUS_XFAIL_ROOTS_${_PROFILE_SOURCE_KEY}")
      set(_XFAIL_DIAGNOSTICS_VARIABLE
        "_LOOM_CORPUS_XFAIL_DIAGNOSTICS_${_PROFILE_SOURCE_KEY}")
      set(_XFAIL_ROOTS "${${_XFAIL_ROOTS_VARIABLE}}")
      set(_XFAIL_DIAGNOSTICS "${${_XFAIL_DIAGNOSTICS_VARIABLE}}")
      set(_ALL_ROOTS_XFAIL_VARIABLE
        "_LOOM_CORPUS_ALL_ROOTS_XFAIL_${_PROFILE_SOURCE_KEY}")
      if(DEFINED ${_ALL_ROOTS_XFAIL_VARIABLE})
        set(_REQUIRE_ALL_ROOTS TRUE)
      else()
        set(_REQUIRE_ALL_ROOTS FALSE)
      endif()

      if(NOT _REQUIRE_ALL_ROOTS)
        set(_ARTIFACT "${_OUTPUT_DIR}/${_PROFILE_STEM}.artifact")
        set(_REPORT "${_OUTPUT_DIR}/${_PROFILE_STEM}.compile.json")
        set(_EXCLUDE_ARGS)
        foreach(_ROOT IN LISTS _XFAIL_ROOTS)
          list(APPEND _EXCLUDE_ARGS "--exclude-root=${_ROOT}")
        endforeach()
        add_custom_command(
          OUTPUT "${_ARTIFACT}" "${_REPORT}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_OUTPUT_DIR}"
          COMMAND "$<TARGET_FILE:loom::tools::loom-compile>"
            "${_SUBJECT_MODULE}"
            "--target=${_COMPILER_TARGET}"
            ${_EXCLUDE_ARGS}
            "--output=${_ARTIFACT}"
            "--compile-report=details"
            "--compile-report-output=${_REPORT}"
          DEPENDS loom::tools::loom-compile "${_SUBJECT_MODULE}"
          COMMENT "Compiling corpus program ${_SOURCE_ID} for ${_COMPILER_TARGET}"
          VERBATIM
        )
        list(APPEND _PROGRAM_OUTPUTS "${_ARTIFACT}" "${_REPORT}")
        list(APPEND _PROGRAM_ARTIFACTS "${_ARTIFACT}")
        list(APPEND _PROGRAM_REPORTS "${_REPORT}")
      endif()

      list(LENGTH _XFAIL_ROOTS _PROFILE_XFAIL_COUNT)
      if(_PROFILE_XFAIL_COUNT GREATER 0)
        set(_XFAIL_ARGS)
        if(_REQUIRE_ALL_ROOTS)
          list(APPEND _XFAIL_ARGS "--require-all-roots")
        endif()
        set(_PROFILE_XFAIL_INDEX 0)
        while(_PROFILE_XFAIL_INDEX LESS _PROFILE_XFAIL_COUNT)
          list(GET _XFAIL_ROOTS ${_PROFILE_XFAIL_INDEX} _ROOT)
          list(GET _XFAIL_DIAGNOSTICS ${_PROFILE_XFAIL_INDEX} _DIAGNOSTIC)
          math(EXPR _PROFILE_XFAIL_INDEX "${_PROFILE_XFAIL_INDEX} + 1")
          list(APPEND _XFAIL_ARGS
            "--expected-root=${_ROOT}"
            "--expected-diagnostic=${_DIAGNOSTIC}"
          )
        endwhile()
        set(_XFAIL_RESULT "${_OUTPUT_DIR}/${_PROFILE_STEM}.xfails")
        add_custom_command(
          OUTPUT "${_XFAIL_RESULT}"
          COMMAND "${CMAKE_COMMAND}" -E make_directory "${_OUTPUT_DIR}"
          COMMAND "$<TARGET_FILE:loom::build_tools::corpus::loom-corpus-compile-xfails>"
            "--compiler=$<TARGET_FILE:loom::tools::loom-compile>"
            "--stamp-output=${_XFAIL_RESULT}"
            ${_XFAIL_ARGS}
            "${_SUBJECT_MODULE}"
            "--target=${_COMPILER_TARGET}"
          DEPENDS
            loom::build_tools::corpus::loom-corpus-compile-xfails
            loom::tools::loom-compile
            "${_SUBJECT_MODULE}"
          COMMENT "Probing corpus diagnostic xfails in ${_SOURCE_ID} for ${_COMPILER_TARGET}"
          VERBATIM
        )
        list(APPEND _PROGRAM_OUTPUTS "${_XFAIL_RESULT}")
        list(APPEND _PROGRAM_XFAIL_RESULTS "${_XFAIL_RESULT}")
      endif()
    endforeach()

    add_custom_target("${_PROGRAM_TARGET}" DEPENDS ${_PROGRAM_OUTPUTS})
    set_target_properties("${_PROGRAM_TARGET}" PROPERTIES
      LOOM_CORPUS_ARTIFACTS "${_PROGRAM_ARTIFACTS}"
      LOOM_CORPUS_COMPILE_REPORTS "${_PROGRAM_REPORTS}"
      LOOM_CORPUS_XFAIL_RESULTS "${_PROGRAM_XFAIL_RESULTS}"
      LOOM_CORPUS_SOURCE_ID "${_SOURCE_ID}"
    )
    list(APPEND _PROGRAM_TARGETS "${_PROGRAM_TARGET}")
    list(APPEND "_SEMANTIC_PROGRAMS_${_SEMANTIC_NAME}" "${_PROGRAM_TARGET}")
    list(APPEND "_SEMANTIC_ARTIFACTS_${_SEMANTIC_NAME}" ${_PROGRAM_ARTIFACTS})
    list(APPEND "_SEMANTIC_REPORTS_${_SEMANTIC_NAME}" ${_PROGRAM_REPORTS})
    list(APPEND "_SEMANTIC_XFAIL_RESULTS_${_SEMANTIC_NAME}"
      ${_PROGRAM_XFAIL_RESULTS})
    list(APPEND _ALL_ARTIFACTS ${_PROGRAM_ARTIFACTS})
    list(APPEND _ALL_REPORTS ${_PROGRAM_REPORTS})
    list(APPEND _ALL_XFAIL_RESULTS ${_PROGRAM_XFAIL_RESULTS})
  endforeach()

  list(REMOVE_DUPLICATES _SEMANTIC_NAMES)
  foreach(_SEMANTIC_NAME IN LISTS _SEMANTIC_NAMES)
    set(_SEMANTIC_TARGET "${_PACKAGE_NAME}_${_SEMANTIC_NAME}")
    add_custom_target("${_SEMANTIC_TARGET}"
      DEPENDS ${_SEMANTIC_PROGRAMS_${_SEMANTIC_NAME}})
    set_target_properties("${_SEMANTIC_TARGET}" PROPERTIES
      LOOM_CORPUS_ARTIFACTS "${_SEMANTIC_ARTIFACTS_${_SEMANTIC_NAME}}"
      LOOM_CORPUS_COMPILE_REPORTS "${_SEMANTIC_REPORTS_${_SEMANTIC_NAME}}"
      LOOM_CORPUS_XFAIL_RESULTS "${_SEMANTIC_XFAIL_RESULTS_${_SEMANTIC_NAME}}"
    )
  endforeach()

  set(_CORPUS_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  add_custom_target("${_CORPUS_TARGET}" DEPENDS ${_PROGRAM_TARGETS})
  set_target_properties("${_CORPUS_TARGET}" PROPERTIES
    LOOM_CORPUS_ARTIFACTS "${_ALL_ARTIFACTS}"
    LOOM_CORPUS_COMPILE_REPORTS "${_ALL_REPORTS}"
    LOOM_CORPUS_XFAIL_RESULTS "${_ALL_XFAIL_RESULTS}"
  )
endfunction()

# Declares one registered correctness profile after semantic sources exist.
function(_loom_declare_corpus_test ID)
  foreach(_VARIABLE
      CMAKE_CURRENT_LIST_DIR CMAKE_CURRENT_SOURCE_DIR CMAKE_CURRENT_BINARY_DIR
      IREE_IDE_FOLDER)
    get_property("${_VARIABLE}" GLOBAL PROPERTY "${ID}_${_VARIABLE}")
  endforeach()
  set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
  get_property(IREE_PACKAGE_ROOT_PREFIX GLOBAL PROPERTY "${ID}_PACKAGE_PATH")
  get_property(_ARGUMENTS GLOBAL PROPERTY "${ID}_ARGUMENTS")
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;PROFILE;RESOURCE_GROUP"
    "MANIFESTS;SOURCES;XFAILS;ALLOWED_FAILURES;EXCLUDES;ARGS;RUNNER_ARGS;LABELS;REQUIRES"
    ${_ARGUMENTS}
  )
  if(_RULE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR
      "Unknown loom_corpus_test arguments: ${_RULE_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT _RULE_MANIFESTS)
    message(FATAL_ERROR
      "loom_corpus_test(${_RULE_NAME}) requires MANIFESTS")
  endif()
  if(DEFINED _RULE_REQUIRES)
    if(NOT (${_RULE_REQUIRES}))
      return()
    endif()
  endif()

  _loom_corpus_resolve_manifests(
    _SOURCE_IDS _SOURCES _SOURCE_MODULE_TARGETS ${_RULE_MANIFESTS})
  list(LENGTH _SOURCE_IDS _SOURCE_ID_COUNT)
  list(LENGTH _SOURCES _SOURCE_COUNT)
  list(LENGTH _SOURCE_MODULE_TARGETS _SOURCE_MODULE_TARGET_COUNT)
  if(NOT _SOURCE_ID_COUNT EQUAL _SOURCE_COUNT OR
     NOT _SOURCE_ID_COUNT EQUAL _SOURCE_MODULE_TARGET_COUNT)
    message(FATAL_ERROR
      "loom_corpus_test(${_RULE_NAME}) source identities, paths, and modules "
      "must correspond")
  endif()

  set(_SELECTED_SOURCES)
  foreach(_SOURCE_ID IN LISTS _RULE_SOURCES)
    if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) selects unknown source ${_SOURCE_ID}")
    endif()
    if(_SOURCE_ID IN_LIST _SELECTED_SOURCES)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) repeats selected source ${_SOURCE_ID}")
    endif()
    list(APPEND _SELECTED_SOURCES "${_SOURCE_ID}")
  endforeach()

  list(LENGTH _RULE_EXCLUDES _EXCLUDE_VALUE_COUNT)
  math(EXPR _EXCLUDE_REMAINDER "${_EXCLUDE_VALUE_COUNT} % 2")
  if(NOT _EXCLUDE_REMAINDER EQUAL 0)
    message(FATAL_ERROR
      "loom_corpus_test(${_RULE_NAME}) EXCLUDES must contain source/reason pairs")
  endif()
  set(_EXCLUDED_SOURCES)
  set(_INDEX 0)
  while(_INDEX LESS _EXCLUDE_VALUE_COUNT)
    list(GET _RULE_EXCLUDES ${_INDEX} _SOURCE_ID)
    math(EXPR _INDEX "${_INDEX} + 1")
    list(GET _RULE_EXCLUDES ${_INDEX} _REASON)
    math(EXPR _INDEX "${_INDEX} + 1")
    if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) excludes unknown source ${_SOURCE_ID}")
    endif()
    if(NOT _REASON)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) exclusion for ${_SOURCE_ID} requires a reason")
    endif()
    if(_SOURCE_ID IN_LIST _EXCLUDED_SOURCES)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) repeats exclusion for ${_SOURCE_ID}")
    endif()
    if(_SOURCE_ID IN_LIST _SELECTED_SOURCES)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) source ${_SOURCE_ID} cannot be both selected and excluded")
    endif()
    list(APPEND _EXCLUDED_SOURCES "${_SOURCE_ID}")
  endwhile()

  set(_FAILURE_QUALIFICATION_KEYS)
  foreach(_QUALIFICATION_KIND XFAIL ALLOWED_FAILURE)
    set(_QUALIFICATION_ARGUMENT "${_QUALIFICATION_KIND}S")
    set(_QUALIFICATION_VALUES_VARIABLE "_RULE_${_QUALIFICATION_ARGUMENT}")
    set(_QUALIFICATION_VALUES ${${_QUALIFICATION_VALUES_VARIABLE}})
    if(_QUALIFICATION_KIND STREQUAL "XFAIL")
      set(_QUALIFICATION_NAME "xfail")
      set(_QUALIFICATION_FLAG "--xfail")
    else()
      set(_QUALIFICATION_NAME "allowed failure")
      set(_QUALIFICATION_FLAG "--allow-failure")
    endif()
    list(LENGTH _QUALIFICATION_VALUES _QUALIFICATION_VALUE_COUNT)
    math(EXPR _QUALIFICATION_REMAINDER "${_QUALIFICATION_VALUE_COUNT} % 3")
    if(NOT _QUALIFICATION_REMAINDER EQUAL 0)
      message(FATAL_ERROR
        "loom_corpus_test(${_RULE_NAME}) ${_QUALIFICATION_ARGUMENT} must contain source/record/diagnostic triples")
    endif()
    set(_INDEX 0)
    while(_INDEX LESS _QUALIFICATION_VALUE_COUNT)
      list(GET _QUALIFICATION_VALUES ${_INDEX} _SOURCE_ID)
      math(EXPR _INDEX "${_INDEX} + 1")
      list(GET _QUALIFICATION_VALUES ${_INDEX} _RECORD)
      math(EXPR _INDEX "${_INDEX} + 1")
      list(GET _QUALIFICATION_VALUES ${_INDEX} _DIAGNOSTIC)
      math(EXPR _INDEX "${_INDEX} + 1")
      if(NOT _SOURCE_ID IN_LIST _SOURCE_IDS)
        message(FATAL_ERROR
          "loom_corpus_test(${_RULE_NAME}) ${_QUALIFICATION_NAME} names unknown source ${_SOURCE_ID}")
      endif()
      if(_SOURCE_ID IN_LIST _EXCLUDED_SOURCES)
        message(FATAL_ERROR
          "loom_corpus_test(${_RULE_NAME}) source ${_SOURCE_ID} cannot be both excluded and qualified by ${_QUALIFICATION_NAME}")
      endif()
      if(_SELECTED_SOURCES AND NOT _SOURCE_ID IN_LIST _SELECTED_SOURCES)
        message(FATAL_ERROR
          "loom_corpus_test(${_RULE_NAME}) ${_QUALIFICATION_NAME} names unselected source ${_SOURCE_ID}")
      endif()
      if(NOT _RECORD MATCHES "^@" OR NOT _DIAGNOSTIC)
        message(FATAL_ERROR
          "loom_corpus_test(${_RULE_NAME}) has malformed ${_QUALIFICATION_NAME} for ${_SOURCE_ID}")
      endif()
      set(_QUALIFICATION_KEY "${_SOURCE_ID}=${_RECORD}")
      if(_QUALIFICATION_KEY IN_LIST _FAILURE_QUALIFICATION_KEYS)
        message(FATAL_ERROR
          "loom_corpus_test(${_RULE_NAME}) repeats failure qualification ${_RECORD} for ${_SOURCE_ID}")
      endif()
      list(APPEND _FAILURE_QUALIFICATION_KEYS "${_QUALIFICATION_KEY}")
      set_property(GLOBAL APPEND PROPERTY
        "${ID}_FAILURE_QUALIFICATION_${_SOURCE_ID}"
        "${_QUALIFICATION_FLAG}=${_RECORD}=${_DIAGNOSTIC}")
    endwhile()
  endforeach()

  set(_PROFILE_STEM "${_RULE_PROFILE}")
  if(_PROFILE_STEM MATCHES ":")
    string(REGEX REPLACE "^.*:" "" _PROFILE_STEM "${_PROFILE_STEM}")
  elseif(_PROFILE_STEM MATCHES "/")
    string(REGEX REPLACE "^.*/" "" _PROFILE_STEM "${_PROFILE_STEM}")
  endif()
  string(REGEX REPLACE "[-.+]" "_" _PROFILE_STEM "${_PROFILE_STEM}")

  math(EXPR _LAST_SOURCE_INDEX "${_SOURCE_COUNT} - 1")
  foreach(_SOURCE_INDEX RANGE ${_LAST_SOURCE_INDEX})
    list(GET _SOURCE_IDS ${_SOURCE_INDEX} _SOURCE_ID)
    if(_SELECTED_SOURCES AND NOT _SOURCE_ID IN_LIST _SELECTED_SOURCES)
      continue()
    endif()
    if(_SOURCE_ID IN_LIST _EXCLUDED_SOURCES)
      continue()
    endif()
    get_property(_SOURCE_FAILURE_QUALIFICATION_ARGS GLOBAL PROPERTY
      "${ID}_FAILURE_QUALIFICATION_${_SOURCE_ID}")
    list(GET _SOURCE_MODULE_TARGETS ${_SOURCE_INDEX} _MODULE_TARGET)
    _loom_corpus_program_stem(_PROGRAM_STEM "${_SOURCE_ID}")
    loom_execution_test(
      NAME "${_PROGRAM_STEM}_test_execute_${_PROFILE_STEM}_test"
      CORRECTNESS_ONLY
      MODULE "${_MODULE_TARGET}"
      ARGS ${_RULE_ARGS} ${_SOURCE_FAILURE_QUALIFICATION_ARGS}
      RUNNER_ARGS ${_RULE_RUNNER_ARGS}
      LABELS ${_RULE_LABELS}
      RESOURCE_GROUP "${_RULE_RESOURCE_GROUP}"
    )
  endforeach()
endfunction()

# Expands target-owned corpus declarations after the complete tree is known.
function(loom_finalize_corpus)
  get_property(_BUILDS GLOBAL PROPERTY LOOM_CORPUS_BUILDS)
  foreach(_BUILD IN LISTS _BUILDS)
    _loom_declare_corpus_build("${_BUILD}")
  endforeach()
  get_property(_TESTS GLOBAL PROPERTY LOOM_CORPUS_TESTS)
  foreach(_TEST IN LISTS _TESTS)
    _loom_declare_corpus_test("${_TEST}")
  endforeach()
endfunction()
