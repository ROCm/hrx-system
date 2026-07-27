# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

###############################################################################
# Main user rules
###############################################################################

function(_iree_py_library_source_target OUTPUT_TARGET SOURCE_FILE)
  iree_package_name(_PACKAGE_NAME)
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _SOURCE_TARGET_SUFFIX "${SOURCE_FILE}")
  set(_SOURCE_TARGET "${_PACKAGE_NAME}_${_SOURCE_TARGET_SUFFIX}_py_source")
  if(NOT TARGET "${_SOURCE_TARGET}")
    set(_SOURCE_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}")
    get_filename_component(_SOURCE_BIN_DIR "${_SOURCE_BIN_PATH}" DIRECTORY)
    add_custom_command(
      OUTPUT
        "${_SOURCE_BIN_PATH}"
      COMMAND
        ${CMAKE_COMMAND} -E make_directory "${_SOURCE_BIN_DIR}"
      COMMAND
        ${CMAKE_COMMAND} -E create_symlink
          "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}"
          "${_SOURCE_BIN_PATH}"
      DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}"
      VERBATIM
    )
    add_custom_target("${_SOURCE_TARGET}"
      DEPENDS
        "${_SOURCE_BIN_PATH}"
    )
  endif()
  set(${OUTPUT_TARGET} "${_SOURCE_TARGET}" PARENT_SCOPE)
endfunction()

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# MAIN: optional executable Python entry point for py_binary-style targets
# SRCS: List of source files for the library
# IMPORTS: List of package import directories relative to the current package
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MAIN"
    "SRCS;IMPORTS;DEPS;PYEXT_DEPS"
    ${ARGN}
  )

  iree_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::iree::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  iree_package_target_names(_RULE_DEP_TARGETS ${_RULE_DEPS})

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_SOURCE_FILES)
  set(_SOURCE_TARGETS)
  foreach(_SRC_FILE ${_RULE_SRCS})
    list(APPEND _SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
    _iree_py_library_source_target(_SOURCE_TARGET "${_SRC_FILE}")
    list(APPEND _SOURCE_TARGETS "${_SOURCE_TARGET}")
  endforeach()

  set(_IMPORT_DIRS)
  foreach(_IMPORT ${_RULE_IMPORTS})
    get_filename_component(
      _SOURCE_IMPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${_IMPORT}" ABSOLUTE
    )
    get_filename_component(
      _BINARY_IMPORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_IMPORT}" ABSOLUTE
    )
    list(APPEND _IMPORT_DIRS "${_SOURCE_IMPORT_DIR}" "${_BINARY_IMPORT_DIR}")
  endforeach()

  add_custom_target(${_NAME} ALL
    DEPENDS
      ${_RULE_DEP_TARGETS}
      ${_SOURCE_TARGETS}
  )

  set_target_properties(${_NAME} PROPERTIES
    IREE_PY_SOURCE_FILES "${_SOURCE_FILES}"
    IREE_PY_IMPORT_DIRS "${_IMPORT_DIRS}"
    IREE_PY_DEPS "${_RULE_DEPS}"
  )
  if(_RULE_MAIN)
    set_target_properties(${_NAME} PROPERTIES
      IREE_PY_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_MAIN}"
    )
  endif()

  # Add PYEXT_DEPS if any.
  if(_RULE_PYEXT_DEPS)
    list(TRANSFORM _RULE_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
    iree_package_target_names(_RULE_PYEXT_DEP_TARGETS ${_RULE_PYEXT_DEPS})
    add_dependencies(${_NAME} ${_RULE_PYEXT_DEP_TARGETS})
  endif()
endfunction()

function(iree_py_library_main OUTPUT_MAIN TARGET_NAME)
  iree_package_target_name(_TARGET_NAME "${TARGET_NAME}")
  if(NOT TARGET "${_TARGET_NAME}")
    message(FATAL_ERROR "iree_py_library target ${TARGET_NAME} was not found")
  endif()
  get_target_property(_MAIN "${_TARGET_NAME}" IREE_PY_MAIN)
  if(NOT _MAIN)
    message(FATAL_ERROR "iree_py_library target ${TARGET_NAME} does not declare MAIN")
  endif()
  set(${OUTPUT_MAIN} "${_MAIN}" PARENT_SCOPE)
endfunction()

function(iree_py_library_collect_sources OUTPUT_SOURCE_FILES TARGET_NAME)
  iree_package_target_name(_TARGET_NAME "${TARGET_NAME}")
  if(NOT TARGET "${_TARGET_NAME}")
    message(FATAL_ERROR "iree_py_library target ${TARGET_NAME} was not found")
  endif()

  get_target_property(_SOURCE_FILES "${_TARGET_NAME}" IREE_PY_SOURCE_FILES)
  if(NOT _SOURCE_FILES)
    set(_SOURCE_FILES)
  endif()

  get_target_property(_DEPS "${_TARGET_NAME}" IREE_PY_DEPS)
  if(_DEPS)
    foreach(_DEP ${_DEPS})
      iree_py_library_collect_sources(_DEP_SOURCE_FILES "${_DEP}")
      list(APPEND _SOURCE_FILES ${_DEP_SOURCE_FILES})
    endforeach()
  endif()

  if(_SOURCE_FILES)
    list(REMOVE_DUPLICATES _SOURCE_FILES)
  endif()
  set(${OUTPUT_SOURCE_FILES} "${_SOURCE_FILES}" PARENT_SCOPE)
endfunction()

function(iree_py_library_collect_package_dirs OUTPUT_PACKAGE_DIRS TARGET_NAME)
  iree_package_target_name(_TARGET_NAME "${TARGET_NAME}")
  if(NOT TARGET "${_TARGET_NAME}")
    message(FATAL_ERROR "iree_py_library target ${TARGET_NAME} was not found")
  endif()

  get_target_property(_PACKAGE_DIRS "${_TARGET_NAME}" IREE_PY_IMPORT_DIRS)
  if(NOT _PACKAGE_DIRS)
    set(_PACKAGE_DIRS)
  endif()

  get_target_property(_DEPS "${_TARGET_NAME}" IREE_PY_DEPS)
  if(_DEPS)
    foreach(_DEP ${_DEPS})
      iree_py_library_collect_package_dirs(_DEP_PACKAGE_DIRS "${_DEP}")
      list(APPEND _PACKAGE_DIRS ${_DEP_PACKAGE_DIRS})
    endforeach()
  endif()

  if(_PACKAGE_DIRS)
    list(REMOVE_DUPLICATES _PACKAGE_DIRS)
  endif()
  set(${OUTPUT_PACKAGE_DIRS} "${_PACKAGE_DIRS}" PARENT_SCOPE)
endfunction()

# iree_local_py_test()
#
# CMake function to run python test with provided python package paths.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# GENERATED_IN_BINARY_DIR: If present, indicates that the srcs have been
#   in the CMAKE_CURRENT_BINARY_DIR.
# PACKAGE_DIRS: Python package paths to be added to PYTHONPATH.
function(iree_local_py_test)
  if(NOT IREE_BUILD_TESTS OR ANDROID OR EMSCRIPTEN)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRC"
    "ARGS;LABELS;PACKAGE_DIRS;TIMEOUT"
    ${ARGN}
  )

  # Switch between source and generated tests.
  set(_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  if(_RULE_GENERATED_IN_BINARY_DIR)
    set(_SRC_DIR "${CMAKE_CURRENT_BINARY_DIR}")
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  iree_package_ns(_PACKAGE_NS)
  string(REPLACE "::" "/" _PACKAGE_PATH ${_PACKAGE_NS})
  set(_NAME_PATH "${_PACKAGE_PATH}/${_RULE_NAME}")
  list(APPEND _RULE_LABELS "${_PACKAGE_PATH}")
  if(NOT DEFINED _RULE_TIMEOUT)
    set(_RULE_TIMEOUT 60)
  endif()

  add_test(
    NAME ${_NAME_PATH}
    COMMAND
      "${Python3_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_SRC}"
      ${_RULE_ARGS}
  )

  set_property(TEST ${_NAME_PATH} PROPERTY LABELS "${_RULE_LABELS}")
  set_property(TEST ${_NAME_PATH} PROPERTY TIMEOUT ${_RULE_TIMEOUT})

  set(_IREE_INSTALL_PACKAGE_DIRS ${_RULE_PACKAGE_DIRS})

  # Extend the PYTHONPATH environment variable with _RULE_PACKAGE_DIRS.
  list(APPEND _RULE_PACKAGE_DIRS "$ENV{PYTHONPATH}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    # Windows uses semi-colon delimiters, but so does CMake, so escape them.
    list(JOIN _RULE_PACKAGE_DIRS "\\;" _PYTHONPATH)
  else()
    list(JOIN _RULE_PACKAGE_DIRS ":" _PYTHONPATH)
  endif()
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
      "PYTHONPATH=${_PYTHONPATH}"
      "PYTHONDONTWRITEBYTECODE=1"
  )

  iree_configure_test(${_NAME_PATH})

  if(IREE_PYTHON_TEST_REGISTRATION_FUNCTION AND
     NOT IREE_SKIP_TEST_REGISTRATION)
    if(COMMAND ${IREE_PYTHON_TEST_REGISTRATION_FUNCTION})
      cmake_language(CALL ${IREE_PYTHON_TEST_REGISTRATION_FUNCTION}
        NAME
          "${_NAME_PATH}"
        SRC
          "${_RULE_SRC}"
        ARGS
          ${_RULE_ARGS}
        LABELS
          ${_RULE_LABELS}
        PACKAGE_DIRS
          ${_IREE_INSTALL_PACKAGE_DIRS}
        TIMEOUT
          ${_RULE_TIMEOUT}
      )
    endif()
  endif()

  # TODO(marbre): Find out how to add deps to tests.
endfunction()

# iree_py_test()
#
# CMake function to imitate Bazel's iree_py_test rule.
#
# Parameters:
# NAME: name of test
# SRCS: Test source file (single file only, despite name)
# ARGS: Command line arguments to the Python source file.
# LABELS: Additional labels to apply to the test. The package path is added
#     automatically.
# IMPORTS: List of package import directories relative to the current package.
# DEPS: List of iree_py_library targets needed by the test.
# GENERATED_IN_BINARY_DIR: If present, indicates that the srcs have been
#   in the CMAKE_CURRENT_BINARY_DIR.
function(iree_py_test)
  cmake_parse_arguments(
    _RULE
    "GENERATED_IN_BINARY_DIR"
    "NAME;SRCS"
    "ARGS;LABELS;PACKAGE_DIRS;IMPORTS;DEPS;TIMEOUT"
    ${ARGN}
  )
  set(_HAS_EXPLICIT_PACKAGE_DIRS FALSE)
  if(NOT _RULE_PACKAGE_DIRS)
    set(_RULE_PACKAGE_DIRS
      "${IREE_BINARY_DIR}/compiler/bindings/python"
      "${IREE_BINARY_DIR}/runtime/bindings/python"
    )
  else()
    set(_HAS_EXPLICIT_PACKAGE_DIRS TRUE)
  endif()

  foreach(_IMPORT ${_RULE_IMPORTS})
    get_filename_component(
      _SOURCE_IMPORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/${_IMPORT}" ABSOLUTE
    )
    get_filename_component(
      _BINARY_IMPORT_DIR "${CMAKE_CURRENT_BINARY_DIR}/${_IMPORT}" ABSOLUTE
    )
    list(APPEND _RULE_PACKAGE_DIRS "${_SOURCE_IMPORT_DIR}" "${_BINARY_IMPORT_DIR}")
  endforeach()

  iree_package_ns(_PACKAGE_NS)
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  if(NOT _HAS_EXPLICIT_PACKAGE_DIRS)
    foreach(_DEP ${_RULE_DEPS})
      iree_py_library_collect_package_dirs(_DEP_PACKAGE_DIRS "${_DEP}")
      list(APPEND _RULE_PACKAGE_DIRS ${_DEP_PACKAGE_DIRS})
    endforeach()
  endif()
  if(_RULE_PACKAGE_DIRS)
    list(REMOVE_DUPLICATES _RULE_PACKAGE_DIRS)
  endif()

  iree_local_py_test(
    NAME
      "${_RULE_NAME}"
    SRC
      "${_RULE_SRCS}"
    ARGS
      ${_RULE_ARGS}
    LABELS
      ${_RULE_LABELS}
    PACKAGE_DIRS
      ${_RULE_PACKAGE_DIRS}
    GENERATED_IN_BINARY_DIR
      "${_RULE_GENERATED_IN_BINARY_DIR}"
    TIMEOUT
      ${_RULE_TIMEOUT}
  )
endfunction()
