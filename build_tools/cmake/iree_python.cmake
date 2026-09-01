# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

###############################################################################
# Main user rules
###############################################################################

function(_iree_finalize_windows_python_sources)
  get_property(
    _SOURCE_FILES GLOBAL PROPERTY IREE_WINDOWS_PYTHON_SOURCE_FILES
  )
  get_property(
    _DESTINATION_FILES GLOBAL PROPERTY IREE_WINDOWS_PYTHON_DESTINATION_FILES
  )
  list(LENGTH _SOURCE_FILES _SOURCE_FILE_COUNT)
  list(LENGTH _DESTINATION_FILES _DESTINATION_FILE_COUNT)
  if(_SOURCE_FILE_COUNT EQUAL 0 OR
     NOT _SOURCE_FILE_COUNT EQUAL _DESTINATION_FILE_COUNT)
    message(FATAL_ERROR
      "invalid Windows Python source materialization registration"
    )
  endif()

  set(_MANIFEST_CONTENT "")
  math(EXPR _LAST_SOURCE_FILE_INDEX "${_SOURCE_FILE_COUNT} - 1")
  foreach(_FILE_INDEX RANGE ${_LAST_SOURCE_FILE_INDEX})
    list(GET _SOURCE_FILES ${_FILE_INDEX} _SOURCE_FILE)
    list(GET _DESTINATION_FILES ${_FILE_INDEX} _DESTINATION_FILE)
    string(APPEND _MANIFEST_CONTENT
      "${_SOURCE_FILE}|${_DESTINATION_FILE}\n"
    )
  endforeach()

  set(_MATERIALIZATION_MANIFEST
    "${CMAKE_BINARY_DIR}/CMakeFiles/iree_windows_python_sources.txt"
  )
  set(_MATERIALIZATION_STAMP
    "${CMAKE_BINARY_DIR}/CMakeFiles/iree_windows_python_sources.stamp"
  )
  set(_MATERIALIZE_FILES_SCRIPT
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/iree_materialize_files.cmake"
  )
  file(CONFIGURE
    OUTPUT "${_MATERIALIZATION_MANIFEST}"
    CONTENT "${_MANIFEST_CONTENT}"
    @ONLY
    NEWLINE_STYLE UNIX
  )

  add_custom_command(
    OUTPUT
      "${_MATERIALIZATION_STAMP}"
    BYPRODUCTS
      ${_DESTINATION_FILES}
    COMMAND
      ${CMAKE_COMMAND}
        "-DIREE_MATERIALIZATION_MANIFEST=${_MATERIALIZATION_MANIFEST}"
        "-DIREE_MATERIALIZATION_STAMP=${_MATERIALIZATION_STAMP}"
        -P "${_MATERIALIZE_FILES_SCRIPT}"
    DEPENDS
      ${_SOURCE_FILES}
      "${_MATERIALIZATION_MANIFEST}"
      "${_MATERIALIZE_FILES_SCRIPT}"
    COMMENT
      "Materializing Windows Python sources"
    VERBATIM
  )
  add_custom_target(iree_windows_python_sources
    DEPENDS
      "${_MATERIALIZATION_STAMP}"
  )
endfunction()

function(_iree_register_windows_python_source SOURCE_PATH DESTINATION_PATH)
  set_property(GLOBAL APPEND PROPERTY
    IREE_WINDOWS_PYTHON_SOURCE_FILES "${SOURCE_PATH}"
  )
  set_property(GLOBAL APPEND PROPERTY
    IREE_WINDOWS_PYTHON_DESTINATION_FILES "${DESTINATION_PATH}"
  )
  get_property(_FINALIZATION_SCHEDULED GLOBAL PROPERTY
    IREE_WINDOWS_PYTHON_FINALIZATION_SCHEDULED
  )
  if(NOT _FINALIZATION_SCHEDULED)
    set_property(GLOBAL PROPERTY
      IREE_WINDOWS_PYTHON_FINALIZATION_SCHEDULED TRUE
    )
    cmake_language(DEFER DIRECTORY "${CMAKE_SOURCE_DIR}"
      CALL _iree_finalize_windows_python_sources
    )
  endif()
endfunction()

function(_iree_py_library_source_target OUTPUT_TARGET SOURCE_FILE)
  iree_package_name(_PACKAGE_NAME)
  string(REGEX REPLACE "[^A-Za-z0-9_]" "_" _SOURCE_TARGET_SUFFIX "${SOURCE_FILE}")
  set(_SOURCE_TARGET "${_PACKAGE_NAME}_${_SOURCE_TARGET_SUFFIX}_py_source")
  if(NOT TARGET "${_SOURCE_TARGET}")
    set(_SOURCE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/${SOURCE_FILE}")
    set(_SOURCE_BIN_PATH "${CMAKE_CURRENT_BINARY_DIR}/${SOURCE_FILE}")
    get_filename_component(_SOURCE_BIN_DIR "${_SOURCE_BIN_PATH}" DIRECTORY)
    if(WIN32)
      # Windows file symlinks require Developer Mode or elevated privileges.
      # Materialize all Python sources through one batch edge so a clean build
      # launches one process while retaining tracked outputs and repair.
      _iree_register_windows_python_source(
        "${_SOURCE_PATH}" "${_SOURCE_BIN_PATH}"
      )
      add_custom_target("${_SOURCE_TARGET}")
      add_dependencies("${_SOURCE_TARGET}" iree_windows_python_sources)
    else()
      add_custom_command(
        OUTPUT
          "${_SOURCE_BIN_PATH}"
        COMMAND
          ${CMAKE_COMMAND} -E make_directory "${_SOURCE_BIN_DIR}"
        COMMAND
          ${CMAKE_COMMAND} -E create_symlink
            "${_SOURCE_PATH}"
            "${_SOURCE_BIN_PATH}"
        DEPENDS
          "${_SOURCE_PATH}"
        VERBATIM
      )
      add_custom_target("${_SOURCE_TARGET}"
        DEPENDS
          "${_SOURCE_BIN_PATH}"
      )
    endif()
  endif()
  set(${OUTPUT_TARGET} "${_SOURCE_TARGET}" PARENT_SCOPE)
endfunction()

# iree_py_library()
#
# CMake function to imitate Bazel's iree_py_library rule.
#
# Parameters:
# NAME: name of target
# MAIN: optional executable Python source entry point for py_binary-style targets
# MAIN_MODULE: optional executable Python module entry point for py_binary-style targets
# SRCS: List of source files for the library
# IMPORTS: List of package import directories relative to the current package
# DEPS: List of other targets the test python libraries require
# PYEXT_DEPS: List of deps of extensions built with iree_pyext_module
function(iree_py_library)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;MAIN;MAIN_MODULE"
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
    if(IS_ABSOLUTE "${_SRC_FILE}")
      list(APPEND _SOURCE_FILES "${_SRC_FILE}")
      list(APPEND _SOURCE_TARGETS "${_SRC_FILE}")
    else()
      list(APPEND _SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/${_SRC_FILE}")
      _iree_py_library_source_target(_SOURCE_TARGET "${_SRC_FILE}")
      list(APPEND _SOURCE_TARGETS "${_SOURCE_TARGET}")
    endif()
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
  if(_RULE_MAIN AND _RULE_MAIN_MODULE)
    message(FATAL_ERROR "iree_py_library accepts either MAIN or MAIN_MODULE, not both")
  elseif(_RULE_MAIN)
    if(IS_ABSOLUTE "${_RULE_MAIN}")
      set(_MAIN "${_RULE_MAIN}")
    else()
      set(_MAIN "${CMAKE_CURRENT_SOURCE_DIR}/${_RULE_MAIN}")
    endif()
    set_target_properties(${_NAME} PROPERTIES
      IREE_PY_ENTRYPOINT_ARGUMENTS "${_MAIN}"
    )
  elseif(_RULE_MAIN_MODULE)
    set_target_properties(${_NAME} PROPERTIES
      IREE_PY_ENTRYPOINT_ARGUMENTS "-m;${_RULE_MAIN_MODULE}"
    )
  endif()

  # Add PYEXT_DEPS if any.
  if(_RULE_PYEXT_DEPS)
    list(TRANSFORM _RULE_PYEXT_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
    iree_package_target_names(_RULE_PYEXT_DEP_TARGETS ${_RULE_PYEXT_DEPS})
    add_dependencies(${_NAME} ${_RULE_PYEXT_DEP_TARGETS})
  endif()
endfunction()

function(iree_py_library_entrypoint OUTPUT_ARGUMENTS TARGET_NAME)
  iree_package_target_name(_TARGET_NAME "${TARGET_NAME}")
  if(NOT TARGET "${_TARGET_NAME}")
    message(FATAL_ERROR "iree_py_library target ${TARGET_NAME} was not found")
  endif()
  get_target_property(
    _ENTRYPOINT_ARGUMENTS
    "${_TARGET_NAME}"
    IREE_PY_ENTRYPOINT_ARGUMENTS
  )
  if(NOT _ENTRYPOINT_ARGUMENTS)
    message(FATAL_ERROR
      "iree_py_library target ${TARGET_NAME} does not declare MAIN or MAIN_MODULE")
  endif()
  set(${OUTPUT_ARGUMENTS} "${_ENTRYPOINT_ARGUMENTS}" PARENT_SCOPE)
endfunction()

function(iree_py_library_collect_sources OUTPUT_SOURCE_FILES TARGET_NAME)
  if(TARGET "${TARGET_NAME}")
    set(_TARGET_NAME "${TARGET_NAME}")
    get_target_property(_ALIASED_TARGET "${_TARGET_NAME}" ALIASED_TARGET)
    if(_ALIASED_TARGET)
      set(_TARGET_NAME "${_ALIASED_TARGET}")
    endif()
  elseif("${TARGET_NAME}" MATCHES "^[^:].*::")
    string(REPLACE "::" "_" _TARGET_NAME "${TARGET_NAME}")
  else()
    iree_package_target_name(_TARGET_NAME "${TARGET_NAME}")
  endif()
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

# Prepends Python package directories to the PYTHONPATH of a CTest test.
#
# CTest applies ENVIRONMENT_MODIFICATION entries sequentially. Reverse the
# directories before prepending them so their declared precedence is preserved.
# The path_list_prepend operation selects the native path separator at test time
# and retains any PYTHONPATH inherited by CTest.
function(iree_python_test_add_package_dirs TEST_NAME)
  set(_PACKAGE_DIRS ${ARGN})
  if(NOT _PACKAGE_DIRS)
    return()
  endif()

  list(REMOVE_DUPLICATES _PACKAGE_DIRS)
  list(REVERSE _PACKAGE_DIRS)
  set(_PYTHONPATH_MODIFICATIONS)
  foreach(_PACKAGE_DIR IN LISTS _PACKAGE_DIRS)
    list(APPEND _PYTHONPATH_MODIFICATIONS
      "PYTHONPATH=path_list_prepend:${_PACKAGE_DIR}"
    )
  endforeach()
  set_property(TEST "${TEST_NAME}" APPEND PROPERTY ENVIRONMENT_MODIFICATION
    ${_PYTHONPATH_MODIFICATIONS}
  )
endfunction()

# iree_py_generated_files()
#
# Runs one Python generator action to produce one or more declared build-tree
# files. The generator must preserve output timestamps when contents are
# unchanged. A private freshness stamp tracks generator inputs independently of
# output contents so an authority-only edit can rerun generation without
# forcing every C/C++ consumer to rebuild.
#
# Parameters:
# NAME: name of the generated-file target.
# GENERATOR: iree_py_library target with an executable entry point.
# OUTPUTS: build-tree-relative output paths.
# OUTPUT_FLAGS: generator flags paired one-to-one with OUTPUTS.
# ARGS: arguments passed before the paired output arguments.
# INPUTS: additional non-Python generator inputs.
# COMMENT: optional build progress message.
function(iree_py_generated_files)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;GENERATOR;COMMENT"
    "OUTPUTS;OUTPUT_FLAGS;ARGS;INPUTS"
    ${ARGN}
  )

  if(NOT _RULE_NAME)
    message(FATAL_ERROR "generated file actions require NAME")
  endif()
  if(NOT _RULE_GENERATOR)
    message(FATAL_ERROR "generated file actions require GENERATOR")
  endif()
  list(LENGTH _RULE_OUTPUTS _OUTPUT_COUNT)
  if(_OUTPUT_COUNT EQUAL 0)
    message(FATAL_ERROR "generated file actions require at least one output")
  endif()
  list(LENGTH _RULE_OUTPUT_FLAGS _OUTPUT_FLAG_COUNT)
  if(NOT _OUTPUT_FLAG_COUNT EQUAL _OUTPUT_COUNT)
    message(FATAL_ERROR
      "generated file output flags and outputs must be paired")
  endif()

  set(_OUTPUTS)
  set(_OUTPUT_ARGS)
  math(EXPR _OUTPUT_LAST "${_OUTPUT_COUNT} - 1")
  foreach(_INDEX RANGE 0 ${_OUTPUT_LAST})
    list(GET _RULE_OUTPUTS ${_INDEX} _OUTPUT)
    list(GET _RULE_OUTPUT_FLAGS ${_INDEX} _OUTPUT_FLAG)
    set(_OUTPUT_PATH "${CMAKE_CURRENT_BINARY_DIR}/${_OUTPUT}")
    list(APPEND _OUTPUTS "${_OUTPUT_PATH}")
    list(APPEND _OUTPUT_ARGS "${_OUTPUT_FLAG}=${_OUTPUT_PATH}")
  endforeach()
  if(NOT _RULE_COMMENT)
    set(_RULE_COMMENT "Generating ${_RULE_NAME}")
  endif()

  iree_py_library_entrypoint(_GENERATOR_ENTRYPOINT "${_RULE_GENERATOR}")
  iree_py_library_collect_sources(_GENERATOR_INPUTS "${_RULE_GENERATOR}")
  iree_py_library_collect_package_dirs(
    _GENERATOR_PACKAGE_DIRS "${_RULE_GENERATOR}"
  )
  list(APPEND _GENERATOR_PACKAGE_DIRS "$ENV{PYTHONPATH}")
  if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
    list(JOIN _GENERATOR_PACKAGE_DIRS "\\;" _GENERATOR_PYTHONPATH)
  else()
    list(JOIN _GENERATOR_PACKAGE_DIRS ":" _GENERATOR_PYTHONPATH)
  endif()

  iree_package_name(_PACKAGE_NAME)
  set(_GEN_TARGET "${_PACKAGE_NAME}_${_RULE_NAME}")
  set(_GEN_STAMP
    "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${_GEN_TARGET}.stamp")
  add_custom_command(
    OUTPUT
      "${_GEN_STAMP}"
    BYPRODUCTS
      ${_OUTPUTS}
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${_GENERATOR_PYTHONPATH}"
      "PYTHONDONTWRITEBYTECODE=1"
      "${Python3_EXECUTABLE}"
      ${_GENERATOR_ENTRYPOINT}
      ${_RULE_ARGS}
      ${_OUTPUT_ARGS}
    COMMAND
      "${CMAKE_COMMAND}" -E touch "${_GEN_STAMP}"
    DEPENDS
      ${_GENERATOR_INPUTS}
      ${_RULE_INPUTS}
    COMMENT
      "${_RULE_COMMENT}"
    VERBATIM
  )
  set_source_files_properties(
    ${_RULE_OUTPUTS}
    ${_OUTPUTS}
    PROPERTIES GENERATED TRUE
  )

  add_custom_target("${_GEN_TARGET}"
    DEPENDS
      "${_GEN_STAMP}"
  )
  iree_register_generated_compile_input("${_GEN_TARGET}"
    OUTPUTS ${_OUTPUTS}
  )
endfunction()

# iree_local_py_test()
#
# CMake function to run python test with provided python package paths.
#
# Parameters:
# NAME: name of test
# SRC: Test source file
# SOURCES: All Python sources required by the test.
# DEPS: Python library targets required by the test.
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
    "ARGS;DEPS;LABELS;PACKAGE_DIRS;SOURCES;TIMEOUT"
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

  iree_python_test_add_package_dirs("${_NAME_PATH}" ${_RULE_PACKAGE_DIRS})
  set_property(TEST ${_NAME_PATH} PROPERTY ENVIRONMENT
    "PYTHONDONTWRITEBYTECODE=1"
  )

  set(_TEST_BUILD_TARGETS)
  if(_RULE_DEPS)
    set(_TEST_BUILD_TARGET "${_NAME}_test_deps")
    add_custom_target("${_TEST_BUILD_TARGET}" ALL)
    set_property(
      TARGET "${_TEST_BUILD_TARGET}"
      PROPERTY FOLDER ${IREE_IDE_FOLDER}/test
    )
    foreach(_TEST_DEPENDENCY IN LISTS _RULE_DEPS)
      iree_register_target_dependency(
        TARGET "${_TEST_BUILD_TARGET}"
        DEPENDENCY "${_TEST_DEPENDENCY}"
      )
    endforeach()
    list(APPEND _TEST_BUILD_TARGETS "${_TEST_BUILD_TARGET}")
  endif()

  iree_configure_test(${_NAME_PATH})
  iree_register_test_build_targets(
    "${_NAME_PATH}"
    TARGETS ${_TEST_BUILD_TARGETS}
  )

  if(IREE_PYTHON_TEST_REGISTRATION_FUNCTION AND
     NOT IREE_SKIP_TEST_REGISTRATION)
    if(COMMAND ${IREE_PYTHON_TEST_REGISTRATION_FUNCTION})
      cmake_language(CALL ${IREE_PYTHON_TEST_REGISTRATION_FUNCTION}
        NAME
          "${_NAME_PATH}"
        SRC
          "${_RULE_SRC}"
        SOURCES
          ${_RULE_SOURCES}
        DEPS
          ${_RULE_DEPS}
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
# MAIN: Python source file to execute.
# SRCS: All Python sources required by the test.
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
    "MAIN;NAME"
    "ARGS;LABELS;PACKAGE_DIRS;IMPORTS;DEPS;SRCS;TIMEOUT"
    ${ARGN}
  )
  if(_RULE_MAIN)
    set(_RULE_MAIN_SOURCE "${_RULE_MAIN}")
  elseif("${_RULE_SRCS}" MATCHES "^[^;]+$")
    set(_RULE_MAIN_SOURCE "${_RULE_SRCS}")
  else()
    message(FATAL_ERROR
      "iree_py_test ${_RULE_NAME} requires MAIN when declaring multiple SRCS")
  endif()

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
  set(_RULE_SOURCE_FILES ${_RULE_SRCS})
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
      "${_RULE_MAIN_SOURCE}"
    SOURCES
      ${_RULE_SOURCE_FILES}
    DEPS
      ${_RULE_DEPS}
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
