# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Final binary composition mirrors Bazel dynamic_deps. Component libraries keep
# their ordinary static link and compilation contracts in both configurations.
include_guard(GLOBAL)
cmake_policy(SET CMP0099 NEW)

function(loomc_cc_binary)
  cmake_parse_arguments(_LINK "RUNTIME_ONLY" "" "" ${ARGN})
  if(LOOM_LINK_SHARED_TOOLS)
    loom_cc_binary(${_LINK_UNPARSED_ARGUMENTS} SETUP_INSTALL_RPATH)
    cmake_parse_arguments(_RULE "" "NAME;INSTALL_COMPONENT" "" ${ARGN})
    iree_package_name(_PACKAGE_NAME)
    set(_target "${_PACKAGE_NAME}_${_RULE_NAME}")
    if(TARGET "${_target}")
      set_property(GLOBAL APPEND PROPERTY LOOMC_SHARED_TOOLS "${_target}")
      set_property(TARGET "${_target}" PROPERTY LOOMC_RUNTIME_ONLY
        "${_LINK_RUNTIME_ONLY}")
      if(NOT _RULE_INSTALL_COMPONENT)
        set(_RULE_INSTALL_COMPONENT "IREETool-${_RULE_NAME}")
      endif()
      set_property(TARGET "${_target}" PROPERTY LOOMC_INSTALL_COMPONENT
        "${_RULE_INSTALL_COMPONENT}")
    endif()
  else()
    loom_cc_binary(${_LINK_UNPARSED_ARGUMENTS})
  endif()
endfunction()

function(loomc_interop_headers)
  cmake_parse_arguments(_RULE "" "NAME" "DEPS" ${ARGN})
  iree_cc_library(NAME "${_RULE_NAME}" DEPS ${_RULE_DEPS})
  if(LOOM_LINK_SHARED_TOOLS)
    iree_package_name(_PACKAGE_NAME)
    set(_target "${_PACKAGE_NAME}_${_RULE_NAME}")
    set_property(TARGET "${_target}" PROPERTY IREE_CC_DEPS "")
    set_property(TARGET "${_target}" PROPERTY INTERFACE_LINK_LIBRARIES
      "$<COMPILE_ONLY:${_RULE_DEPS}>")
  endif()
endfunction()

# The generated BUILD inventory is also the installed SDK's binding inventory.
macro(loomc_shared_libraries)
  cmake_parse_arguments(_LOOMC "" "NAME" "COMPONENTS;RUNTIME_DEPS;TOOL_COMPONENTS" ${ARGN})
  if(LOOM_LINK_SHARED_TOOLS)
    list(APPEND _LOOMC_COMPONENTS ${_LOOMC_TOOL_COMPONENTS})
  endif()
  list(TRANSFORM _LOOMC_COMPONENTS REPLACE "^::" "loom::binding::c::")
  if(NOT IREE_TARGET_HAS_THREADS)
    list(REMOVE_ITEM _LOOMC_COMPONENTS
      loom::binding::c::task_pool loom::binding::c::task_queue)
  endif()
endmacro()

function(_loomc_resolve out_target target)
  if(TARGET "${target}")
    get_target_property(_alias "${target}" ALIASED_TARGET)
    if(_alias)
      set(target "${_alias}")
    endif()
  endif()
  set(${out_target} "${target}" PARENT_SCOPE)
endfunction()

# Walk producer-retained dependencies, never generated linker expressions.
# External targets and opaque expressions retain their own CMake link contract.
function(_loomc_collect roots mode out_objects out_libraries out_owned out_options)
  set(_pending ${roots})
  set(_visited)
  set(_owned)
  set(_objects)
  set(_libraries)
  set(_options)
  while(_pending)
    list(POP_FRONT _pending _dependency)
    if(NOT _dependency)
      continue()
    endif()
    _loomc_resolve(_target "${_dependency}")
    if(_target IN_LIST _visited)
      continue()
    endif()
    list(APPEND _visited "${_target}")
    if(NOT mode STREQUAL "runtime" AND _target IN_LIST _runtime_owned)
      list(APPEND _libraries loomc_runtime)
      continue()
    endif()
    if(mode STREQUAL "tool" AND _target IN_LIST _compiler_exports)
      list(APPEND _libraries loomc_shared)
      continue()
    endif()
    if(NOT TARGET "${_target}")
      if(_target MATCHES "::" AND NOT _target MATCHES "^\\$<")
        message(FATAL_ERROR "Unresolved shared-tool dependency: ${_target}")
      endif()
      list(APPEND _libraries "${_target}")
      continue()
    endif()
    get_property(_object TARGET "${_target}" PROPERTY IREE_CC_OBJECT_TARGET)
    get_property(_duplicate_safe TARGET "${_target}"
      PROPERTY IREE_CC_LINKABLE_MORE_THAN_ONCE)
    if(mode STREQUAL "tool" AND _target IN_LIST _compiler_owned AND
        _object AND NOT _duplicate_safe)
      message(FATAL_ERROR
        "Shared tool reaches unexported compiler implementation: ${_target}")
    endif()
    get_property(_recorded TARGET "${_target}" PROPERTY IREE_CC_DEPS SET)
    if(NOT _recorded)
      get_property(_type TARGET "${_target}" PROPERTY TYPE)
      get_property(_imported TARGET "${_target}" PROPERTY IMPORTED)
      if(NOT _imported AND _type STREQUAL "STATIC_LIBRARY")
        # Opaque support libraries, including the allocator, share this owner.
        # The final export list still limits the companion's symbol surface.
        list(APPEND _owned "${_target}")
        if(mode STREQUAL "runtime" AND NOT WIN32)
          target_compile_options("${_target}" PRIVATE -fvisibility=default)
        endif()
      endif()
      list(APPEND _libraries "${_target}")
      continue()
    endif()
    # Only composed implementations and their recorded dependency groups have
    # this owner. Visiting an external link requirement does not absorb it.
    list(APPEND _owned "${_target}")
    if(_object)
      get_property(_target_objects TARGET "${_target}" PROPERTY IREE_CC_OBJECTS)
      list(APPEND _objects ${_target_objects})
      if(mode STREQUAL "runtime" AND NOT WIN32)
        # IREE's default compile options explicitly hide symbols. The companion
        # exports its C interfaces through the final platform export list.
        target_compile_options("${_object}" PRIVATE -fvisibility=default)
      endif()
    endif()
    get_property(_deps TARGET "${_target}" PROPERTY IREE_CC_DEPS)
    get_property(_linkopts TARGET "${_target}" PROPERTY IREE_CC_LINKOPTS)
    list(APPEND _pending ${_deps})
    list(APPEND _options ${_linkopts})
  endwhile()
  list(REMOVE_DUPLICATES _objects)
  list(REMOVE_DUPLICATES _libraries)
  list(REMOVE_DUPLICATES _options)
  set(${out_objects} "${_objects}" PARENT_SCOPE)
  set(${out_libraries} "${_libraries}" PARENT_SCOPE)
  set(${out_owned} "${_owned}" PARENT_SCOPE)
  set(${out_options} "${_options}" PARENT_SCOPE)
endfunction()

function(_loomc_windows_runtime_exports objects libraries)
  set(_inputs ${objects})
  foreach(_library IN LISTS libraries)
    if(TARGET "${_library}")
      get_property(_type TARGET "${_library}" PROPERTY TYPE)
      if(_type STREQUAL "STATIC_LIBRARY")
        list(APPEND _inputs "$<TARGET_LINKER_FILE:${_library}>")
      endif()
    endif()
  endforeach()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    cmake_path(GET CMAKE_CXX_COMPILER PARENT_PATH _compiler_dir)
    find_program(_loomc_dumpbin NAMES dumpbin HINTS "${_compiler_dir}" REQUIRED)
    set(_symbol_tool "--dumpbin\n${_loomc_dumpbin}")
  else()
    if(NOT CMAKE_NM)
      message(FATAL_ERROR "Shared tool exports require the compiler's nm tool.")
    endif()
    set(_symbol_tool "--nm\n${CMAKE_NM}")
  endif()
  set(_exports "${CMAKE_CURRENT_BINARY_DIR}/iree-exports-$<CONFIG>.def")
  set(_arguments "${CMAKE_CURRENT_BINARY_DIR}/iree-exports-$<CONFIG>.args")
  set(_generator "${IREE_ROOT_DIR}/loom/build_tools/shared_exports.py")
  file(GENERATE OUTPUT "${_arguments}" CONTENT
    "${_symbol_tool}\n--output\n${_exports}\n--prefix\niree_\n--prefix\n___tracy_\n--prefix\nmi_\n$<JOIN:${_inputs},\n>\n")
  add_custom_command(OUTPUT "${_exports}"
    COMMAND "${Python3_EXECUTABLE}" "${_generator}" "@${_arguments}"
    DEPENDS ${_inputs} "${_arguments}" "${_generator}"
    COMMENT "Collecting shared IREE runtime exports"
    VERBATIM)
  target_sources(loomc_runtime PRIVATE "${_exports}")
endfunction()

function(loomc_configure_shared_tools)
  _loomc_collect("${_LOOMC_RUNTIME_DEPS}" runtime
    _objects _libraries _runtime_owned _options)
  add_library(loomc_runtime SHARED ${_objects})
  target_link_libraries(loomc_runtime PRIVATE ${_libraries})
  target_link_options(loomc_runtime PRIVATE ${IREE_DEFAULT_LINKOPTS} ${_options})
  set_target_properties(loomc_runtime PROPERTIES
    OUTPUT_NAME iree EXPORT_NAME runtime LINKER_LANGUAGE CXX)
  iree_target_sanitizer_suppressions(loomc_runtime DEPS ${_LOOMC_RUNTIME_DEPS})
  if(WIN32)
    _loomc_windows_runtime_exports("${_objects}" "${_libraries}")
  elseif(APPLE)
    set_target_properties(loomc_runtime PROPERTIES
      BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
    set(_exports "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/iree_exports.macos")
    target_link_options(loomc_runtime PRIVATE "-Wl,-exported_symbols_list,${_exports}")
  elseif(UNIX)
    set_target_properties(loomc_runtime PROPERTIES
      BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
    set(_exports "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/iree_exports.lds")
    target_link_options(loomc_runtime PRIVATE "-Wl,--version-script=${_exports}")
    # Exact-build companions own their function implementations. Data symbols
    # and undefined external hooks retain ordinary ELF resolution.
    foreach(_companion loomc_runtime loomc_shared)
      target_link_options("${_companion}" PRIVATE "-Wl,-Bsymbolic-functions")
    endforeach()
  endif()
  if(_exports)
    set_property(TARGET loomc_runtime APPEND PROPERTY LINK_DEPENDS "${_exports}")
  endif()

  set(_compiler_exports)
  foreach(_root IN LISTS _LOOMC_COMPONENTS)
    _loomc_resolve(_target "${_root}")
    list(APPEND _compiler_exports "${_target}")
  endforeach()
  _loomc_collect("${_LOOMC_COMPONENTS}" compiler
    _objects _libraries _compiler_owned _options)
  set_property(TARGET loomc_shared PROPERTY SOURCES "${_objects}")
  set_property(TARGET loomc_shared PROPERTY LINK_LIBRARIES "${_libraries}")
  target_link_options(loomc_shared PRIVATE ${IREE_DEFAULT_LINKOPTS} ${_options})
  iree_target_sanitizer_suppressions(loomc_shared
    DEPS ${_LOOMC_COMPONENTS} loomc_runtime)

  get_property(_tools GLOBAL PROPERTY LOOMC_SHARED_TOOLS)
  if(WIN32)
    # Windows resolves dependent DLLs beside the executable. Keep the build
    # bundle together so tools run directly as well as through CTest.
    set_target_properties(loomc_shared loomc_runtime ${_tools} PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${IREE_BINARY_DIR}/bin")
  endif()
  set(_install_components "${HRX_PUBLIC_DIST_COMPONENT}")
  set(_compiler_install_components)
  foreach(_tool IN LISTS _tools)
    get_property(_deps TARGET "${_tool}" PROPERTY IREE_CC_DEPS)
    get_property(_runtime_only TARGET "${_tool}" PROPERTY LOOMC_RUNTIME_ONLY)
    set(_mode tool)
    if(_runtime_only)
      # The compiler and its native consumers have one executable-local owner.
      set(_mode runtime_tool)
    endif()
    _loomc_collect("${_deps}" "${_mode}" _objects _libraries _owned _options)
    set_property(TARGET "${_tool}" PROPERTY LINK_LIBRARIES
      "$<COMPILE_ONLY:${_deps}>;${_libraries};${IREE_DEFAULT_EXECUTABLE_LINK_LIBRARIES}")
    target_sources("${_tool}" PRIVATE ${_objects})
    target_link_options("${_tool}" PRIVATE ${_options})
    set_property(TARGET "${_tool}" APPEND PROPERTY IREE_SANITIZER_DEPENDENCIES
      loomc_runtime)
    get_property(_component TARGET "${_tool}" PROPERTY LOOMC_INSTALL_COMPONENT)
    list(APPEND _install_components "${_component}")
    if(NOT _runtime_only)
      set_property(TARGET "${_tool}" APPEND PROPERTY IREE_SANITIZER_DEPENDENCIES
        loomc_shared)
      list(APPEND _compiler_install_components "${_component}")
    endif()
  endforeach()

  install(TARGETS loomc_runtime EXPORT loomc-targets
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT "${HRX_PUBLIC_DIST_COMPONENT}")
  list(REMOVE_DUPLICATES _install_components)
  list(REMOVE_ITEM _install_components "${HRX_PUBLIC_DIST_COMPONENT}")
  foreach(_component IN LISTS _install_components)
    set(_companions loomc_runtime)
    if(_component IN_LIST _compiler_install_components)
      list(APPEND _companions loomc_shared)
    endif()
    install(TARGETS ${_companions}
      LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT "${_component}"
      RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT "${_component}")
  endforeach()
endfunction()
