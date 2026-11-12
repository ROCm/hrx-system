# Copyright 2026 The IREE Authors
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Linux-hosted clang-cl toolchain for x86-64 Windows targets.
#
# The selected Windows SDK must use the xwin winsysroot layout produced by
# .github/scripts/setup_windows_cross.sh. LLVM_ROOT names a Linux LLVM
# installation with clang-cl, lld-link, llvm-lib, llvm-ml, llvm-mt, llvm-nm,
# and llvm-rc. WINSDK_ROOT names the versioned MSVC and Windows Kits sysroot.

if(NOT CMAKE_HOST_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "The clang-cl Windows cross toolchain requires a Linux build host; "
    "the current host is ${CMAKE_HOST_SYSTEM_NAME}.")
endif()

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_C_COMPILER_TARGET "x86_64-pc-windows-msvc")
set(CMAKE_CXX_COMPILER_TARGET "x86_64-pc-windows-msvc")

function(_hrx_windows_cross_input NAME DESCRIPTION)
  if(NOT DEFINED ${NAME} AND DEFINED ENV{${NAME}})
    file(TO_CMAKE_PATH "$ENV{${NAME}}" _INPUT_PATH)
    set(${NAME} "${_INPUT_PATH}" CACHE PATH "${DESCRIPTION}")
  else()
    set(${NAME} "${${NAME}}" CACHE PATH "${DESCRIPTION}")
  endif()
  if(NOT IS_DIRECTORY "${${NAME}}")
    message(FATAL_ERROR
      "Windows cross-compilation requires ${NAME} to name ${DESCRIPTION}; "
      "got '${${NAME}}'.")
  endif()
endfunction()

function(_hrx_windows_cross_version_directory PARENT OUTPUT_VARIABLE)
  file(GLOB _CANDIDATES LIST_DIRECTORIES TRUE "${PARENT}/*")
  set(_DIRECTORIES)
  foreach(_CANDIDATE IN LISTS _CANDIDATES)
    if(IS_DIRECTORY "${_CANDIDATE}")
      list(APPEND _DIRECTORIES "${_CANDIDATE}")
    endif()
  endforeach()
  list(LENGTH _DIRECTORIES _DIRECTORY_COUNT)
  if(NOT _DIRECTORY_COUNT EQUAL 1)
    message(FATAL_ERROR
      "The Windows cross sysroot requires exactly one version under "
      "${PARENT}; found ${_DIRECTORY_COUNT}: ${_DIRECTORIES}")
  endif()
  list(GET _DIRECTORIES 0 _DIRECTORY)
  set(${OUTPUT_VARIABLE} "${_DIRECTORY}" PARENT_SCOPE)
endfunction()

function(_hrx_windows_cross_tool NAME OUTPUT_VARIABLE)
  set(_PATH "${LLVM_ROOT}/bin/${NAME}")
  if(NOT EXISTS "${_PATH}")
    message(FATAL_ERROR
      "Windows cross-compilation requires LLVM tool ${NAME}: ${_PATH}")
  endif()
  set(${OUTPUT_VARIABLE} "${_PATH}" PARENT_SCOPE)
endfunction()

_hrx_windows_cross_input(LLVM_ROOT "a Linux LLVM installation")
_hrx_windows_cross_input(WINSDK_ROOT "an xwin Windows SDK sysroot")

_hrx_windows_cross_version_directory(
  "${WINSDK_ROOT}/VC/Tools/MSVC" _HRX_MSVC_ROOT)
_hrx_windows_cross_version_directory(
  "${WINSDK_ROOT}/Windows Kits/10/Include" _HRX_WINSDK_INCLUDE_ROOT)
cmake_path(GET _HRX_MSVC_ROOT FILENAME _HRX_MSVC_VERSION)
cmake_path(GET _HRX_WINSDK_INCLUDE_ROOT FILENAME _HRX_WINSDK_VERSION)
set(_HRX_WINSDK_LIBRARY_ROOT
  "${WINSDK_ROOT}/Windows Kits/10/Lib/${_HRX_WINSDK_VERSION}")
if(NOT IS_DIRECTORY "${_HRX_WINSDK_LIBRARY_ROOT}")
  message(FATAL_ERROR
    "Windows SDK ${_HRX_WINSDK_VERSION} has no matching library directory: "
    "${_HRX_WINSDK_LIBRARY_ROOT}")
endif()
set(CMAKE_SYSTEM_VERSION "${_HRX_WINSDK_VERSION}")

_hrx_windows_cross_tool(clang-cl _HRX_CLANG_CL)
_hrx_windows_cross_tool(lld-link _HRX_LLD_LINK)
_hrx_windows_cross_tool(llvm-lib _HRX_LLVM_LIB)
_hrx_windows_cross_tool(llvm-ml _HRX_LLVM_ML)
_hrx_windows_cross_tool(llvm-mt _HRX_LLVM_MT)
_hrx_windows_cross_tool(llvm-nm _HRX_LLVM_NM)
_hrx_windows_cross_tool(llvm-rc _HRX_LLVM_RC)

execute_process(
  COMMAND "${_HRX_CLANG_CL}" --no-default-config -print-resource-dir
  RESULT_VARIABLE _HRX_RESOURCE_RESULT
  OUTPUT_VARIABLE _HRX_RESOURCE_ROOT
  ERROR_VARIABLE _HRX_RESOURCE_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT _HRX_RESOURCE_RESULT EQUAL 0 OR
   NOT IS_DIRECTORY "${_HRX_RESOURCE_ROOT}/include")
  message(FATAL_ERROR
    "Could not query the selected clang resource directory: "
    "${_HRX_RESOURCE_ERROR}")
endif()

set(_HRX_BUILTINS_ROOT "${_HRX_RESOURCE_ROOT}/lib/windows")
if(NOT EXISTS "${_HRX_BUILTINS_ROOT}/clang_rt.builtins-x86_64.lib")
  if(NOT DEFINED WINDOWS_COMPILER_RT_ROOT AND
     DEFINED ENV{WINDOWS_COMPILER_RT_ROOT})
    file(TO_CMAKE_PATH "$ENV{WINDOWS_COMPILER_RT_ROOT}"
      WINDOWS_COMPILER_RT_ROOT)
  endif()
  if(NOT EXISTS
      "${WINDOWS_COMPILER_RT_ROOT}/clang_rt.builtins-x86_64.lib")
    message(FATAL_ERROR
      "The selected clang has no Windows x86-64 builtins. Install "
      "clang_rt.builtins-x86_64.lib under its resource directory or set "
      "WINDOWS_COMPILER_RT_ROOT to a matching compiler-rt directory.")
  endif()
  set(_HRX_BUILTINS_ROOT "${WINDOWS_COMPILER_RT_ROOT}")
endif()
set(WINDOWS_COMPILER_RT_ROOT "${WINDOWS_COMPILER_RT_ROOT}" CACHE PATH
  "Optional matching Windows compiler-rt directory")

set(CMAKE_C_COMPILER "${_HRX_CLANG_CL}")
set(CMAKE_CXX_COMPILER "${_HRX_CLANG_CL}")
set(CMAKE_ASM_COMPILER "${_HRX_CLANG_CL}")
set(CMAKE_ASM_MASM_COMPILER "${_HRX_LLVM_ML}" CACHE FILEPATH
  "MASM-compatible assembler for Windows target objects")
set(CMAKE_AR "${_HRX_LLVM_LIB}")
set(CMAKE_LINKER "${_HRX_LLD_LINK}")
set(CMAKE_MT "${_HRX_LLVM_MT}")
set(CMAKE_NM "${_HRX_LLVM_NM}")
set(CMAKE_RC_COMPILER "${_HRX_LLVM_RC}")

set(_HRX_INCLUDE_DIRECTORIES
  "${_HRX_RESOURCE_ROOT}/include"
  "${_HRX_MSVC_ROOT}/include"
  "${_HRX_WINSDK_INCLUDE_ROOT}/ucrt"
  "${_HRX_WINSDK_INCLUDE_ROOT}/shared"
  "${_HRX_WINSDK_INCLUDE_ROOT}/um"
  "${_HRX_WINSDK_INCLUDE_ROOT}/winrt"
  "${_HRX_WINSDK_INCLUDE_ROOT}/cppwinrt")
set(_HRX_LIBRARY_DIRECTORIES
  "${_HRX_MSVC_ROOT}/lib/x64"
  "${_HRX_WINSDK_LIBRARY_ROOT}/ucrt/x64"
  "${_HRX_WINSDK_LIBRARY_ROOT}/um/x64"
  "${_HRX_BUILTINS_ROOT}")

# MSVC toolset 14.x implements compiler compatibility version 19.x. The STL
# uses this version to select the supported clang builtin surface.
string(REPLACE "." ";" _HRX_MSVC_VERSION_COMPONENTS
  "${_HRX_MSVC_VERSION}")
list(GET _HRX_MSVC_VERSION_COMPONENTS 0 _HRX_MSVC_TOOLSET_MAJOR)
list(GET _HRX_MSVC_VERSION_COMPONENTS 1 _HRX_MSVC_TOOLSET_MINOR)
math(EXPR _HRX_MSVC_COMPILER_MAJOR "${_HRX_MSVC_TOOLSET_MAJOR} + 5")

string(CONCAT _HRX_COMPILE_FLAGS
  "--no-default-config --target=x86_64-pc-windows-msvc "
  "-fms-compatibility-version=${_HRX_MSVC_COMPILER_MAJOR}.${_HRX_MSVC_TOOLSET_MINOR} "
  "/Z7 /X /clang:-fdebug-compilation-dir=.")
foreach(_DIRECTORY IN LISTS _HRX_INCLUDE_DIRECTORIES)
  string(APPEND _HRX_COMPILE_FLAGS " /imsvc\"${_DIRECTORY}\"")
endforeach()
set(CMAKE_C_FLAGS_INIT "${_HRX_COMPILE_FLAGS}")
set(CMAKE_CXX_FLAGS_INIT "${_HRX_COMPILE_FLAGS}")
set(CMAKE_ASM_FLAGS_INIT "${_HRX_COMPILE_FLAGS}")

set(_HRX_LINK_FLAGS "/MACHINE:X64 /DEFAULTLIB:clang_rt.builtins-x86_64.lib")
foreach(_DIRECTORY IN LISTS _HRX_LIBRARY_DIRECTORIES)
  string(APPEND _HRX_LINK_FLAGS " /LIBPATH:\"${_DIRECTORY}\"")
endforeach()
set(CMAKE_EXE_LINKER_FLAGS_INIT "${_HRX_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_HRX_LINK_FLAGS}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_HRX_LINK_FLAGS}")
set(CMAKE_STATIC_LINKER_FLAGS_INIT "/MACHINE:X64")

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  LLVM_ROOT
  WINSDK_ROOT
  WINDOWS_COMPILER_RT_ROOT)
