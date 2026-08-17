# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

function(_iree_runtime_amdgpu_resolve_tool path_var)
  set(_path "${${path_var}}")
  if(WIN32 AND _path AND NOT EXISTS "${_path}" AND
     EXISTS "${_path}${CMAKE_EXECUTABLE_SUFFIX}")
    set(${path_var} "${_path}${CMAKE_EXECUTABLE_SUFFIX}" PARENT_SCOPE)
  endif()
endfunction()

function(iree_runtime_configure_amdgpu_toolchain)
  if(NOT IREE_HAL_DRIVER_AMDGPU)
    return()
  endif()
  if(NOT CMAKE_C_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
      "IREE_HAL_DRIVER_AMDGPU=ON requires the configured C compiler to be "
      "Clang-family and capable of targeting amdgcn-amd-amdhsa. Configure "
      "with the host LLVM Clang toolchain; IREE_ROCM_PATH supplies device "
      "libraries and runtime components, not the host compiler.")
  endif()

  set(_iree_runtime_amdgpu_clang_binary "${CMAKE_C_COMPILER}")
  if(CMAKE_C_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    get_filename_component(
      _iree_runtime_amdgpu_clang_dir "${CMAKE_C_COMPILER}" DIRECTORY)
    unset(_iree_runtime_amdgpu_clang_binary)
    find_program(_iree_runtime_amdgpu_clang_binary
      NAMES clang
      HINTS "${_iree_runtime_amdgpu_clang_dir}"
      NO_DEFAULT_PATH
      REQUIRED)
  endif()
  set(IREE_CLANG_BINARY "${_iree_runtime_amdgpu_clang_binary}" CACHE FILEPATH
    "Clang used by IREE runtime AMDGPU device binary builds." FORCE)

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-ar
    OUTPUT_VARIABLE _llvm_ar_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  _iree_runtime_amdgpu_resolve_tool(_llvm_ar_from_clang)
  if(EXISTS "${_llvm_ar_from_clang}")
    set(IREE_LLVM_AR_BINARY "${_llvm_ar_from_clang}" CACHE FILEPATH
      "llvm-ar used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_llvm_ar_fallback llvm-ar REQUIRED)
    set(IREE_LLVM_AR_BINARY "${_llvm_ar_fallback}" CACHE FILEPATH
      "llvm-ar used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  set(_probe_src "${CMAKE_BINARY_DIR}/CMakeFiles/iree-runtime-amdgcn-probe.c")
  set(_probe_obj "${CMAKE_BINARY_DIR}/CMakeFiles/iree-runtime-amdgcn-probe.bc")
  file(WRITE "${_probe_src}" "void iree_runtime_amdgcn_probe(void) {}\n")
  execute_process(
    COMMAND "${IREE_CLANG_BINARY}"
      -target amdgcn-amd-amdhsa -mcpu=gfx900 -nogpulib
      -x c -std=c11 -c -emit-llvm "${_probe_src}" -o "${_probe_obj}"
    RESULT_VARIABLE _probe_result
    OUTPUT_VARIABLE _probe_stdout
    ERROR_VARIABLE _probe_stderr
  )
  if(NOT _probe_result EQUAL 0)
    message(FATAL_ERROR
      "Configured C compiler cannot compile a minimal amdgcn-amd-amdhsa object.\n"
      "Compiler: ${IREE_CLANG_BINARY}\n"
      "stderr:\n${_probe_stderr}")
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-link
    OUTPUT_VARIABLE _llvm_link_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  _iree_runtime_amdgpu_resolve_tool(_llvm_link_from_clang)
  if(EXISTS "${_llvm_link_from_clang}")
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_from_clang}" CACHE FILEPATH
      "llvm-link used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_llvm_link_fallback llvm-link REQUIRED)
    set(IREE_LLVM_LINK_BINARY "${_llvm_link_fallback}" CACHE FILEPATH
      "llvm-link used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=ld.lld
    OUTPUT_VARIABLE _lld_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  _iree_runtime_amdgpu_resolve_tool(_lld_from_clang)
  if(EXISTS "${_lld_from_clang}")
    set(IREE_LLD_BINARY "${_lld_from_clang}" CACHE FILEPATH
      "lld used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_lld_fallback NAMES ld.lld lld REQUIRED)
    set(IREE_LLD_BINARY "${_lld_fallback}" CACHE FILEPATH
      "lld used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=llvm-objcopy
    OUTPUT_VARIABLE _objcopy_from_clang
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  _iree_runtime_amdgpu_resolve_tool(_objcopy_from_clang)
  if(EXISTS "${_objcopy_from_clang}")
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_from_clang}" CACHE FILEPATH
      "llvm-objcopy used by IREE runtime AMDGPU device binary builds." FORCE)
  else()
    find_program(_objcopy_fallback llvm-objcopy REQUIRED)
    set(IREE_LLVM_OBJCOPY_BINARY "${_objcopy_fallback}" CACHE FILEPATH
      "llvm-objcopy used by IREE runtime AMDGPU device binary builds." FORCE)
  endif()

  execute_process(
    COMMAND "${IREE_CLANG_BINARY}" -print-resource-dir
    OUTPUT_VARIABLE _clang_resource_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _clang_resource_dir OR NOT EXISTS "${_clang_resource_dir}/include")
    message(FATAL_ERROR
      "Could not determine clang resource include directory from ${IREE_CLANG_BINARY}")
  endif()
  set(IREE_CLANG_BUILTIN_HEADERS_PATH "${_clang_resource_dir}/include" CACHE PATH
    "Clang resource include directory used by AMDGPU device binary builds." FORCE)

  set(IREE_ROCM_DEVICE_LIBRARIES_PATH
    "${IREE_ROCM_DEVICE_LIBRARIES_PATH}" CACHE PATH
    "ROCm bitcode libraries used by HIP device fixture builds.")
  if(NOT IREE_ROCM_DEVICE_LIBRARIES_PATH)
    get_filename_component(
      _clang_llvm_root "${_clang_resource_dir}/../../.." ABSOLUTE)
    set(_rocm_device_library_candidates
      "${_clang_llvm_root}/amdgcn/bitcode")
    foreach(_rocm_root
        "${IREE_HAL_AMDGPU_DEVICE_TOOLCHAIN_ROCM_PATH}"
        "${IREE_ROCM_PATH}"
        "$ENV{ROCM_PATH}"
        "$ENV{ROCM_ROOT}"
        "$ENV{ROCM_HOME}"
        "$ENV{HIP_PATH}")
      if(NOT _rocm_root)
        continue()
      endif()
      list(APPEND _rocm_device_library_candidates
        "${_rocm_root}/amdgcn/bitcode"
        "${_rocm_root}/lib/amdgcn/bitcode"
        "${_rocm_root}/llvm/amdgcn/bitcode"
        "${_rocm_root}/lib/llvm/amdgcn/bitcode")
      file(GLOB _packaged_rocm_device_library_candidates
        LIST_DIRECTORIES true
        "${_rocm_root}/lib/python*/site-packages/_rocm_sdk_*/llvm/amdgcn/bitcode"
        "${_rocm_root}/lib64/python*/site-packages/_rocm_sdk_*/llvm/amdgcn/bitcode")
      list(APPEND _rocm_device_library_candidates
        ${_packaged_rocm_device_library_candidates})
    endforeach()
    list(REMOVE_DUPLICATES _rocm_device_library_candidates)
    foreach(_candidate ${_rocm_device_library_candidates})
      if(EXISTS "${_candidate}/hip.bc" AND
         EXISTS "${_candidate}/ockl.bc" AND
         EXISTS "${_candidate}/ocml.bc")
        set(IREE_ROCM_DEVICE_LIBRARIES_PATH "${_candidate}" CACHE PATH
          "ROCm bitcode libraries used by HIP device fixture builds." FORCE)
        break()
      endif()
    endforeach()
  endif()

  set(IREE_AMDGPU_HIP_DEVICE_LIBRARIES_AVAILABLE OFF CACHE INTERNAL
    "Whether the AMDGPU toolchain can link HIP device fixtures." FORCE)
  if(IREE_ROCM_DEVICE_LIBRARIES_PATH)
    foreach(_device_library hip.bc ockl.bc ocml.bc)
      if(NOT EXISTS
          "${IREE_ROCM_DEVICE_LIBRARIES_PATH}/${_device_library}")
        message(FATAL_ERROR
          "ROCm device library directory is incomplete: "
          "${IREE_ROCM_DEVICE_LIBRARIES_PATH}/${_device_library} is missing")
      endif()
    endforeach()

    execute_process(
      COMMAND "${IREE_CLANG_BINARY}" -print-prog-name=clang-offload-bundler
      OUTPUT_VARIABLE _offload_bundler_from_clang
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    _iree_runtime_amdgpu_resolve_tool(_offload_bundler_from_clang)
    if(EXISTS "${_offload_bundler_from_clang}")
      set(IREE_CLANG_OFFLOAD_BUNDLER_BINARY
        "${_offload_bundler_from_clang}" CACHE FILEPATH
        "clang-offload-bundler used by HIP device fixture builds." FORCE)
    else()
      find_program(_offload_bundler_fallback clang-offload-bundler REQUIRED)
      set(IREE_CLANG_OFFLOAD_BUNDLER_BINARY
        "${_offload_bundler_fallback}" CACHE FILEPATH
        "clang-offload-bundler used by HIP device fixture builds." FORCE)
    endif()
    set(IREE_AMDGPU_HIP_DEVICE_LIBRARIES_AVAILABLE ON CACHE INTERNAL
      "Whether the AMDGPU toolchain can link HIP device fixtures." FORCE)
  endif()
endfunction()
