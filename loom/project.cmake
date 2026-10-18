# Copyright 2026 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

list(APPEND CMAKE_MODULE_PATH
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/amdgpu"
  "${CMAKE_CURRENT_LIST_DIR}/build_tools/cmake"
)

include(loom_amdgpu_selectors)

if(NOT DEFINED LOOM_BUILD)
  option(LOOM_BUILD
    "Build Loom compiler libraries, tools, bindings, and tests." ON)
endif()

option(LOOM_LINK_SHARED_TOOLS
  "Link LoomC tools to one shared compiler and its exact-build IREE runtime." OFF)
if(LOOM_LINK_SHARED_TOOLS AND CMAKE_VERSION VERSION_LESS 3.27)
  message(FATAL_ERROR "LOOM_LINK_SHARED_TOOLS requires CMake 3.27 or newer.")
endif()
if(LOOM_LINK_SHARED_TOOLS AND BUILD_SHARED_LIBS)
  message(FATAL_ERROR
    "LOOM_LINK_SHARED_TOOLS requires BUILD_SHARED_LIBS=OFF so the bundle owns its component implementations.")
endif()

option(LOOM_TARGET_DEFAULTS
  "Default value for dependency-satisfied Loom production targets." ON)
if(NOT DEFINED LOOM_TARGET_AMDGPU_DEFAULT)
  set(LOOM_TARGET_AMDGPU_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_SPIRV_DEFAULT)
  set(LOOM_TARGET_SPIRV_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_WASM_DEFAULT)
  set(LOOM_TARGET_WASM_DEFAULT OFF)
endif()
if(NOT DEFINED LOOM_TARGET_VM_DEFAULT)
  set(LOOM_TARGET_VM_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_XDNA_DEFAULT)
  set(LOOM_TARGET_XDNA_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_X86_DEFAULT)
  set(LOOM_TARGET_X86_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
option(LOOM_TARGET_AMDGPU
  "Enables Loom AMDGPU target support."
  ${LOOM_TARGET_AMDGPU_DEFAULT})
if(NOT DEFINED LOOM_TARGET_AMDGPU_TARGETS)
  set(LOOM_TARGET_AMDGPU_TARGETS
    ${_LOOM_AMDGPU_DEFAULT_TARGET_SELECTORS}
    CACHE STRING
    "Loom AMDGPU target selectors to compile into AMDGPU target support.")
endif()
option(LOOM_TARGET_SPIRV
  "Enables Loom SPIR-V target support."
  ${LOOM_TARGET_SPIRV_DEFAULT})
option(LOOM_TARGET_WASM
  "Enables Loom WebAssembly target support."
  ${LOOM_TARGET_WASM_DEFAULT})
option(LOOM_TARGET_VM
  "Enables Loom VM target support."
  ${LOOM_TARGET_VM_DEFAULT})
option(LOOM_TARGET_XDNA
  "Enables Loom AMD XDNA target support."
  ${LOOM_TARGET_XDNA_DEFAULT})
option(LOOM_TARGET_X86
  "Enables Loom x86 target support."
  ${LOOM_TARGET_X86_DEFAULT})

option(LOOM_TARGET_ARCH_AMDGPU
  "Enables the AMDGPU Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_SPIRV
  "Enables the SPIR-V Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_WASM
  "Enables the WebAssembly Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_VM
  "Enables the VM Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_XDNA
  "Enables the AMD XDNA Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_X86
  "Enables the x86 Loom target architecture slice."
  OFF)
mark_as_advanced(
  LOOM_TARGET_ARCH_AMDGPU
  LOOM_TARGET_ARCH_SPIRV
  LOOM_TARGET_ARCH_VM
  LOOM_TARGET_ARCH_WASM
  LOOM_TARGET_ARCH_XDNA
  LOOM_TARGET_ARCH_X86
)

option(LOOM_EMIT_AMDGPU
  "Enables the AMDGPU Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_SPIRV
  "Enables the SPIR-V Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_WASM
  "Enables the WebAssembly Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_XDNA
  "Enables the AMD XDNA Loom artifact emitter slice."
  OFF)
mark_as_advanced(
  LOOM_EMIT_AMDGPU
  LOOM_EMIT_SPIRV
  LOOM_EMIT_WASM
  LOOM_EMIT_XDNA
)

if(LOOM_TARGET_AMDGPU)
  set(LOOM_TARGET_ARCH_AMDGPU ON)
  set(LOOM_EMIT_AMDGPU ON)
endif()
if(LOOM_TARGET_SPIRV)
  set(LOOM_TARGET_ARCH_SPIRV ON)
  set(LOOM_EMIT_SPIRV ON)
endif()
if(LOOM_TARGET_WASM)
  set(LOOM_TARGET_ARCH_WASM ON)
  set(LOOM_EMIT_WASM ON)
endif()
if(LOOM_TARGET_VM)
  set(LOOM_TARGET_ARCH_VM ON)
endif()
if(LOOM_TARGET_XDNA)
  set(LOOM_TARGET_ARCH_XDNA ON)
  set(LOOM_EMIT_XDNA ON)
endif()
if(LOOM_TARGET_X86)
  set(LOOM_TARGET_ARCH_X86 ON)
endif()

if(LOOM_TARGET_ARCH_AMDGPU)
  loom_amdgpu_configure_target_selectors()
endif()

option(LOOM_EXECUTE_DEFAULTS
  "Default value for dependency-satisfied Loom execution substrates." ON)
if(NOT DEFINED LOOM_EXECUTE_IREE_HAL_DEFAULT)
  set(LOOM_EXECUTE_IREE_HAL_DEFAULT ${LOOM_EXECUTE_DEFAULTS})
endif()
option(LOOM_EXECUTE_IREE_HAL
  "Enables Loom execution providers that use IREE HAL runtime support."
  ${LOOM_EXECUTE_IREE_HAL_DEFAULT})

option(LOOM_IMPORT_CXX
  "Enables the native Loom C/C++ importer, tools, and binding." OFF)
option(LOOM_IMPORT_CXX_EMBED_INCLUDES
  "Embeds Loom C++ and HIP source headers in the C++ importer." ON)
option(LOOM_IMPORT_MLIR
  "Enables the Loom MLIR importer package and importer tests." OFF)
option(LOOM_IMPORT_TILELANG
  "Enables the Loom TileLang importer package and importer tests." OFF)

set(_LOOM_SERVE_MEDIA_HOST OFF)
if(CMAKE_SYSTEM_NAME MATCHES "^(Linux|Darwin|Windows)$")
  set(_LOOM_SERVE_MEDIA_HOST ON)
endif()
option(LOOM_SERVE_IMAGE_INPUT
  "Enables image inputs to Loom serving deployments."
  ${_LOOM_SERVE_MEDIA_HOST})
set(LOOM_SERVE_IMAGE_CODECS "png;jpeg;gif;webp" CACHE STRING
  "Image input codecs registered in Loom serving deployments.")
if(LOOM_BUILD AND LOOM_TARGET_ARCH_VM AND LOOM_SERVE_IMAGE_INPUT AND
    NOT _LOOM_SERVE_MEDIA_HOST)
  message(FATAL_ERROR
    "Loom image input requires Linux, macOS, or Windows. Set LOOM_SERVE_IMAGE_INPUT=OFF to build other serving modalities.")
endif()
foreach(_codec png jpeg gif webp)
  string(TOUPPER "${_codec}" _codec_upper)
  set(LOOM_SERVE_IMAGE_CODEC_${_codec_upper} OFF)
  if(LOOM_BUILD AND LOOM_TARGET_ARCH_VM AND LOOM_SERVE_IMAGE_INPUT AND
      _codec IN_LIST LOOM_SERVE_IMAGE_CODECS)
    set(LOOM_SERVE_IMAGE_CODEC_${_codec_upper} ON)
  endif()
endforeach()
foreach(_codec IN LISTS LOOM_SERVE_IMAGE_CODECS)
  if(NOT _codec MATCHES "^(png|jpeg|gif|webp)$")
    message(FATAL_ERROR "Unknown Loom image input codec: ${_codec}")
  endif()
endforeach()

set(_LOOM_SERVE_AUDIO_CODECS "")
if(_LOOM_SERVE_MEDIA_HOST)
  set(_LOOM_SERVE_AUDIO_CODECS "wav;mp3;flac")
endif()
set(LOOM_SERVE_AUDIO_CODECS "${_LOOM_SERVE_AUDIO_CODECS}" CACHE STRING
  "Optional encoded-audio input codecs; empty retains raw PCM transcription.")
if(LOOM_BUILD AND LOOM_TARGET_ARCH_VM AND LOOM_SERVE_AUDIO_CODECS AND
    NOT _LOOM_SERVE_MEDIA_HOST)
  message(FATAL_ERROR
    "Loom encoded audio requires Linux, macOS, or Windows. Set LOOM_SERVE_AUDIO_CODECS= to retain raw PCM.")
endif()
set(LOOM_SERVE_AUDIO_DECODING OFF)
foreach(_codec wav mp3 flac)
  string(TOUPPER "${_codec}" _codec_upper)
  set(LOOM_SERVE_AUDIO_CODEC_${_codec_upper} OFF)
  if(LOOM_BUILD AND LOOM_TARGET_ARCH_VM AND _LOOM_SERVE_MEDIA_HOST AND
      _codec IN_LIST LOOM_SERVE_AUDIO_CODECS)
    set(LOOM_SERVE_AUDIO_CODEC_${_codec_upper} ON)
    set(LOOM_SERVE_AUDIO_DECODING ON)
  endif()
endforeach()
foreach(_codec IN LISTS LOOM_SERVE_AUDIO_CODECS)
  if(NOT _codec MATCHES "^(wav|mp3|flac)$")
    message(FATAL_ERROR "Unknown Loom audio input codec: ${_codec}")
  endif()
endforeach()

if(LOOM_EMIT_AMDGPU AND NOT LOOM_TARGET_ARCH_AMDGPU)
  message(FATAL_ERROR
    "LOOM_EMIT_AMDGPU=ON requires LOOM_TARGET_ARCH_AMDGPU=ON.")
endif()

if(LOOM_EMIT_SPIRV AND NOT LOOM_TARGET_ARCH_SPIRV)
  message(FATAL_ERROR
    "LOOM_EMIT_SPIRV=ON requires LOOM_TARGET_ARCH_SPIRV=ON.")
endif()

if(LOOM_EMIT_WASM AND NOT LOOM_TARGET_ARCH_WASM)
  message(FATAL_ERROR
    "LOOM_EMIT_WASM=ON requires LOOM_TARGET_ARCH_WASM=ON.")
endif()

if(LOOM_EMIT_XDNA AND NOT LOOM_TARGET_ARCH_XDNA)
  message(FATAL_ERROR
    "LOOM_EMIT_XDNA=ON requires LOOM_TARGET_ARCH_XDNA=ON.")
endif()

function(loom_configure_project)
  if(LOOM_SERVE_AUDIO_DECODING)
    iree_configure_miniaudio()
  endif()
  if(LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU)
    iree_configure_rocm_hsa_runtime_headers()
  endif()
  if(LOOM_SERVE_IMAGE_CODEC_PNG)
    iree_configure_libpng()
  endif()
  if(LOOM_SERVE_IMAGE_CODEC_JPEG)
    iree_configure_libjpeg_turbo()
  endif()
  if(LOOM_SERVE_IMAGE_CODEC_GIF)
    iree_configure_giflib()
  endif()
  if(LOOM_SERVE_IMAGE_CODEC_WEBP)
    iree_configure_libwebp()
  endif()
endfunction()
