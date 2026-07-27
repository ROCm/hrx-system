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

option(LOOM_TARGET_DEFAULTS
  "Default value for dependency-satisfied Loom production targets." ON)
if(NOT DEFINED LOOM_TARGET_AMDGPU_DEFAULT)
  set(LOOM_TARGET_AMDGPU_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_IREE_VM_DEFAULT)
  set(LOOM_TARGET_IREE_VM_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_LLVMIR_DEFAULT)
  set(LOOM_TARGET_LLVMIR_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_SPIRV_DEFAULT)
  set(LOOM_TARGET_SPIRV_DEFAULT ${LOOM_TARGET_DEFAULTS})
endif()
if(NOT DEFINED LOOM_TARGET_WASM_DEFAULT)
  set(LOOM_TARGET_WASM_DEFAULT OFF)
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
option(LOOM_TARGET_IREE_VM
  "Enables Loom IREE VM target support."
  ${LOOM_TARGET_IREE_VM_DEFAULT})
option(LOOM_TARGET_LLVMIR
  "Enables Loom LLVM IR oracle target support."
  ${LOOM_TARGET_LLVMIR_DEFAULT})
option(LOOM_TARGET_SPIRV
  "Enables Loom SPIR-V target support."
  ${LOOM_TARGET_SPIRV_DEFAULT})
option(LOOM_TARGET_WASM
  "Enables Loom WebAssembly target support."
  ${LOOM_TARGET_WASM_DEFAULT})
option(LOOM_TARGET_X86
  "Enables Loom x86 target support."
  ${LOOM_TARGET_X86_DEFAULT})

option(LOOM_TARGET_ARCH_AMDGPU
  "Enables the AMDGPU Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_IREE_VM
  "Enables the IREE VM Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_LLVMIR
  "Enables the LLVM IR Loom oracle target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_SPIRV
  "Enables the SPIR-V Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_WASM
  "Enables the WebAssembly Loom target architecture slice."
  OFF)
option(LOOM_TARGET_ARCH_X86
  "Enables the x86 Loom target architecture slice."
  OFF)
mark_as_advanced(
  LOOM_TARGET_ARCH_AMDGPU
  LOOM_TARGET_ARCH_IREE_VM
  LOOM_TARGET_ARCH_LLVMIR
  LOOM_TARGET_ARCH_SPIRV
  LOOM_TARGET_ARCH_WASM
  LOOM_TARGET_ARCH_X86
)

option(LOOM_EMIT_AMDGPU
  "Enables the AMDGPU Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_IREE_VM
  "Enables the IREE VM Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_LLVMIR
  "Enables the LLVM IR Loom debug artifact emitter slice." OFF)
option(LOOM_EMIT_SPIRV
  "Enables the SPIR-V Loom artifact emitter slice."
  OFF)
option(LOOM_EMIT_WASM
  "Enables the WebAssembly Loom artifact emitter slice."
  OFF)
mark_as_advanced(
  LOOM_EMIT_AMDGPU
  LOOM_EMIT_IREE_VM
  LOOM_EMIT_LLVMIR
  LOOM_EMIT_SPIRV
  LOOM_EMIT_WASM
)

if(LOOM_TARGET_AMDGPU)
  set(LOOM_TARGET_ARCH_AMDGPU ON)
  set(LOOM_EMIT_AMDGPU ON)
endif()
if(LOOM_TARGET_IREE_VM)
  set(LOOM_TARGET_ARCH_IREE_VM ON)
  set(LOOM_EMIT_IREE_VM ON)
endif()
if(LOOM_TARGET_LLVMIR)
  set(LOOM_TARGET_ARCH_LLVMIR ON)
  set(LOOM_EMIT_LLVMIR ON)
endif()
if(LOOM_TARGET_SPIRV)
  set(LOOM_TARGET_ARCH_SPIRV ON)
  set(LOOM_EMIT_SPIRV ON)
endif()
if(LOOM_TARGET_WASM)
  set(LOOM_TARGET_ARCH_WASM ON)
  set(LOOM_EMIT_WASM ON)
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
if(NOT DEFINED LOOM_EXECUTE_IREE_VM_DEFAULT)
  set(LOOM_EXECUTE_IREE_VM_DEFAULT ${LOOM_EXECUTE_DEFAULTS})
endif()
option(LOOM_EXECUTE_IREE_HAL
  "Enables Loom execution providers that use IREE HAL runtime support."
  ${LOOM_EXECUTE_IREE_HAL_DEFAULT})
option(LOOM_EXECUTE_IREE_VM
  "Enables Loom execution tests that use IREE VM runtime support."
  ${LOOM_EXECUTE_IREE_VM_DEFAULT})

option(LOOM_IMPORT_MLIR
  "Enables the Loom MLIR importer package and importer tests." OFF)
option(LOOM_IMPORT_TILELANG
  "Enables the Loom TileLang importer package and importer tests." OFF)

if(LOOM_EMIT_AMDGPU AND NOT LOOM_TARGET_ARCH_AMDGPU)
  message(FATAL_ERROR
    "LOOM_EMIT_AMDGPU=ON requires LOOM_TARGET_ARCH_AMDGPU=ON.")
endif()

if(LOOM_EMIT_IREE_VM AND NOT LOOM_TARGET_ARCH_IREE_VM)
  message(FATAL_ERROR
    "LOOM_EMIT_IREE_VM=ON requires LOOM_TARGET_ARCH_IREE_VM=ON.")
endif()

if(LOOM_EMIT_LLVMIR AND NOT LOOM_TARGET_ARCH_LLVMIR)
  message(FATAL_ERROR
    "LOOM_EMIT_LLVMIR=ON requires LOOM_TARGET_ARCH_LLVMIR=ON.")
endif()

if(LOOM_EMIT_SPIRV AND NOT LOOM_TARGET_ARCH_SPIRV)
  message(FATAL_ERROR
    "LOOM_EMIT_SPIRV=ON requires LOOM_TARGET_ARCH_SPIRV=ON.")
endif()

if(LOOM_EMIT_WASM AND NOT LOOM_TARGET_ARCH_WASM)
  message(FATAL_ERROR
    "LOOM_EMIT_WASM=ON requires LOOM_TARGET_ARCH_WASM=ON.")
endif()

function(loom_configure_project)
  if(LOOM_TARGET_ARCH_AMDGPU AND LOOM_EMIT_AMDGPU)
    iree_configure_rocm_hsa_runtime_headers()
  endif()
endfunction()
