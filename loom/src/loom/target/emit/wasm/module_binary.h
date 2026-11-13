// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly binary module emission from planned physical programs.
//
// This target-owned layer is the Wasm artifact boundary: it serializes planned
// function/type/export indices, physical locals, and structured Low bodies into
// one binary module. Tool validation, disassembly, allocation, and diagnostics
// remain outside this production emitter.

#ifndef LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
#define LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/target/emit/wasm/program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_module_binary_flag_bits_e {
  // Module defines and exports one default linear memory as `memory`.
  LOOM_WASM_MODULE_BINARY_FLAG_DEFINES_MEMORY = 1u << 0,
} loom_wasm_module_binary_flag_bits_t;

// Bitset of loom_wasm_module_binary_flag_bits_t values.
typedef uint32_t loom_wasm_module_binary_flags_t;

typedef struct loom_wasm_module_binary_t {
  // Allocator-owned Wasm module binary bytes.
  uint8_t* data;
  // Number of bytes in |data|.
  iree_host_size_t data_length;
  // Structural facts represented in the emitted module binary.
  loom_wasm_module_binary_flags_t flags;
} loom_wasm_module_binary_t;

// Releases storage owned by |module|. Safe to call on a zero-initialized
// module object.
void loom_wasm_module_binary_deinitialize(loom_wasm_module_binary_t* module,
                                          iree_allocator_t allocator);

// Emits one complete WebAssembly binary from a compiler-planned physical
// program. The writer preserves planned function/type/local indices and walks
// trusted structured Low bodies in source order. Allocation and target
// resolution belong to program planning. Encoding rejection emits structured
// diagnostics and leaves |out_accepted| false without publishing partial bytes.
iree_status_t loom_wasm_program_emit_binary(
    const loom_wasm_program_plan_t* plan,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_allocator_t allocator,
    bool* out_accepted, loom_wasm_module_binary_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
