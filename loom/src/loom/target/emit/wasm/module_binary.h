// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly binary module emission from prepared target-low modules.
//
// This target-owned layer is the Wasm artifact boundary: it walks the input
// module, assigns Wasm function/type/export indices, builds low emission frames
// for each wasm.core.simd128 low.func.def, and writes one binary module. Tool
// validation, disassembly, and execution remain outside this production
// emitter.

#ifndef LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
#define LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/ir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_module_binary_flag_bits_e {
  // Module defines one default linear memory section.
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

// Emits a complete Wasm binary module for every wasm.core.simd128 low.func.def
// in |module|. The emitter preserves module symbol order for Wasm function
// indices, emits direct low.func.call instructions against those indices, and
// exports low functions marked public or carrying an explicit export symbol.
//
// Imports, kernel entries, and non-wasm low functions currently fail loud. The
// body emitter walks structured regions in source order, so module emission
// requires source-priority frame scheduling. The caller owns source-to-low
// lowering, target verification, and the remaining frame options used by the
// scheduler/allocator.
iree_status_t loom_wasm_emit_module(
    loom_module_t* module, const loom_low_emission_frame_options_t* options,
    iree_arena_allocator_t* arena, iree_allocator_t allocator,
    loom_wasm_module_binary_t* out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_MODULE_BINARY_H_
