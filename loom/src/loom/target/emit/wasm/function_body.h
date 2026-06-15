// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly function-body emission from allocated structured target-low IR.
//
// This emits the size-prefixed function body stored in a Wasm code section,
// not a complete Wasm module. Module sections, imports/exports, validation
// tools, and runtime execution adapters are target-owned layers above this
// body emitter.

#ifndef LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
#define LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_

#include "iree/base/api.h"
#include "loom/codegen/low/allocation.h"
#include "loom/ir/attribute.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_function_body_flag_bits_e {
  // Body contains at least one descriptor-backed memory access.
  LOOM_WASM_FUNCTION_BODY_FLAG_USES_MEMORY = 1u << 0,
} loom_wasm_function_body_flag_bits_t;

// Bitset of loom_wasm_function_body_flag_bits_t values.
typedef uint32_t loom_wasm_function_body_flags_t;

typedef struct loom_wasm_function_body_t {
  // Allocator-owned size-prefixed Wasm function-body bytes.
  uint8_t* data;
  // Number of bytes in |data|, including the leading body-size LEB.
  iree_host_size_t data_length;
  // Number of semantic body bytes after the leading body-size LEB.
  iree_host_size_t body_length;
  // Number of Wasm function parameters.
  uint32_t parameter_count;
  // Number of Wasm local indices including parameters.
  uint32_t local_count;
  // Structural facts observed while emitting the function body.
  loom_wasm_function_body_flags_t flags;
} loom_wasm_function_body_t;

// Resolves a direct low.func.call callee to a Wasm function index owned by the
// enclosing module emitter.
typedef iree_status_t (*loom_wasm_resolve_function_index_fn_t)(
    void* user_data, loom_symbol_ref_t callee, uint32_t* out_function_index);

typedef struct loom_wasm_function_body_options_t {
  // Optional callback used to resolve low.func.call callees.
  loom_wasm_resolve_function_index_fn_t resolve_function_index;
  // Caller-owned data passed to |resolve_function_index|.
  void* resolve_function_index_user_data;
} loom_wasm_function_body_options_t;

// Releases storage owned by |body|. Safe to call on a zero-initialized body.
void loom_wasm_function_body_deinitialize(loom_wasm_function_body_t* body,
                                          iree_allocator_t allocator);

// Emits a size-prefixed Wasm code-section function body for one allocated
// low.func.def. The allocation table must use the wasm.core.simd128 descriptor
// set and describe lifetimes for structured source-order body emission.
// Unsupported structural forms or descriptor packets fail loud instead of
// producing partial Wasm.
iree_status_t loom_wasm_emit_function_body(
    const loom_low_allocation_table_t* allocation,
    const loom_wasm_function_body_options_t* options,
    iree_allocator_t allocator, loom_wasm_function_body_t* out_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
