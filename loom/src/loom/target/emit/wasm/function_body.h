// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WebAssembly function-body emission from planned structured target-low IR.
//
// This emits the size-prefixed function body stored in a Wasm code section,
// not a complete Wasm module. Module sections, imports/exports, validation
// tools, and runtime execution adapters are target-owned layers above this
// body emitter.

#ifndef LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
#define LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_

#include "iree/base/api.h"
#include "loom/error/emitter.h"
#include "loom/target/emit/wasm/program.h"

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

// Releases storage owned by |body|. Safe to call on a zero-initialized body.
void loom_wasm_function_body_deinitialize(loom_wasm_function_body_t* body,
                                          iree_allocator_t allocator);

// Emits a size-prefixed Wasm code-section function body from one planned
// function plan. Physical locals and direct-call targets are trusted facts in
// |program|; the writer may append private locals required by Wasm encoding.
// Rejection emits source diagnostics and leaves |out_accepted| false. Statuses
// report only allocation or diagnostic-sink failures.
iree_status_t loom_wasm_emit_function_body(
    const loom_wasm_program_plan_t* program,
    const loom_wasm_function_plan_t* function,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_allocator_t allocator,
    bool* out_accepted, loom_wasm_function_body_t* out_body);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_FUNCTION_BODY_H_
