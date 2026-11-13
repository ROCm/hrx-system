// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Planned physical WebAssembly program consumed by binary emission.

#ifndef LOOM_TARGET_EMIT_WASM_PROGRAM_H_
#define LOOM_TARGET_EMIT_WASM_PROGRAM_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/wasm/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // No WebAssembly local is assigned to a module value or symbol.
  LOOM_WASM_PROGRAM_INDEX_NONE = UINT32_MAX,
};

// One interned WebAssembly function signature.
typedef struct loom_wasm_function_type_t {
  // WebAssembly value types for function parameters.
  const loom_wasm_value_type_t* parameters;
  // Number of parameter entries.
  uint32_t parameter_count;
  // WebAssembly value types for function results.
  const loom_wasm_value_type_t* results;
  // Number of result entries.
  uint32_t result_count;
} loom_wasm_function_type_t;

// One planned Low function ready for WebAssembly body emission.
typedef struct loom_wasm_function_plan_t {
  // Structured Low function body walked by the WebAssembly writer.
  const loom_region_t* body;
  // Machine schedule model resolved from this function's target facts.
  const loom_low_schedule_model_t* schedule_model;
  // Function-local value IDs indexed by planned value ordinal.
  const loom_value_id_t* value_ids;
  // WebAssembly local index by planned value ordinal. Entries without a
  // physical local contain LOOM_WASM_PROGRAM_INDEX_NONE.
  const uint32_t* local_indices_by_value_ordinal;
  // Number of entries in |value_ids| and |local_indices_by_value_ordinal|.
  loom_value_ordinal_t value_count;
  // Borrowed module symbol name.
  iree_string_view_t name;
  // Original export identity, or the authored export for direct Low input.
  iree_string_view_t source_export_name;
  // Borrowed export name, or empty when the function is module-private.
  iree_string_view_t export_name;
  // WebAssembly function index assigned by module symbol order.
  uint32_t function_index;
  // Interned WebAssembly function type index.
  uint32_t type_index;
  // Interned WebAssembly function signature.
  loom_wasm_function_type_t type;
  // Planned parameter and physical-local types in local-index order.
  const loom_wasm_value_type_t* local_types;
  // Number of parameter entries at the start of |local_types|.
  uint32_t parameter_count;
  // Number of planned entries in |local_types|.
  uint32_t local_count;
} loom_wasm_function_plan_t;

// Immutable physical WebAssembly program produced by target planning.
//
// All tables are arena-owned and borrow the semantically immutable Low module.
// The plan contains only target-consumable physical facts; compiler liveness,
// allocation intervals, diagnostics, and rejected candidates do not cross the
// emission boundary.
typedef struct loom_wasm_program_plan_t {
  // Low module containing the planned function bodies. Emission preserves
  // semantic IR and uses only the module-owned value-ordinal scratch map.
  loom_module_t* module;
  // Functions in WebAssembly function-index order.
  const loom_wasm_function_plan_t* functions;
  // Number of entries in |functions|.
  iree_host_size_t function_count;
  // Interned WebAssembly function signatures in type-index order.
  const loom_wasm_function_type_t* types;
  // Number of entries in |types|.
  iree_host_size_t type_count;
  // WebAssembly function index by module symbol ID.
  const uint32_t* function_indices_by_symbol;
  // Number of entries in |function_indices_by_symbol|.
  iree_host_size_t symbol_count;
  // Number of exported functions.
  iree_host_size_t export_count;
} loom_wasm_program_plan_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_PROGRAM_H_
