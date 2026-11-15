// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V typed logical storage materialization.
//
// Low reservations retain their canonical byte placements. Workgroup arrays
// share an explicitly laid-out entry block when the target supports their
// element widths. Function declarations enter the function's entry prefix.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_

#include "iree/base/api.h"
#include "loom/codegen/low/storage_layout.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/binary_writer.h"
#include "loom/target/emit/spirv/entry_interface.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"
#include "loom/target/emit/spirv/program.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_workgroup_array_t loom_spirv_workgroup_array_t;

// Function-owned storage collected by its ordinary emission traversal.
typedef struct loom_spirv_module_storage_t {
  // Canonical placements captured at reservation declarations.
  loom_low_storage_layout_builder_t layout;
  // Arena-owned typed Workgroup arrays, in first-address order.
  loom_spirv_workgroup_array_t* arrays;
  // Number of initialized arrays.
  iree_host_size_t array_count;
  // Allocated capacity of |arrays|.
  iree_host_size_t array_capacity;
  // Explicit-layout capabilities required by the arrays' actual element widths.
  loom_spirv_feature_bits_t layout_features;
} loom_spirv_module_storage_t;

// Borrowed resources for one function's storage emission.
typedef struct loom_spirv_module_storage_context_t {
  // Prepared function identity and target contract.
  const loom_spirv_function_plan_t* function;
  // Function-local storage/value identity map.
  loom_spirv_module_value_table_t* values;
  // Module type cache, builder, and scratch arena.
  loom_spirv_type_context_t* types;
  // Entry prefix receiving private declarations, followed by storage addresses.
  loom_spirv_binary_writer_t* function_prefix;
  // Retained globals referenced by this function.
  loom_spirv_entry_interface_t* entry_interface;
  // Unique emitted function ID for interface membership.
  uint32_t function_id;
  // Transitive implicit Workgroup use, updated when this function declares it.
  bool* uses_implicit_workgroup;
  // Source-scoped sink for unsupported storage and encoded-format limits.
  iree_diagnostic_emitter_t diagnostic_emitter;
  // Borrowed acceptance cell cleared by a structured rejection.
  bool* accepted;
} loom_spirv_module_storage_context_t;

// Captures one reservation during its owning emission traversal.
iree_status_t loom_spirv_module_storage_reserve(
    loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context, const loom_op_t* op);

// Materializes a low.storage.address as a typed Workgroup or Function array.
iree_status_t loom_spirv_module_storage_emit_address(
    loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context, const loom_op_t* op,
    loom_spirv_module_value_ref_t* out_value_ref);

// Completes Workgroup declarations after all typed uses are known. Explicit
// layout projects canonical byte offsets; implicit layout remains available
// only when no emitted operation requires stronger physical alignment.
iree_status_t loom_spirv_module_storage_finish(
    const loom_spirv_module_storage_t* storage,
    const loom_spirv_module_storage_context_t* context,
    loom_spirv_feature_bits_t* required_features);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
