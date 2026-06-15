// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V module-scope storage materialization.
//
// Low storage is target-independent byte storage. SPIR-V logical Workgroup
// storage is typed module-scope OpVariable storage, so this layer owns the
// narrow bridge from low.storage.reserve/address to typed Workgroup variables
// and typed SSA value refs.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/target/arch/spirv/value_types.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_spirv_module_workgroup_storage_entry_t
    loom_spirv_module_workgroup_storage_entry_t;

typedef struct loom_spirv_module_workgroup_storage_state_t {
  // Workgroup storage entries indexed by function-local value ordinal.
  loom_spirv_module_workgroup_storage_entry_t* entries;
  // Number of entries addressable through entries.
  iree_host_size_t entry_count;
} loom_spirv_module_workgroup_storage_state_t;

// Initializes function-local Workgroup storage state over |value_domain|.
iree_status_t loom_spirv_module_workgroup_storage_initialize(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* out_state,
    iree_arena_allocator_t* scratch_arena);

// Records a low.storage.reserve Workgroup storage declaration.
iree_status_t loom_spirv_module_workgroup_storage_emit_reserve(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* state, const loom_op_t* op);

// Materializes a low.storage.address result as a typed Workgroup array pointer.
iree_status_t loom_spirv_module_workgroup_storage_emit_address(
    const loom_local_value_domain_t* value_domain,
    loom_spirv_module_workgroup_storage_state_t* state, const loom_op_t* op,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_value_ref_t* out_value_ref);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
