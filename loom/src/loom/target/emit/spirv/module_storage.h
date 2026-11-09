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
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes a low.storage.address result as a typed Workgroup array pointer.
iree_status_t loom_spirv_module_workgroup_storage_emit_address(
    loom_spirv_module_value_table_t* value_table, const loom_op_t* op,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_module_value_ref_t* out_value_ref);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
