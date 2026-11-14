// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V typed logical storage materialization.
//
// Low storage is byte storage. Workgroup and Function storage become typed
// OpVariable arrays. Workgroup declarations enter the module declaration
// section; Function declarations are collected for the entry-block prefix.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/binary_writer.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/emit/spirv/module_types.h"
#include "loom/target/emit/spirv/module_values.h"

#ifdef __cplusplus
extern "C" {
#endif

// Materializes a low.storage.address as a typed Workgroup or Function array.
iree_status_t loom_spirv_module_storage_emit_address(
    loom_spirv_module_value_table_t* value_table, const loom_op_t* op,
    loom_spirv_type_context_t* type_context,
    loom_spirv_module_builder_t* builder,
    loom_spirv_binary_writer_t* function_variables,
    loom_spirv_module_value_ref_t* out_value_ref);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_STORAGE_H_
