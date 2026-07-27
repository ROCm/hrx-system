// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V logical register-class helpers.
//
// These helpers keep target-owned register classes mapped to compact semantic
// value types without exposing SPIR-V register-class IDs to core Loom code.

#ifndef LOOM_TARGET_ARCH_SPIRV_REGISTERS_H_
#define LOOM_TARGET_ARCH_SPIRV_REGISTERS_H_

#include "iree/base/api.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/target/arch/spirv/scalar_types.h"
#include "loom/target/arch/spirv/value_types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the typed Workgroup pointer register class for |scalar_type|, or
// LOOM_LOW_REG_CLASS_NONE when the scalar is not represented by the logical
// core descriptor set.
uint16_t loom_spirv_ptr_workgroup_reg_class_id(
    loom_spirv_scalar_type_t scalar_type);

// Returns the typed Workgroup array pointer register class for |scalar_type|,
// or LOOM_LOW_REG_CLASS_NONE when the scalar is not represented by the logical
// core descriptor set.
uint16_t loom_spirv_ptr_workgroup_array_reg_class_id(
    loom_spirv_scalar_type_t scalar_type);

// Maps non-payload SPIR-V register classes to exact semantic value types.
//
// `spirv.id` intentionally has no row here: callers must use ABI metadata or
// descriptor packet rows to recover its payload type.
bool loom_spirv_value_type_from_reg_class_id(
    uint16_t reg_class_id, loom_spirv_value_type_t* out_value_type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_REGISTERS_H_
