// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU collective payload shape and register binding support.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PAYLOAD_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PAYLOAD_H_

#include <stdbool.h>
#include <stdint.h>

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amdgpu/lower/plan.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when the source value can be represented as one or more native
// 32-bit VGPR payload registers for AMDGPU collective packets.
bool loom_amdgpu_collective_payload_is_supported(
    const loom_module_t* module, loom_value_id_t value_id,
    loom_amdgpu_subgroup_payload_kind_t* out_kind,
    uint32_t* out_register_count);

// Returns true when the payload kind represents 32-bit integer data.
bool loom_amdgpu_collective_payload_is_integer(
    loom_amdgpu_subgroup_payload_kind_t payload_kind);

// Returns true when the payload kind represents 32-bit floating-point data.
bool loom_amdgpu_collective_payload_is_float(
    loom_amdgpu_subgroup_payload_kind_t payload_kind);

// Looks up or materializes a source payload value in low IR form.
iree_status_t loom_amdgpu_collective_lookup_payload(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t value, loom_amdgpu_subgroup_payload_kind_t payload_kind,
    loom_value_id_t* out_low_value);

// Extracts one 32-bit payload register from a scalar or vector low value.
iree_status_t loom_amdgpu_collective_payload_register(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    uint32_t register_count, loom_value_id_t low_value, uint32_t register_index,
    loom_type_t lane_type, loom_value_id_t* out_register);

// Binds scalar or concatenated vector payload registers to a source result.
iree_status_t loom_amdgpu_collective_bind_payload_result(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_value_id_t source_result, uint32_t register_count,
    const loom_value_id_t* result_registers);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PAYLOAD_H_
