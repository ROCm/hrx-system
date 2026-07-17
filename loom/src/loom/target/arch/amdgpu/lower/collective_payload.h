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
#include "loom/target/low_legality.h"

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

typedef uint32_t loom_amdgpu_workgroup_collective_shape_flags_t;

enum loom_amdgpu_workgroup_collective_shape_flag_bits_e {
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_MULTI_WAVE = 1u << 0,
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_PARTIAL_TAIL = 1u << 1,
};

typedef enum loom_amdgpu_workgroup_collective_shape_failure_e {
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_NONE = 0,
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WORKGROUP_SIZE = 1,
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_WAVE_COUNT = 2,
  LOOM_AMDGPU_WORKGROUP_COLLECTIVE_SHAPE_FAILURE_SCRATCH_BYTE_LENGTH = 3,
} loom_amdgpu_workgroup_collective_shape_failure_t;

typedef struct loom_amdgpu_workgroup_collective_shape_t {
  // Exact flattened workgroup size selected by launch configuration.
  uint32_t flat_workgroup_size;
  // Number of cross-wave partitions spanned by the workgroup.
  uint32_t wave_count;
  // Shape properties derived from the workgroup and partition sizes.
  loom_amdgpu_workgroup_collective_shape_flags_t flags;
} loom_amdgpu_workgroup_collective_shape_t;

// Resolves the bounded cross-wave workgroup shape used by workgroup
// collective selection and legality. Returns false with |out_failure| set when
// source IR or target facts do not prove an encodable shape.
bool loom_amdgpu_collective_resolve_workgroup_shape(
    const loom_module_t* module, loom_func_like_t function,
    const loom_target_bundle_t* bundle, uint32_t partition_lane_count,
    uint32_t register_count,
    loom_amdgpu_workgroup_collective_shape_t* out_shape,
    loom_amdgpu_workgroup_collective_shape_failure_t* out_failure);

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

// Resolves descriptors required for cross-wave LDS exchange in workgroup
// collective lowering.
iree_status_t loom_amdgpu_collective_resolve_cross_wave_descriptors(
    loom_low_lower_context_t* context,
    loom_amdgpu_workgroup_collective_cross_wave_descriptors_t* out_descriptors,
    bool* out_present);

// Emits the target-selected barrier sequence for cross-wave LDS publication.
iree_status_t loom_amdgpu_collective_emit_cross_wave_barrier(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    const loom_amdgpu_workgroup_collective_cross_wave_descriptors_t*
        descriptors);

// Verifies descriptors required for cross-wave LDS exchange in workgroup
// collective lowering.
iree_status_t loom_amdgpu_collective_verify_cross_wave_descriptor_requirements(
    loom_target_low_legality_context_t* context, const loom_op_t* op);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_COLLECTIVE_PAYLOAD_H_
