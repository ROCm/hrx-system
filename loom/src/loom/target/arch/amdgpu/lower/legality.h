// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Common AMDGPU source-to-low legality helpers.

#ifndef LOOM_TARGET_ARCH_AMDGPU_LOWER_LEGALITY_H_
#define LOOM_TARGET_ARCH_AMDGPU_LOWER_LEGALITY_H_

#include "loom/target/arch/amdgpu/refs/target_refs.h"
#include "loom/target/low_legality.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // target_key, export_name, config_key, function_name, and op_name.
  LOOM_AMDGPU_LOW_LEGALITY_CONTEXT_PARAM_COUNT = 5,
};

typedef struct loom_amdgpu_low_legality_descriptor_requirement_t {
  // Constraint key reported when this descriptor ref is missing.
  iree_string_view_t constraint_key;
  // Descriptor ref required by the source-to-low legality path.
  loom_amdgpu_descriptor_ref_t descriptor_ref;
} loom_amdgpu_low_legality_descriptor_requirement_t;

// Populates the common AMDGPU legality diagnostic context params.
void loom_amdgpu_low_legality_make_context_params(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_diagnostic_param_t* params);

// Emits ERR_AMDGPU_023 for a source-to-low legality constraint owned by the
// AMDGPU lowering provider.
iree_status_t loom_amdgpu_low_legality_reject(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    iree_string_view_t constraint_key);

// Verifies one descriptor requirement, rejecting with the matching constraint
// key when the descriptor ref is missing.
iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirement(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    loom_amdgpu_descriptor_ref_t descriptor_ref,
    iree_string_view_t constraint_key);

// Verifies an ordered descriptor requirement set, rejecting on the first
// missing descriptor ref with the matching constraint key.
iree_status_t loom_amdgpu_low_legality_verify_descriptor_requirements(
    loom_target_low_legality_context_t* context, const loom_op_t* op,
    const loom_amdgpu_low_legality_descriptor_requirement_t* requirements,
    iree_host_size_t requirement_count);

// Returns true when the target bundle belongs to an AMDGPU contract set.
bool loom_amdgpu_low_legality_bundle_is_amdgpu(
    const loom_target_bundle_t* bundle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_LOWER_LEGALITY_H_
