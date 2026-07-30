// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU target op interpretation and typed fact projection.

#ifndef LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/target/facts.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/arch/amdgpu/target_info.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable facts projected once from a verified amdgpu.target witness.
typedef struct loom_amdgpu_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // Canonical target and normalized AMDHSA feature identity.
  loom_amdgpu_target_identity_t identity;

  // Compiler-semantic properties resolved from |identity| and |base|.
  loom_amdgpu_target_properties_t properties;

  // True when subgroup_size was explicitly present in the authored target.
  bool subgroup_size_authored;

  // True when contract_set_key was explicitly present in the authored target.
  bool contract_set_key_authored;
} loom_amdgpu_target_facts_t;

// Static fact type used by generated amdgpu.target metadata.
extern const loom_target_fact_type_t loom_amdgpu_target_fact_type;

// Returns |facts| as AMDGPU facts, or NULL for another target family.
static inline const loom_amdgpu_target_facts_t* loom_amdgpu_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_amdgpu_target_fact_type
             ? (const loom_amdgpu_target_facts_t*)facts
             : NULL;
}

// Returns the canonical AMDGPU target name selected by |target_op|, or empty.
iree_string_view_t loom_amdgpu_target_record_target_name(
    const loom_op_t* target_op);

// Returns the AMDGPU target row selected by |target_op|, or NULL.
const loom_amdgpu_target_info_t* loom_amdgpu_target_record_target(
    const loom_op_t* target_op);

// Returns the AMDGPU processor name selected by |target_op|, or empty.
iree_string_view_t loom_amdgpu_target_record_processor_name(
    const loom_op_t* target_op);

// Returns the AMDGPU processor row selected by |target_op|, or NULL.
const loom_amdgpu_processor_info_t* loom_amdgpu_target_record_processor(
    const loom_op_t* target_op);

// Resolves the structured AMDGPU identity carried by |target_op|.
//
// Unspecified target-ID features resolve from generated processor defaults.
// Callers consume this same identity shape as live target profiles.
void loom_amdgpu_target_record_resolve_identity(
    const loom_op_t* target_op, loom_amdgpu_target_identity_t* out_identity);

// Resolves the compiler-semantic AMDGPU properties carried by |target_op|.
//
// |common| is the indexed target bundle for the record and may contain
// authored target-neutral overrides.
void loom_amdgpu_target_record_resolve_properties(
    const loom_op_t* target_op, const loom_target_bundle_t* common,
    loom_amdgpu_target_properties_t* out_properties);

// Builds a target record carrying every durable fact from |profile| and the
// function-local fields preserved from |authored_target_op|.
iree_status_t loom_amdgpu_target_record_build_for_profile(
    loom_builder_t* builder, const loom_amdgpu_target_profile_t* profile,
    const loom_op_t* authored_target_op, loom_symbol_ref_t symbol,
    loom_location_id_t location, loom_op_t** out_target_op);

iree_status_t loom_amdgpu_target_record_verify(
    const loom_module_t* module, const loom_op_t* op,
    iree_diagnostic_emitter_t emitter);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_OPS_TARGET_H_
