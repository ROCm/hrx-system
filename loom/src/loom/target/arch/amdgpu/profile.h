// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Structured AMDGPU target profile.

#ifndef LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_
#define LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_

#include "iree/base/api.h"
#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/profile.h"

#ifdef __cplusplus
extern "C" {
#endif

// AMDGPU family profile identity.
extern const loom_target_profile_type_t loom_amdgpu_target_profile_type;

// Complete structured identity retained by an AMDGPU target profile.
typedef struct loom_amdgpu_target_identity_t {
  // Exact, generic, or overlay target selected for this profile.
  const loom_amdgpu_target_info_t* target;

  // Normalized AMDHSA target-ID feature states.
  loom_amdgpu_amdhsa_feature_states_t amdhsa_features;
} loom_amdgpu_target_identity_t;

// Immutable compiler-semantic properties resolved for one AMDGPU target.
//
// Processor identity and compatibility remain outside this view. Static
// processor properties provide target capabilities, scheduling, ABI, and ELF
// policy. |common| carries the target-neutral capability and limit projection,
// including function-local overrides when this view is resolved from indexed
// IR. AMDHSA feature states retain target-ID and code-object policy without
// reparsing an external string.
typedef struct loom_amdgpu_target_properties_t {
  // Canonical target selecting all target-local semantic overlays.
  const loom_amdgpu_target_info_t* target;

  // Static compiler properties selected by the exact or generic processor.
  const loom_amdgpu_processor_properties_t* processor;

  // Common target capability, ABI, and numeric-limit projection.
  const loom_target_bundle_t* common;

  // Normalized AMDHSA target-ID feature states.
  loom_amdgpu_amdhsa_feature_states_t amdhsa_features;

  // Active instruction restrictions requiring legalization or hazard handling.
  loom_amdgpu_instruction_constraint_bits_t instruction_constraints;

  // Immutable LDS bank-service model set available for target analysis.
  loom_amdgpu_lds_bank_service_model_set_ordinal_t
      lds_bank_service_model_set_ordinal;

  // Target-required AMDHSA kernel metadata extensions.
  loom_amdgpu_metadata_string_property_set_t kernel_metadata_extensions;
} loom_amdgpu_target_properties_t;

// Immutable AMDGPU target facts used by specialization and emission.
typedef struct loom_amdgpu_target_profile_t {
  // Target-neutral family identity and bundle projection.
  loom_target_profile_t base;

  // Structured target and AMDHSA feature identity.
  loom_amdgpu_target_identity_t identity;

  // Compiler-semantic projection resolved from |identity|.
  loom_amdgpu_target_properties_t properties;
} loom_amdgpu_target_profile_t;

// Initializes the normalized default identity for |target|.
//
// Supported AMDHSA modes remain unconstrained and unsupported modes are
// explicit.
void loom_amdgpu_target_identity_initialize(
    const loom_amdgpu_target_info_t* target,
    loom_amdgpu_target_identity_t* out_identity);

// Returns whether two identities select the same canonical target and every
// known target-ID feature state.
bool loom_amdgpu_target_identity_equal(
    const loom_amdgpu_target_identity_t* lhs,
    const loom_amdgpu_target_identity_t* rhs);

// Returns whether |effective| refines every processor and target-ID feature
// requirement carried by |requirement|.
bool loom_amdgpu_target_identity_satisfies_requirement(
    const loom_amdgpu_target_identity_t* effective,
    const loom_amdgpu_target_identity_t* requirement);

// Resolves a normalized AMDHSA identity and common target projection into
// compiler-semantic AMDGPU properties.
void loom_amdgpu_target_properties_resolve(
    const loom_amdgpu_target_identity_t* identity,
    const loom_target_bundle_t* common,
    loom_amdgpu_target_properties_t* out_properties);

// Initializes an AMDGPU profile and its target-neutral bundle projection.
iree_status_t loom_amdgpu_target_profile_initialize(
    const loom_amdgpu_target_identity_t* identity,
    loom_amdgpu_target_profile_t* out_profile);

// Returns true when |properties| support native AMDHSA HSACO emission.
static inline bool loom_amdgpu_target_properties_support_hsaco(
    const loom_amdgpu_target_properties_t* properties) {
  return properties != NULL &&
         loom_amdgpu_processor_properties_support_hsaco(properties->processor);
}

// Returns true when |properties| activate |constraint|.
static inline bool loom_amdgpu_target_properties_have_instruction_constraint(
    const loom_amdgpu_target_properties_t* properties,
    loom_amdgpu_instruction_constraint_bit_t constraint) {
  return properties != NULL &&
         iree_any_bit_set(properties->instruction_constraints, constraint);
}

// Returns |profile| as an AMDGPU profile, or NULL for another family.
static inline const loom_amdgpu_target_profile_t*
loom_amdgpu_target_profile_cast(const loom_target_profile_t* profile) {
  return loom_target_profile_has_type(profile, &loom_amdgpu_target_profile_type)
             ? (const loom_amdgpu_target_profile_t*)profile
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_PROFILE_H_
