// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable AMDGPU target facts.

#ifndef LOOM_TARGET_ARCH_AMDGPU_FACTS_H_
#define LOOM_TARGET_ARCH_AMDGPU_FACTS_H_

#include "loom/target/arch/amdgpu/target_info_defs.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_resolved_target_t loom_low_resolved_target_t;

// Complete structured identity retained by AMDGPU facts and profiles.
typedef struct loom_amdgpu_target_identity_t {
  // Exact, generic, or overlay target selected for this identity.
  const loom_amdgpu_target_info_t* target;

  // Normalized AMDHSA target-ID feature states.
  loom_amdgpu_amdhsa_feature_states_t amdhsa_features;
} loom_amdgpu_target_identity_t;

// Immutable compiler-semantic properties resolved for one AMDGPU target.
//
// Processor identity and compatibility remain outside this view. Static
// processor properties provide target capabilities, scheduling, ABI, and ELF
// policy. |common| carries the target-neutral capability and limit projection,
// including function-local overrides. AMDHSA feature states retain target-ID
// and code-object policy without reparsing an external string.
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

typedef struct loom_amdgpu_target_facts_t {
  // Target-neutral facts shared by all target families.
  loom_target_facts_t base;

  // Canonical target and normalized AMDHSA feature identity.
  loom_amdgpu_target_identity_t identity;

  // Compiler-semantic properties resolved from |identity| and |base|.
  loom_amdgpu_target_properties_t properties;

  // True when subgroup_size was supplied as an explicit semantic input.
  bool subgroup_size_explicit;

  // True when contract_set_key was supplied as an explicit semantic input.
  bool contract_set_key_explicit;
} loom_amdgpu_target_facts_t;

// Static fact type used by AMDGPU target projection and structured profiles.
extern const loom_target_fact_type_t loom_amdgpu_target_fact_type;

// Returns the AMDGPU processor selected by a resolved low target, or NULL.
const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_low_resolved_target_t* target);

// Returns the compiler-semantic processor properties selected by a resolved
// low target, or NULL.
const loom_amdgpu_processor_properties_t*
loom_amdgpu_target_processor_properties_from_resolved_target(
    const loom_low_resolved_target_t* target);

// Initializes the normalized default identity for |target|.
//
// Supported AMDHSA modes remain unconstrained and unsupported modes are
// explicit.
void loom_amdgpu_target_identity_initialize(
    const loom_amdgpu_target_info_t* target,
    loom_amdgpu_target_identity_t* out_identity);

// Initializes one identity and applies packed positive/negative target-ID
// feature assertions. |feature_words| contains |feature_word_count| positive
// words followed by the same number of negative words.
void loom_amdgpu_target_identity_initialize_with_features(
    const loom_amdgpu_target_info_t* target, const uint64_t* feature_words,
    uint16_t feature_word_count, loom_amdgpu_target_identity_t* out_identity);

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

// Returns |facts| as AMDGPU facts, or NULL for another target family.
static inline const loom_amdgpu_target_facts_t* loom_amdgpu_target_facts_cast(
    const loom_target_facts_t* facts) {
  return facts != NULL && facts->fact_type == &loom_amdgpu_target_fact_type
             ? (const loom_amdgpu_target_facts_t*)facts
             : NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMDGPU_FACTS_H_
