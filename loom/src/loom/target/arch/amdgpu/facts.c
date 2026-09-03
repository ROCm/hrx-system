// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/facts.h"

#include "loom/codegen/low/target_binding.h"
#include "loom/target/arch/amdgpu/target_info.h"

const loom_amdgpu_processor_info_t*
loom_amdgpu_target_processor_from_resolved_target(
    const loom_low_resolved_target_t* target) {
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(target->target_facts);
  return target_facts != NULL ? loom_amdgpu_target_info_target_processor(
                                    target_facts->identity.target)
                              : NULL;
}

const loom_amdgpu_processor_properties_t*
loom_amdgpu_target_processor_properties_from_resolved_target(
    const loom_low_resolved_target_t* target) {
  const loom_amdgpu_target_facts_t* target_facts =
      loom_amdgpu_target_facts_cast(target->target_facts);
  return target_facts != NULL ? target_facts->properties.processor : NULL;
}

void loom_amdgpu_target_identity_initialize(
    const loom_amdgpu_target_info_t* target,
    loom_amdgpu_target_identity_t* out_identity) {
  IREE_ASSERT_ARGUMENT(target);
  IREE_ASSERT_ARGUMENT(out_identity);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(target);
  IREE_ASSERT(processor != NULL);
  *out_identity = (loom_amdgpu_target_identity_t){.target = target};
  loom_amdgpu_amdhsa_feature_states_initialize(processor,
                                               &out_identity->amdhsa_features);
}

void loom_amdgpu_target_identity_initialize_with_features(
    const loom_amdgpu_target_info_t* target, const uint64_t* feature_words,
    uint16_t feature_word_count, loom_amdgpu_target_identity_t* out_identity) {
  loom_amdgpu_target_identity_initialize(target, out_identity);
  for (uint8_t stable_value = 0; stable_value < 32; ++stable_value) {
    const loom_amdgpu_target_id_feature_support_bit_t support_bit =
        (loom_amdgpu_target_id_feature_support_bit_t)(UINT32_C(1)
                                                      << stable_value);
    if (!iree_any_bit_set(LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS,
                          support_bit)) {
      continue;
    }
    const iree_host_size_t word_index = stable_value / 64u;
    const uint64_t bit = UINT64_C(1) << (stable_value % 64u);
    const bool positive = word_index < feature_word_count &&
                          iree_any_bit_set(feature_words[word_index], bit);
    const bool negative =
        word_index < feature_word_count &&
        iree_any_bit_set(feature_words[feature_word_count + word_index], bit);
    if (!positive && !negative) continue;
    IREE_ASSERT(!(positive && negative));
    loom_amdgpu_target_feature_state_t* state =
        loom_amdgpu_amdhsa_feature_state_select(&out_identity->amdhsa_features,
                                                support_bit);
    IREE_ASSERT(state != NULL);
    *state = positive ? LOOM_AMDGPU_TARGET_FEATURE_ON
                      : LOOM_AMDGPU_TARGET_FEATURE_OFF;
  }
}

bool loom_amdgpu_target_identity_equal(
    const loom_amdgpu_target_identity_t* lhs,
    const loom_amdgpu_target_identity_t* rhs) {
  if (lhs == NULL || rhs == NULL || lhs->target == NULL ||
      lhs->target != rhs->target) {
    return false;
  }

  loom_amdgpu_target_id_feature_support_flags_t remaining_features =
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS;
  while (remaining_features != 0) {
    const loom_amdgpu_target_id_feature_support_bit_t feature =
        (loom_amdgpu_target_id_feature_support_bit_t)(remaining_features &
                                                      (0u -
                                                       remaining_features));
    remaining_features &= ~feature;
    if (loom_amdgpu_amdhsa_feature_state_query(&lhs->amdhsa_features,
                                               feature) !=
        loom_amdgpu_amdhsa_feature_state_query(&rhs->amdhsa_features,
                                               feature)) {
      return false;
    }
  }
  return true;
}

bool loom_amdgpu_target_identity_satisfies_requirement(
    const loom_amdgpu_target_identity_t* effective,
    const loom_amdgpu_target_identity_t* requirement) {
  if (effective == NULL || effective->target == NULL || requirement == NULL ||
      requirement->target == NULL ||
      !loom_amdgpu_target_satisfies_code_object_requirement(
          effective->target, requirement->target)) {
    return false;
  }

  const loom_amdgpu_processor_info_t* requirement_processor =
      loom_amdgpu_target_info_target_processor(requirement->target);
  loom_amdgpu_target_id_feature_support_flags_t remaining_features =
      requirement_processor->target_id.supported_features;
  IREE_ASSERT(iree_all_bits_set(
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_KNOWN_FLAGS, remaining_features));
  while (remaining_features != 0) {
    const loom_amdgpu_target_id_feature_support_bit_t feature =
        (loom_amdgpu_target_id_feature_support_bit_t)(remaining_features &
                                                      (0u -
                                                       remaining_features));
    remaining_features &= ~feature;
    const loom_amdgpu_target_feature_state_t required_state =
        loom_amdgpu_amdhsa_feature_state_query(&requirement->amdhsa_features,
                                               feature);
    if (required_state == LOOM_AMDGPU_TARGET_FEATURE_ANY) {
      continue;
    }
    if (loom_amdgpu_amdhsa_feature_state_query(&effective->amdhsa_features,
                                               feature) != required_state) {
      return false;
    }
  }
  return true;
}

void loom_amdgpu_target_properties_resolve(
    const loom_amdgpu_target_identity_t* identity,
    const loom_target_bundle_t* common,
    loom_amdgpu_target_properties_t* out_properties) {
  IREE_ASSERT_ARGUMENT(identity);
  IREE_ASSERT_ARGUMENT(identity->target);
  IREE_ASSERT_ARGUMENT(common);
  IREE_ASSERT_ARGUMENT(out_properties);
  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(identity->target);
  IREE_ASSERT(processor != NULL);
  const loom_amdgpu_processor_properties_t* processor_properties =
      &processor->properties;
  *out_properties = (loom_amdgpu_target_properties_t){
      .target = identity->target,
      .processor = processor_properties,
      .common = common,
      .amdhsa_features = identity->amdhsa_features,
      .instruction_constraints = identity->target->instruction_constraints,
      .lds_bank_service_model_set_ordinal =
          identity->target->lds_bank_service_model_set_ordinal,
      .kernel_metadata_extensions =
          identity->target->kernel_metadata_extensions,
  };
}

static bool loom_amdgpu_target_facts_satisfy_identity_requirement(
    const loom_target_facts_t* effective_base,
    const loom_target_facts_t* requirement_base) {
  const loom_amdgpu_target_facts_t* effective =
      (const loom_amdgpu_target_facts_t*)effective_base;
  const loom_amdgpu_target_facts_t* requirement =
      (const loom_amdgpu_target_facts_t*)requirement_base;
  return loom_amdgpu_target_identity_satisfies_requirement(
      &effective->identity, &requirement->identity);
}

static bool loom_amdgpu_target_facts_satisfy_specialization_requirement(
    const loom_target_facts_t* effective_base,
    const loom_target_facts_t* requirement_base) {
  const loom_amdgpu_target_facts_t* effective =
      (const loom_amdgpu_target_facts_t*)effective_base;
  const loom_amdgpu_target_facts_t* requirement =
      (const loom_amdgpu_target_facts_t*)requirement_base;
  if (!loom_amdgpu_target_identity_satisfies_requirement(
          &effective->identity, &requirement->identity)) {
    return false;
  }

  // A processor may support more than one wavefront size. An omitted authored
  // choice uses the row default without constraining later specialization. An
  // explicit choice remains a requirement even when it equals that default.
  loom_target_snapshot_t effective_snapshot = effective->base.storage.snapshot;
  loom_target_snapshot_t requirement_snapshot =
      requirement->base.storage.snapshot;
  if (requirement->subgroup_size_explicit) {
    const uint32_t required_subgroup_size = requirement_snapshot.subgroup_size;
    if (effective->subgroup_size_explicit &&
        effective_snapshot.subgroup_size != required_subgroup_size) {
      return false;
    }
    const loom_amdgpu_processor_properties_t* processor =
        effective->properties.processor;
    IREE_ASSERT(processor != NULL);
    if (!loom_amdgpu_processor_properties_support_wavefront_size(
            processor, required_subgroup_size)) {
      return false;
    }
    effective_snapshot.subgroup_size = required_subgroup_size;
  } else {
    requirement_snapshot.subgroup_size = 0;
  }

  // Processor refinement deliberately replaces a generic descriptor contract
  // with the exact processor contract. Only an explicit override constrains
  // the effective contract key.
  if (requirement->contract_set_key_explicit &&
      !iree_string_view_equal(
          effective->base.storage.config.contract_set_key,
          requirement->base.storage.config.contract_set_key)) {
    return false;
  }

  // Product fields constrain specialization only after either side selects
  // them explicitly. Machine limits, memory spaces, and required target
  // feature bits always participate in the target relation.
  return loom_target_facts_product_contract_satisfies_specialization_requirement(
             effective_base, requirement_base) &&
         loom_target_snapshot_satisfies_specialization_requirement(
             &effective_snapshot, &requirement_snapshot) &&
         iree_all_bits_set(
             effective->base.storage.config.contract_feature_bits,
             requirement->base.storage.config.contract_feature_bits);
}

static void loom_amdgpu_target_facts_rebind(loom_target_facts_t* base_facts) {
  loom_amdgpu_target_facts_t* facts = (loom_amdgpu_target_facts_t*)base_facts;
  loom_amdgpu_target_properties_resolve(
      &facts->identity, &facts->base.storage.bundle, &facts->properties);
  facts->subgroup_size_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_SUBGROUP_SIZE);
  facts->contract_set_key_explicit = loom_target_facts_field_is_explicit(
      &facts->base, LOOM_TARGET_FACT_FIELD_CONTRACT_SET_KEY);
}

static iree_string_view_t loom_amdgpu_target_facts_identity_name(
    const loom_target_facts_t* base_facts) {
  const loom_amdgpu_target_facts_t* facts =
      (const loom_amdgpu_target_facts_t*)base_facts;
  return facts->identity.target != NULL ? facts->identity.target->name
                                        : iree_string_view_empty();
}

const loom_target_fact_type_t loom_amdgpu_target_fact_type = {
    .name = IREE_SVL("amdgpu"),
    .storage_size = sizeof(loom_amdgpu_target_facts_t),
    .satisfies_identity_requirement =
        loom_amdgpu_target_facts_satisfy_identity_requirement,
    .satisfies_specialization_requirement =
        loom_amdgpu_target_facts_satisfy_specialization_requirement,
    .rebind = loom_amdgpu_target_facts_rebind,
    .identity_name = loom_amdgpu_target_facts_identity_name,
};
