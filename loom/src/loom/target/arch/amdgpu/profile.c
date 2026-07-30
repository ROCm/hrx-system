// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/profile.h"

#include "loom/target/arch/amdgpu/records/target_records.h"

const loom_target_profile_type_t loom_amdgpu_target_profile_type = {
    .name = IREE_SVL("amdgpu"),
};

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

static iree_status_t loom_amdgpu_target_profile_normalize_feature(
    const loom_amdgpu_processor_info_t* processor, iree_string_view_t name,
    loom_amdgpu_target_id_feature_support_bit_t feature,
    loom_amdgpu_target_feature_state_t default_state,
    loom_amdgpu_target_feature_state_t requested_state,
    loom_amdgpu_target_feature_state_t* out_state) {
  if (requested_state == LOOM_AMDGPU_TARGET_FEATURE_ANY) {
    *out_state = default_state;
    return iree_ok_status();
  }
  if (requested_state != LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED &&
      requested_state != LOOM_AMDGPU_TARGET_FEATURE_OFF &&
      requested_state != LOOM_AMDGPU_TARGET_FEATURE_ON) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target feature '%.*s' has invalid state %u",
                            (int)name.size, name.data, requested_state);
  }
  const bool supported =
      loom_amdgpu_processor_supports_target_id_features(processor, feature);
  if (supported && requested_state == LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' supports target feature '%.*s'",
        (int)processor->name.size, processor->name.data, (int)name.size,
        name.data);
  }
  if (!supported && requested_state != LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "AMDGPU processor '%.*s' does not support target feature '%.*s'",
        (int)processor->name.size, processor->name.data, (int)name.size,
        name.data);
  }
  *out_state = requested_state;
  return iree_ok_status();
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

iree_status_t loom_amdgpu_target_profile_initialize(
    const loom_amdgpu_target_identity_t* identity,
    loom_amdgpu_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(identity);
  IREE_ASSERT_ARGUMENT(identity->target);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_amdgpu_target_profile_t){0};

  const loom_amdgpu_processor_info_t* processor =
      loom_amdgpu_target_info_target_processor(identity->target);
  IREE_ASSERT(processor != NULL);
  loom_amdgpu_target_identity_t normalized_identity = {0};
  loom_amdgpu_target_identity_initialize(identity->target,
                                         &normalized_identity);
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_normalize_feature(
      processor, IREE_SV("sramecc"),
      LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_SRAMECC,
      normalized_identity.amdhsa_features.sramecc,
      identity->amdhsa_features.sramecc,
      &normalized_identity.amdhsa_features.sramecc));
  IREE_RETURN_IF_ERROR(loom_amdgpu_target_profile_normalize_feature(
      processor, IREE_SV("xnack"), LOOM_AMDGPU_TARGET_ID_FEATURE_SUPPORT_XNACK,
      normalized_identity.amdhsa_features.xnack,
      identity->amdhsa_features.xnack,
      &normalized_identity.amdhsa_features.xnack));
  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          processor->properties.descriptor_set.ordinal);
  if (target_bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU processor '%.*s' has no Loom target bundle for descriptor "
        "set '%.*s'",
        (int)processor->name.size, processor->name.data,
        (int)processor->properties.descriptor_set.key.size,
        processor->properties.descriptor_set.key.data);
  }

  loom_amdgpu_target_properties_t properties = {0};
  loom_amdgpu_target_properties_resolve(&normalized_identity, target_bundle,
                                        &properties);
  *out_profile = (loom_amdgpu_target_profile_t){
      .base =
          {
              .type = &loom_amdgpu_target_profile_type,
              .target_bundle = target_bundle,
          },
      .identity = normalized_identity,
      .properties = properties,
  };
  return iree_ok_status();
}
