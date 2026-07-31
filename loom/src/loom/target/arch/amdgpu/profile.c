// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/profile.h"

#include "loom/target/arch/amdgpu/records/target_records.h"
#include "loom/target/arch/amdgpu/target_info.h"

static iree_status_t loom_amdgpu_target_profile_project_facts(
    const loom_target_profile_t* base_profile, iree_arena_allocator_t* arena,
    loom_target_facts_t* base_facts) {
  (void)arena;
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(base_profile);
  IREE_ASSERT(profile != NULL);
  IREE_ASSERT_LE(profile->identity.target->target_kind, UINT8_MAX);
  loom_amdgpu_target_facts_t* facts = (loom_amdgpu_target_facts_t*)base_facts;
  facts->base.selector = (uint8_t)profile->identity.target->target_kind;
  facts->identity = profile->identity;
  loom_amdgpu_target_properties_resolve(
      &facts->identity, &facts->base.storage.bundle, &facts->properties);
  return iree_ok_status();
}

const loom_target_profile_type_t loom_amdgpu_target_profile_type = {
    .name = IREE_SVL("amdgpu"),
    .fact_type = &loom_amdgpu_target_fact_type,
    .project_facts = loom_amdgpu_target_profile_project_facts,
};

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
          identity->target->descriptor_set_ordinal);
  if (target_bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU target '%.*s' has no Loom target bundle for descriptor "
        "set '%.*s'",
        (int)identity->target->name.size, identity->target->name.data,
        (int)identity->target->descriptor_set_key.size,
        identity->target->descriptor_set_key.data);
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
