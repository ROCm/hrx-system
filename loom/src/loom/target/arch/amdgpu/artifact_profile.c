// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/artifact_profile.h"

#include "loom/target/arch/amdgpu/target_info.h"

static loom_amdgpu_target_feature_state_t
loom_amdgpu_artifact_target_profile_map_feature_state(
    iree_hal_amdgpu_target_feature_state_t state) {
  switch (state) {
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ANY:
      return LOOM_AMDGPU_TARGET_FEATURE_ANY;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_UNSUPPORTED:
      return LOOM_AMDGPU_TARGET_FEATURE_UNSUPPORTED;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_OFF:
      return LOOM_AMDGPU_TARGET_FEATURE_OFF;
    case IREE_HAL_AMDGPU_TARGET_FEATURE_STATE_ON:
      return LOOM_AMDGPU_TARGET_FEATURE_ON;
    default:
      IREE_ASSERT_UNREACHABLE("unknown AMDGPU target feature state");
      return LOOM_AMDGPU_TARGET_FEATURE_ANY;
  }
}

iree_status_t loom_amdgpu_artifact_target_profile_parse(
    iree_string_view_t target_key, loom_amdgpu_target_profile_t* out_profile,
    iree_hal_amdgpu_target_kind_t* out_target_kind) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_amdgpu_target_profile_t){0};

  iree_hal_amdgpu_target_identity_t hal_identity = {0};
  IREE_RETURN_IF_ERROR(iree_hal_amdgpu_target_identity_parse_artifact_key(
      target_key, &hal_identity));

  const loom_amdgpu_target_info_t* target = NULL;
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_info_lookup_target(hal_identity.target, &target));
  const loom_amdgpu_target_identity_t identity = {
      .target = target,
      .amdhsa_features =
          {
              .sramecc = loom_amdgpu_artifact_target_profile_map_feature_state(
                  hal_identity.amdhsa_features.sramecc),
              .xnack = loom_amdgpu_artifact_target_profile_map_feature_state(
                  hal_identity.amdhsa_features.xnack),
          },
  };
  loom_amdgpu_target_profile_t profile = {0};
  IREE_RETURN_IF_ERROR(
      loom_amdgpu_target_profile_initialize(&identity, &profile));
  *out_profile = profile;
  if (out_target_kind != NULL) {
    *out_target_kind = hal_identity.kind;
  }
  return iree_ok_status();
}
