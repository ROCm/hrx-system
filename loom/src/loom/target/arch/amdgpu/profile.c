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

iree_status_t loom_amdgpu_target_profile_initialize(
    const loom_amdgpu_amdhsa_profile_facts_t* amdhsa,
    loom_amdgpu_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(amdhsa);
  IREE_ASSERT_ARGUMENT(amdhsa->processor);
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_amdgpu_target_profile_t){0};

  const bool is_gfx1250 =
      iree_string_view_equal(amdhsa->processor->name, IREE_SV("gfx1250"));
  if (is_gfx1250 &&
      amdhsa->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_A0 &&
      amdhsa->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_B0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "gfx1250 target profile requires an A0 or B0 silicon revision");
  }
  if (!is_gfx1250 &&
      amdhsa->gfx1250_revision != LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "gfx1250 silicon revision cannot qualify processor '%.*s'",
        (int)amdhsa->processor->name.size, amdhsa->processor->name.data);
  }

  const loom_target_bundle_t* target_bundle =
      loom_amdgpu_target_bundle_for_descriptor_set(
          amdhsa->processor->descriptor_set.ordinal);
  if (target_bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "AMDGPU processor '%.*s' has no Loom target bundle for descriptor "
        "set '%.*s'",
        (int)amdhsa->processor->name.size, amdhsa->processor->name.data,
        (int)amdhsa->processor->descriptor_set.key.size,
        amdhsa->processor->descriptor_set.key.data);
  }

  *out_profile = (loom_amdgpu_target_profile_t){
      .base =
          {
              .type = &loom_amdgpu_target_profile_type,
              .target_bundle = target_bundle,
          },
      .amdhsa = *amdhsa,
  };
  return iree_ok_status();
}
