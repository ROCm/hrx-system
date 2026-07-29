// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/profile.h"

const loom_target_profile_type_t loom_spirv_target_profile_type = {
    .name = IREE_SVL("spirv"),
};

void loom_spirv_target_profile_initialize(
    const loom_target_bundle_t* target_bundle,
    const loom_spirv_cooperative_property_set_t* cooperative_properties,
    loom_spirv_target_profile_t* out_profile) {
  IREE_ASSERT_ARGUMENT(out_profile);
  *out_profile = (loom_spirv_target_profile_t){
      .base =
          {
              .type = &loom_spirv_target_profile_type,
              .target_bundle = target_bundle,
          },
      .cooperative_properties = cooperative_properties,
  };
}
