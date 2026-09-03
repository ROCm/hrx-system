// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/legalization.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/math_policy.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/target/arch/spirv/ops/target.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/arch/spirv/records/target_records.h"

typedef struct loom_spirv_profile_selection_storage_t {
  // Structured SPIR-V target profile.
  loom_spirv_target_profile_t profile;
} loom_spirv_profile_selection_storage_t;

static iree_status_t loom_spirv_target_provider_select_profile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  (void)provider;
  if (!iree_string_view_equal(selector, IREE_SV("vulkan1.3+bda"))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "SPIR-V target selector '%.*s' is not available; expected "
        "'vulkan1.3+bda'",
        (int)selector.size, selector.data);
  }

  loom_spirv_profile_selection_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_spirv_profile_selection_storage_t){0};
  loom_spirv_target_profile_initialize(
      &loom_spirv_target_profile_bundle_vulkan1_3,
      /*cooperative_properties=*/NULL, &storage->profile);
  *out_selection = (loom_target_profile_selection_t){
      .profile = &storage->profile.base,
      .selector = IREE_SV("vulkan1.3+bda"),
      .storage = storage,
  };
  return iree_ok_status();
}

static void loom_spirv_target_provider_release_profile_selection(
    const loom_target_provider_t* provider,
    loom_target_profile_selection_t* selection) {
  (void)provider;
  iree_allocator_free(selection->allocator, selection->storage);
}

static const loom_low_verify_provider_t* const kLoomSpirvLowVerifyProviders[] =
    {
        &loom_spirv_low_verify_provider,
};

static const loom_target_legalizer_provider_t* const
    kLoomSpirvLegalizerProviders[] = {
        &loom_spirv_target_legalizer_provider_storage,
};

const loom_target_provider_t loom_spirv_target_provider = {
    .profile_type = &loom_spirv_target_profile_type,
    .materialize_definition = loom_spirv_target_materialize_definition,
    .register_context = loom_spirv_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_spirv_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_spirv_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_spirv_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLegalizerProviders),
            .values = kLoomSpirvLegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLowVerifyProviders),
            .values = kLoomSpirvLowVerifyProviders,
        },
    .select_profile = loom_spirv_target_provider_select_profile,
    .release_profile_selection =
        loom_spirv_target_provider_release_profile_selection,
};

static const loom_target_provider_t* const kLoomSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
};

const loom_target_provider_set_t loom_spirv_target_provider_set = {
    .providers = kLoomSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomSpirvTargetProviders),
};
