// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/spirv/provider.h"

#include "loom/target/arch/spirv/descriptors/low_registry.h"
#include "loom/target/arch/spirv/low_verify.h"
#include "loom/target/arch/spirv/lower/lower.h"
#include "loom/target/arch/spirv/math_policy.h"
#include "loom/target/arch/spirv/ops/ops.h"
#include "loom/target/arch/spirv/ops/registry.h"
#include "loom/target/arch/spirv/profile.h"
#include "loom/target/materialization.h"

static const loom_low_verify_provider_t* const kLoomSpirvLowVerifyProviders[] =
    {
        &loom_spirv_low_verify_provider,
};

static bool loom_spirv_provider_profile_bundle_is_valid(
    const loom_target_bundle_t* bundle) {
  return bundle != NULL && bundle->snapshot != NULL &&
         bundle->export_plan != NULL && bundle->config != NULL &&
         bundle->snapshot->codegen_format == LOOM_TARGET_CODEGEN_FORMAT_SPIRV &&
         bundle->snapshot->artifact_format ==
             LOOM_TARGET_ARTIFACT_FORMAT_SPIRV_BINARY;
}

static iree_string_view_t loom_spirv_provider_materialization_symbol_stem(
    const loom_target_profile_t* base_profile) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_cast(base_profile);
  const loom_target_bundle_t* bundle =
      profile != NULL ? profile->base.target_bundle : NULL;
  if (!loom_spirv_provider_profile_bundle_is_valid(bundle)) {
    return iree_string_view_empty();
  }
  return iree_string_view_is_empty(bundle->name) ? IREE_SV("spirv_vulkan1_3")
                                                 : bundle->name;
}

static bool loom_spirv_provider_record_matches_effective_target(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_profile_t* base_profile,
    const loom_op_t* authored_target_op) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_cast(base_profile);
  if (profile == NULL ||
      loom_spirv_target_kind(target_op) != LOOM_SPIRV_TARGET_KIND_VULKAN1_3) {
    return false;
  }
  return loom_target_record_projection_matches_bundle(
      module, target_op, profile->base.target_bundle, authored_target_op);
}

static iree_status_t loom_spirv_provider_build_effective_target_record(
    loom_builder_t* builder, const loom_target_profile_t* base_profile,
    const loom_op_t* authored_target_op, loom_symbol_ref_t symbol,
    loom_location_id_t location, loom_op_t** out_target_op) {
  const loom_spirv_target_profile_t* profile =
      loom_spirv_target_profile_cast(base_profile);
  if (profile == NULL || !loom_spirv_provider_profile_bundle_is_valid(
                             profile->base.target_bundle)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "SPIR-V target materialization requires a complete SPIR-V profile");
  }
  return loom_target_record_projection_build(
      builder, LOOM_OP_SPIRV_TARGET, LOOM_SPIRV_TARGET_KIND_VULKAN1_3, symbol,
      profile->base.target_bundle, authored_target_op,
      /*extension_attrs=*/NULL,
      /*extension_attr_count=*/0, location, out_target_op);
}

const loom_target_provider_t loom_spirv_target_provider = {
    .profile_type = &loom_spirv_target_profile_type,
    .register_context = loom_spirv_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_spirv_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_spirv_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_spirv_math_policy_registry_initialize,
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomSpirvLowVerifyProviders),
            .values = kLoomSpirvLowVerifyProviders,
        },
    .materialization =
        {
            .op_kind = LOOM_OP_SPIRV_TARGET,
            .symbol_stem = loom_spirv_provider_materialization_symbol_stem,
            .record_matches_effective_target =
                loom_spirv_provider_record_matches_effective_target,
            .build_effective_target_record =
                loom_spirv_provider_build_effective_target_record,
        },
};

static const loom_target_provider_t* const kLoomSpirvTargetProviders[] = {
    &loom_spirv_target_provider,
};

const loom_target_provider_set_t loom_spirv_target_provider_set = {
    .providers = kLoomSpirvTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomSpirvTargetProviders),
};
