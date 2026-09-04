// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/facts.h"
#include "loom/target/arch/amd/xdna/aie2p/legalization.h"
#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/registry.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/target.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/pass.h"
#include "loom/target/arch/amd/xdna/aie2p/profile.h"
#include "loom/target/arch/amd/xdna/device/profile.h"

typedef struct loom_aie2p_profile_selection_storage_t {
  // Structured AIE2P target profile.
  loom_aie2p_target_profile_t profile;
} loom_aie2p_profile_selection_storage_t;

static iree_status_t loom_aie2p_target_provider_select_profile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  (void)provider;
  const loom_xdna_device_profile_t* device_profile =
      loom_xdna_device_profile_lookup(selector);
  if (device_profile == NULL ||
      loom_xdna_device_profile_array_family(device_profile)->architecture !=
          LOOM_XDNA_ARCHITECTURE_AIE2P) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "unsupported AIE2P target profile '%.*s'",
                            (int)selector.size, selector.data);
  }

  loom_aie2p_profile_selection_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_aie2p_profile_selection_storage_t){0};
  iree_status_t status =
      loom_aie2p_target_profile_initialize(device_profile, &storage->profile);
  if (iree_status_is_ok(status)) {
    *out_selection = (loom_target_profile_selection_t){
        .profile = &storage->profile.base,
        .selector = iree_make_cstring_view(device_profile->key),
        .storage = storage,
    };
  } else {
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static void loom_aie2p_target_provider_release_profile_selection(
    const loom_target_provider_t* provider,
    loom_target_profile_selection_t* selection) {
  (void)provider;
  iree_allocator_free(selection->allocator, selection->storage);
}

static const loom_low_verify_provider_t* const kAie2pLowVerifyProviders[] = {
    &loom_aie2p_low_verify_provider,
};

static const loom_target_legalizer_provider_t* const
    kAie2pLegalizerProviders[] = {
        &loom_aie2p_target_legalizer_provider_storage,
};

static iree_status_t loom_aie2p_provider_build_string_attr(
    loom_builder_t* builder, iree_string_view_t name, iree_string_view_t value,
    loom_named_attr_t* out_attr) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, name, &name_id));
  loom_string_id_t value_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(builder->module, value, &value_id));
  *out_attr = (loom_named_attr_t){
      .name_id = name_id,
      .value = loom_attr_string(value_id),
  };
  return iree_ok_status();
}

static iree_status_t loom_aie2p_provider_build_pipeline_lower_run(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, IREE_SV("aie2p-lower-pipeline"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_aie2p_provider_build_pipeline_target_where(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_named_attr_t contract_attr = {0};
  IREE_RETURN_IF_ERROR(loom_aie2p_provider_build_string_attr(
      builder, IREE_SV("contract"), IREE_SV("amd.xdna.aie2p.array"),
      &contract_attr));
  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS, IREE_SV("target"),
      loom_make_named_attr_slice(&contract_attr, 1),
      loom_aie2p_provider_build_pipeline_lower_run, NULL, &where_op);
}

static iree_status_t loom_aie2p_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase != LOOM_TARGET_PIPELINE_PHASE_SOURCE_TO_LOW) {
    return iree_ok_status();
  }
  loom_op_t* for_op = NULL;
  return loom_pass_ir_build_for(contribution->builder, LOOM_PASS_ANCHOR_FUNC,
                                loom_aie2p_provider_build_pipeline_target_where,
                                NULL, &for_op);
}

const loom_target_provider_t loom_aie2p_target_provider = {
    .profile_type = &loom_aie2p_target_profile_type,
    .materialize_definition = loom_aie2p_target_materialize_definition,
    .register_context = loom_aie2p_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_aie2p_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_aie2p_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_aie2p_math_policy_registry_initialize,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLegalizerProviders),
            .values = kAie2pLegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLowVerifyProviders),
            .values = kAie2pLowVerifyProviders,
        },
    .pass_registry = &loom_aie2p_pipeline_pass_registry,
    .contribute_pipeline = loom_aie2p_provider_contribute_pipeline,
    .select_profile = loom_aie2p_target_provider_select_profile,
    .release_profile_selection =
        loom_aie2p_target_provider_release_profile_selection,
    .target_fact_type = &loom_aie2p_target_fact_type,
};

static const loom_target_provider_t* const kAie2pTargetProviders[] = {
    &loom_aie2p_target_provider,
};

const loom_target_provider_set_t loom_aie2p_target_provider_set = {
    .providers = kAie2pTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pTargetProviders),
};
