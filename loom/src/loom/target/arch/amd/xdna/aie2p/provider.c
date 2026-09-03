// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/low_registry.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/artifact_emitter.h"
#include "loom/target/arch/amd/xdna/aie2p/low_verify.h"
#include "loom/target/arch/amd/xdna/aie2p/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/math_policy.h"
#include "loom/target/arch/amd/xdna/aie2p/ops/registry.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/pass.h"

static const loom_target_emitter_t* const kAie2pTargetEmitters[] = {
    &loom_aie2p_xdna_emitter,
};

static const loom_low_verify_provider_t* const kAie2pLowVerifyProviders[] = {
    &loom_aie2p_low_verify_provider,
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
    .register_context = loom_aie2p_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_aie2p_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_aie2p_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_aie2p_math_policy_registry_initialize,
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kAie2pLowVerifyProviders),
            .values = kAie2pLowVerifyProviders,
        },
    .emitter_list =
        {
            .values = kAie2pTargetEmitters,
            .count = IREE_ARRAYSIZE(kAie2pTargetEmitters),
        },
    .pass_registry = &loom_aie2p_pipeline_pass_registry,
    .contribute_pipeline = loom_aie2p_provider_contribute_pipeline,
};

static const loom_target_provider_t* const kAie2pTargetProviders[] = {
    &loom_aie2p_target_provider,
};

const loom_target_provider_set_t loom_aie2p_target_provider_set = {
    .providers = kAie2pTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kAie2pTargetProviders),
};
