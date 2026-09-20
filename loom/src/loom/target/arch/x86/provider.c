// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/provider.h"

#include "loom/pass/builder.h"
#include "loom/target/arch/x86/descriptors/low_registry.h"
#include "loom/target/arch/x86/legalization.h"
#include "loom/target/arch/x86/low_verify.h"
#include "loom/target/arch/x86/lower/lower.h"
#include "loom/target/arch/x86/math_policy.h"
#include "loom/target/arch/x86/ops/ops.h"
#include "loom/target/arch/x86/ops/registry.h"
#include "loom/target/arch/x86/pass_registry.h"
#include "loom/target/emit/native/x86/object.h"

static const loom_target_legalizer_provider_t* kLoomX86LegalizerProviders[] = {
    &loom_x86_target_legalizer_provider_storage,
};

static const loom_low_verify_provider_t* const kLoomX86LowVerifyProviders[] = {
    &loom_x86_low_verify_provider,
};

static const loom_target_emitter_t* const kLoomX86Emitters[] = {
    &loom_x86_elf_object_emitter,
};

static iree_status_t loom_x86_provider_build_sysv_abi_materialization(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, IREE_SV("x86-materialize-sysv-abi"),
                                loom_named_attr_slice_empty(), &run_op);
}

static iree_status_t loom_x86_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  if (contribution->phase !=
      LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_SYMBOL_MATERIALIZATION) {
    return iree_ok_status();
  }
  return loom_x86_provider_build_sysv_abi_materialization(contribution->builder,
                                                          NULL);
}

const loom_target_provider_t loom_x86_target_provider = {
    .select_low_call_policy = loom_target_select_low_call_policy_direct,
    .emitter_list =
        {
            .values = kLoomX86Emitters,
            .count = IREE_ARRAYSIZE(kLoomX86Emitters),
        },
    .canonical_module_emitter = &loom_x86_elf_object_emitter,
    .register_context = loom_x86_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_x86_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_x86_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry = loom_x86_math_policy_registry_initialize,
    .pass_registry = &loom_x86_pass_registry,
    .contribute_pipeline = loom_x86_provider_contribute_pipeline,
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomX86LegalizerProviders),
            .values = kLoomX86LegalizerProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomX86LowVerifyProviders),
            .values = kLoomX86LowVerifyProviders,
        },
    .target_fact_type = &loom_x86_target_fact_type,
};

static const loom_target_provider_t* const kLoomX86TargetProviders[] = {
    &loom_x86_target_provider,
};

const loom_target_provider_set_t loom_x86_target_provider_set = {
    .providers = kLoomX86TargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomX86TargetProviders),
};
