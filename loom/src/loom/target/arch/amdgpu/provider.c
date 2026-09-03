// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amdgpu/artifact_key.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/diagnostics/packet_diagnostics.h"
#include "loom/target/arch/amdgpu/legalization.h"
#include "loom/target/arch/amdgpu/low_asm_diagnostics.h"
#include "loom/target/arch/amdgpu/low_verify.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/math_policy.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/pass_registry.h"
#include "loom/target/arch/amdgpu/profile.h"

typedef struct loom_amdgpu_profile_selection_storage_t {
  // Structured AMDGPU target profile.
  loom_amdgpu_target_profile_t profile;

  // Owned canonical selector storage.
  iree_string_builder_t selector_builder;
} loom_amdgpu_profile_selection_storage_t;

static iree_status_t loom_amdgpu_target_provider_select_profile(
    const loom_target_provider_t* provider, iree_string_view_t selector,
    iree_allocator_t allocator,
    loom_target_profile_selection_t* out_selection) {
  (void)provider;
  loom_amdgpu_target_identity_t identity = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_artifact_key_parse(selector, &identity));

  loom_amdgpu_profile_selection_storage_t* storage = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(allocator, sizeof(*storage), (void**)&storage));
  *storage = (loom_amdgpu_profile_selection_storage_t){0};
  iree_string_builder_initialize(allocator, &storage->selector_builder);

  iree_status_t status =
      loom_amdgpu_target_profile_initialize(&identity, &storage->profile);
  if (iree_status_is_ok(status)) {
    status = loom_amdgpu_artifact_key_append(&storage->profile.identity,
                                             &storage->selector_builder);
  }
  if (iree_status_is_ok(status)) {
    *out_selection = (loom_target_profile_selection_t){
        .profile = &storage->profile.base,
        .selector = iree_string_builder_view(&storage->selector_builder),
        .storage = storage,
    };
  } else {
    iree_string_builder_deinitialize(&storage->selector_builder);
    iree_allocator_free(allocator, storage);
  }
  return status;
}

static void loom_amdgpu_target_provider_release_profile_selection(
    const loom_target_provider_t* provider,
    loom_target_profile_selection_t* selection) {
  (void)provider;
  loom_amdgpu_profile_selection_storage_t* storage =
      (loom_amdgpu_profile_selection_storage_t*)selection->storage;
  iree_string_builder_deinitialize(&storage->selector_builder);
  iree_allocator_free(selection->allocator, storage);
}

static const loom_target_low_legality_provider_t*
    kLoomAmdgpuLowLegalityProviders[] = {
        &loom_amdgpu_low_legality_provider_storage,
};

static const loom_target_legalizer_provider_t* kLoomAmdgpuLegalizerProviders[] =
    {
        &loom_amdgpu_target_legalizer_provider_storage,
};

static const loom_target_low_packet_diagnostic_provider_t*
    kLoomAmdgpuLowPacketDiagnosticProviders[] = {
        &loom_amdgpu_low_packet_diagnostic_provider_storage,
};

static const loom_target_low_asm_diagnostic_provider_t*
    kLoomAmdgpuLowAsmDiagnosticProviders[] = {
        &loom_amdgpu_low_asm_diagnostic_provider,
};

static const loom_low_verify_provider_t* kLoomAmdgpuLowVerifyProviders[] = {
    &loom_amdgpu_low_verify_provider,
};

static loom_target_low_call_policy_t loom_amdgpu_select_low_call_policy(
    const loom_resolved_target_t* resolved_target) {
  (void)resolved_target;
  return LOOM_TARGET_LOW_CALL_POLICY_REQUIRE_INLINE;
}

static iree_status_t loom_amdgpu_provider_build_string_attr(
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

static iree_status_t loom_amdgpu_provider_build_run_pass(
    loom_builder_t* builder, iree_string_view_t key) {
  loom_op_t* run_op = NULL;
  return loom_pass_ir_build_run(builder, 0, key, loom_named_attr_slice_empty(),
                                &run_op);
}

static iree_status_t loom_amdgpu_provider_build_hal_buffer_descriptors_pass(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_amdgpu_provider_build_run_pass(
      builder, IREE_SV("amdgpu-materialize-hal-buffer-descriptors"));
}

static iree_status_t loom_amdgpu_provider_build_hal_kernel_abi_pass(
    loom_builder_t* builder, void* user_data) {
  (void)user_data;
  return loom_amdgpu_provider_build_run_pass(
      builder, IREE_SV("amdgpu-materialize-hal-kernel-abi"));
}

static iree_status_t loom_amdgpu_provider_contribute_pipeline(
    const loom_target_pipeline_contribution_t* contribution) {
  loom_pass_ir_body_build_fn_t build_body = NULL;
  if (contribution->phase ==
      LOOM_TARGET_PIPELINE_PHASE_SOURCE_LOW_ARTIFACT_PREPARATION) {
    build_body = loom_amdgpu_provider_build_hal_buffer_descriptors_pass;
  } else if (contribution->phase ==
             LOOM_TARGET_PIPELINE_PHASE_TARGET_LOW_MATERIALIZATION) {
    build_body = loom_amdgpu_provider_build_hal_kernel_abi_pass;
  } else {
    return iree_ok_status();
  }

  loom_named_attr_t attrs[3] = {0};
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("family"), IREE_SV("amdgpu"), &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("codegen"), IREE_SV("low_native"),
      &attrs[1]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("abi"), IREE_SV("hal_kernel"), &attrs[2]));

  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      contribution->builder, LOOM_PASS_WHERE_BUILD_FLAG_HAS_ATTRS,
      IREE_SV("target"),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), build_body,
      NULL, &where_op);
}

const loom_target_provider_t loom_amdgpu_target_provider = {
    .profile_type = &loom_amdgpu_target_profile_type,
    .materialize_definition = loom_amdgpu_target_materialize_definition,
    .select_low_call_policy = loom_amdgpu_select_low_call_policy,
    .register_context = loom_amdgpu_ops_register_dialect,
    .initialize_low_descriptor_registry =
        loom_amdgpu_low_descriptor_registry_initialize,
    .initialize_low_lower_policy_registry =
        loom_amdgpu_low_lower_policy_registry_initialize,
    .initialize_math_policy_registry =
        loom_amdgpu_math_policy_registry_initialize,
    .low_legality_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowLegalityProviders),
            .values = kLoomAmdgpuLowLegalityProviders,
        },
    .legalizer_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLegalizerProviders),
            .values = kLoomAmdgpuLegalizerProviders,
        },
    .low_packet_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowPacketDiagnosticProviders),
            .values = kLoomAmdgpuLowPacketDiagnosticProviders,
        },
    .low_asm_diagnostic_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowAsmDiagnosticProviders),
            .values = kLoomAmdgpuLowAsmDiagnosticProviders,
        },
    .low_verify_provider_list =
        {
            .count = IREE_ARRAYSIZE(kLoomAmdgpuLowVerifyProviders),
            .values = kLoomAmdgpuLowVerifyProviders,
        },
    .pass_registry = &loom_amdgpu_pass_registry,
    .contribute_pipeline = loom_amdgpu_provider_contribute_pipeline,
    .select_profile = loom_amdgpu_target_provider_select_profile,
    .release_profile_selection =
        loom_amdgpu_target_provider_release_profile_selection,
};

static const loom_target_provider_t* const kLoomAmdgpuTargetProviders[] = {
    &loom_amdgpu_target_provider,
};

const loom_target_provider_set_t loom_amdgpu_target_provider_set = {
    .providers = kLoomAmdgpuTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomAmdgpuTargetProviders),
};
