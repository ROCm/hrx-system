// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amdgpu/provider.h"

#include "loom/ir/module.h"
#include "loom/pass/builder.h"
#include "loom/target/arch/amdgpu/descriptors/low_registry.h"
#include "loom/target/arch/amdgpu/diagnostics/packet_diagnostics.h"
#include "loom/target/arch/amdgpu/legalization.h"
#include "loom/target/arch/amdgpu/low_asm_diagnostics.h"
#include "loom/target/arch/amdgpu/low_verify.h"
#include "loom/target/arch/amdgpu/lower/lower.h"
#include "loom/target/arch/amdgpu/math_policy.h"
#include "loom/target/arch/amdgpu/ops/ops.h"
#include "loom/target/arch/amdgpu/ops/registry.h"
#include "loom/target/arch/amdgpu/ops/target.h"
#include "loom/target/arch/amdgpu/pass_registry.h"
#include "loom/target/arch/amdgpu/profile.h"
#include "loom/target/materialization.h"

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

// Processor rows establish AMDGPU code-object refinement. Target-ID features
// then refine that relation without coupling policy to a particular feature or
// stepping name. Common indexed facts preserve structured representation,
// subgroup, and capacity requirements independently of the family relation.
static bool loom_amdgpu_provider_satisfies_requirement(
    loom_target_record_view_t effective_target,
    loom_target_record_view_t target_requirement) {
  loom_amdgpu_target_identity_t effective_identity = {0};
  loom_amdgpu_target_record_resolve_identity(effective_target.facts->target.op,
                                             &effective_identity);
  loom_amdgpu_target_identity_t requirement_identity = {0};
  loom_amdgpu_target_record_resolve_identity(
      target_requirement.facts->target.op, &requirement_identity);
  if (!loom_amdgpu_target_identity_satisfies_requirement(
          &effective_identity, &requirement_identity)) {
    return false;
  }

  const loom_target_bundle_storage_t* effective_storage =
      &effective_target.facts->storage;
  const loom_target_bundle_storage_t* requirement_storage =
      &target_requirement.facts->storage;

  // Processor refinement deliberately changes a generic descriptor-set
  // contract into the exact processor contract. An explicitly overridden
  // contract key remains a requirement instead of being silently discarded.
  const loom_attribute_t required_contract_set_key =
      loom_op_attrs(target_requirement.facts->target
                        .op)[loom_amdgpu_target_contract_set_key_ATTR_INDEX];
  if (!loom_attr_is_absent(required_contract_set_key) &&
      !iree_string_view_equal(effective_storage->config.contract_set_key,
                              requirement_storage->config.contract_set_key)) {
    return false;
  }

  // ABI and export facts belong to each function contract. Target
  // applicability preserves structural target constraints and requires all
  // authored target feature bits.
  return loom_target_snapshot_satisfies_requirement(
             &effective_storage->snapshot, &requirement_storage->snapshot) &&
         iree_all_bits_set(effective_storage->config.contract_feature_bits,
                           requirement_storage->config.contract_feature_bits);
}

static iree_string_view_t loom_amdgpu_provider_materialization_symbol_stem(
    const loom_target_profile_t* base_profile) {
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(base_profile);
  return profile != NULL && profile->identity.processor != NULL
             ? profile->identity.processor->name
             : iree_string_view_empty();
}

static bool loom_amdgpu_provider_record_matches_profile(
    const loom_module_t* module, const loom_op_t* target_op,
    const loom_target_profile_t* base_profile) {
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(base_profile);
  if (profile == NULL || profile->identity.processor == NULL ||
      loom_amdgpu_target_record_processor(target_op) !=
          profile->identity.processor) {
    return false;
  }

  loom_amdgpu_target_identity_t record_identity = {0};
  loom_amdgpu_target_record_resolve_identity(target_op, &record_identity);
  return loom_amdgpu_target_identity_equal(&record_identity,
                                           &profile->identity) &&
         loom_target_record_projection_matches_bundle(
             module, target_op, profile->base.target_bundle);
}

static iree_status_t loom_amdgpu_provider_build_profile_record(
    loom_builder_t* builder, const loom_target_profile_t* base_profile,
    loom_symbol_ref_t symbol, loom_location_id_t location,
    loom_op_t** out_target_op) {
  const loom_amdgpu_target_profile_t* profile =
      loom_amdgpu_target_profile_cast(base_profile);
  if (profile == NULL || profile->identity.processor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target materialization requires a complete AMDGPU profile");
  }
  return loom_amdgpu_target_record_build_for_profile(builder, profile, symbol,
                                                     location, out_target_op);
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
  return loom_pass_ir_build_run(builder, key, loom_named_attr_slice_empty(),
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
      contribution->builder, IREE_SV("target_op"), IREE_SV("amdgpu.target"),
      &attrs[0]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("codegen"), IREE_SV("low_native"),
      &attrs[1]));
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_build_string_attr(
      contribution->builder, IREE_SV("abi"), IREE_SV("hal_kernel"), &attrs[2]));

  loom_op_t* where_op = NULL;
  return loom_pass_ir_build_where(
      contribution->builder, IREE_SV("target"),
      loom_make_named_attr_slice(attrs, IREE_ARRAYSIZE(attrs)), build_body,
      NULL, &where_op);
}

const loom_target_provider_t loom_amdgpu_target_provider = {
    .profile_type = &loom_amdgpu_target_profile_type,
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
    .materialization =
        {
            .symbol_stem = loom_amdgpu_provider_materialization_symbol_stem,
            .record_matches_profile =
                loom_amdgpu_provider_record_matches_profile,
            .build_profile_record = loom_amdgpu_provider_build_profile_record,
        },
    .record_semantics =
        {
            .op_kind = LOOM_OP_AMDGPU_TARGET,
            .satisfies_requirement = loom_amdgpu_provider_satisfies_requirement,
        },
};

static const loom_target_provider_t* const kLoomAmdgpuTargetProviders[] = {
    &loom_amdgpu_target_provider,
};

const loom_target_provider_set_t loom_amdgpu_target_provider_set = {
    .providers = kLoomAmdgpuTargetProviders,
    .provider_count = IREE_ARRAYSIZE(kLoomAmdgpuTargetProviders),
};
