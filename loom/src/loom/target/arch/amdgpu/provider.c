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
#include "loom/target/arch/amdgpu/records/target_records.h"

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

// Processor rows establish AMDGPU code-object refinement. Exact stepping
// qualifiers remain exact requirements, while generic processor records carry
// no stepping requirement. Common indexed facts then preserve structured
// representation, subgroup, and capacity requirements independently of that
// family relation.
static bool loom_amdgpu_provider_satisfies_requirement(
    loom_target_record_view_t effective_target,
    loom_target_record_view_t target_requirement) {
  if (!loom_amdgpu_processor_satisfies_code_object_requirement(
          loom_amdgpu_target_record_processor(
              effective_target.facts->target.op),
          loom_amdgpu_target_record_processor(
              target_requirement.facts->target.op))) {
    return false;
  }

  const loom_amdgpu_gfx1250_revision_t required_revision =
      loom_amdgpu_target_record_effective_gfx1250_revision(
          target_requirement.facts->target.op);
  if (required_revision != LOOM_AMDGPU_GFX1250_REVISION_UNSPECIFIED &&
      loom_amdgpu_target_record_effective_gfx1250_revision(
          effective_target.facts->target.op) != required_revision) {
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

static iree_status_t loom_amdgpu_provider_validate_materialized_target_symbol(
    const loom_module_t* module, iree_string_view_t symbol_name,
    const loom_amdgpu_target_profile_t* profile, loom_symbol_ref_t target_ref,
    bool* out_reusable) {
  *out_reusable = false;

  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_ok_status();
  }
  if (!loom_amdgpu_target_isa(symbol->defining_op)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target materialization symbol '@%.*s' already names a "
        "non-AMDGPU target op",
        (int)symbol_name.size, symbol_name.data);
  }

  const iree_string_view_t existing_processor =
      loom_amdgpu_target_record_processor_name(symbol->defining_op);
  if (!iree_string_view_equal(existing_processor,
                              profile->amdhsa.processor->name)) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target materialization symbol '@%.*s' selects processor "
        "'%.*s', but the selected profile requires '%.*s'",
        (int)symbol_name.size, symbol_name.data, (int)existing_processor.size,
        existing_processor.data, (int)profile->amdhsa.processor->name.size,
        profile->amdhsa.processor->name.data);
  }

  const loom_amdgpu_gfx1250_revision_t existing_revision =
      loom_amdgpu_target_record_effective_gfx1250_revision(symbol->defining_op);
  if (existing_revision != profile->amdhsa.gfx1250_revision) {
    return iree_make_status(
        IREE_STATUS_ALREADY_EXISTS,
        "AMDGPU target materialization symbol '@%.*s' selects gfx1250 "
        "revision %u, but the selected profile requires revision %u",
        (int)symbol_name.size, symbol_name.data, (unsigned)existing_revision,
        (unsigned)profile->amdhsa.gfx1250_revision);
  }

  *out_reusable = true;
  return iree_ok_status();
}

static iree_status_t loom_amdgpu_provider_resolve_profile_target_ref(
    loom_module_t* module, const loom_amdgpu_target_profile_t* profile,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  if (module == NULL || module->body == NULL ||
      module->body->block_count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "AMDGPU target materialization requires a module "
                            "with a body block");
  }
  if (profile == NULL || profile->amdhsa.processor == NULL) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target materialization requires a selected processor");
  }
  const loom_amdgpu_processor_info_t* processor = profile->amdhsa.processor;
  if (iree_string_view_is_empty(processor->name)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "AMDGPU target materialization selected a processor row with no "
        "processor name");
  }

  loom_string_id_t symbol_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, processor->name, &symbol_name_id));
  uint16_t symbol_id = loom_module_find_symbol(module, symbol_name_id);
  if (symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(
        loom_module_add_symbol(module, symbol_name_id, &symbol_id));
  }
  *out_target_ref = (loom_symbol_ref_t){.module_id = 0, .symbol_id = symbol_id};

  bool reusable = false;
  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_validate_materialized_target_symbol(
      module, processor->name, profile, *out_target_ref, &reusable));
  if (reusable) {
    return iree_ok_status();
  }

  loom_block_t* module_block = loom_module_block(module);
  loom_builder_t builder = {0};
  loom_builder_initialize(module, &module->arena, module_block, &builder);
  if (module_block->first_op != NULL) {
    loom_builder_set_before(&builder, module_block->first_op);
  }

  loom_op_t* target_op = NULL;
  return loom_amdgpu_target_record_build_for_profile(
      &builder, profile, *out_target_ref, LOOM_LOCATION_UNKNOWN, &target_op);
}

static iree_status_t loom_amdgpu_provider_materialize_selection(
    const loom_target_provider_t* provider,
    const loom_target_selection_materialization_request_t* request,
    loom_symbol_ref_t* out_target_ref) {
  (void)provider;
  *out_target_ref = loom_symbol_ref_null();
  const loom_amdgpu_target_profile_t* target_profile =
      (const loom_amdgpu_target_profile_t*)request->target_selection.profile;

  IREE_RETURN_IF_ERROR(loom_amdgpu_provider_resolve_profile_target_ref(
      request->module, target_profile, out_target_ref));
  return iree_ok_status();
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
    .materialize_selection = loom_amdgpu_provider_materialize_selection,
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
