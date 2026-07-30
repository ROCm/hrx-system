// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

#include "iree/base/string_builder.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

bool loom_target_satisfies_requirement(
    const loom_target_environment_t* environment,
    loom_target_record_view_t effective_target,
    loom_target_record_view_t target_requirement) {
  if (!loom_target_record_view_is_valid(effective_target) ||
      !loom_target_record_view_is_valid(target_requirement)) {
    return false;
  }
  if (effective_target.module == target_requirement.module &&
      effective_target.facts->target.op ==
          target_requirement.facts->target.op) {
    return true;
  }
  if (environment == NULL || effective_target.facts->target.op->kind !=
                                 target_requirement.facts->target.op->kind) {
    return false;
  }

  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->record_semantics.op_kind !=
        effective_target.facts->target.op->kind) {
      continue;
    }
    return provider->record_semantics.satisfies_requirement(effective_target,
                                                            target_requirement);
  }
  return false;
}

static bool loom_target_limit_satisfies(uint64_t effective_limit,
                                        uint64_t required_limit) {
  return required_limit == 0 || effective_limit >= required_limit;
}

bool loom_target_snapshot_satisfies_requirement(
    const loom_target_snapshot_t* effective_snapshot,
    const loom_target_snapshot_t* target_requirement) {
  IREE_ASSERT_ARGUMENT(effective_snapshot);
  IREE_ASSERT_ARGUMENT(target_requirement);
  return effective_snapshot->codegen_format ==
             target_requirement->codegen_format &&
         effective_snapshot->artifact_format ==
             target_requirement->artifact_format &&
         effective_snapshot->default_pointer_bitwidth ==
             target_requirement->default_pointer_bitwidth &&
         effective_snapshot->index_bitwidth ==
             target_requirement->index_bitwidth &&
         effective_snapshot->offset_bitwidth ==
             target_requirement->offset_bitwidth &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.x,
             target_requirement->max_workgroup_size.x) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.y,
             target_requirement->max_workgroup_size.y) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_size.z,
             target_requirement->max_workgroup_size.z) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_flat_workgroup_size,
             target_requirement->max_flat_workgroup_size) &&
         (target_requirement->subgroup_size == 0 ||
          effective_snapshot->subgroup_size ==
              target_requirement->subgroup_size) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.x,
                                     target_requirement->max_grid_size.x) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.y,
                                     target_requirement->max_grid_size.y) &&
         loom_target_limit_satisfies(effective_snapshot->max_grid_size.z,
                                     target_requirement->max_grid_size.z) &&
         loom_target_limit_satisfies(effective_snapshot->max_flat_grid_size,
                                     target_requirement->max_flat_grid_size) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.x,
             target_requirement->max_workgroup_count.x) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.y,
             target_requirement->max_workgroup_count.y) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_count.z,
             target_requirement->max_workgroup_count.z) &&
         loom_target_limit_satisfies(
             effective_snapshot->max_workgroup_storage_bytes,
             target_requirement->max_workgroup_storage_bytes) &&
         effective_snapshot->memory_spaces.generic ==
             target_requirement->memory_spaces.generic &&
         effective_snapshot->memory_spaces.global ==
             target_requirement->memory_spaces.global &&
         effective_snapshot->memory_spaces.workgroup ==
             target_requirement->memory_spaces.workgroup &&
         effective_snapshot->memory_spaces.constant ==
             target_requirement->memory_spaces.constant &&
         effective_snapshot->memory_spaces.private_memory ==
             target_requirement->memory_spaces.private_memory &&
         effective_snapshot->memory_spaces.host ==
             target_requirement->memory_spaces.host &&
         effective_snapshot->memory_spaces.descriptor ==
             target_requirement->memory_spaces.descriptor;
}

static iree_status_t loom_target_provider_set_validate_contracts(
    const loom_target_provider_set_t* provider_set) {
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    const loom_target_provider_record_semantics_t semantics =
        provider->record_semantics;
    const bool has_op_kind = semantics.op_kind != LOOM_OP_KIND_UNKNOWN;
    const bool has_relation = semantics.satisfies_requirement != NULL;
    if (has_op_kind != has_relation) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target provider %" PRIhsz
          " record semantics must define both an op kind and a satisfaction "
          "relation",
          i);
    }
    const loom_target_provider_materialization_t materialization =
        provider->materialization;
    const bool has_symbol_stem = materialization.symbol_stem != NULL;
    const bool has_record_match =
        materialization.record_matches_effective_target != NULL;
    const bool has_record_builder =
        materialization.build_effective_target_record != NULL;
    const bool has_any_materialization =
        has_symbol_stem || has_record_match || has_record_builder;
    const bool has_complete_materialization =
        has_symbol_stem && has_record_match && has_record_builder;
    if (has_any_materialization != has_complete_materialization) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target provider %" PRIhsz
          " must define every structured materialization hook",
          i);
    }
    if (has_complete_materialization &&
        (provider->profile_type == NULL || !has_op_kind)) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target provider %" PRIhsz
          " structured materialization requires a profile type and record "
          "semantics",
          i);
    }
    if (!has_op_kind) {
      continue;
    }
    for (iree_host_size_t j = 0; j < i; ++j) {
      if (provider_set->providers[j]->record_semantics.op_kind ==
          semantics.op_kind) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target op kind 0x%04X has semantics providers at indices %" PRIhsz
            " and %" PRIhsz,
            (unsigned)semantics.op_kind, j, i);
      }
    }
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_low_descriptor_registry(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (provider->initialize_low_descriptor_registry == NULL) {
    return iree_ok_status();
  }
  loom_target_low_descriptor_registry_t provider_registry = {0};
  provider->initialize_low_descriptor_registry(&provider_registry);
  return loom_target_low_descriptor_registry_append_to_tables(
      &provider_registry, environment->descriptor_set_providers,
      IREE_ARRAYSIZE(environment->descriptor_set_providers),
      &environment->descriptor_set_provider_count);
}

static iree_status_t loom_target_environment_append_low_lower_policy_registry(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (provider->initialize_low_lower_policy_registry == NULL) {
    return iree_ok_status();
  }
  loom_low_lower_policy_registry_t provider_registry = {0};
  provider->initialize_low_lower_policy_registry(&provider_registry);
  if (environment->low_lower_policy_entry_count +
          provider_registry.entry_count >
      IREE_ARRAYSIZE(environment->low_lower_policy_entries)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target source-to-low policy capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider_registry.entry_count; ++i) {
    environment->low_lower_policy_entries
        [environment->low_lower_policy_entry_count++] =
        provider_registry.entries[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_math_policy_registry(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (provider->initialize_math_policy_registry == NULL) {
    return iree_ok_status();
  }
  loom_target_math_policy_registry_t provider_registry = {0};
  provider->initialize_math_policy_registry(&provider_registry);
  if (environment->math_policy_entry_count + provider_registry.entry_count >
      IREE_ARRAYSIZE(environment->math_policy_entries)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target math policy capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider_registry.entry_count; ++i) {
    environment->math_policy_entries[environment->math_policy_entry_count++] =
        provider_registry.entries[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_low_legality_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_legality_provider_count +
          provider->low_legality_provider_list.count >
      IREE_ARRAYSIZE(environment->low_legality_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target low legality provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->low_legality_provider_list.count;
       ++i) {
    environment
        ->low_legality_providers[environment->low_legality_provider_count++] =
        provider->low_legality_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_legalizer_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->legalizer_provider_count +
          provider->legalizer_provider_list.count >
      IREE_ARRAYSIZE(environment->legalizer_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target legalizer provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->legalizer_provider_list.count;
       ++i) {
    environment->legalizer_providers[environment->legalizer_provider_count++] =
        provider->legalizer_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_environment_append_low_packet_diagnostic_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_packet_diagnostic_provider_count +
          provider->low_packet_diagnostic_provider_list.count >
      IREE_ARRAYSIZE(environment->low_packet_diagnostic_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target low packet diagnostic provider capacity exceeded");
  }
  for (iree_host_size_t i = 0;
       i < provider->low_packet_diagnostic_provider_list.count; ++i) {
    environment->low_packet_diagnostic_providers
        [environment->low_packet_diagnostic_provider_count++] =
        provider->low_packet_diagnostic_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t
loom_target_environment_append_low_asm_diagnostic_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_asm_diagnostic_provider_count +
          provider->low_asm_diagnostic_provider_list.count >
      IREE_ARRAYSIZE(environment->low_asm_diagnostic_providers)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target low asm diagnostic provider capacity exceeded");
  }
  for (iree_host_size_t i = 0;
       i < provider->low_asm_diagnostic_provider_list.count; ++i) {
    environment->low_asm_diagnostic_providers
        [environment->low_asm_diagnostic_provider_count++] =
        provider->low_asm_diagnostic_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_low_verify_providers(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->low_verify_provider_count +
          provider->low_verify_provider_list.count >
      IREE_ARRAYSIZE(environment->low_verify_providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target low verifier provider capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->low_verify_provider_list.count;
       ++i) {
    const iree_host_size_t index = environment->low_verify_provider_count++;
    environment->low_verify_providers[index] =
        provider->low_verify_provider_list.values[i];
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_append_emitters(
    loom_target_environment_t* environment,
    const loom_target_provider_t* provider) {
  if (environment->emitter_count + provider->emitter_list.count >
      IREE_ARRAYSIZE(environment->emitters)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target emitter capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->emitter_list.count; ++i) {
    const iree_host_size_t index = environment->emitter_count++;
    environment->emitters[index] = provider->emitter_list.values[i];
  }
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize(
    const loom_target_provider_set_t* provider_set,
    loom_target_environment_t* out_environment) {
  IREE_ASSERT_ARGUMENT(provider_set);
  IREE_ASSERT_ARGUMENT(out_environment);
  *out_environment = (loom_target_environment_t){
      .provider_set = provider_set,
  };
  IREE_RETURN_IF_ERROR(
      loom_target_provider_set_validate_contracts(provider_set));

  const loom_pass_registry_t*
      pass_registries[LOOM_TARGET_PROVIDER_PASS_REGISTRY_CAPACITY] = {0};
  iree_host_size_t pass_registry_count = 0;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->pass_registry != NULL) {
      if (pass_registry_count >= IREE_ARRAYSIZE(pass_registries)) {
        return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                "target pass registry capacity exceeded");
      }
      pass_registries[pass_registry_count++] = provider->pass_registry;
    }
    IREE_RETURN_IF_ERROR(loom_target_environment_append_low_descriptor_registry(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_low_lower_policy_registry(
            out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_math_policy_registry(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_low_legality_providers(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_legalizer_providers(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_low_packet_diagnostic_providers(
            out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_low_asm_diagnostic_providers(
            out_environment, provider));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_low_verify_providers(
        out_environment, provider));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_append_emitters(out_environment, provider));
  }
  IREE_RETURN_IF_ERROR(loom_pass_registry_storage_initialize_from_registries(
      pass_registries, pass_registry_count,
      &out_environment->pass_registry_storage));
  return iree_ok_status();
}

void loom_target_environment_deinitialize(
    loom_target_environment_t* environment) {
  if (environment == NULL) {
    return;
  }
  *environment = (loom_target_environment_t){0};
}

iree_status_t loom_target_environment_register_context(
    const loom_target_environment_t* environment, loom_context_t* context) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(context);
  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->register_context == NULL) {
      continue;
    }
    IREE_RETURN_IF_ERROR(provider->register_context(context));
  }
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize_low_descriptor_registry(
    const loom_target_environment_t* environment,
    loom_target_low_descriptor_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_target_low_descriptor_registry_initialize_from_tables(
      out_registry, environment->descriptor_set_providers,
      environment->descriptor_set_provider_count);
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize_low_lower_policy_registry(
    const loom_target_environment_t* environment,
    loom_low_lower_policy_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_low_lower_policy_registry_initialize_from_entries(
      out_registry, environment->low_lower_policy_entries,
      environment->low_lower_policy_entry_count);
  return iree_ok_status();
}

iree_status_t loom_target_environment_initialize_math_policy_registry(
    const loom_target_environment_t* environment,
    loom_target_math_policy_registry_t* out_registry) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(out_registry);
  loom_target_math_policy_registry_initialize_from_entries(
      out_registry, environment->math_policy_entries,
      environment->math_policy_entry_count);
  return iree_ok_status();
}

loom_target_low_legality_provider_list_t
loom_target_environment_low_legality_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_low_legality_provider_list_make(
      environment->low_legality_providers,
      environment->low_legality_provider_count);
}

loom_target_legalizer_provider_list_t
loom_target_environment_legalizer_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_legalizer_provider_list_make(
      environment->legalizer_providers, environment->legalizer_provider_count);
}

loom_target_low_packet_diagnostic_provider_list_t
loom_target_environment_low_packet_diagnostic_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_low_packet_diagnostic_provider_list_make(
      environment->low_packet_diagnostic_providers,
      environment->low_packet_diagnostic_provider_count);
}

loom_target_low_asm_diagnostic_provider_list_t
loom_target_environment_low_asm_diagnostic_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_low_asm_diagnostic_provider_list_make(
      environment->low_asm_diagnostic_providers,
      environment->low_asm_diagnostic_provider_count);
}

loom_low_verify_provider_list_t
loom_target_environment_low_verify_provider_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_low_verify_provider_list_make(
      environment->low_verify_providers,
      environment->low_verify_provider_count);
}

loom_target_emitter_list_t loom_target_environment_emitter_list(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_target_emitter_list_make(environment->emitters,
                                       environment->emitter_count);
}

const loom_pass_registry_t* loom_target_environment_pass_registry(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_pass_registry_storage_registry(
      &environment->pass_registry_storage);
}

iree_status_t loom_target_environment_contribute_pipeline(
    const loom_target_environment_t* environment,
    loom_target_pipeline_phase_t phase,
    loom_pass_environment_t pass_environment, loom_builder_t* builder) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(builder);
  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->contribute_pipeline == NULL) {
      continue;
    }
    const loom_target_pipeline_contribution_t contribution = {
        .target_environment = environment,
        .phase = phase,
        .builder = builder,
        .pass_environment = pass_environment,
    };
    IREE_RETURN_IF_ERROR(provider->contribute_pipeline(&contribution));
  }
  return iree_ok_status();
}

static iree_status_t loom_target_environment_validate_materialized_target_ref(
    const loom_module_t* module, loom_symbol_ref_t target_ref) {
  if (!loom_symbol_ref_is_valid(target_ref)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target provider materialized a null target ref");
  }
  if (target_ref.module_id != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target provider materialized non-local target ref {%u, %u}",
        (unsigned)target_ref.module_id, (unsigned)target_ref.symbol_id);
  }
  if (target_ref.symbol_id >= module->symbols.count) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target provider materialized target symbol id %u, but module has "
        "only %" PRIhsz " symbols",
        (unsigned)target_ref.symbol_id, module->symbols.count);
  }

  const loom_symbol_t* symbol = &module->symbols.entries[target_ref.symbol_id];
  if (symbol->defining_op == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target provider materialized unbound target symbol id %u",
        (unsigned)target_ref.symbol_id);
  }
  const loom_target_like_t target =
      loom_target_like_cast(module, symbol->defining_op);
  if (!loom_target_like_isa(target)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target provider materialized symbol id %u, but its defining op is "
        "not target-like",
        (unsigned)target_ref.symbol_id);
  }
  const loom_symbol_ref_t defining_ref = loom_target_like_symbol(target);
  if (defining_ref.module_id != target_ref.module_id ||
      defining_ref.symbol_id != target_ref.symbol_id) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target provider materialized symbol id %u, but the target-like op "
        "defines {%u, %u}",
        (unsigned)target_ref.symbol_id, (unsigned)defining_ref.module_id,
        (unsigned)defining_ref.symbol_id);
  }
  return iree_ok_status();
}

static bool loom_target_provider_find_equal_materialized_record(
    const loom_target_provider_t* provider, const loom_module_t* module,
    const loom_target_profile_t* profile, const loom_op_t* authored_target_op,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();
  for (iree_host_size_t i = 0; i < module->symbols.count; ++i) {
    const loom_symbol_t* symbol = &module->symbols.entries[i];
    const loom_op_t* defining_op = symbol->defining_op;
    if (defining_op == NULL ||
        defining_op->kind != provider->record_semantics.op_kind) {
      continue;
    }
    if (!provider->materialization.record_matches_effective_target(
            module, defining_op, profile, authored_target_op)) {
      continue;
    }
    *out_target_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = (loom_symbol_id_t)i,
    };
    return true;
  }
  return false;
}

static iree_status_t loom_target_provider_add_materialized_symbol(
    loom_module_t* module, iree_string_view_t symbol_stem,
    loom_symbol_ref_t* out_target_ref) {
  *out_target_ref = loom_symbol_ref_null();

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(module, symbol_stem, &name_id));
  if (loom_module_find_symbol(module, name_id) == LOOM_SYMBOL_ID_INVALID) {
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(module, name_id, &symbol_id));
    *out_target_ref = (loom_symbol_ref_t){
        .module_id = 0,
        .symbol_id = symbol_id,
    };
    return iree_ok_status();
  }

  iree_string_builder_t name_builder;
  iree_string_builder_initialize(module->allocator, &name_builder);
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t ordinal = 1; ordinal <= module->symbols.count + 1 &&
                                     !loom_symbol_ref_is_valid(*out_target_ref);
       ++ordinal) {
    iree_string_builder_reset(&name_builder);
    status = iree_string_builder_append_string(&name_builder, symbol_stem);
    if (iree_status_is_ok(status)) {
      status = iree_string_builder_append_format(&name_builder,
                                                 "_target_%" PRIhsz, ordinal);
    }
    if (!iree_status_is_ok(status)) {
      break;
    }
    status = loom_module_intern_string(
        module, iree_string_builder_view(&name_builder), &name_id);
    if (!iree_status_is_ok(status)) {
      break;
    }
    if (loom_module_find_symbol(module, name_id) != LOOM_SYMBOL_ID_INVALID) {
      continue;
    }
    loom_symbol_id_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    status = loom_module_add_symbol(module, name_id, &symbol_id);
    if (iree_status_is_ok(status)) {
      *out_target_ref = (loom_symbol_ref_t){
          .module_id = 0,
          .symbol_id = symbol_id,
      };
    }
  }
  iree_string_builder_deinitialize(&name_builder);
  if (iree_status_is_ok(status) && !loom_symbol_ref_is_valid(*out_target_ref)) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "could not allocate a unique target materialization symbol");
  }
  return status;
}

iree_status_t loom_target_environment_materialize_effective_target(
    const loom_target_environment_t* environment, loom_module_t* module,
    const loom_target_profile_t* target_profile,
    const loom_op_t* authored_target_op, loom_symbol_ref_t* out_target_ref) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(target_profile);
  IREE_ASSERT_ARGUMENT(out_target_ref);
  *out_target_ref = loom_symbol_ref_null();
  if (module->body == NULL || module->body->block_count == 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target materialization requires a module with a body block");
  }

  if (target_profile->target_bundle == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "target materialization requires a complete target profile");
  }
  const loom_target_provider_set_t* provider_set = environment->provider_set;
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->profile_type != target_profile->type ||
        provider->materialization.build_effective_target_record == NULL) {
      continue;
    }

    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    if (loom_target_provider_find_equal_materialized_record(
            provider, module, target_profile, authored_target_op,
            &target_ref)) {
      *out_target_ref = target_ref;
      return iree_ok_status();
    }

    const iree_string_view_t symbol_stem =
        provider->materialization.symbol_stem(target_profile);
    if (iree_string_view_is_empty(symbol_stem)) {
      return iree_make_status(
          IREE_STATUS_FAILED_PRECONDITION,
          "target provider '%.*s' produced an empty materialization symbol "
          "stem",
          (int)target_profile->type->name.size,
          target_profile->type->name.data);
    }
    IREE_RETURN_IF_ERROR(loom_target_provider_add_materialized_symbol(
        module, symbol_stem, &target_ref));

    loom_block_t* module_block = loom_module_block(module);
    loom_builder_t builder = {0};
    loom_builder_initialize(module, &module->arena, module_block, &builder);
    if (module_block->first_op != NULL) {
      loom_builder_set_before(&builder, module_block->first_op);
    }
    loom_op_t* target_op = NULL;
    IREE_RETURN_IF_ERROR(
        provider->materialization.build_effective_target_record(
            &builder, target_profile, authored_target_op, target_ref,
            LOOM_LOCATION_UNKNOWN, &target_op));
    IREE_RETURN_IF_ERROR(
        loom_target_environment_validate_materialized_target_ref(module,
                                                                 target_ref));
    *out_target_ref = target_ref;
    return iree_ok_status();
  }

  const iree_string_view_t target_name = target_profile->type->name;
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no linked target provider can materialize selected target family "
      "'%.*s'",
      (int)target_name.size, target_name.data);
}
