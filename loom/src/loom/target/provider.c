// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"

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

iree_status_t loom_target_environment_materialize_selection(
    const loom_target_environment_t* environment, loom_module_t* module,
    loom_target_selection_t target_selection,
    loom_symbol_ref_t* out_target_ref) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(module);
  IREE_ASSERT_ARGUMENT(out_target_ref);
  *out_target_ref = loom_symbol_ref_null();
  if (loom_target_selection_is_empty(target_selection)) {
    return iree_ok_status();
  }

  const loom_target_provider_set_t* provider_set = environment->provider_set;
  const loom_target_selection_materialization_request_t request = {
      .target_environment = environment,
      .module = module,
      .target_selection = target_selection,
  };
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider->materialize_selection == NULL) {
      continue;
    }
    bool materialized = false;
    loom_symbol_ref_t target_ref = loom_symbol_ref_null();
    IREE_RETURN_IF_ERROR(provider->materialize_selection(
        provider, &request, &materialized, &target_ref));
    if (!materialized) {
      continue;
    }
    IREE_RETURN_IF_ERROR(
        loom_target_environment_validate_materialized_target_ref(module,
                                                                 target_ref));
    *out_target_ref = target_ref;
    return iree_ok_status();
  }

  const iree_string_view_t target_name = target_selection.bundle
                                             ? target_selection.bundle->name
                                             : IREE_SV("<target-data-only>");
  return iree_make_status(
      IREE_STATUS_UNIMPLEMENTED,
      "no linked target provider can materialize selected target '%.*s'",
      (int)target_name.size, target_name.data);
}
