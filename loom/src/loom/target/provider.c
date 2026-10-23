// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/provider.h"

#include <stddef.h>
#include <string.h>

typedef struct loom_target_artifact_metadata_t {
  // Allocator owning the metadata, descriptors, and trailing string bytes.
  iree_allocator_t allocator;
  // Target facts whose structural pointers and strings refer to this
  // allocation.
  loom_target_bundle_storage_t target;
} loom_target_artifact_metadata_t;

static void loom_target_artifact_metadata_release(void* storage) {
  loom_target_artifact_metadata_t* metadata = storage;
  iree_allocator_free(metadata->allocator, metadata);
}

static iree_string_view_t loom_target_artifact_copy_string(
    iree_string_view_t source, char** cursor) {
  if (iree_string_view_is_empty(source)) {
    return iree_string_view_empty();
  }
  memcpy(*cursor, source.data, source.size);
  const iree_string_view_t result = iree_make_string_view(*cursor, source.size);
  *cursor += source.size;
  return result;
}

loom_target_call_policy_t loom_target_provider_select_call_policy(
    const loom_resolved_target_t* resolved_target, const loom_module_t* module,
    loom_call_like_kind_t kind, loom_call_like_t call,
    loom_func_like_t callee) {
  if (resolved_target == NULL || resolved_target->provider == NULL ||
      resolved_target->provider->select_call_policy == NULL) {
    return LOOM_TARGET_CALL_POLICY_DIRECT;
  }
  return resolved_target->provider->select_call_policy(resolved_target, module,
                                                       kind, call, callee);
}

iree_status_t loom_target_emit_artifact_retain_metadata(
    const loom_target_bundle_t* target_bundle, iree_host_size_t sidecar_count,
    const loom_target_emit_sidecar_artifact_t* sidecars,
    iree_allocator_t allocator, loom_target_emit_artifact_t* artifact) {
  if (target_bundle == NULL && sidecar_count == 0) {
    return iree_ok_status();
  }
  loom_target_bundle_storage_t target = {0};
  if (target_bundle != NULL) {
    target = (loom_target_bundle_storage_t){
        .snapshot = *target_bundle->snapshot,
        .export_plan = *target_bundle->export_plan,
        .config = *target_bundle->config,
        .bundle = *target_bundle,
    };
  }
  static const uint16_t string_offsets[] = {
      offsetof(loom_target_bundle_storage_t, bundle.name),
      offsetof(loom_target_bundle_storage_t, snapshot.name),
      offsetof(loom_target_bundle_storage_t, export_plan.name),
      offsetof(loom_target_bundle_storage_t, export_plan.export_symbol),
      offsetof(loom_target_bundle_storage_t, export_plan.calling_convention),
      offsetof(loom_target_bundle_storage_t, config.name),
      offsetof(loom_target_bundle_storage_t, config.contract_set.spelling),
  };
  iree_host_size_t descriptor_bytes = 0;
  iree_host_size_t allocation_size = 0;
  bool size_valid =
      iree_host_size_checked_mul(sidecar_count, sizeof(*sidecars),
                                 &descriptor_bytes) &&
      iree_host_size_checked_add(sizeof(loom_target_artifact_metadata_t),
                                 descriptor_bytes, &allocation_size);
  for (iree_host_size_t i = 0; size_valid && i < IREE_ARRAYSIZE(string_offsets);
       ++i) {
    const iree_string_view_t* field =
        (const iree_string_view_t*)((const uint8_t*)&target +
                                    string_offsets[i]);
    size_valid = iree_host_size_checked_add(allocation_size, field->size,
                                            &allocation_size);
  }
  for (iree_host_size_t i = 0; size_valid && i < sidecar_count; ++i) {
    size_valid = iree_host_size_checked_add(
        allocation_size, sidecars[i].identifier.size, &allocation_size);
  }
  if (!size_valid) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "artifact metadata allocation size overflow");
  }

  loom_target_artifact_metadata_t* metadata = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc_uninitialized(
      allocator, allocation_size, (void**)&metadata));
  *metadata = (loom_target_artifact_metadata_t){
      .allocator = allocator,
      .target = target,
  };
  loom_target_bundle_storage_rebind(&metadata->target);
  loom_target_emit_sidecar_artifact_t* owned_sidecars =
      (loom_target_emit_sidecar_artifact_t*)(metadata + 1);
  char* cursor = (char*)(owned_sidecars + sidecar_count);
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(string_offsets); ++i) {
    iree_string_view_t* field =
        (iree_string_view_t*)((uint8_t*)&metadata->target + string_offsets[i]);
    *field = loom_target_artifact_copy_string(*field, &cursor);
  }
  for (iree_host_size_t i = 0; i < sidecar_count; ++i) {
    owned_sidecars[i] = sidecars[i];
    owned_sidecars[i].identifier =
        loom_target_artifact_copy_string(sidecars[i].identifier, &cursor);
    iree_byte_sequence_retain(owned_sidecars[i].contents);
  }
  artifact->target_bundle =
      target_bundle != NULL ? &metadata->target.bundle : NULL;
  artifact->sidecars = sidecar_count != 0 ? owned_sidecars : NULL;
  artifact->sidecar_count = sidecar_count;
  artifact->storage = metadata;
  artifact->release_storage = loom_target_artifact_metadata_release;
  return iree_ok_status();
}

void loom_target_emit_artifact_release(loom_target_emit_artifact_t* artifact) {
  if (artifact == NULL) {
    return;
  }
  IREE_ASSERT(artifact->sidecar_count == 0 || artifact->sidecars != NULL);
  IREE_ASSERT(artifact->storage == NULL || artifact->release_storage != NULL);
  iree_byte_sequence_release(artifact->contents);
  iree_byte_sequence_release(artifact->target_listing_contents);
  if (artifact->sidecars != NULL) {
    for (iree_host_size_t i = 0; i < artifact->sidecar_count; ++i) {
      iree_byte_sequence_release(artifact->sidecars[i].contents);
    }
  }
  if (artifact->storage != NULL && artifact->release_storage != NULL) {
    artifact->release_storage(artifact->storage);
  }
  *artifact = (loom_target_emit_artifact_t){0};
}

void loom_target_provider_set_storage_initialize(
    loom_target_provider_set_storage_t* storage) {
  IREE_ASSERT_ARGUMENT(storage);
  *storage = (loom_target_provider_set_storage_t){
      .provider_set = loom_target_provider_set_make(storage->providers, 0),
  };
}

iree_status_t loom_target_provider_set_storage_append(
    loom_target_provider_set_storage_t* storage,
    const loom_target_provider_t* provider) {
  IREE_ASSERT_ARGUMENT(storage);
  if (provider == NULL) {
    return iree_ok_status();
  }
  if (storage->provider_set.provider_count >=
      IREE_ARRAYSIZE(storage->providers)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target provider set storage capacity exceeded");
  }
  storage->providers[storage->provider_set.provider_count++] = provider;
  return iree_ok_status();
}

iree_status_t loom_target_provider_set_storage_append_set(
    loom_target_provider_set_storage_t* storage,
    const loom_target_provider_set_t* provider_set) {
  IREE_ASSERT_ARGUMENT(storage);
  IREE_ASSERT_ARGUMENT(provider_set);
  const iree_host_size_t append_count = provider_set->provider_count;
  if (append_count > IREE_ARRAYSIZE(storage->providers) -
                         storage->provider_set.provider_count) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target provider set storage capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < append_count; ++i) {
    storage->providers[storage->provider_set.provider_count++] =
        provider_set->providers[i];
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
  if (provider->emitter_list.count != 0 &&
      provider->emitter_list.values == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target emitter list has no storage");
  }
  if (environment->emitter_count + provider->emitter_list.count >
      IREE_ARRAYSIZE(environment->emitters)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "target emitter capacity exceeded");
  }
  for (iree_host_size_t i = 0; i < provider->emitter_list.count; ++i) {
    const loom_target_emitter_t* emitter = provider->emitter_list.values[i];
    if (emitter == NULL || iree_string_view_is_empty(emitter->name) ||
        iree_string_view_is_empty(emitter->public_artifact_format) ||
        emitter->emit == NULL ||
        loom_target_environment_lookup_emitter(
            environment, emitter->public_artifact_format) != NULL) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target emitter must have a name, unique format, and emit function");
    }
    const iree_host_size_t index = environment->emitter_count++;
    environment->emitters[index] = emitter;
  }
  return iree_ok_status();
}

static const loom_target_fact_type_t* loom_target_provider_fact_type(
    const loom_target_provider_t* provider) {
  return provider->target_fact_type != NULL ? provider->target_fact_type
         : provider->profile_type != NULL   ? provider->profile_type->fact_type
                                            : NULL;
}

static iree_status_t loom_target_environment_append_canonical_emitter_fact(
    const loom_target_provider_t* provider,
    const loom_target_emitter_t* emitter,
    const loom_target_fact_type_t* target_fact_type, const char* product_name,
    const loom_target_fact_type_t** fact_types,
    iree_host_size_t fact_type_capacity, iree_host_size_t* inout_count) {
  if (emitter == NULL && target_fact_type == NULL) {
    return iree_ok_status();
  }
  if (emitter == NULL || target_fact_type == NULL) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "canonical %s emitter and target fact type must be paired",
        product_name);
  }
  bool emitter_contributed = false;
  for (iree_host_size_t i = 0; i < provider->emitter_list.count; ++i) {
    emitter_contributed |= provider->emitter_list.values[i] == emitter;
  }
  if (!emitter_contributed) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "canonical %s emitter '%.*s' must be in the provider's %" PRIhsz
        "-entry emitter list",
        product_name, (int)emitter->name.size, emitter->name.data,
        provider->emitter_list.count);
  }
  for (iree_host_size_t i = 0; i < *inout_count; ++i) {
    if (fact_types[i] == target_fact_type) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target fact type has multiple canonical %s emitters", product_name);
    }
  }
  if (*inout_count >= fact_type_capacity) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "canonical %s emitter capacity exceeded",
                            product_name);
  }
  fact_types[(*inout_count)++] = target_fact_type;
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
  const loom_target_fact_type_t* canonical_module_fact_types
      [LOOM_TARGET_PROVIDER_CANONICAL_EMITTER_CAPACITY] = {0};
  iree_host_size_t canonical_module_fact_type_count = 0;
  const loom_target_fact_type_t* canonical_kernel_fact_types
      [LOOM_TARGET_PROVIDER_CANONICAL_EMITTER_CAPACITY] = {0};
  iree_host_size_t canonical_kernel_fact_type_count = 0;
  if (provider_set->provider_count != 0 && provider_set->providers == NULL) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "target provider set has no storage");
  }
  for (iree_host_size_t i = 0; i < provider_set->provider_count; ++i) {
    const loom_target_provider_t* provider = provider_set->providers[i];
    if (provider == NULL) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "target provider set contains a NULL provider");
    }
    const loom_target_fact_type_t* fact_type =
        loom_target_provider_fact_type(provider);
    for (iree_host_size_t j = 0; j < i; ++j) {
      const loom_target_provider_t* previous = provider_set->providers[j];
      const bool duplicate_profile =
          provider->profile_type != NULL && previous->profile_type != NULL &&
          iree_string_view_equal(provider->profile_type->name,
                                 previous->profile_type->name);
      const bool duplicate_facts =
          fact_type != NULL &&
          fact_type == loom_target_provider_fact_type(previous);
      if (duplicate_profile || duplicate_facts) {
        return iree_make_status(
            IREE_STATUS_INVALID_ARGUMENT,
            "target profile family and fact type ownership must be unique");
      }
    }
    if (provider->target_fact_type != NULL && provider->profile_type != NULL &&
        provider->target_fact_type != provider->profile_type->fact_type) {
      return iree_make_status(
          IREE_STATUS_INVALID_ARGUMENT,
          "target provider fact type must match its profile fact type");
    }
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
    IREE_RETURN_IF_ERROR(loom_target_environment_append_canonical_emitter_fact(
        provider, provider->canonical_module_emitter,
        provider->canonical_module_fact_type, "module",
        canonical_module_fact_types,
        IREE_ARRAYSIZE(canonical_module_fact_types),
        &canonical_module_fact_type_count));
    IREE_RETURN_IF_ERROR(loom_target_environment_append_canonical_emitter_fact(
        provider, provider->canonical_kernel_emitter,
        provider->canonical_kernel_fact_type, "kernel",
        canonical_kernel_fact_types,
        IREE_ARRAYSIZE(canonical_kernel_fact_types),
        &canonical_kernel_fact_type_count));
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

loom_target_low_descriptor_registry_t
loom_target_environment_low_descriptor_registry(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  loom_target_low_descriptor_registry_t registry;
  loom_target_low_descriptor_registry_initialize_from_tables(
      &registry, environment->descriptor_set_providers,
      environment->descriptor_set_provider_count);
  return registry;
}

loom_low_lower_policy_registry_t
loom_target_environment_low_lower_policy_registry(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  loom_low_lower_policy_registry_t registry;
  loom_low_lower_policy_registry_initialize_from_entries(
      &registry, environment->low_lower_policy_entries,
      environment->low_lower_policy_entry_count);
  return registry;
}

loom_target_math_policy_registry_t loom_target_environment_math_policy_registry(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  loom_target_math_policy_registry_t registry;
  loom_target_math_policy_registry_initialize_from_entries(
      &registry, environment->math_policy_entries,
      environment->math_policy_entry_count);
  return registry;
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

const loom_target_emitter_t* loom_target_environment_lookup_emitter(
    const loom_target_environment_t* environment,
    iree_string_view_t public_artifact_format) {
  IREE_ASSERT_ARGUMENT(environment);
  if (iree_string_view_is_empty(public_artifact_format)) {
    return environment->emitter_count == 1 ? environment->emitters[0] : NULL;
  }
  for (iree_host_size_t i = 0; i < environment->emitter_count; ++i) {
    const loom_target_emitter_t* emitter = environment->emitters[i];
    if (iree_string_view_equal(emitter->public_artifact_format,
                               public_artifact_format)) {
      return emitter;
    }
  }
  return NULL;
}

const loom_pass_registry_t* loom_target_environment_pass_registry(
    const loom_target_environment_t* environment) {
  IREE_ASSERT_ARGUMENT(environment);
  return loom_pass_registry_storage_registry(
      &environment->pass_registry_storage);
}

const loom_target_provider_t* loom_target_environment_lookup_profile_provider(
    const loom_target_environment_t* environment,
    const loom_target_profile_type_t* profile_type) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(profile_type);
  for (iree_host_size_t i = 0; i < environment->provider_set->provider_count;
       ++i) {
    if (environment->provider_set->providers[i]->profile_type == profile_type) {
      return environment->provider_set->providers[i];
    }
  }
  return NULL;
}

const loom_target_provider_t* loom_target_environment_lookup_fact_provider(
    const loom_target_environment_t* environment,
    const loom_target_fact_type_t* fact_type) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(fact_type);
  for (iree_host_size_t i = 0; i < environment->provider_set->provider_count;
       ++i) {
    const loom_target_provider_t* provider =
        environment->provider_set->providers[i];
    const loom_target_fact_type_t* provider_fact_type =
        loom_target_provider_fact_type(provider);
    if (provider_fact_type == fact_type) {
      return provider;
    }
  }
  return NULL;
}

const loom_target_emitter_t*
loom_target_environment_lookup_canonical_module_emitter(
    const loom_target_environment_t* environment,
    const loom_target_fact_type_t* fact_type) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(fact_type);
  for (iree_host_size_t i = 0; i < environment->provider_set->provider_count;
       ++i) {
    const loom_target_provider_t* provider =
        environment->provider_set->providers[i];
    if (provider->canonical_module_fact_type == fact_type) {
      return provider->canonical_module_emitter;
    }
  }
  return NULL;
}

const loom_target_emitter_t*
loom_target_environment_lookup_canonical_kernel_emitter(
    const loom_target_environment_t* environment,
    const loom_target_fact_type_t* fact_type) {
  IREE_ASSERT_ARGUMENT(environment);
  IREE_ASSERT_ARGUMENT(fact_type);
  for (iree_host_size_t i = 0; i < environment->provider_set->provider_count;
       ++i) {
    const loom_target_provider_t* provider =
        environment->provider_set->providers[i];
    if (provider->canonical_kernel_fact_type == fact_type) {
      return provider->canonical_kernel_emitter;
    }
  }
  return NULL;
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
