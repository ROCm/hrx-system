// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/low_descriptor_registry.h"

#include <inttypes.h>

void loom_target_low_descriptor_registry_initialize_from_tables(
    const loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_count,
    loom_target_low_descriptor_registry_t* out_registry) {
  *out_registry = (loom_target_low_descriptor_registry_t){
      .descriptor_set_providers = descriptor_set_providers,
      .descriptor_set_provider_count = descriptor_set_provider_count,
      .registry =
          {
              .descriptor_set_providers = descriptor_set_providers,
              .descriptor_set_provider_count = descriptor_set_provider_count,
          },
  };
}

iree_status_t loom_target_low_descriptor_registry_append_to_tables(
    const loom_target_low_descriptor_registry_t* source,
    loom_low_descriptor_set_provider_t* descriptor_set_providers,
    iree_host_size_t descriptor_set_provider_capacity,
    iree_host_size_t* descriptor_set_provider_count) {
  if (*descriptor_set_provider_count > descriptor_set_provider_capacity) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target-low registry destination counts exceed destination capacity");
  }
  if (source->descriptor_set_provider_count >
      descriptor_set_provider_capacity - *descriptor_set_provider_count) {
    return iree_make_status(
        IREE_STATUS_RESOURCE_EXHAUSTED,
        "target-low registry descriptor-set provider table capacity exceeded");
  }

  for (iree_host_size_t i = 0; i < source->descriptor_set_provider_count; ++i) {
    descriptor_set_providers[*descriptor_set_provider_count + i] =
        source->descriptor_set_providers[i];
  }
  *descriptor_set_provider_count += source->descriptor_set_provider_count;
  return iree_ok_status();
}

iree_status_t loom_target_low_source_binding_select(
    const loom_low_descriptor_registry_t* registry,
    const loom_target_bundle_t* bundle,
    loom_target_low_source_binding_t* out_binding) {
  *out_binding = (loom_target_low_source_binding_t){0};
  const loom_target_contract_set_t contract_set = bundle->config->contract_set;
  const loom_low_descriptor_set_t* descriptor_set =
      loom_low_descriptor_registry_lookup_contract_set(registry, contract_set);
  if (descriptor_set == NULL) {
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected low descriptor set '%.*s' "
        "(0x%016" PRIx64 ") that is not linked",
        (int)bundle->name.size, bundle->name.data,
        (int)contract_set.spelling.size, contract_set.spelling.data,
        contract_set.stable_id);
  }
  const uint64_t schedule_model_stable_id =
      bundle->config->schedule_model_stable_id;
  const loom_low_schedule_model_t* schedule_model =
      loom_low_descriptor_set_lookup_schedule_model_by_id(
          descriptor_set, schedule_model_stable_id);
  if (schedule_model == NULL) {
    const iree_string_view_t descriptor_set_key =
        loom_low_descriptor_set_string(descriptor_set,
                                       descriptor_set->key_string_ref);
    return iree_make_status(
        IREE_STATUS_NOT_FOUND,
        "target bundle '%.*s' selected schedule model 0x%016" PRIx64
        " that is not "
        "available in low descriptor set '%.*s'",
        (int)bundle->name.size, bundle->name.data, schedule_model_stable_id,
        (int)descriptor_set_key.size, descriptor_set_key.data);
  }
  *out_binding = (loom_target_low_source_binding_t){
      .descriptor_set = descriptor_set,
      .schedule_model = schedule_model,
  };
  return iree_ok_status();
}
