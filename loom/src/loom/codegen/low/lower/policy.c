// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower.h"

void loom_low_lower_policy_registry_initialize_from_entries(
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count,
    loom_low_lower_policy_registry_t* out_registry) {
  *out_registry = (loom_low_lower_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

const loom_low_lower_policy_t* loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    uint64_t contract_set_stable_id) {
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    if (entry->contract_set_stable_id != contract_set_stable_id) {
      continue;
    }
    return entry->policy;
  }
  return NULL;
}

const loom_low_lower_policy_t* loom_low_lower_policy_registry_lookup_for_bundle(
    const loom_low_lower_policy_registry_t* registry,
    const loom_target_bundle_t* bundle) {
  return loom_low_lower_policy_registry_lookup(
      registry, bundle->config->contract_set.stable_id);
}
