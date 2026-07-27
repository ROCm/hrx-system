// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/lower/lower.h"

void loom_low_lower_policy_registry_initialize_from_entries(
    loom_low_lower_policy_registry_t* out_registry,
    const loom_low_lower_policy_registry_entry_t* entries,
    iree_host_size_t entry_count) {
  *out_registry = (loom_low_lower_policy_registry_t){
      .entries = entries,
      .entry_count = entry_count,
  };
}

const loom_low_lower_policy_t* loom_low_lower_policy_registry_lookup(
    const loom_low_lower_policy_registry_t* registry,
    iree_string_view_t contract_set_key) {
  contract_set_key = iree_string_view_trim(contract_set_key);
  for (iree_host_size_t i = 0; i < registry->entry_count; ++i) {
    const loom_low_lower_policy_registry_entry_t* entry = &registry->entries[i];
    if (!iree_string_view_equal(entry->contract_set_key, contract_set_key)) {
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
      registry, bundle->config->contract_set_key);
}
