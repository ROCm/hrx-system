// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/type_registry.h"

#include "loom/ops/type_registry_tables.h"
#include "loom/util/fact_table.h"

iree_host_size_t loom_type_registry_count(void) {
  return loom_type_registry_entry_count;
}

const loom_type_registry_entry_t* loom_type_registry_entries(void) {
  return loom_type_registry_entries_storage;
}

const loom_type_descriptor_t* loom_type_registry_lookup_builtin(
    loom_type_kind_t kind) {
  if (!loom_type_kind_is_valid(kind)) {
    return NULL;
  }
  return loom_type_registry_builtin_descriptors[kind];
}

const loom_type_descriptor_t* loom_type_registry_lookup(
    iree_string_view_t name) {
  iree_host_size_t low = 0;
  iree_host_size_t high = loom_type_registry_entry_count;
  while (low < high) {
    const iree_host_size_t mid = low + (high - low) / 2;
    const int comparison = iree_string_view_compare(
        loom_type_registry_entries_storage[mid].name, name);
    if (comparison < 0) {
      low = mid + 1;
    } else if (comparison > 0) {
      high = mid;
    } else {
      return loom_type_registry_entries_storage[mid].descriptor;
    }
  }
  return NULL;
}

const loom_value_fact_domain_t* loom_type_registry_resolve_fact_domain(
    void* user_data, const loom_fact_context_t* context,
    const loom_module_t* module, loom_type_t type) {
  (void)user_data;
  (void)context;
  if (loom_type_is_dialect(type)) {
    const loom_string_id_t name_id = loom_type_dialect_name_id(type);
    if (module == NULL || name_id == LOOM_STRING_ID_INVALID ||
        (iree_host_size_t)name_id >= module->strings.count) {
      return NULL;
    }
    const loom_type_descriptor_t* descriptor =
        loom_type_registry_lookup(module->strings.entries[name_id]);
    return descriptor != NULL ? descriptor->fact_domain : NULL;
  }
  const loom_type_descriptor_t* descriptor =
      loom_type_registry_lookup_builtin(loom_type_kind(type));
  return descriptor != NULL ? descriptor->fact_domain : NULL;
}

void loom_type_registry_configure_fact_context(loom_fact_context_t* context) {
  if (context == NULL) {
    return;
  }
  context->resolve_type_domain =
      loom_value_fact_type_domain_resolver_callback_make(
          loom_type_registry_resolve_fact_domain, NULL);
}
