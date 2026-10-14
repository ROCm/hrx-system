// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/pass/registry.h"

#include <string.h>

const loom_pass_descriptor_t* loom_pass_registry_lookup(
    const loom_pass_registry_t* registry, iree_string_view_t key) {
  iree_host_size_t low = 0;
  iree_host_size_t high = registry->descriptor_count;
  while (low < high) {
    iree_host_size_t mid = low + (high - low) / 2;
    const loom_pass_descriptor_t* descriptor =
        loom_pass_registry_at(registry, mid);
    int comparison = iree_string_view_compare(key, descriptor->key);
    if (comparison == 0) {
      return descriptor;
    } else if (comparison < 0) {
      high = mid;
    } else {
      low = mid + 1;
    }
  }
  return NULL;
}

static iree_status_t loom_pass_registry_storage_insert(
    loom_pass_registry_storage_t* storage,
    const loom_pass_descriptor_t* descriptor) {
  if (storage->registry.descriptor_count >=
      IREE_ARRAYSIZE(storage->descriptor_refs)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "pass registry storage capacity exceeded");
  }

  iree_host_size_t insert_index = 0;
  while (insert_index < storage->registry.descriptor_count) {
    const loom_pass_descriptor_t* existing =
        storage->descriptor_refs[insert_index];
    int comparison = iree_string_view_compare(descriptor->key, existing->key);
    if (comparison == 0) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "duplicate pass descriptor key '%.*s'",
                              (int)descriptor->key.size, descriptor->key.data);
    }
    if (comparison < 0) {
      break;
    }
    ++insert_index;
  }

  const iree_host_size_t move_count =
      storage->registry.descriptor_count - insert_index;
  if (move_count != 0) {
    memmove(&storage->descriptor_refs[insert_index + 1],
            &storage->descriptor_refs[insert_index],
            move_count * sizeof(storage->descriptor_refs[0]));
  }
  storage->descriptor_refs[insert_index] = descriptor;
  ++storage->registry.descriptor_count;
  return iree_ok_status();
}

iree_status_t loom_pass_registry_storage_initialize_from_registries(
    const loom_pass_registry_t* const* registries,
    iree_host_size_t registry_count,
    loom_pass_registry_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  out_storage->registry.descriptors = NULL;
  out_storage->registry.descriptor_count = 0;
  out_storage->registry.descriptor_refs = out_storage->descriptor_refs;

  for (iree_host_size_t registry_index = 0; registry_index < registry_count;
       ++registry_index) {
    const loom_pass_registry_t* registry = registries[registry_index];
    if (registry == NULL) {
      continue;
    }
    for (iree_host_size_t descriptor_index = 0;
         descriptor_index < registry->descriptor_count; ++descriptor_index) {
      IREE_RETURN_IF_ERROR(loom_pass_registry_storage_insert(
          out_storage, loom_pass_registry_at(registry, descriptor_index)));
    }
  }
  return iree_ok_status();
}

const loom_pass_registry_t* loom_pass_registry_storage_registry(
    const loom_pass_registry_storage_t* storage) {
  return &storage->registry;
}

bool loom_pass_descriptor_is_available(
    const loom_pass_descriptor_t* descriptor) {
  return descriptor &&
         !iree_any_bit_set(descriptor->flags, LOOM_PASS_DESCRIPTOR_UNAVAILABLE);
}
