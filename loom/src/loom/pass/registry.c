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

// Merges sorted provider registries through alternating reference buffers.
// Each level visits every descriptor once, bounding composition by
// O(k + n log k) for n descriptors from k registries. Descriptors stay
// borrowed.
static iree_host_size_t loom_pass_registry_merge(
    const loom_pass_registry_t* const* registries,
    iree_host_size_t registry_count, const loom_pass_descriptor_t** output,
    const loom_pass_descriptor_t** scratch) {
  if (registry_count == 1) {
    const loom_pass_registry_t* registry = registries[0];
    if (registry == NULL) {
      return 0;
    }
    for (iree_host_size_t i = 0; i < registry->descriptor_count; ++i) {
      output[i] = loom_pass_registry_at(registry, i);
    }
    return registry->descriptor_count;
  }

  const iree_host_size_t midpoint = registry_count / 2;
  const iree_host_size_t left_count =
      loom_pass_registry_merge(registries, midpoint, scratch, output);
  const iree_host_size_t right_count =
      loom_pass_registry_merge(registries + midpoint, registry_count - midpoint,
                               scratch + left_count, output + left_count);
  const iree_host_size_t count = left_count + right_count;
  // Provider namespaces are often disjoint. Concatenate ordered runs without
  // comparing each descriptor again; this also handles empty providers.
  if (left_count == 0 || right_count == 0 ||
      iree_string_view_compare(scratch[left_count - 1]->key,
                               scratch[left_count]->key) < 0) {
    memcpy(output, scratch, count * sizeof(*output));
    return count;
  }
  if (iree_string_view_compare(scratch[count - 1]->key, scratch[0]->key) < 0) {
    memcpy(output, scratch + left_count, right_count * sizeof(*output));
    memcpy(output + right_count, scratch, left_count * sizeof(*output));
    return count;
  }
  iree_host_size_t left = 0;
  iree_host_size_t right = left_count;
  iree_host_size_t position = 0;
  while (left < left_count && right < count) {
    if (iree_string_view_compare(scratch[left]->key, scratch[right]->key) < 0) {
      output[position++] = scratch[left++];
    } else {
      output[position++] = scratch[right++];
    }
  }
  memcpy(output + position, scratch + left,
         (left_count - left) * sizeof(*output));
  position += left_count - left;
  memcpy(output + position, scratch + right, (count - right) * sizeof(*output));
  return count;
}

iree_status_t loom_pass_registry_storage_initialize_from_registries(
    const loom_pass_registry_t* const* registries,
    iree_host_size_t registry_count,
    loom_pass_registry_storage_t* out_storage) {
  IREE_ASSERT_ARGUMENT(out_storage);
  out_storage->registry.descriptors = NULL;
  out_storage->registry.descriptor_count = 0;
  out_storage->registry.descriptor_refs = out_storage->descriptor_refs;

  const loom_pass_registry_t* single_registry = NULL;
  iree_host_size_t nonempty_registry_count = 0;
  for (iree_host_size_t registry_index = 0; registry_index < registry_count;
       ++registry_index) {
    const loom_pass_registry_t* registry = registries[registry_index];
    if (registry == NULL || registry->descriptor_count == 0) {
      continue;
    }
    if (registry->descriptor_count >
        IREE_ARRAYSIZE(out_storage->descriptor_refs) -
            out_storage->registry.descriptor_count) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "pass registry storage capacity exceeded");
    }
    out_storage->registry.descriptor_count += registry->descriptor_count;
    single_registry = registry;
    ++nonempty_registry_count;
  }

  if (nonempty_registry_count == 1) {
    for (iree_host_size_t i = 0; i < single_registry->descriptor_count; ++i) {
      out_storage->descriptor_refs[i] =
          loom_pass_registry_at(single_registry, i);
    }
  } else if (nonempty_registry_count > 1) {
    const loom_pass_descriptor_t*
        scratch[LOOM_PASS_REGISTRY_STORAGE_DESCRIPTOR_CAPACITY];
    loom_pass_registry_merge(registries, registry_count,
                             out_storage->descriptor_refs, scratch);
  }
  for (iree_host_size_t i = 1; i < out_storage->registry.descriptor_count;
       ++i) {
    const loom_pass_descriptor_t* previous =
        out_storage->descriptor_refs[i - 1];
    const loom_pass_descriptor_t* descriptor = out_storage->descriptor_refs[i];
    if (iree_string_view_equal(previous->key, descriptor->key)) {
      return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                              "duplicate pass descriptor key '%.*s'",
                              (int)descriptor->key.size, descriptor->key.data);
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
