// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/handle_registry.h"

#include <string.h>

enum iree_hip_handle_registry_slot_state_e {
  IREE_HIP_HANDLE_REGISTRY_SLOT_EMPTY = 0,
  IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE = 1,
  IREE_HIP_HANDLE_REGISTRY_SLOT_TOMBSTONE = 2,
};

static iree_host_size_t iree_hip_handle_registry_hash(uintptr_t handle) {
#if UINTPTR_MAX > UINT32_MAX
  handle ^= handle >> 33;
  handle *= UINT64_C(0xff51afd7ed558ccd);
  handle ^= handle >> 33;
#else
  handle ^= handle >> 16;
  handle *= UINT32_C(0x7feb352d);
  handle ^= handle >> 15;
#endif
  return (iree_host_size_t)handle;
}

static void iree_hip_handle_registry_insert_unchecked(
    iree_hip_handle_registry_t* registry, uintptr_t handle) {
  const iree_host_size_t mask = registry->capacity - 1;
  iree_host_size_t slot = iree_hip_handle_registry_hash(handle) & mask;
  while (registry->states[slot] == IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE) {
    slot = (slot + 1) & mask;
  }
  registry->handles[slot] = handle;
  registry->states[slot] = IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE;
  ++registry->count;
}

static iree_status_t iree_hip_handle_registry_rehash(
    iree_hip_handle_registry_t* registry, iree_host_size_t new_capacity) {
  iree_host_size_t handles_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_mul(
          new_capacity, sizeof(*registry->handles), &handles_size))) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "opaque handle registry capacity overflow");
  }

  uintptr_t* new_handles = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      iree_allocator_system(), handles_size, (void**)&new_handles));
  uint8_t* new_states = NULL;
  iree_status_t status = iree_allocator_malloc(
      iree_allocator_system(), new_capacity * sizeof(*new_states),
      (void**)&new_states);
  if (!iree_status_is_ok(status)) {
    iree_allocator_free(iree_allocator_system(), new_handles);
    return status;
  }
  memset(new_states, 0, new_capacity * sizeof(*new_states));

  uintptr_t* old_handles = registry->handles;
  uint8_t* old_states = registry->states;
  const iree_host_size_t old_capacity = registry->capacity;
  registry->handles = new_handles;
  registry->states = new_states;
  registry->capacity = new_capacity;
  registry->count = 0;
  registry->tombstone_count = 0;
  for (iree_host_size_t i = 0; i < old_capacity; ++i) {
    if (old_states[i] == IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE) {
      iree_hip_handle_registry_insert_unchecked(registry, old_handles[i]);
    }
  }
  iree_allocator_free(iree_allocator_system(), old_states);
  iree_allocator_free(iree_allocator_system(), old_handles);
  return iree_ok_status();
}

void iree_hip_handle_registry_initialize(iree_hip_handle_registry_t* registry) {
  IREE_ASSERT_ARGUMENT(registry);
  iree_slim_mutex_initialize(&registry->mutex);
  registry->handles = NULL;
  registry->states = NULL;
  registry->capacity = 0;
  registry->count = 0;
  registry->tombstone_count = 0;
}

void iree_hip_handle_registry_deinitialize(
    iree_hip_handle_registry_t* registry) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT(registry->count == 0);
  iree_allocator_free(iree_allocator_system(), registry->states);
  iree_allocator_free(iree_allocator_system(), registry->handles);
  iree_slim_mutex_deinitialize(&registry->mutex);
}

iree_status_t iree_hip_handle_registry_insert(
    iree_hip_handle_registry_t* registry, uintptr_t handle) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(handle);

  iree_slim_mutex_lock(&registry->mutex);
  iree_status_t status = iree_ok_status();
  if (registry->capacity == 0) {
    status = iree_hip_handle_registry_rehash(registry, 16);
  } else if (registry->count + registry->tombstone_count >=
             registry->capacity - registry->capacity / 4 - 1) {
    iree_host_size_t new_capacity = registry->capacity;
    if (registry->count >= registry->capacity / 2) {
      if (IREE_UNLIKELY(!iree_host_size_checked_mul(registry->capacity, 2,
                                                    &new_capacity))) {
        status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                                  "opaque handle registry size overflow");
      }
    }
    if (iree_status_is_ok(status)) {
      status = iree_hip_handle_registry_rehash(registry, new_capacity);
    }
  }

  iree_host_size_t insertion_slot = 0;
  if (iree_status_is_ok(status)) {
    const iree_host_size_t mask = registry->capacity - 1;
    iree_host_size_t slot = iree_hip_handle_registry_hash(handle) & mask;
    insertion_slot = slot;
    bool found_tombstone = false;
    while (registry->states[slot] != IREE_HIP_HANDLE_REGISTRY_SLOT_EMPTY) {
      if (registry->states[slot] == IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE &&
          registry->handles[slot] == handle) {
        status = iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                                  "opaque handle is already registered");
        break;
      }
      if (!found_tombstone &&
          registry->states[slot] == IREE_HIP_HANDLE_REGISTRY_SLOT_TOMBSTONE) {
        insertion_slot = slot;
        found_tombstone = true;
      }
      slot = (slot + 1) & mask;
      if (!found_tombstone) insertion_slot = slot;
    }
  }
  if (iree_status_is_ok(status)) {
    if (registry->states[insertion_slot] ==
        IREE_HIP_HANDLE_REGISTRY_SLOT_TOMBSTONE) {
      --registry->tombstone_count;
    }
    registry->handles[insertion_slot] = handle;
    registry->states[insertion_slot] = IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE;
    ++registry->count;
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return status;
}

bool iree_hip_handle_registry_lookup_retain(
    iree_hip_handle_registry_t* registry, uintptr_t handle,
    iree_hip_handle_registry_retain_fn_t retain_fn) {
  IREE_ASSERT_ARGUMENT(registry);
  IREE_ASSERT_ARGUMENT(retain_fn);
  if (!handle) return false;

  bool found = false;
  iree_slim_mutex_lock(&registry->mutex);
  if (registry->capacity != 0) {
    const iree_host_size_t mask = registry->capacity - 1;
    iree_host_size_t slot = iree_hip_handle_registry_hash(handle) & mask;
    while (registry->states[slot] != IREE_HIP_HANDLE_REGISTRY_SLOT_EMPTY) {
      if (registry->states[slot] == IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE &&
          registry->handles[slot] == handle) {
        retain_fn(handle);
        found = true;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return found;
}

bool iree_hip_handle_registry_remove(iree_hip_handle_registry_t* registry,
                                     uintptr_t handle) {
  IREE_ASSERT_ARGUMENT(registry);
  if (!handle) return false;

  bool found = false;
  iree_slim_mutex_lock(&registry->mutex);
  if (registry->capacity != 0) {
    const iree_host_size_t mask = registry->capacity - 1;
    iree_host_size_t slot = iree_hip_handle_registry_hash(handle) & mask;
    while (registry->states[slot] != IREE_HIP_HANDLE_REGISTRY_SLOT_EMPTY) {
      if (registry->states[slot] == IREE_HIP_HANDLE_REGISTRY_SLOT_LIVE &&
          registry->handles[slot] == handle) {
        registry->states[slot] = IREE_HIP_HANDLE_REGISTRY_SLOT_TOMBSTONE;
        --registry->count;
        ++registry->tombstone_count;
        found = true;
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  iree_slim_mutex_unlock(&registry->mutex);
  return found;
}
