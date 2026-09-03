// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/function_version.h"

#include <string.h>

void loom_function_version_owner_initialize(
    iree_arena_allocator_t* arena, loom_function_version_owner_t* out_owner) {
  IREE_ASSERT_ARGUMENT(arena);
  IREE_ASSERT_ARGUMENT(out_owner);
  *out_owner = (loom_function_version_owner_t){
      .arena = arena,
  };
}

iree_status_t loom_function_version_owner_reserve(
    loom_function_version_owner_t* owner, iree_host_size_t minimum_capacity) {
  IREE_ASSERT_ARGUMENT(owner);
  IREE_ASSERT_ARGUMENT(owner->arena);
  if (minimum_capacity <= owner->capacity) {
    return iree_ok_status();
  }

  iree_host_size_t new_capacity = owner->capacity > 0 ? owner->capacity : 8;
  while (new_capacity < minimum_capacity) {
    if (!iree_host_size_checked_mul(new_capacity, 2, &new_capacity)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "function-version capacity overflow");
    }
  }

  loom_function_version_t** new_storage = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      owner->arena, new_capacity, sizeof(*new_storage), (void**)&new_storage));
  if (owner->list.count > 0) {
    memcpy(new_storage, owner->storage,
           owner->list.count * sizeof(*new_storage));
  }
  owner->storage = new_storage;
  owner->capacity = new_capacity;
  owner->list.values = new_storage;
  return iree_ok_status();
}

iree_status_t loom_function_version_owner_append(
    loom_function_version_owner_t* owner, loom_function_version_t* version) {
  IREE_ASSERT_ARGUMENT(owner);
  IREE_ASSERT_ARGUMENT(version);
  iree_host_size_t required_capacity = 0;
  if (!iree_host_size_checked_add(owner->list.count, 1, &required_capacity)) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "function-version count overflow");
  }
  IREE_RETURN_IF_ERROR(
      loom_function_version_owner_reserve(owner, required_capacity));
  owner->storage[owner->list.count++] = version;
  return iree_ok_status();
}

bool loom_function_version_owner_remove(loom_function_version_owner_t* owner,
                                        loom_function_version_t* version) {
  if (!owner || !version) return false;
  for (iree_host_size_t i = 0; i < owner->list.count; ++i) {
    if (owner->storage[i] != version) continue;
    const iree_host_size_t trailing_count = owner->list.count - i - 1;
    if (trailing_count > 0) {
      memmove(&owner->storage[i], &owner->storage[i + 1],
              trailing_count * sizeof(*owner->storage));
    }
    --owner->list.count;
    owner->storage[owner->list.count] = NULL;
    return true;
  }
  return false;
}

loom_function_version_t* loom_function_version_list_find(
    const loom_function_version_list_t* list, loom_func_like_t function) {
  if (list == NULL || function.op == NULL) {
    return NULL;
  }
  for (iree_host_size_t i = 0; i < list->count; ++i) {
    loom_function_version_t* version = list->values[i];
    if (version != NULL && version->function.op == function.op &&
        version->function.vtable == function.vtable) {
      return version;
    }
  }
  return NULL;
}

void loom_function_version_update(loom_function_version_t* version,
                                  loom_func_like_t function) {
  IREE_ASSERT_ARGUMENT(version);
  IREE_ASSERT(function.op != NULL);
  version->function = function;
}
