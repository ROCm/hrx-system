// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_HANDLE_REGISTRY_H_
#define HRX_BINDING_HIP_HANDLE_REGISTRY_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Tracks live opaque API handles without dereferencing caller-provided values.
typedef struct iree_hip_handle_registry_t {
  // Serializes all registry operations.
  iree_slim_mutex_t mutex;
  // Open-addressed table of live handle values.
  uintptr_t* handles;
  // Slot state array parallel to |handles|.
  uint8_t* states;
  // Power-of-two slot count in |handles| and |states|.
  iree_host_size_t capacity;
  // Number of live handles.
  iree_host_size_t count;
  // Number of removed slots awaiting reuse or rehashing.
  iree_host_size_t tombstone_count;
} iree_hip_handle_registry_t;

// Retains a handle while its registry entry is locked and known to be live.
typedef void (*iree_hip_handle_registry_retain_fn_t)(uintptr_t handle);

void iree_hip_handle_registry_initialize(iree_hip_handle_registry_t* registry);

// Requires the registry to be empty.
void iree_hip_handle_registry_deinitialize(
    iree_hip_handle_registry_t* registry);

iree_status_t iree_hip_handle_registry_insert(
    iree_hip_handle_registry_t* registry, uintptr_t handle);

// Looks up and retains a live handle before releasing the registry lock.
bool iree_hip_handle_registry_lookup_retain(
    iree_hip_handle_registry_t* registry, uintptr_t handle,
    iree_hip_handle_registry_retain_fn_t retain_fn);

// Removes a live handle, transferring its existing ownership to the caller.
bool iree_hip_handle_registry_remove(iree_hip_handle_registry_t* registry,
                                     uintptr_t handle);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_HANDLE_REGISTRY_H_
