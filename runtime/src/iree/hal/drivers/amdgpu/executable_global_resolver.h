// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_GLOBAL_RESOLVER_H_
#define IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_GLOBAL_RESOLVER_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/util/global_table.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// HSA-backed resolver state for one loaded executable variant.
typedef struct iree_hal_amdgpu_executable_global_resolver_t {
  // Host allocator used for executable-owned buffer aliases.
  iree_allocator_t host_allocator;

  // Physical device ordinal selected by the executable's queue family.
  iree_host_size_t physical_device_ordinal;

  // Borrowed HSA dynamic symbol table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Borrowed logical device used in global buffer placements.
  iree_hal_device_t* device;

  // Borrowed HSA executable containing variable symbols.
  hsa_executable_t executable;

  // Physical device agent selected by the executable's queue family.
  hsa_agent_t device_agent;
} iree_hal_amdgpu_executable_global_resolver_t;

// Initializes |out_table| with callbacks backed by |resolver|.
// |resolver| must remain valid until |out_table| is deinitialized.
iree_status_t iree_hal_amdgpu_executable_global_resolver_initialize_table(
    iree_hal_amdgpu_executable_global_resolver_t* resolver,
    iree_hal_amdgpu_global_table_t* out_table);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_EXECUTABLE_GLOBAL_RESOLVER_H_
