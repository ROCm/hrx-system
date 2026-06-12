// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_UTIL_GLOBAL_TABLE_H_
#define IREE_HAL_DRIVERS_AMDGPU_UTIL_GLOBAL_TABLE_H_

#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_global_table_entry_t
    iree_hal_amdgpu_global_table_entry_t;

typedef struct iree_hal_amdgpu_global_table_resolver_t {
  // Resolver-specific state passed to all callbacks.
  void* user_data;

  // Verifies that |name| resolves to an executable variable and returns its
  // byte length. The table calls this on the first loaded physical device only.
  iree_status_t (*verify)(void* user_data, iree_string_view_t name,
                          iree_host_size_t verification_physical_device_ordinal,
                          iree_device_size_t* out_byte_length);

  // Creates an executable-owned buffer alias for |name| on one physical device.
  iree_status_t (*create_buffer)(
      void* user_data, iree_string_view_t name,
      iree_device_size_t expected_byte_length,
      iree_hal_queue_affinity_t selected_queue_affinity,
      iree_host_size_t physical_device_ordinal, iree_hal_buffer_t** out_buffer);
} iree_hal_amdgpu_global_table_resolver_t;

typedef struct iree_hal_amdgpu_global_table_params_t {
  // Host allocator used for table storage.
  iree_allocator_t host_allocator;

  // Queue affinity domain of the owning AMDGPU logical device.
  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain;

  // Bitmask of physical device ordinals this executable was loaded on.
  uint64_t loaded_physical_device_mask;

  // Number of physical devices in the owning AMDGPU logical device.
  iree_host_size_t physical_device_count;

  // Backend resolver used for HSA symbol verification and buffer creation.
  iree_hal_amdgpu_global_table_resolver_t resolver;
} iree_hal_amdgpu_global_table_params_t;

typedef struct iree_hal_amdgpu_global_table_hsa_params_t {
  // Host allocator used for table storage and cached buffer wrappers.
  iree_allocator_t host_allocator;

  // Queue affinity domain of the owning AMDGPU logical device.
  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain;

  // Bitmask of physical device ordinals this executable was loaded on.
  uint64_t loaded_physical_device_mask;

  // Borrowed HSA dynamic symbol table.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Borrowed logical device used in global buffer placements.
  iree_hal_device_t* device;

  // Borrowed HSA executable containing variable symbols.
  hsa_executable_t executable;

  // Number of physical device agents in |device_agents|.
  iree_host_size_t device_agent_count;

  // Borrowed physical device agent table owned by the executable.
  const hsa_agent_t* device_agents;
} iree_hal_amdgpu_global_table_hsa_params_t;

typedef struct iree_hal_amdgpu_global_table_t {
  // True when the table has been initialized and must be deinitialized.
  bool initialized;

  // Host allocator used for all table-owned allocations.
  iree_allocator_t host_allocator;

  // Queue affinity domain of the owning logical device.
  iree_hal_amdgpu_queue_affinity_domain_t queue_affinity_domain;

  // Bitmask of physical device ordinals with loaded code objects.
  uint64_t loaded_physical_device_mask;

  // Number of physical devices in the owning logical device.
  iree_host_size_t physical_device_count;

  // Backend resolver used for HSA symbol verification and buffer creation.
  iree_hal_amdgpu_global_table_resolver_t resolver;

  // HSA-backed resolver state used by the production AMDGPU executable path.
  struct {
    // Borrowed HSA dynamic symbol table.
    const iree_hal_amdgpu_libhsa_t* libhsa;

    // Borrowed logical device used in global buffer placements.
    iree_hal_device_t* device;

    // Borrowed HSA executable containing variable symbols.
    hsa_executable_t executable;

    // Number of physical device agents in |device_agents|.
    iree_host_size_t device_agent_count;

    // Borrowed physical device agent table owned by the executable.
    const hsa_agent_t* device_agents;
  } hsa;

  // Guards entry publication and cached buffer publication.
  iree_slim_mutex_t mutex;

  // Dense table indexed by executable-local global handle value.
  iree_hal_amdgpu_global_table_entry_t** entries;

  // Number of populated entries.
  iree_host_size_t entry_count;

  // Capacity of |entries|.
  iree_host_size_t entry_capacity;
} iree_hal_amdgpu_global_table_t;

// Initializes |out_table| for executable global resolution.
iree_status_t iree_hal_amdgpu_global_table_initialize(
    const iree_hal_amdgpu_global_table_params_t* params,
    iree_hal_amdgpu_global_table_t* out_table);

// Initializes |out_table| with the built-in HSA executable global resolver.
iree_status_t iree_hal_amdgpu_global_table_initialize_hsa(
    const iree_hal_amdgpu_global_table_hsa_params_t* params,
    iree_hal_amdgpu_global_table_t* out_table);

// Releases cached buffer aliases and table storage.
void iree_hal_amdgpu_global_table_deinitialize(
    iree_hal_amdgpu_global_table_t* table);

// Finds a variable symbol by name and interns a global handle for it.
iree_status_t iree_hal_amdgpu_global_table_lookup(
    iree_hal_amdgpu_global_table_t* table, iree_string_view_t name,
    iree_hal_executable_global_t* out_global);

// Returns metadata for |global|. Returned string storage is table-owned.
iree_status_t iree_hal_amdgpu_global_table_info(
    iree_hal_amdgpu_global_table_t* table, iree_hal_executable_global_t global,
    iree_hal_executable_global_info_t* out_info);

// Returns an executable-owned buffer alias for |global| on the selected device.
iree_status_t iree_hal_amdgpu_global_table_buffer(
    iree_hal_amdgpu_global_table_t* table, iree_hal_executable_global_t global,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_buffer_t** out_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_UTIL_GLOBAL_TABLE_H_
