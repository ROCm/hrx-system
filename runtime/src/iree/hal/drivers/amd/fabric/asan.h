// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_FABRIC_ASAN_H_
#define IREE_HAL_DRIVERS_AMD_FABRIC_ASAN_H_

#include "iree/base/threading/mutex.h"
#include "iree/hal/device.h"
#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/memory/asan_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_hal_amd_fabric_asan_shadow_mode_e {
  // Commit precise shadow slabs only when application storage is published.
  IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_SPARSE = 0,
  // Initially alias one poisoned slab throughout the reservation, replacing
  // aliases with precise slabs as application storage is published.
  IREE_HAL_AMD_FABRIC_ASAN_SHADOW_MODE_PREMAPPED = 1,
} iree_hal_amd_fabric_asan_shadow_mode_t;

typedef struct iree_hal_amd_fabric_asan_options_t {
  // Sparse commitment or initial shared poison aliases.
  iree_hal_amd_fabric_asan_shadow_mode_t shadow_mode;
  // Log2 application bytes represented by one shadow byte, at most seven.
  uint32_t shadow_scale_shift;
  // Reserved shadow address space, independent of committed physical bytes.
  uint64_t shadow_size;
  // Physical allocation granularity of precise shadow backing.
  uint64_t shadow_slab_size;
  // Reserved application window for owned device-local allocations.
  uint64_t owned_application_size;
  // Minimum poisoned bytes on either side of public pool allocations.
  uint64_t redzone_size;
  // Maximum freed application backing retained for stale-pointer checks.
  uint64_t quarantine_size;
} iree_hal_amd_fabric_asan_options_t;

// Existing 96-byte shader ABI, published through iree_asan_config.
typedef struct iree_hal_amd_fabric_asan_config_t {
  // Complete record size in bytes.
  uint32_t record_length;
  // Shader ABI version, currently zero.
  uint32_t abi_version;
  // Bit zero enables address checking.
  uint32_t flags;
  // Log2 application bytes represented by one shadow byte.
  uint32_t shadow_scale_shift;
  // Device address added to each shifted application address.
  uint64_t shadow_base;
  // First covered application address, currently zero.
  uint64_t application_window_base;
  // Complete application address coverage in bytes.
  uint64_t application_window_size;
  // Complete reserved shadow extent in bytes.
  uint64_t shadow_size;
  // Physical shadow slab extent in bytes.
  uint64_t shadow_slab_size;
  // Zero reserved shader fields.
  uint64_t reserved[5];
} iree_hal_amd_fabric_asan_config_t;
static_assert(sizeof(iree_hal_amd_fabric_asan_config_t) == 96,
              "ASAN configuration is a shader ABI");

typedef struct iree_hal_amd_fabric_asan_application_range_t {
  // Stable device address within the owned application reservation.
  uint64_t address;
  // Native mapping extent in bytes.
  uint64_t length;
  // Sorted free-list link while this range is not allocated.
  struct iree_hal_amd_fabric_asan_application_range_t* next;
} iree_hal_amd_fabric_asan_application_range_t;

typedef struct iree_hal_amd_fabric_asan_quarantine_entry_t {
  // FIFO link while the owner retains poisoned application backing.
  struct iree_hal_amd_fabric_asan_quarantine_entry_t* next;
  // Retained backing bytes charged to the quarantine budget.
  uint64_t mapped_size;
  // Releases the retained mapping and its enclosing entry storage.
  void(IREE_API_PTR* release_fn)(void* user_data);
  // Borrowed callback context kept alive by the enclosing entry owner.
  void* user_data;
} iree_hal_amd_fabric_asan_quarantine_entry_t;

typedef struct iree_hal_amd_fabric_asan_shadow_slab_t
    iree_hal_amd_fabric_asan_shadow_slab_t;

// Device-owned address shadow namespace. Native allocation/mapping is cold;
// pool advice only writes already committed shadow bytes. The sealed device
// group dominates this state and every provider borrowing it.
typedef struct iree_hal_amd_fabric_asan_t {
  // Borrowed native API table, or NULL before initialization.
  const amdf_api_t* api;
  // Borrowed device whose application and shadow addresses are represented.
  amdf_device_t* device;
  // Host allocator owning shadow metadata and application range nodes.
  iree_allocator_t host_allocator;
  // Borrowed diagnostic sink for native cleanup/cache-publication failures.
  iree_hal_device_event_sink_t event_sink;
  // Qualified shadow allocation policy, borrowing the scope but owning access.
  iree_hal_amd_fabric_memory_options_t memory;
  // Prepared native shadow mapping source, referenced by memory.accesses.
  amdf_memory_device_access_t access;
  // Immutable shader configuration and coverage geometry.
  iree_hal_amd_fabric_asan_config_t config;
  // Immutable pool layout policy advertised by the device.
  iree_hal_asan_pool_options_t pool_options;
  // Owned sparse shadow reservation, removed before backing is freed.
  amdf_memory_reservation_t* shadow_reservation;
  // Owned application reservation for compact device-local placement.
  amdf_memory_reservation_t* application_reservation;
  // Stable native application interval and mapping granularity.
  amdf_memory_reservation_access_info_t application_info;
  // Whole-backing requirements for aliases in the application reservation.
  amdf_memory_flags_t application_source_flags;
  // Optional shared poison backing used by PREMAPPED mode.
  iree_hal_amd_fabric_memory_t alias_memory;
  // Serializes physical shadow commitment and host poison updates.
  iree_slim_mutex_t shadow_mutex;
  // Dense table of independently host-mapped precise shadow slabs.
  iree_hal_amd_fabric_asan_shadow_slab_t* slabs;
  // Number of initialized entries in slabs.
  iree_host_size_t slab_count;
  // Allocated entry capacity of slabs.
  iree_host_size_t slab_capacity;
  // Serializes application free-list mutation and native alias updates.
  iree_slim_mutex_t application_mutex;
  // Sorted list of unused intervals within application_reservation.
  iree_hal_amd_fabric_asan_application_range_t* application_free_ranges;
  // Serializes the quarantine FIFO; native releases run outside this lock.
  iree_slim_mutex_t quarantine_mutex;
  // Oldest retained poisoned application backing.
  iree_hal_amd_fabric_asan_quarantine_entry_t* quarantine_head;
  // Newest retained poisoned application backing.
  iree_hal_amd_fabric_asan_quarantine_entry_t* quarantine_tail;
  // Current retained backing bytes.
  uint64_t quarantine_size;
  // Maximum retained backing bytes.
  uint64_t quarantine_limit;
  // Cumulative backing releases due to quarantine pressure.
  uint64_t quarantine_eviction_count;
  // Sticky native publication failure; instrumented admission must stop when
  // shadow updates or translations can no longer represent allocation state.
  iree_atomic_int64_t failure;
  // Optional owner notification when poison publication becomes unreliable.
  void(IREE_API_PTR* failure_callback)(void* user_data,
                                       iree_status_code_t status_code);
  // Borrowed context passed to failure_callback.
  void* failure_user_data;
  // A failed native alias removal prevents releasing borrowed native parents.
  bool cleanup_blocked;
} iree_hal_amd_fabric_asan_t;

// Initializes the established 32 TiB shadow/8 TiB application defaults, with
// 128 MiB precise slabs, 256-byte redzones, and a 256 MiB quarantine.
void iree_hal_amd_fabric_asan_options_initialize(
    iree_hal_amd_fabric_asan_options_t* out_options);

// Prepares native reservations and the selected shadow backing scope. The
// caller owns out_state before calling and deinitializes it on both success
// and failure. Native parents remain borrowed until deinitialization succeeds.
iree_status_t iree_hal_amd_fabric_asan_initialize(
    const amdf_api_t* api, amdf_instance_t* instance, amdf_device_t* device,
    uint32_t address_domain_ordinal, amdf_memory_scope_t* shadow_scope,
    const iree_hal_amd_fabric_asan_options_t* options,
    iree_hal_device_event_sink_t event_sink, iree_allocator_t host_allocator,
    iree_hal_amd_fabric_asan_t* out_state);

// Releases quarantined mappings, reservations, and backing after all users
// join. False preserves the remaining graph after a native removal failure.
bool iree_hal_amd_fabric_asan_deinitialize(iree_hal_amd_fabric_asan_t* state);

// Samples quarantine and physical shadow accounting under their owning locks.
void iree_hal_amd_fabric_asan_query_statistics(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_device_asan_observation_t* out_statistics);

// Commits shadow covering an external/native address range before publication.
iree_status_t iree_hal_amd_fabric_asan_map_range(
    iree_hal_amd_fabric_asan_t* state, uint64_t address, uint64_t length);

// Commits and enables an imported logical range. Partial boundary granules may
// widen addressability but never shrink another already-live imported range.
iree_status_t iree_hal_amd_fabric_asan_publish_imported_range(
    iree_hal_amd_fabric_asan_t* state, uint64_t address, uint64_t length);

// Advises a verified pool layout whose shadow was committed by slab acquire.
// A zero accessible_length poisons the complete backing range on release.
void iree_hal_amd_fabric_asan_advise_range(iree_hal_amd_fabric_asan_t* state,
                                           uint64_t backing_address,
                                           uint64_t backing_length,
                                           uint64_t accessible_offset,
                                           uint64_t accessible_length);

// Maps owned backing into a compact application interval. The native source
// access has already been prepared for MAPPING_SOURCE on state->device. A
// failed native update can return an owned range; retain its backing until
// unmap_application succeeds, including on construction rollback.
iree_status_t iree_hal_amd_fabric_asan_map_application(
    iree_hal_amd_fabric_asan_t* state, amdf_memory_t* memory,
    uint32_t memory_access_ordinal, uint64_t length, uint64_t alignment,
    iree_hal_amd_fabric_asan_application_range_t** out_range);

// Removes a retired alias and returns its interval. Failure retains backing
// ownership at the caller and marks the native parents as non-releasable.
bool iree_hal_amd_fabric_asan_unmap_application(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_application_range_t* range);

// Transfers an owned poisoned mapping to the bounded FIFO. Imports retain
// their external lifetime contract and are never quarantined here.
void iree_hal_amd_fabric_asan_quarantine(
    iree_hal_amd_fabric_asan_t* state,
    iree_hal_amd_fabric_asan_quarantine_entry_t* entry, uint64_t mapped_size,
    void(IREE_API_PTR* release_fn)(void* user_data), void* user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_FABRIC_ASAN_H_
