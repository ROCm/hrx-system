// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_
#define IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/asan.h"
#include "iree/hal/drivers/amdgpu/api.h"
#include "iree/hal/drivers/amdgpu/util/shadow_map.h"

typedef struct iree_hal_amdgpu_physical_device_t
    iree_hal_amdgpu_physical_device_t;
typedef struct iree_hal_amdgpu_system_t iree_hal_amdgpu_system_t;

typedef struct iree_hal_amdgpu_asan_application_range_t {
  // Device-visible application address assigned to this range.
  uint64_t address;

  // Length of the application range in bytes.
  iree_device_size_t length;

  // Next range in the sorted free list while owned by ASAN state.
  struct iree_hal_amdgpu_asan_application_range_t* next;
} iree_hal_amdgpu_asan_application_range_t;

typedef void(IREE_API_PTR* iree_hal_amdgpu_asan_quarantine_release_fn_t)(
    void* user_data);

typedef struct iree_hal_amdgpu_asan_quarantine_entry_t {
  // Next entry in the ASAN state quarantine FIFO while retained.
  struct iree_hal_amdgpu_asan_quarantine_entry_t* next;

  // Mapping byte length counted against the quarantine budget.
  iree_device_size_t mapped_size;

  // Callback invoked to release the retained mapping and entry storage.
  iree_hal_amdgpu_asan_quarantine_release_fn_t release_fn;

  // User data passed to |release_fn|.
  void* user_data;
} iree_hal_amdgpu_asan_quarantine_entry_t;

typedef struct iree_hal_amdgpu_asan_state_statistics_t {
  // Current total bytes retained by the quarantine FIFO.
  iree_device_size_t quarantine_size;

  // Cumulative count of mappings released due to quarantine pressure.
  uint64_t quarantine_eviction_count;

  // Number of precise physical shadow slabs mapped in the shadow map.
  iree_host_size_t shadow_mapped_slab_count;

  // Physical shadow bytes committed by the shadow map.
  iree_device_size_t shadow_committed_size;
} iree_hal_amdgpu_asan_state_statistics_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_hal_amdgpu_asan_state_t
//===----------------------------------------------------------------------===//

// Logical-device ASAN state shared by allocator, executable, and queue paths.
typedef struct iree_hal_amdgpu_asan_state_t {
  // True when ASAN support is enabled for the logical device.
  uint32_t is_enabled : 1;

  // Sparse shadow map used by ASAN-instrumented device code.
  iree_hal_amdgpu_shadow_map_t shadow_map;

  // Borrowed HSA API table used to release the application reservation.
  const iree_hal_amdgpu_libhsa_t* libhsa;

  // Host allocator used for ASAN state-owned range nodes.
  iree_allocator_t host_allocator;

  // Base pointer of the ASAN-owned application virtual address reservation.
  IREE_AMDGPU_DEVICE_PTR void* owned_application_base_ptr;

  // Size of the ASAN-owned application virtual address reservation.
  iree_device_size_t owned_application_size;

  // Guards application free-list mutation.
  iree_slim_mutex_t application_mutex;

  // Sorted list of free ranges within the ASAN-owned application window.
  iree_hal_amdgpu_asan_application_range_t* application_free_ranges;

  // Guards quarantine FIFO mutation.
  iree_slim_mutex_t quarantine_mutex;

  // Oldest freed poisoned mapping retained for stale-pointer checks.
  iree_hal_amdgpu_asan_quarantine_entry_t* quarantine_head;

  // Newest freed poisoned mapping retained for stale-pointer checks.
  iree_hal_amdgpu_asan_quarantine_entry_t* quarantine_tail;

  // Current total bytes retained by the quarantine FIFO.
  iree_device_size_t quarantine_size;

  // Maximum total bytes to retain in the quarantine FIFO.
  iree_device_size_t quarantine_limit;

  // Cumulative count of mappings released due to quarantine pressure.
  uint64_t quarantine_eviction_count;
} iree_hal_amdgpu_asan_state_t;

// Initializes |out_state| from logical-device options.
//
// When ASAN is disabled this leaves |out_state| zeroed and returns OK. When
// enabled the state reserves a device-visible application virtual address
// window for HAL-owned allocations, reserves the configured shadow virtual
// address coverage, and configures later physical shadow slab mappings from
// the selected shadow backing policy with shared topology access.
iree_status_t iree_hal_amdgpu_asan_state_initialize(
    const iree_hal_amdgpu_logical_device_options_t* options,
    iree_hal_amdgpu_system_t* system, iree_host_size_t physical_device_count,
    iree_hal_amdgpu_physical_device_t* const* physical_devices,
    iree_allocator_t host_allocator, iree_hal_amdgpu_asan_state_t* out_state);

// Releases any ASAN application-window and shadow resources owned by |state|.
void iree_hal_amdgpu_asan_state_deinitialize(
    iree_hal_amdgpu_asan_state_t* state);

// Returns true when |state| owns enabled ASAN resources.
bool iree_hal_amdgpu_asan_state_is_enabled(
    const iree_hal_amdgpu_asan_state_t* state);

// Snapshots ASAN memory-pressure counters.
//
// Disabled ASAN state reports all-zero counters. The snapshot uses the
// quarantine and shadow-map locks independently and may race with later
// allocation, release, or shadow-slab mapping work.
void iree_hal_amdgpu_asan_state_query_statistics(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_state_statistics_t* out_statistics);

// Returns the mutable ASAN shadow map, or NULL when ASAN is disabled.
iree_hal_amdgpu_shadow_map_t* iree_hal_amdgpu_asan_state_shadow_map(
    iree_hal_amdgpu_asan_state_t* state);

// Ensures shadow slabs are mapped for the given device-visible application byte
// range.
//
// When ASAN is disabled this is a no-op. When enabled the range must fit within
// the configured application coverage; out-of-coverage ranges fail loudly
// because instrumented kernels would otherwise fault when loading shadow bytes.
iree_status_t iree_hal_amdgpu_asan_state_map_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length);

// Marks an application allocation with one addressable subrange.
//
// The entire mapped range beginning at |mapped_address| is first poisoned as a
// heap redzone. |accessible_length| bytes beginning at |accessible_address|
// are then marked addressable, with a final partial shadow byte when the
// accessible length is not shadow-granule aligned. The mapped range must fit
// inside the configured application coverage.
iree_status_t iree_hal_amdgpu_asan_state_publish_allocated_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t mapped_address,
    iree_device_size_t mapped_length, uint64_t accessible_address,
    iree_device_size_t accessible_length);

// Marks an application allocation after its shadow slabs are committed.
//
// This is the infallible companion used by ASAN slab-provider advice. It
// assumes all validation and shadow-slab allocation happened on a prior
// fallible setup path such as iree_hal_amdgpu_asan_state_map_range.
void iree_hal_amdgpu_asan_state_publish_allocated_range_raw(
    iree_hal_amdgpu_asan_state_t* state, uint64_t mapped_address,
    iree_device_size_t mapped_length, uint64_t accessible_address,
    iree_device_size_t accessible_length);

// Marks an externally-owned device-visible application range addressable.
//
// Imported ranges may begin or end inside a shadow granule. The ASAN shadow
// format can precisely encode partial tail granules, while a partial leading
// granule is widened to addressable to avoid false positives for valid imports.
iree_status_t iree_hal_amdgpu_asan_state_publish_imported_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t application_length);

// Re-poisons a previously published allocation before teardown.
//
// This performs no host allocations and asserts HSA shadow writes succeed.
// |application_address| and |mapped_length| must match a range successfully
// published by iree_hal_amdgpu_asan_state_publish_allocated_range.
void iree_hal_amdgpu_asan_state_publish_released_range(
    iree_hal_amdgpu_asan_state_t* state, uint64_t application_address,
    iree_device_size_t mapped_length);

// Retains a released poisoned mapping for stale-pointer checks.
//
// |entry| must stay valid until |release_fn| is invoked. This function performs
// no host allocations. When the quarantine budget is zero or the FIFO exceeds
// the configured budget, one or more oldest entries are released before return.
void iree_hal_amdgpu_asan_state_quarantine_entry(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_quarantine_entry_t* entry,
    iree_device_size_t mapped_size,
    iree_hal_amdgpu_asan_quarantine_release_fn_t release_fn, void* user_data);

// Assigns an application virtual address range from the ASAN-owned window.
//
// The state owns the overall HSA VMM reservation. Callers must map physical HSA
// VMM handles into the returned subrange before exposing it to kernels. The
// returned address is aligned to |alignment| when non-zero. Callers must return
// |out_application_range| with
// iree_hal_amdgpu_asan_state_release_application_range once the range is no
// longer mapped.
iree_status_t iree_hal_amdgpu_asan_state_reserve_application_range(
    iree_hal_amdgpu_asan_state_t* state, iree_device_size_t length,
    iree_device_size_t alignment, uint64_t* out_application_address,
    iree_hal_amdgpu_asan_application_range_t** out_application_range);

// Releases an application range previously reserved from |state|.
//
// This performs no host allocations and is safe to call from cleanup paths that
// have no status channel. |application_range| is consumed by the ASAN state.
void iree_hal_amdgpu_asan_state_release_application_range(
    iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_application_range_t* application_range);

// Populates |out_config| with the device-visible ASAN configuration.
//
// Callers must only call this when ASAN is enabled. Report fields remain zero
// until the report transport is attached to the logical-device ASAN state.
void iree_hal_amdgpu_asan_state_populate_config(
    const iree_hal_amdgpu_asan_state_t* state,
    iree_hal_amdgpu_asan_config_t* out_config);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_ASAN_STATE_H_
