// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUILDER_H_
#define IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUILDER_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/abi/command_buffer.h"
#include "iree/hal/drivers/amdgpu/util/pm4_barrier.h"
#include "iree/hal/drivers/amdgpu/util/pm4_dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Growable or borrowed PM4 dword storage.
typedef struct iree_hal_amdgpu_pm4_dword_builder_t {
  // Host allocator used to grow |dwords|.
  iree_allocator_t host_allocator;
  // PM4 dwords being emitted into host-owned or resident storage.
  uint32_t* dwords;
  // Number of populated PM4 dwords in |dwords|.
  uint32_t dword_count;
  // Number of allocated PM4 dwords in |dwords|.
  uint32_t capacity;
  // True when |dwords| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_dword_builder_t;

// Growable or borrowed byte storage for compact records and templates.
typedef struct iree_hal_amdgpu_pm4_byte_builder_t {
  // Host allocator used to grow |bytes|.
  iree_allocator_t host_allocator;
  // Bytes being emitted into host-owned or resident storage.
  uint8_t* bytes;
  // Number of populated bytes in |bytes|.
  iree_host_size_t length;
  // Number of allocated bytes in |bytes|.
  iree_host_size_t capacity;
  // True when |bytes| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_byte_builder_t;

// Growable or borrowed dynamic binding fixup storage.
typedef struct iree_hal_amdgpu_pm4_fixup_entry_builder_t {
  // Host allocator used to grow |entries|.
  iree_allocator_t host_allocator;
  // Fixup entries being emitted into host-owned or resident storage.
  iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries;
  // Number of populated fixup entries.
  uint32_t count;
  // Number of allocated fixup entries.
  uint32_t capacity;
  // True when |entries| must be freed by the builder.
  bool owns_storage;
} iree_hal_amdgpu_pm4_fixup_entry_builder_t;

void iree_hal_amdgpu_pm4_dword_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_dword_builder_t* out_builder);

void iree_hal_amdgpu_pm4_dword_builder_deinitialize(
    iree_hal_amdgpu_pm4_dword_builder_t* builder);

// Replaces owned storage with borrowed fixed-capacity |dwords|.
void iree_hal_amdgpu_pm4_dword_builder_borrow_storage(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t* dwords,
    uint32_t capacity);

// Ensures capacity for at least |required_capacity| dwords.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_reserve(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t required_capacity);

// Appends uninitialized dwords and returns their starting address.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_append(
    iree_hal_amdgpu_pm4_dword_builder_t* builder, uint32_t dword_count,
    uint32_t** out_dwords);

// Emits one execution or fixup barrier into |builder|.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_barrier(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    iree_hal_amdgpu_vendor_packet_capability_flags_t capabilities,
    iree_hal_amdgpu_pm4_barrier_flags_t barrier_flags,
    iree_hsa_fence_scope_t acquire_scope, iree_hsa_fence_scope_t release_scope);

// Appends a previously derived static dispatch setup stream.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_setup(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t
        source_dwords[IREE_HAL_AMDGPU_PM4_DISPATCH_SETUP_DWORD_COUNT],
    uint32_t setup_dword_count);

// Emits dispatch user data for |launch_state|.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_user_data(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    uint64_t kernarg_address, const void* kernarg_preload_data);

// Emits one direct dispatch with the given thread counts.
iree_status_t iree_hal_amdgpu_pm4_dword_builder_emit_dispatch_direct(
    iree_hal_amdgpu_pm4_dword_builder_t* builder,
    const uint32_t dispatch_thread_count[3], uint32_t dispatch_initiator);

// Resolves whether a kernarg range is copied into preloaded user-data SGPRs.
iree_status_t iree_hal_amdgpu_pm4_dispatch_kernarg_range_preload_offset(
    const iree_hal_amdgpu_pm4_dispatch_launch_state_t* launch_state,
    uint32_t kernarg_byte_offset, uint32_t kernarg_byte_length,
    bool* out_is_preloaded, uint32_t* out_preload_dword_offset);

void iree_hal_amdgpu_pm4_byte_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_byte_builder_t* out_builder);

void iree_hal_amdgpu_pm4_byte_builder_deinitialize(
    iree_hal_amdgpu_pm4_byte_builder_t* builder);

// Replaces owned storage with borrowed fixed-capacity |bytes|.
void iree_hal_amdgpu_pm4_byte_builder_borrow_storage(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, uint8_t* bytes,
    iree_host_size_t capacity);

// Ensures capacity for at least |required_capacity| bytes.
iree_status_t iree_hal_amdgpu_pm4_byte_builder_reserve(
    iree_hal_amdgpu_pm4_byte_builder_t* builder,
    iree_host_size_t required_capacity);

// Appends zeroed aligned bytes and returns their offset and address.
iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_aligned(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t alignment,
    iree_host_size_t byte_length, uint32_t* out_offset, uint8_t** out_bytes);

// Appends uninitialized compact-record bytes.
iree_status_t iree_hal_amdgpu_pm4_byte_builder_append_record(
    iree_hal_amdgpu_pm4_byte_builder_t* builder, iree_host_size_t byte_length,
    uint8_t** out_bytes);

void iree_hal_amdgpu_pm4_fixup_entry_builder_initialize(
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* out_builder);

void iree_hal_amdgpu_pm4_fixup_entry_builder_deinitialize(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder);

// Replaces owned storage with borrowed fixed-capacity |entries|.
void iree_hal_amdgpu_pm4_fixup_entry_builder_borrow_storage(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t* entries,
    uint32_t capacity);

// Ensures capacity for at least |required_capacity| fixup entries.
iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_reserve(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    uint32_t required_capacity);

// Appends one dynamic binding fixup entry.
iree_status_t iree_hal_amdgpu_pm4_fixup_entry_builder_append(
    iree_hal_amdgpu_pm4_fixup_entry_builder_t* builder,
    iree_hal_amdgpu_command_buffer_pm4_fixup_entry_t entry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_PM4_COMMAND_BUILDER_H_
