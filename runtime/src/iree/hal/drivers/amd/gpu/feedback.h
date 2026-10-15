// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_FEEDBACK_H_
#define IREE_HAL_DRIVERS_AMD_GPU_FEEDBACK_H_

#include "iree/hal/drivers/amd/fabric/memory.h"
#include "iree/hal/drivers/amd/gpu/completion.h"
#include "iree/hal/drivers/amd/gpu/source_context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Loom's existing device feedback ABI. These records contain device addresses;
// the host uses its own mappings instead of assuming unified pointer values.
typedef struct iree_hal_amd_gpu_feedback_config_t {
  // Complete configuration size in bytes.
  uint32_t record_length;
  // Configuration ABI version, currently zero.
  uint32_t abi_version;
  // Bit zero enables shared-channel publication.
  uint32_t flags;
  // Zero padding preserving native 64-bit alignment.
  uint32_t reserved0;
  // Device address of the shared channel control block.
  uint64_t channel_base;
  // Device address of the native 64-byte notification signal ABI.
  uint64_t notify_signal;
  // Opaque host context copied by the shader, never dereferenced on device.
  uint64_t source_context;
  // Zero reserved configuration fields.
  uint64_t reserved[3];
} iree_hal_amd_gpu_feedback_config_t;
static_assert(sizeof(iree_hal_amd_gpu_feedback_config_t) == 64,
              "device feedback configuration ABI");

typedef struct iree_hal_amd_gpu_feedback_control_t {
  // Complete control size in bytes.
  uint32_t record_length;
  // Channel ABI version, currently zero.
  uint32_t abi_version;
  // Bit zero enables publication.
  uint32_t flags;
  // Zero padding preserving native 64-bit alignment.
  uint32_t reserved0;
  // Device base of a twice-mapped circular allocation.
  uint64_t ring_base;
  // Power-of-two physical capacity, excluding the alias.
  uint64_t ring_capacity;
  // Host release-published byte position available for producer reuse.
  uint64_t read_tail;
  // GPU CAS-reserved byte position; reservation does not imply publication.
  uint64_t reservation_head;
  // GPU count of reports dropped because the finite channel was full.
  uint64_t dropped_packet_count;
  // Zero reserved control field.
  uint64_t reserved;
} iree_hal_amd_gpu_feedback_control_t;
static_assert(sizeof(iree_hal_amd_gpu_feedback_control_t) == 64,
              "device feedback control ABI");

typedef struct iree_hal_amd_gpu_feedback_packet_t {
  // Complete 64-byte-aligned packet extent, including payload and padding.
  uint32_t record_length;
  // Header extent, at least 64 bytes.
  uint16_t header_length;
  // Existing schema: ASAN=1, printf=2, host call=3, TSAN=4, UBSAN=5.
  uint16_t kind;
  // Packet behavior; bit zero denotes an asynchronous report.
  uint32_t flags;
  // Release-published READY=1; other values are not consumable.
  uint32_t state;
  // Monotonic reservation byte position, distinguishing wrap generations.
  uint64_t sequence;
  // Shader dispatch pointer supplied as diagnostic context only.
  uint64_t source_dispatch_ptr;
  // Source workgroup X coordinate.
  uint32_t source_workgroup_id_x;
  // Source workitem X coordinate.
  uint32_t source_workitem_id_x;
  // Opaque retained source context from the executable's configuration.
  uint64_t source_context;
  // Zero reserved packet fields.
  uint64_t reserved[2];
} iree_hal_amd_gpu_feedback_packet_t;
static_assert(sizeof(iree_hal_amd_gpu_feedback_packet_t) == 64,
              "device feedback packet ABI");

// One physical device's live shared channel, with a sole completion-thread
// consumer. Producers reserve with CAS, publish READY, and notify the existing
// native event. Executable registration is cold and independently synchronized.
typedef struct iree_hal_amd_gpu_feedback_t {
  // Borrowed native owner, alive through all shader and host accesses.
  iree_hal_amd_gpu_context_t* context;
  // Borrowed public source identity, alive through completion-worker shutdown.
  iree_hal_device_t* device;
  // Borrowed immutable display name, captured during device construction.
  iree_string_view_t device_id;
  // Native control block and notification signal, with a coherent host view.
  iree_hal_amd_fabric_memory_t control_memory;
  // Physical report storage shared by both circular aliases.
  iree_hal_amd_fabric_memory_t ring_memory;
  // Owned host/GPU aliases, removed before physical backing is released.
  amdf_memory_reservation_t* reservation;
  // Host control pointer within control_memory.
  iree_hal_amd_gpu_feedback_control_t* control;
  // Host base of the twice-mapped ring, independent of its device address.
  uint8_t* ring;
  // Immutable device-global values, specialized with each source context.
  iree_hal_amd_gpu_feedback_config_t config;
  // Owned source metadata surviving early executable release.
  iree_hal_amd_gpu_source_context_registry_t sources;
  // First reported channel failure; only the completion worker accesses it.
  iree_status_t failure;
  // Last drop counter reported by the completion worker.
  uint64_t reported_drops;
} iree_hal_amd_gpu_feedback_t;

// Cold device construction on the memory service. Uses the existing feedback
// ABI and requires qualified native shader-wake operands. On failure, rolls
// back acquired resources; a failed native removal preserves its backing and
// latches context->cleanup_blocked. Disabled devices need no channel
// allocation.
iree_status_t iree_hal_amd_gpu_feedback_initialize(
    iree_hal_amd_gpu_context_t* context, iree_hal_device_t* device,
    const iree_hal_amd_fabric_memory_options_t* memory,
    uint32_t address_domain_ordinal, iree_allocator_t host_allocator,
    iree_hal_amd_gpu_feedback_t* out_feedback);

// Called after all queues and the sole drain consumer join. False indicates
// failed native alias removal, which preserves the remaining ownership graph.
bool iree_hal_amd_gpu_feedback_deinitialize(
    iree_hal_amd_gpu_feedback_t* feedback);

// Drains a bounded ready prefix without waiting for unpublished reservations.
// Shader/native publication errors are reported once through the device sink.
bool iree_hal_amd_gpu_feedback_poll(void* user_data, amdf_status_t wait_status);

// Drains the ready prefix observed at entry before publishing host completion.
// Called only by the sole completion worker; never waits for a reserved hole.
void iree_hal_amd_gpu_feedback_drain(iree_hal_amd_gpu_feedback_t* feedback);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_HAL_DRIVERS_AMD_GPU_FEEDBACK_H_
