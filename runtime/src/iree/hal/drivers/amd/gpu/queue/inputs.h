// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_INPUTS_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_INPUTS_H_

#include "iree/hal/api.h"
#include "iree/hal/utils/resource_set.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_amd_gpu_binding_requirement_t {
  // Minimum captured extent across all uses, including private native ABIs.
  uint64_t minimum_length;
  // Length minus the root binding's length, modulo 2^64.
  uint64_t difference;
  // Union root for whole-range copy equalities. Slot zero is constant zero;
  // public bindings occupy indices one through binding_capacity.
  uint32_t parent;
  // Required native sites and whether the logical length is consumed.
  uint32_t sites;
} iree_hal_amd_gpu_binding_requirement_t;

// Captured, qualified binding. Address commitment waits for queue dependencies;
// repeated uses in recorded commands share one dense native input.
typedef struct iree_hal_amd_gpu_command_binding_t {
  // Owned buffer unless the caller explicitly borrowed invocation lifetime.
  iree_hal_buffer_t* buffer;
  // Byte offset from the logical beginning of buffer.
  iree_device_size_t offset;
  // Logical extent after resolving WHOLE_BUFFER.
  iree_device_size_t length;
  // Prepared native slots for the queue (0) and shader program (1) sites.
  iree_hal_buffer_native_binding_slot_t slots[2];
} iree_hal_amd_gpu_command_binding_t;

typedef struct iree_hal_amd_gpu_command_input_t
    iree_hal_amd_gpu_command_input_t;

// Recording-time deduplication produces the dense invocation input vector.
// Recorded operations and fixups never survive as a host replay list.
typedef struct iree_hal_amd_gpu_command_inputs_t {
  // Borrowed public identity, native family, and binding-count owner.
  iree_hal_command_buffer_t* command_buffer;
  // Allocator for recording-only indexing and retained dense descriptors.
  iree_allocator_t host_allocator;
  // Borrowed retained-resource set, or NULL for UNRETAINED recordings.
  iree_hal_resource_set_t* resources;
  // Inline requirements owned by the containing command buffer.
  iree_hal_amd_gpu_binding_requirement_t* requirements;
  // Dense unique inputs consumed once per invocation.
  iree_hal_amd_gpu_command_input_t* entries;
  // Number of initialized descriptors.
  uint32_t count;
  // Allocated descriptor capacity.
  uint32_t capacity;
  // Recording-only open-addressed index; released by seal.
  uint32_t* index;
  // Power-of-two index capacity.
  uint32_t index_capacity;
  // Number of executor-provided values preceding unique external inputs.
  uint32_t value_base;
} iree_hal_amd_gpu_command_inputs_t;

// requirements has binding_capacity + 1 entries; zero is the constant-length
// root. All referenced owners dominate this recorder. Initialization allocates
// nothing; deinitialization releases only descriptor/index storage.
void iree_hal_amd_gpu_command_inputs_initialize(
    iree_hal_command_buffer_t* command_buffer, uint32_t value_base,
    iree_hal_resource_set_t* resources,
    iree_hal_amd_gpu_binding_requirement_t* requirements,
    iree_allocator_t host_allocator,
    iree_hal_amd_gpu_command_inputs_t* out_inputs);
void iree_hal_amd_gpu_command_inputs_deinitialize(
    iree_hal_amd_gpu_command_inputs_t* inputs);

// Registers an address under queue (0) or program (1) native access. All uses
// share one descriptor; minimum extent and alignment constraints accumulate.
// whole_transfer admits an empty remaining range and contributes its minimum
// displacement; otherwise a committed nonzero address is always required.
iree_status_t iree_hal_amd_gpu_command_inputs_add_binding(
    iree_hal_amd_gpu_command_inputs_t* inputs, iree_hal_buffer_ref_t reference,
    uint32_t site, uint32_t minimum_length, uint32_t alignment,
    bool whole_transfer, uint32_t* out_value);
iree_status_t iree_hal_amd_gpu_command_inputs_equal_lengths(
    iree_hal_amd_gpu_command_inputs_t* inputs, iree_hal_buffer_ref_t source,
    iree_hal_buffer_ref_t target);
iree_status_t iree_hal_amd_gpu_command_inputs_add_length(
    iree_hal_amd_gpu_command_inputs_t* inputs, uint32_t binding_slot,
    uint32_t* out_value);
iree_status_t iree_hal_amd_gpu_command_inputs_add_image(
    iree_hal_amd_gpu_command_inputs_t* inputs, const uint64_t* addresses,
    uint32_t* out_value);

// Flattens whole-range constraints and releases the recording-only hash index.
void iree_hal_amd_gpu_command_inputs_seal(
    iree_hal_amd_gpu_command_inputs_t* inputs);

// Captures an externally validated binding table into binding_count entries.
// Checks aggregated native requirements once and retains used buffers unless
// explicitly borrowed. Failure leaves zeroed unused entries and retained
// prefix entries that the caller releases through its ordinary cleanup.
iree_status_t iree_hal_amd_gpu_command_inputs_capture(
    const iree_hal_amd_gpu_command_inputs_t* inputs,
    iree_hal_buffer_binding_table_t bindings,
    iree_hal_queue_execute_flags_t flags,
    iree_hal_amd_gpu_command_binding_t* out_bindings);

// Resolves committed backing after dependencies. values has count uint64_t
// entries and excludes the executor-provided prefix. Native slot selection and
// aggregate alignment/extent facts come from recording/capture, with no search
// or per-operation resolution at submission.
iree_status_t iree_hal_amd_gpu_command_inputs_write(
    const iree_hal_amd_gpu_command_inputs_t* inputs,
    const iree_hal_amd_gpu_command_binding_t* bindings, uint32_t queue_ordinal,
    uint64_t* values);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_INPUTS_H_
