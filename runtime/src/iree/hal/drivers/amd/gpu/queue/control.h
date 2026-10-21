// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTROL_H_
#define IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTROL_H_

#include "iree/hal/drivers/amd/gpu/queue/buffer.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_hal_amd_gpu_control_kind_e {
  IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP,
  IREE_HAL_AMD_GPU_CONTROL_WAIT,
  IREE_HAL_AMD_GPU_CONTROL_STORE,
  IREE_HAL_AMD_GPU_CONTROL_RMW,
} iree_hal_amd_gpu_control_kind_t;

// Scalar command-processor memory operation. PM4 emits this into its ring or
// resident program; AQL carries the same native bytes in a format-1 packet.
// This is a direct-call capture, not a recorded host operation stream.
typedef struct iree_hal_amd_gpu_control_t {
  // Wait reference, store value, or RMW operand.
  uint64_t value;
  // Wait comparison mask; unused by other kinds.
  uint64_t mask;
  // Requested atomic ordering and coherence domain.
  iree_hal_atomic_flags_t flags;
  // Timestamp, wait, store, or RMW.
  uint8_t kind;
  // Naturally aligned target width in bits.
  iree_hal_atomic_width_t width;
  // Wait condition or RMW operation; unused by timestamp and store.
  uint8_t operation;
  // Classification of missing target capabilities or target misalignment.
  iree_hal_atomic_target_error_mode_t target_error_mode;
} iree_hal_amd_gpu_control_t;

static inline iree_hal_amd_gpu_control_t iree_hal_amd_gpu_control_timestamp(
    void) {
  const iree_hal_amd_gpu_control_t command = {
      .kind = IREE_HAL_AMD_GPU_CONTROL_TIMESTAMP,
      .width = IREE_HAL_ATOMIC_WIDTH_64};
  return command;
}
static inline iree_hal_amd_gpu_control_t iree_hal_amd_gpu_control_wait(
    iree_hal_atomic_wait_params_t params) {
  const iree_hal_amd_gpu_control_t command = {
      .value = params.value,
      .mask = params.mask,
      .flags = params.flags & ~IREE_HAL_ATOMIC_FLAG_RELEASE,
      .kind = IREE_HAL_AMD_GPU_CONTROL_WAIT,
      .width = params.width,
      .operation = params.condition,
      .target_error_mode = params.target_error_mode};
  return command;
}
static inline iree_hal_amd_gpu_control_t iree_hal_amd_gpu_control_store(
    iree_hal_atomic_store_params_t params) {
  const iree_hal_amd_gpu_control_t command = {
      .value = params.value,
      .flags = params.flags & ~IREE_HAL_ATOMIC_FLAG_ACQUIRE,
      .kind = IREE_HAL_AMD_GPU_CONTROL_STORE,
      .width = params.width,
      .target_error_mode = params.target_error_mode};
  return command;
}
static inline iree_hal_amd_gpu_control_t iree_hal_amd_gpu_control_rmw(
    iree_hal_atomic_rmw_params_t params) {
  const iree_hal_amd_gpu_control_t command = {
      .value = params.operand,
      .flags = params.flags,
      .kind = IREE_HAL_AMD_GPU_CONTROL_RMW,
      .width = params.width,
      .operation = params.operation,
      .target_error_mode = params.target_error_mode};
  return command;
}

// One bit in the immutable scope row's four byte-sized operation cells.
// Public validation has established the kind, width, and operation values.
uint32_t iree_hal_amd_gpu_control_requirement(
    iree_hal_amd_gpu_control_t command);

// Checks prepared queue-site guarantees without requiring committed backing.
// required and incompatible use the packed scope-row representation; the
// latter classifies only failures not also required with the default mode.
// Native binding admission has already established family membership.
iree_status_t iree_hal_amd_gpu_control_validate_target(
    const iree_hal_queue_family_t* family, const iree_hal_buffer_t* buffer,
    uint32_t required, uint32_t incompatible);

// Resolves after allocation dependencies and checks the actual target address.
iree_status_t iree_hal_amd_gpu_control_resolve(
    const iree_hal_amd_gpu_buffer_reference_t* target,
    iree_hal_amd_gpu_control_t command, uint64_t* out_address);

// Exact scalar packet extent and target-address position in DWORDs. The
// address occupies two complete DWORDs, suitable for ordinary wide fixups.
uint32_t iree_hal_amd_gpu_control_word_count(
    iree_hal_amd_gpu_control_t command);
uint32_t iree_hal_amd_gpu_control_address_word(
    iree_hal_amd_gpu_control_t command);

// Emits complete native DWORDs into caller-provided storage. Capacity, target
// qualification, alignment, payload barriers, and lifetime are already proven.
uint32_t* iree_hal_amd_gpu_control_emit(uint32_t* words, uint64_t address,
                                        iree_hal_amd_gpu_control_t command);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // IREE_HAL_DRIVERS_AMD_GPU_QUEUE_CONTROL_H_
