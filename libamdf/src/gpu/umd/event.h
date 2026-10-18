// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_EVENT_H_
#define AMDF_SRC_GPU_UMD_EVENT_H_

#include "libamdf/src/gpu/umd/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_gpu_umd_event_t amdf_gpu_umd_event_t;

// Producer operands established by native event construction.
typedef struct amdf_gpu_umd_event_result_t {
  // Qualified native producer protocol.
  amdf_gpu_event_format_t format;
  // Version of that producer protocol.
  uint32_t format_version;
  // Native queue publication paths qualified to use the producer operands.
  amdf_queue_publication_modes_t publication_modes;
  // Device-local address of the native mailbox.
  uint64_t mailbox_address;
  // Native mailbox payload.
  uint32_t trigger_data;
  // Native interrupt payload.
  uint32_t interrupt_context;
} amdf_gpu_umd_event_result_t;

// Creates a native event, borrowing device. Failure preserves both outputs;
// shared connection preparation remains with its existing native owner.
amdf_status_t amdf_gpu_umd_event_create(
    amdf_gpu_umd_device_t* device, amdf_gpu_umd_event_t** out_event,
    amdf_gpu_umd_event_result_t* out_result);

// Signals may run concurrently with a caller-serialized wait.
amdf_status_t amdf_gpu_umd_event_signal(amdf_gpu_umd_event_t* event);

// Blocks only the caller. Native failure does not imply producer quiescence.
amdf_status_t amdf_gpu_umd_event_wait(amdf_gpu_umd_event_t* event,
                                      uint64_t timeout_nanoseconds);

// Success consumes event; a native release failure preserves it for retry.
amdf_status_t amdf_gpu_umd_event_destroy(amdf_gpu_umd_event_t* event);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_GPU_UMD_EVENT_H_
