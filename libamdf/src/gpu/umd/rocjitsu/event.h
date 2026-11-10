// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_GPU_UMD_ROCJITSU_EVENT_H_
#define AMDF_SRC_GPU_UMD_ROCJITSU_EVENT_H_

#include "libamdf/src/gpu/umd/event.h"
#include "libamdf/src/gpu/umd/rocjitsu/source.h"

// Releases the source's adopted event page after every event and producer has
// retired and the engine has joined. Failure preserves remaining ownership.
amdf_status_t amdf_rocjitsu_event_source_release(
    amdf_rocjitsu_source_t* source);

amdf_status_t amdf_rocjitsu_event_create(
    amdf_gpu_umd_device_t* device, amdf_gpu_umd_event_t** out_event,
    amdf_gpu_umd_event_result_t* out_result);
amdf_status_t amdf_rocjitsu_event_signal(amdf_gpu_umd_event_t* event);
amdf_status_t amdf_rocjitsu_event_wait(amdf_gpu_umd_event_t* event,
                                       uint64_t timeout_nanoseconds);
amdf_status_t amdf_rocjitsu_event_destroy(amdf_gpu_umd_event_t* event);

#endif  // AMDF_SRC_GPU_UMD_ROCJITSU_EVENT_H_
