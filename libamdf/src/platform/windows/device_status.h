// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_WINDOWS_DEVICE_STATUS_H_
#define AMDF_SRC_PLATFORM_WINDOWS_DEVICE_STATUS_H_

#include "libamdf/src/atomics.h"
#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Observed terminal execution failure shared by a KMT device's queues.
// A rejected operation is not evidence that the device has failed. Only an
// explicit removal/hang/fault result or a terminal execution-state query may
// publish this latch. Publication does not prove retirement or cancel work.
typedef struct amdf_kmt_device_status_t {
  // First confirmed terminal failure; zero until one has been observed.
  amdf_atomic_uint64_t terminal_status;
} amdf_kmt_device_status_t;

// Initializes an unpublished device's failure latch.
void amdf_kmt_device_status_initialize(amdf_kmt_device_status_t* device_status);

// Returns the observed terminal failure without a system call or polling.
amdf_status_t amdf_kmt_device_status_query(
    const amdf_kmt_device_status_t* device_status);

// Classifies a failed operation and preserves its error unless a terminal
// device failure is confirmed. An ambiguous NTSTATUS triggers one execution
// state query on this error path, never on successful submission or progress
// sampling. A failed diagnostic query is returned without marking the device
// lost. A successful ACTIVE query leaves the original operation error intact.
amdf_status_t amdf_kmt_device_status_observe_error(
    amdf_kmt_device_status_t* device_status, const amdf_kmt_api_t* api,
    D3DKMT_HANDLE device, amdf_status_t operation_status);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_WINDOWS_DEVICE_STATUS_H_
