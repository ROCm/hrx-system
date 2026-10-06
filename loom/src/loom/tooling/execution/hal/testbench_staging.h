// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_STAGING_H_
#define LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_STAGING_H_

#include "loom/tooling/execution/hal/runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

// Device-local backing for one synchronous testbench kernel-launch span.
// Host fixtures remain owned by the value table so CPU oracles and observations
// see the same allocations and aliases after readback.
typedef struct loom_run_hal_testbench_staging_t {
  // Allocator owning the transfer array.
  iree_allocator_t host_allocator;
  // Upload copies from borrowed fixture allocations to owned device buffers.
  iree_hal_transfer_operation_t* transfers;
  // Number of owned device buffers and initialized transfer records.
  iree_host_size_t transfer_count;
} loom_run_hal_testbench_staging_t;

// Uploads each distinct host fixture allocation once and redirects |bindings|
// to its device-local backing, preserving all allocation-relative offsets.
// Bindings local to |runtime->device| pass through without allocation or
// transfer. Other bindings must be host-mappable. Fixture
// allocations must remain live and exclusively owned by this execution through
// readback. The caller deinitializes |out_staging| even when staging fails.
iree_status_t loom_run_hal_testbench_staging_initialize(
    const loom_run_hal_runtime_t* runtime, iree_host_size_t binding_count,
    iree_hal_buffer_binding_t* bindings, iree_allocator_t host_allocator,
    loom_run_hal_testbench_staging_t* out_staging);

// Copies completed device results back to the original fixture allocations.
// The caller must wait for kernel completion before readback. Each initialized
// staging object supports one readback, followed by deinitialization.
iree_status_t loom_run_hal_testbench_staging_readback(
    const loom_run_hal_runtime_t* runtime,
    loom_run_hal_testbench_staging_t* staging);

// Releases staged device buffers and transfer storage after all work completes.
void loom_run_hal_testbench_staging_deinitialize(
    loom_run_hal_testbench_staging_t* staging);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_HAL_TESTBENCH_STAGING_H_
