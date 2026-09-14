// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_TEST_UTIL_H_
#define IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_TEST_UTIL_H_

#include "iree/hal/drivers/amdgpu/ipc_event_monitor.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Injects worker-thread creation failure for deterministic testing. The
// monitor must be stopped when changing this setting.
void iree_hal_amdgpu_ipc_event_monitor_set_fail_thread_create_for_testing(
    bool fail_thread_create);

// Returns the number of operation nodes inspected by scheduler list work.
int64_t iree_hal_amdgpu_ipc_event_monitor_schedule_node_visit_count_for_testing(
    void);

// Resets scheduler node-visit instrumentation after a test synchronization
// point at which no prior operation can perform additional list work.
void iree_hal_amdgpu_ipc_event_monitor_reset_schedule_node_visit_count_for_testing(
    void);

// Returns the number of cancellation notifications posted while shutdown was
// active. Ordinary transaction rollback must not wake the worker.
int64_t iree_hal_amdgpu_ipc_event_monitor_cancellation_wake_count_for_testing(
    void);

// Resets cancellation-notification instrumentation.
void iree_hal_amdgpu_ipc_event_monitor_reset_cancellation_wake_count_for_testing(
    void);

// Makes all currently scheduled finite operations immediately due without
// relying on wall-clock delay. Intended for scheduler starvation tests.
void iree_hal_amdgpu_ipc_event_monitor_make_finite_operations_due_for_testing(
    void);

// Moves all currently scheduled dormant operations to an immediately due
// finite bucket for one poll. Intended for scheduler starvation tests that
// must prepare a large due batch without relying on wall-clock delay.
iree_host_size_t
iree_hal_amdgpu_ipc_event_monitor_make_dormant_operations_due_for_testing(void);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_IPC_EVENT_MONITOR_TEST_UTIL_H_
