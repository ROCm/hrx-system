// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_TESTING_DEVICE_H_
#define LOOM_RUNTIME_TESTING_DEVICE_H_

#include "iree/hal/driver_registry.h"
#include "loom/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers one HAL driver factory in the supplied registry.
typedef iree_status_t(IREE_API_PTR* loom_runtime_test_driver_register_fn_t)(
    iree_hal_driver_registry_t* registry);

// Test-only runtime composition supplied by one physical-device leaf. Runtime
// tests consume this without linking any compiler target package.
typedef struct loom_runtime_test_device_t {
  // Exact non-task runtime driver registration, or NULL for the task driver.
  loom_runtime_test_driver_register_fn_t driver_register;
  // Driver registry name. A NULL registration requires the task driver name.
  iree_string_view_t driver_name;
  // Default device URI owned by the device leaf.
  iree_string_view_t device_uri;
  // Storage policy used by source service tests for this physical device.
  loom_runtime_pool_backing_t service_backing;
} loom_runtime_test_device_t;

// Implemented exactly once by the linked physical-device leaf.
const loom_runtime_test_device_t* loom_runtime_test_device(void);

// Acquires the linked physical device's registry-owned runtime driver, or the
// host's task driver. Native process state is shared across test cases. The
// caller releases its reference after creating devices.
iree_status_t loom_runtime_test_driver_acquire(loom_runtime_host_t* host,
                                               iree_hal_driver_t** out_driver);

// Joins raw workspace allocation captures through the normal idle policy.
// Callers first retire model consumers and release all workspace views. Model
// retirement alone does not join queue allocation bookkeeping. This is an
// outer test-thread wait that advances remaining cold acquisitions;
// trim_and_drain then joins physical backing returns.
iree_status_t loom_runtime_test_workspace_await_idle(
    loom_runtime_workspace_t* workspace);

// Trims managed backing and observes native capacity notifications, including
// cascading slab returns, before tests inspect physical accounting. Device work
// remains independent: live reservations survive, and callers retire unused
// buffers before expecting their backing to disappear. Permanently lost backing
// reports its retained cleanup failure instead of waiting for a native return.
iree_status_t loom_runtime_test_pool_trim_and_drain(loom_runtime_pool_t* owner,
                                                    uint64_t target_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_TESTING_DEVICE_H_
