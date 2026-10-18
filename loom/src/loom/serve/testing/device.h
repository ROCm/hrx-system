// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_DEVICE_H_
#define LOOM_SERVE_TESTING_DEVICE_H_

#include "iree/hal/driver_registry.h"
#include "loom/serve/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Registers one HAL driver factory in a temporary registry.
typedef iree_status_t(IREE_API_PTR* loom_serve_test_driver_register_fn_t)(
    iree_hal_driver_registry_t* registry);

// Test-only runtime composition supplied by one physical-device leaf. Runtime
// tests consume this without linking any compiler target package.
typedef struct loom_serve_test_device_t {
  // Exact non-task runtime driver registration, or NULL for the task driver.
  loom_serve_test_driver_register_fn_t driver_register;
  // Driver registry name. A NULL registration requires the task driver name.
  iree_string_view_t driver_name;
  // Default device URI owned by the device leaf.
  iree_string_view_t device_uri;
  // Storage policy used by source service tests for this physical device.
  loom_serve_device_backing_t service_backing;
} loom_serve_test_device_t;

// Implemented exactly once by the linked physical-device leaf.
const loom_serve_test_device_t* loom_serve_test_device(void);

// Creates the linked physical device's runtime driver. The caller releases the
// returned reference after all devices have been created.
iree_status_t loom_serve_test_driver_create(loom_serve_host_t* host,
                                            iree_allocator_t host_allocator,
                                            iree_hal_driver_t** out_driver);

// Trims managed backing and observes native capacity notifications, including
// cascading slab returns, before tests inspect physical accounting. Device work
// remains independent: live reservations survive, and callers retire unused
// buffers before expecting their backing to disappear.
iree_status_t loom_serve_test_device_trim_and_drain(loom_serve_device_t* device,
                                                    uint64_t target_bytes);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TESTING_DEVICE_H_
