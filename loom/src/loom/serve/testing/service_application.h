// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_TESTING_SERVICE_APPLICATION_H_
#define LOOM_SERVE_TESTING_SERVICE_APPLICATION_H_

#include "loom/serve/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// Configuration for the one source-service application owned by a test
// process. Every acquisition in the process must request an identical
// configuration.
typedef struct loom_serve_test_application_options_t {
  // Number of workers in the host task executor.
  iree_host_size_t worker_count;
  // Number of devices in the immutable HAL group.
  iree_host_size_t device_count;
  // Physical backing strategy shared by every device.
  loom_serve_device_backing_t backing;
  // Physical slab bytes; zero selects the allocator recommendation.
  iree_device_size_t slab_size;
  // Maximum committed virtual-pool bytes; zero imposes no serving limit.
  uint64_t memory_limit;
} loom_serve_test_application_options_t;

// Process-owned source-service test application.
typedef struct loom_serve_test_application_t loom_serve_test_application_t;

// Returns the default configuration for the linked physical-device leaf.
loom_serve_test_application_options_t
loom_serve_test_application_options_default(void);

// Acquires the process-owned application with the exact requested
// configuration. The returned application is borrowed until global test
// environment teardown. A later mismatched request fails instead of creating a
// second host or device group.
// Devices bound numerical submissions to two outstanding units. Manual service
// owners grant work after each complete pass and consume granted progress
// before parking, using the same resource scheduling boundary as the router.
iree_status_t loom_serve_test_application_acquire(
    const loom_serve_test_application_options_t* options,
    iree_allocator_t host_allocator,
    loom_serve_test_application_t** out_application);

// Borrows the shared serving host.
loom_serve_host_t* loom_serve_test_application_host(
    const loom_serve_test_application_t* application);

// Borrows the shared immutable HAL device group.
loom_serve_device_group_t* loom_serve_test_application_device_group(
    const loom_serve_test_application_t* application);

// Borrows one device record. device_index must be less than device_count.
loom_serve_device_t* loom_serve_test_application_device_at(
    const loom_serve_test_application_t* application,
    iree_host_size_t device_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_TESTING_SERVICE_APPLICATION_H_
