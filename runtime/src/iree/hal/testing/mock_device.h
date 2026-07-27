// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A lightweight mock HAL device for testing HAL infrastructure that operates
// on devices without needing real hardware or a full driver stack.
//
// The mock device implements the topology-related vtable methods (id,
// device_spec, topology_info, refine_topology_edge, and assign_topology_info)
// with configurable behavior. It can optionally load tiny metadata-only
// executables for tests that need executable objects without real compiled
// kernels. All other vtable methods return IREE_STATUS_UNIMPLEMENTED or
// zero/NULL as appropriate.
//
// This is intended for testing code that coordinates devices (device groups,
// topology construction, multi-device scheduling) rather than code that
// executes work on devices.

#ifndef IREE_HAL_TESTING_MOCK_DEVICE_H_
#define IREE_HAL_TESTING_MOCK_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable virtual target advertised for metadata-only mock executables.
#define IREE_HAL_MOCK_EXECUTABLE_TARGET_FAMILY "mock"
#define IREE_HAL_MOCK_EXECUTABLE_TARGET_KEY "metadata"

// Options for creating a mock device.
typedef struct iree_hal_mock_device_options_t {
  // Identifier returned by iree_hal_device_id(). The default mock spec uses
  // this as its logical id and display name.
  iree_string_view_t identifier;

  // Optional immutable spec returned by iree_hal_device_spec().
  // Retained by the mock when provided; otherwise a default spec is created.
  // Callers enabling executable support with a custom spec must advertise the
  // mock:metadata executable target themselves.
  iree_hal_device_spec_t* device_spec;

  // Optional status returned by assign_topology_info. IREE_STATUS_OK means the
  // mock accepts the assignment normally.
  iree_status_code_t assign_topology_info_status_code;

  // Enables metadata-only mock executable target advertisement and loading.
  bool executable_loading_enabled;
} iree_hal_mock_device_options_t;

// Initializes |out_options| with safe defaults and an empty identifier.
void iree_hal_mock_device_options_initialize(
    iree_hal_mock_device_options_t* out_options);

// Creates a mock HAL device with the given |options|.
// The identifier string is copied into the device's own storage.
// |out_device| must be released by the caller (see iree_hal_device_release).
iree_status_t iree_hal_mock_device_create(
    const iree_hal_mock_device_options_t* options,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TESTING_MOCK_DEVICE_H_
