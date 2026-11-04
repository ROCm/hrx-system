// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_DEVICE_H_
#define AMDF_SRC_DEVICE_H_

#include "libamdf/src/device_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Initializes an unpublished device base and borrows its endpoint.
amdf_status_t amdf_device_initialize(amdf_device_t* device,
                                     const amdf_device_vtable_t* vtable,
                                     amdf_endpoint_t* endpoint,
                                     amdf_engine_kind_t engine_kind);

// Releases the endpoint borrow held by an unpublished or torn-down device.
void amdf_device_deinitialize(amdf_device_t* device);

// Returns true when a device is implemented by `expected_engine_kind`.
bool amdf_device_is_engine(const amdf_device_t* device,
                           amdf_engine_kind_t expected_engine_kind);

// Returns true when both devices share one provider-instance identity scope.
bool amdf_device_shares_provider_instance(const amdf_device_t* lhs,
                                          const amdf_device_t* rhs);

// Returns the host allocator copied by the device.
amdf_allocator_t amdf_device_host_allocator(const amdf_device_t* device);

// Registers one child that borrows `device`.
amdf_status_t amdf_device_register_child(amdf_device_t* device);

// Releases one child borrow.
void amdf_device_unregister_child(amdf_device_t* device);

// Destroys a device after dependent memory and tracked children are released.
amdf_status_t AMDF_CALL amdf_device_destroy(amdf_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_DEVICE_H_
