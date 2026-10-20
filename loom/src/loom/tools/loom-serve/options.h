// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_SERVE_OPTIONS_H_
#define LOOM_TOOLS_LOOM_SERVE_OPTIONS_H_

#include "iree/hal/driver_registry.h"
#include "loom/serve/runtime/device/device.h"

#ifdef __cplusplus
extern "C" {
#endif

// CLI leaf creating shared application workers and I/O infrastructure. Compiler
// options may be NULL for a runtime-only application; source JIT requests then
// fail at the compiler boundary.
iree_status_t loom_serve_host_create_from_flags(
    const loom_serve_compiler_options_t* compiler_options,
    iree_allocator_t host_allocator, loom_serve_host_t** out_host);

// CLI leaf constructing the application group from ordered, explicit device
// URIs. Backing strategy and slab size apply to every record. A nonzero
// physical byte limit requires a single device. Models resolve their own
// bindings; the entry point destroys the group after every model retires. URI
// interpretation belongs to the HAL driver, with no implicit local device or
// backend fallback. Repeated execution scopes and accepted-unit limits resolve
// host and device ordinals before group construction copies the policy.
iree_status_t loom_serve_device_group_allocate_from_flags(
    loom_serve_host_t* host, iree_hal_driver_registry_t* driver_registry,
    iree_allocator_t host_allocator, loom_serve_device_group_t** out_group);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_SERVE_OPTIONS_H_
