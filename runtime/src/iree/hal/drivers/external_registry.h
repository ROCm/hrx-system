// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_EXTERNAL_REGISTRY_H_
#define IREE_HAL_DRIVERS_EXTERNAL_REGISTRY_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers the external HAL driver modules selected by the build graph.
//
// Exactly one implementation is linked into any target using the aggregate
// driver registry. Build systems provide an empty implementation by default
// and generate direct calls to explicitly composed external modules.
iree_status_t iree_hal_register_external_drivers(
    iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_EXTERNAL_REGISTRY_H_
