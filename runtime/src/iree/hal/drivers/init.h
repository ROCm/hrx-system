// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_INIT_H_
#define IREE_HAL_DRIVERS_INIT_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Registers all built-in and externally composed driver modules selected by
// the build configuration. Note that there may be no drivers available.
//
// Built-in modules are registered first. External modules are then registered
// in their declared order, allowing them to override built-in driver names or
// earlier external modules when the registry resolves a driver request.
IREE_API_EXPORT iree_status_t
iree_hal_register_all_available_drivers(iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_INIT_H_
