// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_RUNTIME_BUILD_TOOLS_BAZEL_TEST_HAL_DRIVER_REGISTRATION_FIXTURE_H_
#define IREE_RUNTIME_BUILD_TOOLS_BAZEL_TEST_HAL_DRIVER_REGISTRATION_FIXTURE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Resets the captured registration sequence and selects the module that should
// fail. A value of zero allows every module to register successfully.
void iree_hal_driver_registration_fixture_reset(int failing_module);

// Returns the number of registration modules called since the last reset.
iree_host_size_t iree_hal_driver_registration_fixture_count(void);

// Returns the module identifier captured at the given sequence index.
int iree_hal_driver_registration_fixture_at(iree_host_size_t index);

// Registration functions used to prove ordered external module composition.
iree_status_t iree_hal_driver_registration_fixture_alpha(
    iree_hal_driver_registry_t* registry);
iree_status_t iree_hal_driver_registration_fixture_beta(
    iree_hal_driver_registry_t* registry);
iree_status_t iree_hal_driver_registration_fixture_gamma(
    iree_hal_driver_registry_t* registry);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_RUNTIME_BUILD_TOOLS_BAZEL_TEST_HAL_DRIVER_REGISTRATION_FIXTURE_H_
