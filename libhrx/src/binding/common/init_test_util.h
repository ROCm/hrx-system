// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_INIT_TEST_UTIL_H_
#define LIBHRX_SRC_BINDING_COMMON_INIT_TEST_UTIL_H_

#include "common/internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Installs or removes a test-owned process device registry. Tests must install
// only while the production registry is absent and must release every context
// registered in it before removal.
void iree_hal_streaming_set_device_registry_for_testing(
    iree_hal_streaming_device_registry_t* device_registry);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_INIT_TEST_UTIL_H_
