// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mock HAL executable for testing executable upload and reflection paths
// without a real compiler backend.
//
// The "mock-executable" format stores executable metadata in a tiny binary
// payload used by replay tests. The "mock" format produces a single inert
// default export for tests that only need a prepare_executable object.

#ifndef IREE_HAL_TESTING_MOCK_EXECUTABLE_H_
#define IREE_HAL_TESTING_MOCK_EXECUTABLE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a mock executable from |executable_params|.
iree_status_t iree_hal_mock_executable_create(
    const iree_hal_executable_load_params_t* executable_params,
    iree_allocator_t host_allocator, iree_hal_executable_t** out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_TESTING_MOCK_EXECUTABLE_H_
