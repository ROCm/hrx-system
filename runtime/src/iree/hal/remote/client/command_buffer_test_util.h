// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_TEST_UTIL_H_
#define IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_TEST_UTIL_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns mutable recorded bytes for protocol fault injection in tests.
// The span remains valid until the command buffer is destroyed or rerecorded.
iree_byte_span_t iree_hal_remote_client_command_buffer_test_stream(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_REMOTE_CLIENT_COMMAND_BUFFER_TEST_UTIL_H_
