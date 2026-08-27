// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_
#define HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_

#include "common/internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Validates a caller-supplied native kernarg byte image before it is copied or
// dispatched. The reflected direct-argument extent covers meaningful native
// parameter bytes; callers may omit ABI tail padding but may not truncate that
// extent. The declared span length remains authoritative after validation.
iree_status_t iree_hal_streaming_validate_prepacked_kernel_arguments(
    const iree_hal_streaming_symbol_t* symbol,
    const iree_hal_streaming_dispatch_params_t* params);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_COMMON_KERNEL_ARGUMENTS_H_
