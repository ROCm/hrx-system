// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_
#define LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_t iree_hal_streaming_context_t;

// Returns the current thread's borrowed streaming context, or NULL when no
// context has been selected.
iree_hal_streaming_context_t* iree_hal_streaming_context_current(void);

// Flushes and waits for every stream in every active context. Destructive
// operations use this when device pointers may be hidden from host-side
// resource tracking.
iree_status_t iree_hal_streaming_context_synchronize_all(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LIBHRX_SRC_BINDING_COMMON_CONTEXT_H_
