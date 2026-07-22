// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_REGISTRY_H_
#define IREE_EXPERIMENTAL_STREAMING_REGISTRY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_context_symbol_map_t
    iree_hal_streaming_context_symbol_map_t;

// Refreshes registered host storage from resolved managed device symbols.
iree_status_t iree_hal_streaming_context_symbol_map_synchronize_managed_data(
    iree_hal_streaming_context_symbol_map_t* map);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_REGISTRY_H_
