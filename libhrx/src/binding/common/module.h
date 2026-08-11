// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_EXPERIMENTAL_STREAMING_MODULE_H_
#define IREE_EXPERIMENTAL_STREAMING_MODULE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_hal_streaming_module_t iree_hal_streaming_module_t;

// Resolves a managed global represented by a pointer slot and initializer
// storage in the same executable. The module owns the managed allocation until
// it is destroyed. This also supports executable formats that cannot enumerate
// globals during module load by initializing the pair on first query.
iree_status_t iree_hal_streaming_module_try_initialize_managed_global(
    iree_hal_streaming_module_t* module, const char* pointer_name,
    const char* initializer_name, bool* out_found, void** out_host_pointer,
    iree_device_size_t* out_size);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_STREAMING_MODULE_H_
