// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Validated binding publication for direct experimental XDNA execution.

#ifndef IREE_EXPERIMENTAL_XDNA_DIRECT_BINDING_H_
#define IREE_EXPERIMENTAL_XDNA_DIRECT_BINDING_H_

#include "iree/hal/drivers/amd/xdna/executable_storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Validates direct caller-owned storage and resolved binding addresses before
// patching invocation-private command bytes. The caller retains every logical
// buffer, native allocation, and mapping through terminal native completion.
iree_status_t iree_xdna_executable_storage_bind(
    const iree_hal_amd_xdna_image_t* image, uint32_t entry_ordinal,
    iree_host_size_t storage_count,
    const iree_hal_amd_xdna_executable_storage_t* storage,
    iree_host_size_t binding_count,
    const iree_hal_amd_xdna_executable_binding_t* bindings);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_EXPERIMENTAL_XDNA_DIRECT_BINDING_H_
