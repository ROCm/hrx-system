// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_COMMON_MODULE_H_
#define HRX_BINDING_COMMON_MODULE_H_

#include "common/internal.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Extracts and validates metadata from the executables already owned by
// |module|. Allocated symbol and operation storage is owned by |module| even
// when later metadata validation fails and remains stable until module
// destruction.
// Synchronization: none (module must not yet be published).
iree_status_t iree_hal_streaming_module_extract_metadata(
    iree_hal_streaming_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_COMMON_MODULE_H_
