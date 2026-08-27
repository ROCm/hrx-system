// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LOCAL_EXECUTABLE_DATA_H_
#define IREE_HAL_LOCAL_EXECUTABLE_DATA_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true when |executable_data| begins with ELF or FatELF magic.
bool iree_hal_local_executable_data_is_elf(
    iree_const_byte_span_t executable_data);

// Returns true when a bounded ELF uses features requiring the platform dynamic
// loader instead of the restricted embedded ELF loader.
//
// Malformed or truncated data returns false. Callers still claim recognizable
// ELF bytes and let the selected loader produce the detailed diagnostic.
bool iree_hal_local_elf_data_requires_system_loader(
    iree_const_byte_span_t executable_data);

// Returns true when |executable_data| begins with the native shared-library
// magic for the current platform.
bool iree_hal_local_executable_data_is_system_library(
    iree_const_byte_span_t executable_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LOCAL_EXECUTABLE_DATA_H_
