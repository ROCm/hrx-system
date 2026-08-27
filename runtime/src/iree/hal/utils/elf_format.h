// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_UTILS_ELF_FORMAT_H_
#define IREE_HAL_UTILS_ELF_FORMAT_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Returns true if |elf_data| contains the complete ELF magic prefix.
bool iree_hal_elf_data_starts_with_magic(iree_const_byte_span_t elf_data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_UTILS_ELF_FORMAT_H_
