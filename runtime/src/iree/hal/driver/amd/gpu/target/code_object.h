// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVER_AMD_GPU_TARGET_CODE_OBJECT_H_
#define IREE_HAL_DRIVER_AMD_GPU_TARGET_CODE_OBJECT_H_

#include "iree/hal/driver/amd/gpu/target/identity.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// AMDGPU Code Objects
//===----------------------------------------------------------------------===//

// Recovers the AMDGPU target identity encoded in an HSA code-object ELF
// header. The span must contain at least the complete 64-byte header; a zero
// length does not indicate an unbounded source.
//
// The returned processor string is borrowed from static target tables and
// remains valid for the lifetime of the process.
iree_status_t iree_hal_amd_gpu_code_object_identity_from_elf(
    iree_const_byte_span_t elf_data,
    iree_hal_amd_gpu_target_identity_t* out_identity);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVER_AMD_GPU_TARGET_CODE_OBJECT_H_
