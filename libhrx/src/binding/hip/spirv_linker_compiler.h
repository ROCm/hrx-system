// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_SPIRV_LINKER_COMPILER_H_
#define HRX_BINDING_HIP_SPIRV_LINKER_COMPILER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hip_spirv_linker_input_t {
  // SPIR-V bytes, either raw or in a standard clang offload bundle.
  iree_const_byte_span_t data;
  // NUL-terminated input name used in compiler diagnostics.
  const char* name;
} iree_hip_spirv_linker_input_t;

// Compiles and links |inputs| into one AMDGPU executable for |target_isa|.
// The returned bytes are allocated with |host_allocator|.
iree_status_t iree_hip_spirv_linker_compile(
    iree_string_view_t target_isa, iree_host_size_t input_count,
    const iree_hip_spirv_linker_input_t* inputs, iree_host_size_t option_count,
    const char* const* options, iree_allocator_t host_allocator,
    iree_byte_span_t* out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_SPIRV_LINKER_COMPILER_H_
