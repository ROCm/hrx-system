// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef HRX_BINDING_HIP_LINKER_COMPILER_H_
#define HRX_BINDING_HIP_LINKER_COMPILER_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef enum iree_hip_linker_input_format_e {
  IREE_HIP_LINKER_INPUT_FORMAT_LLVM_BITCODE,
  IREE_HIP_LINKER_INPUT_FORMAT_SPIRV,
} iree_hip_linker_input_format_t;

typedef struct iree_hip_linker_input_t {
  // Format established by the HIP linker input type.
  iree_hip_linker_input_format_t format;
  // Raw LLVM bitcode/IR, or SPIR-V bytes (raw or in a clang offload bundle).
  iree_const_byte_span_t data;
  // NUL-terminated input name used in compiler diagnostics.
  const char* name;
} iree_hip_linker_input_t;

// Compiles and links |inputs| into one AMDGPU executable for |target_isa|.
// Inputs in one invocation share the LLVM or SPIR-V compilation pipeline.
// The returned bytes are allocated with |host_allocator|.
iree_status_t iree_hip_linker_compile(iree_string_view_t target_isa,
                                      iree_host_size_t input_count,
                                      const iree_hip_linker_input_t* inputs,
                                      iree_host_size_t option_count,
                                      const char* const* options,
                                      iree_allocator_t host_allocator,
                                      iree_byte_span_t* out_executable);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // HRX_BINDING_HIP_LINKER_COMPILER_H_
