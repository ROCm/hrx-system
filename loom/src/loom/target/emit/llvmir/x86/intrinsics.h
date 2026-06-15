// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM x86 intrinsic declaration helpers.

#ifndef LOOM_TARGET_LLVMIR_X86_INTRINSICS_H_
#define LOOM_TARGET_LLVMIR_X86_INTRINSICS_H_

#include "loom/target/emit/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_llvmir_declare_x86_rdtsc(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_declare_x86_sse2_pause(
    loom_llvmir_module_t* module, loom_llvmir_function_t** out_function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_X86_INTRINSICS_H_
