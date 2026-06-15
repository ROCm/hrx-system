// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// LLVM target-independent intrinsic declaration helpers.

#ifndef LOOM_TARGET_LLVMIR_INTRINSICS_BUILTIN_H_
#define LOOM_TARGET_LLVMIR_INTRINSICS_BUILTIN_H_

#include "loom/target/emit/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_llvmir_declare_memcpy(loom_llvmir_module_t* module,
                                         uint32_t target_address_space,
                                         uint32_t source_address_space,
                                         uint32_t length_bit_width,
                                         loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_declare_memset(loom_llvmir_module_t* module,
                                         uint32_t target_address_space,
                                         uint32_t length_bit_width,
                                         loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_declare_prefetch(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_declare_lifetime_start(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function);

iree_status_t loom_llvmir_declare_lifetime_end(
    loom_llvmir_module_t* module, uint32_t pointer_address_space,
    loom_llvmir_function_t** out_function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_INTRINSICS_BUILTIN_H_
