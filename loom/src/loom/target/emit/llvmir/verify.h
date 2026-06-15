// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Lightweight verifier for the structured LLVM IR model.

#ifndef LOOM_TARGET_LLVMIR_VERIFY_H_
#define LOOM_TARGET_LLVMIR_VERIFY_H_

#include "loom/target/emit/llvmir/module.h"

#ifdef __cplusplus
extern "C" {
#endif

iree_status_t loom_llvmir_verify_module(const loom_llvmir_module_t* module);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TARGET_LLVMIR_VERIFY_H_
