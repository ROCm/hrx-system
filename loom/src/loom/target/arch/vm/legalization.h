// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM source legalization.

#ifndef LOOM_TARGET_ARCH_VM_LEGALIZATION_H_
#define LOOM_TARGET_ARCH_VM_LEGALIZATION_H_

#include "loom/target/legalization.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned source rewrites that expose Core VM scalar operations.
extern const loom_target_legalizer_provider_t loom_vm_legalizer_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LEGALIZATION_H_
