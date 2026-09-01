// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Core VM target-low packet verification.

#ifndef LOOM_TARGET_ARCH_VM_LOW_VERIFY_H_
#define LOOM_TARGET_ARCH_VM_LOW_VERIFY_H_

#include "loom/codegen/low/verify.h"

#ifdef __cplusplus
extern "C" {
#endif

// Verifies ISA relationships not expressible in generic Low descriptors.
extern const loom_low_verify_provider_t loom_vm_low_verify_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_LOW_VERIFY_H_
