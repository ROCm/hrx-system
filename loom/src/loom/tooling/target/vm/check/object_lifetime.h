// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Test-only execution of compiled object programs against native finalizers.

#ifndef LOOM_TOOLING_TARGET_VM_CHECK_OBJECT_LIFETIME_H_
#define LOOM_TOOLING_TARGET_VM_CHECK_OBJECT_LIFETIME_H_

#include "loom/tools/loom-check/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// Executes @run through the public compiler and VM, recording native payload
// access, reference counts, and destruction. The borrowed-argument option
// supplies a host-anchored pointer for a synchronous invocation. The provider
// is linked only into loom-check-test.
extern const loom_check_provider_t loom_vm_object_lifetime_check_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_VM_CHECK_OBJECT_LIFETIME_H_
