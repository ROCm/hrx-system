// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// loom-check support for Core VM module emission.

#ifndef LOOM_TARGET_EMIT_VM_CHECK_LOOM_CHECK_H_
#define LOOM_TARGET_EMIT_VM_CHECK_LOOM_CHECK_H_

#include "loom/tools/loom-check/execute.h"

#ifdef __cplusplus
extern "C" {
#endif

// Emits, verifies, and dumps Core VM bytecode modules as mnemonic text.
extern const loom_check_emit_provider_t loom_vm_loom_check_emit_provider;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_VM_CHECK_LOOM_CHECK_H_
