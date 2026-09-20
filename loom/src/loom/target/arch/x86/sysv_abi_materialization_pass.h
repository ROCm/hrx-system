// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// x86 SysV ABI materialization pass.

#ifndef LOOM_TARGET_ARCH_X86_SYSV_ABI_MATERIALIZATION_PASS_H_
#define LOOM_TARGET_ARCH_X86_SYSV_ABI_MATERIALIZATION_PASS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for x86-materialize-sysv-abi.
const loom_pass_info_t* loom_x86_materialize_sysv_abi_pass_info(void);

// Materializes canonical SysV layouts on authored x86 Low function symbols.
iree_status_t loom_x86_materialize_sysv_abi_run(loom_pass_t* pass,
                                                loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_X86_SYSV_ABI_MATERIALIZATION_PASS_H_
