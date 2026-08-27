// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM target-family records.

#ifndef LOOM_TARGET_ARCH_VM_RECORDS_TARGET_RECORDS_H_
#define LOOM_TARGET_ARCH_VM_RECORDS_TARGET_RECORDS_H_

#include "loom/target/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Core VM target bundle used for bytecode module functions.
extern const loom_target_bundle_t loom_vm_target_bundle_core;

// Dense target rows indexed by vm.target kind.
extern const loom_target_bundle_table_t loom_vm_target_bundles;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_RECORDS_TARGET_RECORDS_H_
