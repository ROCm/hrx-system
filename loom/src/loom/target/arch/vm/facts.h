// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Immutable VM target facts and target-owned function ABI contracts.

#ifndef LOOM_TARGET_ARCH_VM_FACTS_H_
#define LOOM_TARGET_ARCH_VM_FACTS_H_

#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Static fact type used by VM target projection.
extern const loom_target_fact_type_t loom_vm_target_fact_type;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_VM_FACTS_H_
