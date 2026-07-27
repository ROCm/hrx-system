// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// IREE VM reference lifetime materialization pass.
//
// This pass is the target-owned pass-pipeline surface over the generic
// ownership lifetime materializer. It recognizes ireevm.ref<T> values as
// managed resources and inserts explicit ireevm.ref.release operations for
// owned references whose lifetime ends along a block or CFG edge.

#ifndef LOOM_TARGET_ARCH_IREEVM_REF_LIFETIME_PASS_H_
#define LOOM_TARGET_ARCH_IREEVM_REF_LIFETIME_PASS_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns static pass metadata for ireevm-ref-lifetime.
const loom_pass_info_t* loom_ireevm_ref_lifetime_pass_info(void);

// Runs IREE VM reference lifetime materialization on a module.
iree_status_t loom_ireevm_ref_lifetime_run(loom_pass_t* pass,
                                           loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_IREEVM_REF_LIFETIME_PASS_H_
