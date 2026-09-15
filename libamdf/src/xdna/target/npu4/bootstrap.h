// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_TARGET_NPU4_BOOTSTRAP_H_
#define AMDF_SRC_XDNA_TARGET_NPU4_BOOTSTRAP_H_

#include "libamdf/src/xdna/bootstrap.h"

#ifdef __cplusplus
extern "C" {
#endif

// Audited native bootstrap selected by the central endpoint profile table.
extern const amdf_xdna_bootstrap_t amdf_xdna_npu4_bootstrap;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_TARGET_NPU4_BOOTSTRAP_H_
