// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Target-low pass pipeline fragments.
//
// These helpers append ordinary pass IR into a caller-owned pass.pipeline.
// They do not compile, interpret, or execute pass programs.

#ifndef LOOM_CODEGEN_LOW_PIPELINE_PIPELINE_H_
#define LOOM_CODEGEN_LOW_PIPELINE_PIPELINE_H_

#include "iree/base/api.h"
#include "loom/pass/builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Appends the common low cleanup and operand-form selection sequence used
// before target-low packetization.
iree_status_t loom_low_pipeline_build_packetization_preparation(
    loom_builder_t* builder);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_PIPELINE_PIPELINE_H_
