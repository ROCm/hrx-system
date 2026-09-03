// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V target op interpretation.

#ifndef LOOM_TARGET_ARCH_SPIRV_OPS_TARGET_H_
#define LOOM_TARGET_ARCH_SPIRV_OPS_TARGET_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/target/resolved_target.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_builder_t loom_builder_t;

// Materializes an exact resolved SPIR-V target as a spirv.target definition.
iree_status_t loom_spirv_target_materialize_definition(
    loom_builder_t* builder, const loom_resolved_target_t* resolved_target,
    loom_symbol_ref_t symbol, loom_location_id_t location);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_SPIRV_OPS_TARGET_H_
