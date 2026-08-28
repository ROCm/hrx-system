// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_SYMBOL_TEMPLATE_EXPANSION_PIPELINE_H_
#define LOOM_TRANSFORMS_SYMBOL_TEMPLATE_EXPANSION_PIPELINE_H_

#include "iree/base/api.h"
#include "loom/pass/builder.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds outside-in template selection and callable expansion pass IR.
//
// |cleanup_body| runs only after an iteration changes the module. It must
// expose newly propagated constants and facts before the next nested template
// predicate is evaluated. The expansion finishes with final selection so
// unresolved reachable applications diagnose at the product boundary.
iree_status_t loom_template_expansion_pipeline_build(
    loom_builder_t* builder, loom_pass_ir_body_build_fn_t cleanup_body,
    void* cleanup_user_data);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_SYMBOL_TEMPLATE_EXPANSION_PIPELINE_H_
