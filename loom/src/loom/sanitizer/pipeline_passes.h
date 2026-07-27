// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Sanitizer pass slots selected by target pipeline options.

#ifndef LOOM_SANITIZER_PIPELINE_PASSES_H_
#define LOOM_SANITIZER_PIPELINE_PASSES_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_sanitizer_insert_assertions_pass_info(void);

iree_status_t loom_sanitizer_insert_assertions_create(
    loom_pass_t* pass, iree_string_view_t options);

iree_status_t loom_sanitizer_insert_assertions_run(loom_pass_t* pass,
                                                   loom_module_t* module,
                                                   loom_func_like_t function);

const loom_pass_info_t* loom_sanitizer_materialize_assertions_pass_info(void);

iree_status_t loom_sanitizer_materialize_assertions_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SANITIZER_PIPELINE_PASSES_H_
