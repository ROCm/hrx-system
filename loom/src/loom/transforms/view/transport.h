// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar replacement of views transported over a common buffer.
//
// The view fact domain owns the common materializing buffer. This pass carries
// byte offsets through source control flow and reconstructs the original view
// type where each argument or result is defined. Buffer assumptions and dynamic
// shape/layout references remain ordinary source program values. Offset recipes
// consume the view-region analysis; no target pointer representation is
// assumed.

#ifndef LOOM_TRANSFORMS_VIEW_TRANSPORT_H_
#define LOOM_TRANSFORMS_VIEW_TRANSPORT_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_decompose_view_transports_pass_info(void);

iree_status_t loom_decompose_view_transports_run(loom_pass_t* pass,
                                                 loom_module_t* module,
                                                 loom_func_like_t function);

// Decomposes view selections without a common materializing buffer into a
// correlated materializing-buffer and root-relative-offset selection.
const loom_pass_info_t* loom_decompose_view_root_selections_pass_info(void);

iree_status_t loom_decompose_view_root_selections_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_VIEW_TRANSPORT_H_
