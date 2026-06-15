// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Consumes local scf.for unroll intent.
//
// This pass consumes scf.for unroll / unroll(%factor) policies with
// compile-time constant loop domains and clones their bodies inline. It is
// intentionally separate from scf-to-cfg so structured loop intent can survive
// until passes such as private-fragment promotion have used it, and then be
// removed before target lowering would otherwise materialize dynamic loop
// control.

#ifndef LOOM_TRANSFORMS_SCF_UNROLL_H_
#define LOOM_TRANSFORMS_SCF_UNROLL_H_

#include "iree/base/api.h"
#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

const loom_pass_info_t* loom_scf_unroll_pass_info(void);

iree_status_t loom_scf_unroll_create(loom_pass_t* pass,
                                     iree_string_view_t options_string);

iree_status_t loom_scf_unroll_run(loom_pass_t* pass, loom_module_t* module,
                                  loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_SCF_UNROLL_H_
