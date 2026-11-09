// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Binding and ownership proofs for CFG block argument rewrites.

#ifndef LOOM_TRANSFORMS_CFG_BLOCK_ARGUMENTS_H_
#define LOOM_TRANSFORMS_CFG_BLOCK_ARGUMENTS_H_

#include "loom/ops/op_defs.h"
#include "loom/util/dominance.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns whether binding the arguments of |block| to |replacements|
// preserves dependent types and makes every replacement available immediately
// before |before_op|. All values belong to verified IR; definition-site
// verification already established the availability of their type dependencies.
// A failed proof rejects the proposed rewrite. This query does not allocate.
bool loom_cfg_block_arguments_can_bind(const loom_module_t* module,
                                       const loom_dominance_info_t* dominance,
                                       const loom_block_t* block,
                                       loom_value_slice_t replacements,
                                       const loom_op_t* before_op);

// Returns whether forwarding an argument to an existing equal-valued source
// preserves independent ownership. Equality and availability are separate
// proofs: consuming uses and required storage identities keep the boundary.
// Managed handles require a lifetime proof beyond payload equality and retain
// their transfer boundaries, including when carried by Low register types.
// Uses cached occurrence classifications without following users or aliases.
bool loom_cfg_block_argument_can_forward(const loom_module_t* module,
                                         loom_value_id_t argument,
                                         loom_value_id_t source);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_CFG_BLOCK_ARGUMENTS_H_
