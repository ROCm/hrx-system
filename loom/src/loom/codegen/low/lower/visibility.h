// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Function-wide selection of acquire visibility realization.

#ifndef LOOM_CODEGEN_LOW_LOWER_VISIBILITY_H_
#define LOOM_CODEGEN_LOW_LOWER_VISIBILITY_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_lower_context_t loom_low_lower_context_t;
typedef struct loom_low_lower_visibility_access_t
    loom_low_lower_visibility_access_t;

// Target cost and capability facts for deferring cache visibility to reads.
// A zero invocation count leaves acquisition at its ordinary target recipe.
// An enabled provider implements required visibility on every physical packet
// of a scalar global or generic 32-bit read that may observe acquired writes,
// independently of advisory cache policies.
typedef struct loom_low_lower_visibility_model_t {
  // Invocations sharing a native read issue group.
  uint32_t invocation_count;
  // Working-set envelope within which the reuse cost comparison applies.
  uint32_t reuse_byte_limit;
  // Repeated full working-set reads that favor eager acquisition.
  uint32_t reuse_count;
} loom_low_lower_visibility_model_t;

// Temporary observations supplied by the shared source planning walk. The
// records live in the planning arena and are discarded before op selection.
typedef struct loom_low_lower_visibility_builder_t {
  // Native realization and locality facts, or a disabled zero model.
  loom_low_lower_visibility_model_t model;
  // Source body whose direct operations form the closed selection region.
  const loom_region_t* body;
  // Observed memory reads and acquisition points in traversal order.
  loom_low_lower_visibility_access_t* first_access;
  // Last observation, or NULL before the first observation.
  loom_low_lower_visibility_access_t* last_access;
  // Strongest observed acquire scope; thread scope requires no visibility.
  uint8_t acquire_scope;
  // True when an opaque effect or unsupported read prevents deferral.
  bool requires_eager;
} loom_low_lower_visibility_builder_t;

// Collects one operation's effects and retained storage facts. Called only by
// the common preselection walk; target queries never traverse source IR.
iree_status_t loom_low_lower_visibility_observe(
    loom_low_lower_context_t* context,
    loom_low_lower_visibility_builder_t* builder, const loom_op_t* op);

// Selects a complete function-wide realization. A non-thread result requires
// that scope on every mutable global or generic payload read unless retained
// analysis proves it observes unchanged kernel-entry contents. Kernel entry
// supplies visibility for that prefix; ordinary callable entry does not.
// All other reads retain the scope, including aliases and loop backedges.
// Thread scope keeps the ordinary eager recipe. Completion and release
// ordering are unchanged.
iree_status_t loom_low_lower_visibility_select(
    loom_low_lower_context_t* context,
    const loom_low_lower_visibility_builder_t* builder,
    uint8_t* out_read_visibility_scope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_VISIBILITY_H_
