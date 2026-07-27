// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Pass pipeline IR verifier.
//
// The verifier checks pass.pipeline/pass.* control IR before it is compiled to
// an executable pipeline program. It resolves pass.run descriptors through the
// active C registry and validates control-flow shape, anchor transitions, pass
// availability, descriptor options, and descriptor requirements without running
// any transforms.

#ifndef LOOM_PASS_VERIFY_H_
#define LOOM_PASS_VERIFY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/ir.h"
#include "loom/pass/environment.h"
#include "loom/pass/predicate.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_verify_options_t {
  // Registry used to resolve pass.run keys. Required.
  const loom_pass_registry_t* registry;
  // Typed execution environment capabilities.
  loom_pass_environment_t environment;
  // Optional provider for pass.where predicates outside the core built-ins.
  loom_pass_predicate_provider_t predicate_provider;
} loom_pass_verify_options_t;

// Verifies every pass.pipeline op in |module|.
iree_status_t loom_pass_verify_module(const loom_module_t* module,
                                      const loom_pass_verify_options_t* options,
                                      iree_arena_allocator_t* scratch_arena);

// Verifies one pass.pipeline op.
iree_status_t loom_pass_verify_pipeline_op(
    const loom_module_t* module, const loom_op_t* pipeline_op,
    const loom_pass_verify_options_t* options,
    iree_arena_allocator_t* scratch_arena);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_VERIFY_H_
