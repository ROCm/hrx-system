// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Descriptor-backed pass.run resolution.
//
// This is the bridge between canonical pass IR and the C pass registry. It is
// intentionally a compile-time helper: execution should receive the resolved
// descriptor and decoded immutable options instead of reparsing operation
// attributes on every interpreter step.

#ifndef LOOM_PASS_INVOCATION_H_
#define LOOM_PASS_INVOCATION_H_

#include "iree/base/api.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_pass_invocation_t {
  // Source pass.run operation used for diagnostics and reproducers.
  const loom_op_t* source_op;
  // Descriptor key spelling from the source pass.run operation.
  iree_string_view_t key;
  // Resolved pass descriptor owned by the active registry.
  const loom_pass_descriptor_t* descriptor;
  // Static metadata returned by descriptor->info().
  const loom_pass_info_t* info;
  // Decoded immutable options allocated from the pipeline program arena.
  loom_pass_decoded_options_t decoded_options;
} loom_pass_invocation_t;

// Resolves a canonical pass.run op into an invocation record.
//
// |required_kind| is the current pipeline anchor requirement. Pass
// LOOM_PASS_COUNT_ to resolve without an anchor check.
iree_status_t loom_pass_invocation_resolve_run_op(
    const loom_module_t* module, const loom_pass_registry_t* registry,
    const loom_op_t* op, loom_pass_kind_t required_kind,
    iree_arena_allocator_t* arena, loom_pass_invocation_t* out_invocation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_INVOCATION_H_
