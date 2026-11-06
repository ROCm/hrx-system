// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_CODEGEN_LOW_OBJECT_LIFETIME_H_
#define LOOM_CODEGEN_LOW_OBJECT_LIFETIME_H_

#include "loom/analysis/ownership_lifetime.h"
#include "loom/codegen/low/descriptors.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prepares final-use cleanup for typed ptr carriers. Source ref owners retain
// their explicit consumption obligations; this never inserts their releases.
iree_status_t loom_low_object_lifetime_prepare(
    loom_module_t* module, const loom_low_descriptor_registry_t* registry,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    loom_ownership_lifetime_result_t* out_result);

// Checks prepared object ownership before executable emission, without
// changing IR. Custom pipelines and direct emission use the same contract.
iree_status_t loom_low_object_lifetime_verify(
    loom_module_t* module, const loom_low_descriptor_registry_t* registry,
    iree_arena_allocator_t* arena, iree_diagnostic_emitter_t emitter,
    loom_ownership_lifetime_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_CODEGEN_LOW_OBJECT_LIFETIME_H_
