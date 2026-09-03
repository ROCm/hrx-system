// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Projection of authored Low representations into selected target contracts.

#ifndef LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_
#define LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ops/op_defs.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Projects every register type and descriptor-backed packet in |function|
// from its authored representation into |target_facts|' exact target
// representation. The complete projection is planned before any function IR
// is mutated. User contract failures are emitted through |emitter| and report
// |out_valid| false; infrastructure failures are returned as status.
iree_status_t loom_low_project_function_representation(
    loom_module_t* module, loom_func_like_t function,
    const loom_target_facts_t* target_facts,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t emitter, iree_arena_allocator_t* scratch_arena,
    bool* out_valid, bool* out_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_REPRESENTATION_PROJECTION_H_
