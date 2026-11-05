// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_STAGE_SET_H_
#define LOOM_RUNTIME_PROGRAM_STAGE_SET_H_

#include "loom/runtime/device/device.h"
#include "loom/runtime/program/jit.h"
#include "loom/runtime/program/preparation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable compiler output in preparation declaration order. It owns code,
// not model buffers, prepared weights, command recordings or VM state.
typedef struct loom_runtime_stage_set_t {
  // Allocator owning this set and its stage array.
  iree_allocator_t host_allocator;
  // Terminal copied evidence, including diagnostics on compilation failure.
  loom_runtime_compilation_report_t compilation;
  // Allocated stage slots; uncompiled slots are NULL on failure.
  iree_host_size_t count;
  // Owned compiled command or direct-dispatch entries.
  loom_runtime_jit_stage_t** values;
} loom_runtime_stage_set_t;

// Transfers status and set after all compiler accesses retire. The model may
// validate/record against immutable roots here, then publish to its owner.
typedef void (*loom_runtime_stage_set_completion_fn_t)(
    void* user_data, iree_status_t status, loom_runtime_stage_set_t* stages);

// Optional per-command immutable-layout publication. Ordinals match preparation
// declaration order, but callbacks may overlap and arrive in any order. Direct
// stages publish no command layout. The borrowed program survives until the
// recipient frees the final stage set, including after native compilation
// fails.
typedef void (*loom_runtime_stage_set_layout_fn_t)(
    void* user_data, iree_host_size_t index, const loom_cmd_program_t* program);

// Snapshots providers and concurrently compiles declarations on shared workers,
// enforcing their workspace limits before publishing successful completion.
// Reuses identical kernel requests within the new catalog. Success owes one
// completion, possibly before return; rejection owes none. Device, preparation,
// sources and user_data remain borrowed through completion. No model/queue
// state or parameter residency is mutated and no caller/worker waits.
iree_status_t loom_runtime_stage_set_compile_async(
    loom_runtime_device_t* device,
    const loom_runtime_preparation_t* preparation,
    const loom_runtime_sources_t* sources,
    loom_runtime_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

// Compiles all declarations concurrently against an already indexed catalog.
// Retains jit through the final join, including after submission or stage
// failure. Results remain in declaration order and include successfully built
// stages on failure. The returned set snapshots the catalog's cumulative report
// at completion; independent catalog jobs may still be running. Preparation and
// user_data remain borrowed through completion. Success owes exactly
// one completion, possibly before return; rejection owes none.
iree_status_t loom_runtime_stage_set_compile_catalog_async(
    loom_runtime_jit_t* jit, const loom_runtime_preparation_t* preparation,
    loom_runtime_stage_set_layout_fn_t layout,
    loom_runtime_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

void loom_runtime_stage_set_free(loom_runtime_stage_set_t* stages);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_STAGE_SET_H_
