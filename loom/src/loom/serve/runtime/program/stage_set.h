// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_STAGE_SET_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_STAGE_SET_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/program/preparation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable compiler output in preparation declaration order. It owns code,
// not model buffers, prepared weights, command recordings or VM state.
typedef struct loom_serve_stage_set_t {
  // Allocator owning this set and its stage array.
  iree_allocator_t host_allocator;
  // Terminal copied evidence, including diagnostics on compilation failure.
  loom_serve_compilation_report_t compilation;
  // Allocated stage slots; uncompiled slots are NULL on failure.
  iree_host_size_t count;
  // Owned compiled command or direct-dispatch entries.
  loom_serve_jit_stage_t** values;
} loom_serve_stage_set_t;

// Transfers status and set after all compiler accesses retire. The model may
// validate/record against immutable roots here, then publish to its owner.
typedef void (*loom_serve_stage_set_completion_fn_t)(
    void* user_data, iree_status_t status, loom_serve_stage_set_t* stages);

// Optional per-command immutable-layout publication. Ordinals match preparation
// declaration order, but callbacks may overlap and arrive in any order. Direct
// stages publish no command layout. The borrowed program survives until the
// recipient frees the final stage set, including after native compilation
// fails.
typedef void (*loom_serve_stage_set_layout_fn_t)(
    void* user_data, iree_host_size_t index, const loom_cmd_program_t* program);

// Snapshots providers and concurrently compiles declarations on shared workers,
// enforcing their workspace limits before publishing successful completion.
// devices is the ordered deployment with preparation_execution_count members.
// Each exact device/family has an independent catalog and report accumulator.
// Reuses identical kernel requests within each catalog. Success owes one
// completion, possibly before return; rejection owes none. Devices,
// preparation, sources, optional sanitizer, and user_data remain borrowed
// through completion. No model/queue state or parameter residency is mutated
// and no caller/worker waits.
iree_status_t loom_serve_stage_set_compile_async(
    loom_serve_device_t* const* devices,
    const loom_serve_preparation_t* preparation,
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

// Compiles all declarations against their indexed execution catalogs. jits has
// preparation_execution_count members in the same immutable deployment order;
// its pointer array is borrowed only during the call.
// Retains every catalog through the final join, including after submission or
// stage failure. Results remain in declaration order and include successfully
// built stages on failure. The returned set merges each catalog's cumulative
// report at completion; independent catalog jobs may still be running.
// Preparation and user_data remain borrowed through completion. Success owes
// exactly one completion, possibly before return; rejection owes none.
iree_status_t loom_serve_stage_set_compile_catalog_async(
    loom_serve_jit_t* const* jits, const loom_serve_preparation_t* preparation,
    loom_serve_stage_set_layout_fn_t layout,
    loom_serve_stage_set_completion_fn_t completion, void* user_data,
    iree_allocator_t host_allocator);

void loom_serve_stage_set_free(loom_serve_stage_set_t* stages);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_STAGE_SET_H_
