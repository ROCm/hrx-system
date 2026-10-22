// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_INPUTS_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_INPUTS_H_

#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"
#include "loom/serve/runtime/program/preparation.h"

#ifdef __cplusplus
extern "C" {
#endif

// Independently indexed checkpoint domains and immutable source-selected read
// ranges. Keys and metadata borrow the owned indexes; no payload is loaded.
typedef struct loom_serve_weight_inputs_t loom_serve_weight_inputs_t;

// Opens and indexes the declaration's ordered shard lists on a cold worker.
// Relative paths resolve against sources.base_directory, not the source
// package. Each independent source has its own provider and imported-file
// cache.
iree_status_t loom_serve_weight_inputs_allocate(
    const loom_serve_preparation_parameter_t* parameter,
    const loom_serve_sources_t* sources, iree_host_size_t concurrency,
    iree_allocator_t host_allocator, loom_serve_weight_inputs_t** out_inputs);
void loom_serve_weight_inputs_free(loom_serve_weight_inputs_t* inputs);

// Borrows one indexed source, with ordinal established by declaration capture.
iree_io_parameter_provider_t* loom_serve_weight_inputs_provider(
    const loom_serve_weight_inputs_t* inputs, uint32_t source);
uint32_t loom_serve_weight_inputs_source_count(
    const loom_serve_weight_inputs_t* inputs);

// Checks an implicit source-zero read against the external checkpoint index
// during preparation. A larger entry is valid; only the requested prefix is
// read. Empty input recipes use this for unchanged or in-place prepared bytes.
iree_status_t loom_serve_weight_inputs_validate_direct(
    const loom_serve_weight_inputs_t* inputs, iree_string_view_t key,
    iree_device_size_t byte_length);

// One validated input range in a source-selected tensor transformation.
typedef struct loom_serve_weight_input_t {
  // Canonical source key and metadata, borrowed from its immutable index.
  const iree_io_parameter_index_entry_t* entry;
  // Byte placement relative to this tensor's input staging region.
  iree_io_parameter_span_t span;
  // Independent checkpoint ordinal; identical keys in peers remain distinct.
  uint32_t source;
} loom_serve_weight_input_t;

// Captured input extent for one final tensor. Empty recipes use canonical
// source-zero reads, with optional equal-size in-place preparation.
typedef struct loom_serve_weight_input_recipe_t {
  // First input ordinal in the plan's immutable range table.
  uint32_t first;
  // Number of consecutive input ranges in declaration order.
  uint32_t count;
  // Required input staging bytes, including source-selected alignment gaps.
  iree_device_size_t byte_length;
} loom_serve_weight_input_recipe_t;

// Borrows a validated range after policy capture has completed.
const loom_serve_weight_input_t* loom_serve_weight_inputs_at(
    const loom_serve_weight_inputs_t* inputs, uint32_t ordinal);

// Largest captured tensor input extent, aligned for reuse by independent lanes.
// Zero means no out-of-place preparation. No allocation occurs here.
iree_device_size_t loom_serve_weight_inputs_staging_length(
    const loom_serve_weight_inputs_t* inputs);
iree_device_size_t loom_serve_weight_inputs_staging_alignment(
    const loom_serve_weight_inputs_t* inputs);

// Links the source-facing "weights" capability beside "prepare":
//   count() -> i32
//   parameter(i32 source, buffer key) -> (buffer metadata, i64 byte_length)
//   input(i32 source, buffer key, i64 offset, i64 length, i64 alignment) -> i64
// parameter exposes the index format's opaque metadata; interpretation belongs
// to the model policy. input captures a nonempty, in-bounds range and returns
// its aligned staging offset. Alignment is a positive power of two. The plan
// outlives the module and all source programs borrowing returned metadata.
iree_status_t loom_serve_weight_inputs_module_create(
    loom_serve_weight_inputs_t* inputs, iree_vm_environment_t* environment,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module);

// Serializes one weight policy invocation. take ends capture even after source
// failure; the whole plan is discarded on failure, not reused or retried.
void loom_serve_weight_inputs_module_begin(iree_vm_module_t* module);
loom_serve_weight_input_recipe_t loom_serve_weight_inputs_module_take(
    iree_vm_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_INPUTS_H_
