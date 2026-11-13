// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Borrowed exact snapshots following one indexed materialization.

#ifndef LOOM_LINK_SOURCE_SNAPSHOTS_H_
#define LOOM_LINK_SOURCE_SNAPSHOTS_H_

#include "loom/error/source.h"
#include "loom/link/materialization_environment.h"
#include "loom/link/plan_projection.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_link_source_snapshots_t {
  // Embedding supplying indexed snapshots and receiving diagnostics.
  const loom_link_plan_materialization_environment_t* environment;
  // Indexed input currently being cloned.
  const loom_link_module_index_module_t* input;
  // Optional original-to-intermediate source IDs for a partial projection.
  const loom_source_id_t* projected_sources;
  // Current input's snapshot rows, reused after each synchronous add.
  loom_source_table_resolver_t current;
  // Linked snapshot rows retained in the product arena.
  loom_source_table_resolver_t target;
} loom_link_source_snapshots_t;

// Allocates bounded row arrays from the selected source directory. Filenames
// and source text remain borrowed from the embedding. With no source-entry
// callback this allocates nothing.
iree_status_t loom_link_source_snapshots_initialize(
    const loom_link_module_index_t* index,
    const loom_link_plan_module_projection_t* projection,
    const loom_link_plan_materialization_environment_t* environment,
    iree_arena_allocator_t* scratch_arena,
    iree_arena_allocator_t* product_arena,
    loom_link_source_snapshots_t* out_snapshots);

// Linker callback invoked before cloning the current input. Composes the
// producer's source-ID maps once, retaining direct lookup for diagnostics and
// final publication. Conflicting snapshots are diagnosed and reject the add.
iree_status_t loom_link_source_snapshots_capture(
    void* user_data, const loom_module_t* source_module,
    const loom_module_t* target_module, const loom_source_id_t* target_sources,
    bool* out_accepted);

// Forwards an emission with a resolver borrowing the current and linked rows.
iree_status_t loom_link_source_snapshots_emit(
    void* user_data, const loom_diagnostic_emission_t* emission);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_LINK_SOURCE_SNAPSHOTS_H_
