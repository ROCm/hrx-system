// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Device realization of residual command-program launch counts.

#ifndef LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_COUNT_REALIZATION_H_
#define LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_COUNT_REALIZATION_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/error/emitter.h"
#include "loom/ir/module.h"
#include "loom/transforms/kernel/kernel_class_materializer.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

// One independently closed generated count-cluster producer and its
// plan-module entry.
typedef struct loom_cmd_launch_count_kernel_t {
  // Entry declaration retained in the command plan module.
  loom_symbol_ref_t entry;

  // Independently closed source product published through the request sink.
  loom_kernel_class_product_t product;
} loom_cmd_launch_count_kernel_t;

// Generated device producers created while legalizing command launch counts.
typedef struct loom_cmd_launch_count_realization_t {
  // Scratch-backed generated products in construction order.
  loom_cmd_launch_count_kernel_t* kernels;

  // Number of entries in |kernels|.
  iree_host_size_t kernel_count;

  // Allocated entries in |kernels|.
  iree_host_size_t kernel_capacity;

  // Scratch-backed O(1) generated-product lookup by plan-module symbol ID.
  const loom_cmd_launch_count_kernel_t** kernel_by_symbol;

  // Number of entries in |kernel_by_symbol|.
  iree_host_size_t symbol_count;
} loom_cmd_launch_count_realization_t;

// Realizes residual direct dispatch counts as generated device producers.
//
// |root_ops| names normalized command roots and may contain duplicates. Exact
// counts remain direct. Compatible residual deterministic read-only closures
// in one ordered command block are clustered into a target-matched 1x1x1
// kernel that writes one transient array of XYZ tuples. The producer normally
// precedes the cluster's first consumer, and each consumer retains its authored
// position with an indirect view of its tuple. Possible intervening writes
// split clusters unless explicit alias scopes prove them disjoint. A residual
// consumer outside an ordered block is realized as a singleton nested serial
// command. The generated source products remain owned by |out_realization|
// until deinitialization.
//
// Captured reads must preserve their source observation. Root-local motion
// analysis proves movement to the originating consumer before clustering may
// hoist reads over explicitly disjoint consumer writes. An earlier observation
// instead gets a singleton producer at its source position. A closure with no
// common legal placement is diagnosed rather than reloading changed storage.
// Every root is planned before source mutation; captured source operations
// remain live until all generated products have been built.
//
// Authored contract failures emit diagnostics, clear |out_valid|, and return
// OK. Unsupported compiler mappings and infrastructure failures return a
// non-OK status.
iree_status_t loom_cmd_launch_count_realize(
    loom_module_t* module, loom_op_t* const* root_ops,
    iree_host_size_t root_count, const loom_value_fact_table_t* facts,
    iree_diagnostic_emitter_t diagnostic_emitter,
    iree_arena_block_pool_t* block_pool, iree_arena_allocator_t* scratch_arena,
    bool* out_valid, loom_cmd_launch_count_realization_t* out_realization);

// Releases every generated source module and resets |realization|.
void loom_cmd_launch_count_realization_deinitialize(
    loom_cmd_launch_count_realization_t* realization);

// Returns the generated product for |entry|, or NULL for an ordinary entry.
static inline const loom_cmd_launch_count_kernel_t*
loom_cmd_launch_count_realization_lookup(
    const loom_cmd_launch_count_realization_t* realization,
    loom_symbol_ref_t entry) {
  if (!loom_symbol_ref_is_valid(entry) || entry.module_id != 0 ||
      entry.symbol_id >= realization->symbol_count) {
    return NULL;
  }
  return realization->kernel_by_symbol[entry.symbol_id];
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_CMD_LOWER_LAUNCH_COUNT_REALIZATION_H_
