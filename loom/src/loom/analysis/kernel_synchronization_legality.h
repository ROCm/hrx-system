// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Kernel synchronization legality analysis.
//
// This analysis proves whole-function synchronization contracts after target
// specialization and required inlining. It retains admitted kernel.async
// streams for target lowering. Local op verifiers independently check types,
// attributes, and SSA contracts.

#ifndef LOOM_ANALYSIS_KERNEL_SYNCHRONIZATION_LEGALITY_H_
#define LOOM_ANALYSIS_KERNEL_SYNCHRONIZATION_LEGALITY_H_

#include "iree/base/api.h"
#include "loom/analysis/movement.h"
#include "loom/error/emitter.h"
#include "loom/ir/ir.h"
#include "loom/ir/local_value_domain.h"
#include "loom/ir/module.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_kernel_synchronization_legality_options_t {
  // Shared immutable-source scope owning transitive callable storage effects.
  struct loom_storage_access_scope_t* storage_access;
  // Active local value domain extended by analyses as values are discovered.
  loom_local_value_domain_t* value_domain;
  // Borrowed function-local value facts. The analysis only reads this table.
  loom_value_fact_table_t* fact_table;
  // Structured diagnostic emitter for user legality failures.
  iree_diagnostic_emitter_t emitter;
  // Name of the phase reporting diagnostics, such as "source-low" or
  // "kernel-synchronization-legality".
  iree_string_view_t phase_name;
} loom_kernel_synchronization_legality_options_t;

// One admitted transfer and its canonical movement request. The owning stream
// records the group that commits it; endpoint projections remain in movement.
typedef struct loom_kernel_async_transfer_t {
  // Source transfer instruction and its independently analyzed endpoints.
  loom_movement_request_t request;
  // Group ordinal within this block's stream.
  iree_host_size_t group_index;
} loom_kernel_async_transfer_t;

// One committed group. Completion belongs to an explicit wait, which can also
// complete earlier groups. No target rediscovers this relation from token uses.
typedef struct loom_kernel_async_group_t {
  // Source group instruction.
  const loom_op_t* op;
  // First transfer committed by this group.
  iree_host_size_t first_transfer;
  // Number of consecutive transfers committed by this group.
  iree_host_size_t transfer_count;
  // Source wait retiring this group, including retirement by a later group.
  const loom_op_t* completion;
} loom_kernel_async_group_t;

// Retained straight-line stream for one block. Streams follow region visitation
// order and own no IR. A body rewrite invalidates them unless it translates the
// retained sites and value correspondence as part of that rewrite.
typedef struct loom_kernel_async_stream_t {
  // Next block with an asynchronous stream, or NULL.
  struct loom_kernel_async_stream_t* next;
  // Owning source block.
  const loom_block_t* block;
  // Transfers in issue order.
  loom_kernel_async_transfer_t* transfers;
  // Number of transfers.
  iree_host_size_t transfer_count;
  // Groups in commit order, with their explicit completion sites.
  loom_kernel_async_group_t* groups;
  // Number of groups.
  iree_host_size_t group_count;
  // Explicit waits in source order, retained for erasure after realization.
  loom_op_t** waits;
  // Number of explicit waits.
  iree_host_size_t wait_count;
} loom_kernel_async_stream_t;

typedef struct loom_kernel_synchronization_legality_result_t {
  // Canonical movement facts, including indexed endpoint view projections.
  loom_movement_analysis_t movement;
  // First retained stream; NULL for functions with no asynchronous work.
  loom_kernel_async_stream_t* streams;
  // Number of error diagnostics emitted.
  uint32_t error_count;
  // Number of nonempty blocks checked for synchronization legality.
  uint64_t blocks_checked;
  // Number of kernel barrier operations checked for convergent execution.
  uint64_t barriers_checked;
  // Number of kernel.async.group ops checked.
  uint64_t groups_checked;
  // Number of kernel.async.wait ops checked.
  uint64_t waits_checked;
} loom_kernel_synchronization_legality_result_t;

// Verifies whole-function synchronization contracts after caller, placement,
// and target facts are available. Source verification checks local operation
// contracts independently; this boundary checks control and temporal contracts
// that depend on the selected, expanded program.
//
// User IR failures are emitted through |options->emitter| and counted in
// |out_result|. The analysis stops after the first violation because retained
// synchronization state is unusable after that point. The function returns OK
// for user IR failures so callers can decide whether an illegal program is a
// pass failure, a source-to-low diagnostic, or another production-path gate.
// Infrastructure failures such as arena allocation failures are returned as
// status failures.
// All result storage and symbolic expressions live in arena. Input facts and
// the value domain remain valid through the consuming rewrite. A diagnosed
// failure leaves partial streams unusable for target selection.
iree_status_t loom_kernel_synchronization_legality_analyze_function(
    const loom_module_t* module, loom_func_like_t function,
    const loom_kernel_synchronization_legality_options_t* options,
    iree_arena_allocator_t* arena,
    loom_kernel_synchronization_legality_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_KERNEL_SYNCHRONIZATION_LEGALITY_H_
