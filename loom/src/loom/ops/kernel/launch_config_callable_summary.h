// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cached required-inline callable summaries used by launch configurations.

#ifndef LOOM_OPS_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_H_
#define LOOM_OPS_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_H_

#include "iree/base/internal/arena.h"
#include "loom/ops/kernel/launch_config_contract.h"
#include "loom/util/fact_table.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_kernel_launch_config_callable_summary_t {
  // Strongest callable effect contract established by the helper body.
  loom_callable_effect_contract_t effect_contract;
  // First semantic launch-configuration failure in the helper call tree.
  loom_kernel_launch_config_contract_failure_t failure;
  // Operation establishing failure, or NULL for a recursive inline cycle.
  const loom_op_t* failure_op;
  // Effective traits on failure_op for an effect failure.
  loom_trait_flags_t failure_traits;
  // Number of flattened callable arguments.
  uint16_t argument_count;
  // Number of flattened callable results.
  uint16_t result_count;
  // Number of 64-bit words required to address every argument.
  uint16_t argument_word_count;
  // Number of 64-bit words required to address every result.
  uint16_t result_word_count;
  // Arguments whose entry storage may be read by the helper call tree.
  uint64_t* read_argument_bits;
  // Results whose value depends on a helper call-tree storage read.
  uint64_t* dependent_result_bits;
  // Reference results with at least one source outside the formal arguments.
  uint64_t* nonargument_result_bits;
  // Exact formal storage sources for each reference result. Rows are indexed
  // by result ordinal and contain argument_word_count words.
  uint64_t* result_reference_argument_bits;
  // Joined return facts owned by the cache's summary fact table.
  loom_value_facts_t* result_facts;
  // Internal construction state: zero, visiting, or complete.
  uint8_t state;
} loom_kernel_launch_config_callable_summary_t;

typedef struct loom_kernel_launch_config_callable_cache_t {
  // Verified module containing every summarized function and call.
  const loom_module_t* module;
  // Caller-owned storage retaining summaries through contract analysis.
  iree_arena_allocator_t* arena;
  // Lazily allocated symbol-indexed summary pointers.
  loom_kernel_launch_config_callable_summary_t** summaries;
  // Lazily allocated fact owner for joined results and extension payloads.
  loom_value_fact_table_t* summary_facts;
} loom_kernel_launch_config_callable_cache_t;

// Initializes an allocation-free cache. Symbol storage and fact extensions are
// allocated only after observing a required-inline call.
void loom_kernel_launch_config_callable_cache_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_kernel_launch_config_callable_cache_t* out_cache);

// Returns the cached summary for an explicitly required-inline func.call.
// Ordinary calls return NULL. |call_op| must be a func.call. Semantic failures
// are retained in the summary; only allocation failure is returned as a status.
iree_status_t loom_kernel_launch_config_callable_cache_lookup(
    loom_kernel_launch_config_callable_cache_t* cache, const loom_op_t* call_op,
    const loom_kernel_launch_config_callable_summary_t** out_summary);

// Fact-context callback importing cached call results into a caller scope.
iree_status_t loom_kernel_launch_config_callable_resolve_results(
    void* user_data, loom_fact_context_t* context, const loom_module_t* module,
    const loom_op_t* op, const loom_value_facts_t* operand_facts,
    loom_value_facts_t* result_facts);

// Returns whether the helper reads storage supplied by |argument_index|.
bool loom_kernel_launch_config_callable_reads_argument(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t argument_index);

// Returns whether |result_index| depends on a storage read.
bool loom_kernel_launch_config_callable_result_is_dependent(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t result_index);

// Returns whether reference |result_index| may derive from |argument_index|.
bool loom_kernel_launch_config_callable_result_references_argument(
    const loom_kernel_launch_config_callable_summary_t* summary,
    uint16_t result_index, uint16_t argument_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_KERNEL_LAUNCH_CONFIG_CALLABLE_SUMMARY_H_
