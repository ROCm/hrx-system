// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source workload pipeline runner for low codegen stress tests and fuzzers.
//
// This helper deliberately lives beside the generated workload source instead
// of inside the production low pipeline. It composes the real verifier,
// source-to-low lowering, low verifier, packetizer, scheduler, and allocator so
// tests can put randomized pressure on production APIs without growing a second
// compiler path.

#ifndef LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_PIPELINE_H_
#define LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_PIPELINE_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/lower/lower.h"
#include "loom/codegen/low/testing/source_workload.h"
#include "loom/ir/ir.h"
#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_low_source_workload_pipeline_options_t {
  // Pass registry used to resolve production low-prep passes.
  const loom_pass_registry_t* pass_registry;
  // Low descriptor registry linked into the caller.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Target lowering policies linked into the caller.
  const loom_low_lower_policy_registry_t* policy_registry;
  // Candidate selection strategy used by packetization.
  loom_low_schedule_strategy_t schedule_strategy;
} loom_low_source_workload_pipeline_options_t;

typedef struct loom_low_source_workload_pipeline_counters_t {
  // Source op category counts before lowering.
  loom_low_source_workload_counts_t source_counts;
  // Number of low.op and low.const packets emitted by lowering.
  uint32_t low_descriptor_op_count;
  // Number of source-to-low error diagnostics emitted by lowering.
  uint32_t lower_error_count;
  // Number of source-to-low remark diagnostics emitted by lowering.
  uint32_t lower_remark_count;
  // Number of schedule nodes produced by packetization.
  iree_host_size_t schedule_node_count;
  // Number of schedule dependency edges produced by packetization.
  iree_host_size_t schedule_dependency_count;
  // Number of descriptor resource-use rows produced by packetization.
  iree_host_size_t schedule_resource_use_count;
  // Number of schedule hazard-gap rows produced by packetization.
  iree_host_size_t schedule_hazard_gap_count;
  // Number of register-allocation assignments produced by packetization.
  iree_host_size_t allocation_assignment_count;
  // Number of assignments placed in spill slots.
  iree_host_size_t allocation_spill_count;
  // Number of low.copy ops coalesced by allocation.
  iree_host_size_t allocation_coalesced_copy_count;
  // Number of low.copy ops left materialized by allocation.
  iree_host_size_t allocation_materialized_copy_count;
} loom_low_source_workload_pipeline_counters_t;

// Runs the generated workload through source verification, source-to-low
// lowering, low verification, packetization, scheduling, and allocation.
iree_status_t loom_low_source_workload_run_pipeline(
    loom_module_t* module,
    const loom_low_source_workload_pipeline_options_t* options,
    iree_arena_block_pool_t* block_pool,
    loom_low_source_workload_pipeline_counters_t* out_counters);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_TESTING_SOURCE_WORKLOAD_PIPELINE_H_
