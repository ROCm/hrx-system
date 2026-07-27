// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Benchmark selection and logical-to-physical work planning.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_WORK_PLAN_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_WORK_PLAN_H_

#include "iree/base/api.h"
#include "loom/tools/iree-benchmark-loom/model.h"
#include "loom/tools/iree-benchmark-loom/options.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum iree_benchmark_loom_work_item_kind_e {
  // Invalid or uninitialized work item.
  IREE_BENCHMARK_LOOM_WORK_ITEM_NONE = 0,
  // A case_end_to_end benchmark window over one selected benchmark.
  IREE_BENCHMARK_LOOM_WORK_ITEM_CASE_END_TO_END = 1,
  // One dispatch_complete measurement for one concrete case sample.
  IREE_BENCHMARK_LOOM_WORK_ITEM_DISPATCH_SAMPLE = 2,
} iree_benchmark_loom_work_item_kind_t;

#define IREE_BENCHMARK_LOOM_WORK_PLAN_INDEX_INVALID \
  IREE_BENCHMARK_LOOM_INDEX_INVALID

typedef struct iree_benchmark_loom_dispatch_compile_item_t {
  // Stable compile item ordinal within the plan.
  iree_host_size_t compile_item_index;
  // Representative selected benchmark ordinal used for compilation.
  iree_host_size_t representative_selection_index;
  // Sample-compilation label used for this compiled candidate.
  iree_string_view_t sample_compilation;
  // True when |case_sample_ordinal| is a compile-time case sample fact.
  bool has_case_sample_ordinal;
  // Case-local sample ordinal specialized into this compiled candidate.
  iree_host_size_t case_sample_ordinal;
} iree_benchmark_loom_dispatch_compile_item_t;

typedef struct iree_benchmark_loom_logical_sample_t {
  // Selected benchmark ordinal that owns this logical report sample.
  iree_host_size_t selection_index;
  // First benchmark-local sample ordinal covered by this logical sample.
  iree_host_size_t begin_benchmark_sample;
  // One-past-end benchmark-local sample ordinal covered by this logical sample.
  iree_host_size_t end_benchmark_sample;
  // True when |case_sample_ordinal| identifies one concrete case sample.
  bool has_case_sample_ordinal;
  // Case-local sample ordinal measured by the physical work item.
  iree_host_size_t case_sample_ordinal;
  // Sample-compilation label used for this logical sample.
  iree_string_view_t sample_compilation;
  // Deduplicated physical work item index satisfying this logical sample.
  iree_host_size_t work_item_index;
} iree_benchmark_loom_logical_sample_t;

typedef struct iree_benchmark_loom_work_item_t {
  // Work kind and payload discriminator.
  iree_benchmark_loom_work_item_kind_t kind;
  // Stable work item ordinal within the plan.
  iree_host_size_t work_item_index;
  // Representative selected benchmark ordinal used for execution.
  iree_host_size_t representative_selection_index;
  // Dispatch compile item index, or INDEX_INVALID for non-dispatch work.
  iree_host_size_t dispatch_compile_item_index;
  // Sample-compilation label used for this physical work item.
  iree_string_view_t sample_compilation;
  // First benchmark-local sample ordinal covered by this work item.
  iree_host_size_t begin_benchmark_sample;
  // One-past-end benchmark-local sample ordinal covered by this work item.
  iree_host_size_t end_benchmark_sample;
  // True when |case_sample_ordinal| identifies one concrete case sample.
  bool has_case_sample_ordinal;
  // Case-local sample ordinal measured by this work item.
  iree_host_size_t case_sample_ordinal;
} iree_benchmark_loom_work_item_t;

typedef struct iree_benchmark_loom_work_plan_t {
  // Host allocator used for arrays owned by this plan.
  iree_allocator_t host_allocator;
  // Selected benchmark records in execution/report order.
  iree_benchmark_loom_selected_benchmark_t* selected_benchmarks;
  // Number of populated selected benchmark records.
  iree_host_size_t selected_benchmark_count;
  // Logical report samples in selected benchmark order.
  iree_benchmark_loom_logical_sample_t* logical_samples;
  // Number of populated logical report samples.
  iree_host_size_t logical_sample_count;
  // Deduplicated HAL dispatch compile items needed by dispatch work items.
  iree_benchmark_loom_dispatch_compile_item_t* dispatch_compile_items;
  // Number of populated dispatch compile items.
  iree_host_size_t dispatch_compile_item_count;
  // Deduplicated physical work items needed by |logical_samples|.
  iree_benchmark_loom_work_item_t* work_items;
  // Number of populated physical work items.
  iree_host_size_t work_item_count;
} iree_benchmark_loom_work_plan_t;

// Builds a selected benchmark and deduplicated physical work plan.
iree_status_t iree_benchmark_loom_work_plan_initialize(
    const loom_testbench_module_plan_t* module_plan,
    const iree_benchmark_loom_options_t* options, iree_allocator_t allocator,
    iree_benchmark_loom_work_plan_t* out_plan);

// Releases arrays owned by |plan|.
void iree_benchmark_loom_work_plan_deinitialize(
    iree_benchmark_loom_work_plan_t* plan);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_WORK_PLAN_H_
