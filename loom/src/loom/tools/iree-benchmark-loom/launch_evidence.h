// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Concrete workload and resolved launch evidence for benchmark results.

#ifndef LOOM_TOOLS_IREE_BENCHMARK_LOOM_LAUNCH_EVIDENCE_H_
#define LOOM_TOOLS_IREE_BENCHMARK_LOOM_LAUNCH_EVIDENCE_H_

#include "iree/base/api.h"
#include "loom/analysis/kernel_launch_config.h"
#include "loom/ir/scalar_type.h"
#include "loom/util/json.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_hal_testbench_actual_provider_t
    loom_run_hal_testbench_actual_provider_t;

typedef struct iree_benchmark_loom_workload_value_t {
  // Source scalar type of this ordered kernel workload argument.
  loom_scalar_type_t type;
  // Exact signed value supplied to launch configuration evaluation.
  int64_t value;
} iree_benchmark_loom_workload_value_t;

typedef struct iree_benchmark_loom_launch_record_t {
  // Concrete check.case sample ordinal that produced this launch.
  iree_host_size_t case_sample_ordinal;
  // Source-order kernel launch ordinal within the case sample.
  iree_host_size_t sequence_step_ordinal;
  // Borrowed executable entry symbol submitted for this launch.
  iree_string_view_t entry;
  // Ordered workload values borrowed from the containing evidence storage.
  const iree_benchmark_loom_workload_value_t* workload_values;
  // Number of entries in |workload_values|.
  iree_host_size_t workload_value_count;
  // Fully evaluated source launch configuration for this workload.
  loom_kernel_launch_config_t launch_config;
} iree_benchmark_loom_launch_record_t;

typedef struct iree_benchmark_loom_launch_evidence_t {
  // Host allocator owning records and contiguous workload value storage.
  iree_allocator_t host_allocator;
  // Owned launch records in sample and source invocation order.
  iree_benchmark_loom_launch_record_t* records;
  // Number of entries in |records|.
  iree_host_size_t record_count;
  // Owned contiguous workload values referenced by |records|.
  iree_benchmark_loom_workload_value_t* workload_values;
  // Number of entries in |workload_values|.
  iree_host_size_t workload_value_count;
} iree_benchmark_loom_launch_evidence_t;

// Allocates zero-initialized launch records and workload value storage for one
// benchmark work item.
iree_status_t iree_benchmark_loom_launch_evidence_initialize(
    iree_host_size_t record_count, iree_host_size_t workload_value_count,
    iree_allocator_t host_allocator,
    iree_benchmark_loom_launch_evidence_t* out_evidence);

// Releases launch records and exact workload values owned by |evidence|.
void iree_benchmark_loom_launch_evidence_deinitialize(
    iree_benchmark_loom_launch_evidence_t* evidence);

// Copies one provider's exact workload and most recently resolved launch
// configuration into the preallocated record and workload ranges.
void iree_benchmark_loom_launch_evidence_capture(
    const loom_run_hal_testbench_actual_provider_t* provider,
    iree_host_size_t case_sample_ordinal,
    iree_host_size_t sequence_step_ordinal, iree_host_size_t record_ordinal,
    iree_host_size_t workload_value_ordinal,
    iree_benchmark_loom_launch_evidence_t* evidence);

// Writes launch records as a JSON array without a surrounding field name.
iree_status_t iree_benchmark_loom_write_launch_evidence_json(
    const iree_benchmark_loom_launch_evidence_t* evidence,
    loom_output_stream_t* stream);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_IREE_BENCHMARK_LOOM_LAUNCH_EVIDENCE_H_
