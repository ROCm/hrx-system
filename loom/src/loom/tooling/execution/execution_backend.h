// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution backend registry for Loom run, benchmark, and tuning tools.

#ifndef LOOM_TOOLING_EXECUTION_EXECUTION_BACKEND_H_
#define LOOM_TOOLING_EXECUTION_EXECUTION_BACKEND_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_run_execution_backend_t loom_run_execution_backend_t;
typedef struct loom_run_one_shot_probe_request_t
    loom_run_one_shot_probe_request_t;
typedef struct loom_run_one_shot_request_t loom_run_one_shot_request_t;

typedef uint32_t loom_run_execution_backend_flags_t;

enum {
  // One-shot requests for this backend consume VM invocation options.
  LOOM_RUN_EXECUTION_BACKEND_FLAG_VM_OPTIONS = 1u << 0,
  // One-shot requests for this backend consume HAL invocation options.
  LOOM_RUN_EXECUTION_BACKEND_FLAG_HAL_OPTIONS = 1u << 1,
};

typedef iree_status_t (*loom_run_execution_backend_probe_fn_t)(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_probe_request_t* request);

typedef iree_status_t (*loom_run_execution_backend_run_one_shot_fn_t)(
    const loom_run_execution_backend_t* backend,
    const loom_run_one_shot_request_t* request);

// Runtime backend linked into a run/benchmark/tune process.
struct loom_run_execution_backend_t {
  // User-facing backend name accepted by execution tools.
  iree_string_view_t name;
  // Backend option families consumed by one-shot requests.
  loom_run_execution_backend_flags_t flags;
  // Optional backend probe hook.
  loom_run_execution_backend_probe_fn_t probe;
  // Runs one compile-and-invoke request for CLI and lit adapters.
  loom_run_execution_backend_run_one_shot_fn_t run_one_shot;
};

// Registry of optional execution backends linked into a runner binary.
typedef struct loom_run_execution_backend_registry_t {
  // Linked backend table.
  const loom_run_execution_backend_t* const* backends;
  // Number of entries in |backends|.
  iree_host_size_t backend_count;
} loom_run_execution_backend_registry_t;

// Initializes |out_registry| from a caller-owned backend table.
void loom_run_execution_backend_registry_initialize_from_entries(
    const loom_run_execution_backend_t* const* backends,
    iree_host_size_t backend_count,
    loom_run_execution_backend_registry_t* out_registry);

// Looks up an execution backend by user-facing name.
const loom_run_execution_backend_t* loom_run_execution_backend_registry_lookup(
    const loom_run_execution_backend_registry_t* registry,
    iree_string_view_t name);

// Appends a comma-separated list of registered execution backend names.
iree_status_t loom_run_execution_backend_registry_format_names(
    const loom_run_execution_backend_registry_t* registry,
    iree_string_builder_t* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_EXECUTION_BACKEND_H_
