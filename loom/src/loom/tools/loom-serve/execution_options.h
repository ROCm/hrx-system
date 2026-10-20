// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLS_LOOM_SERVE_EXECUTION_OPTIONS_H_
#define LOOM_TOOLS_LOOM_SERVE_EXECUTION_OPTIONS_H_

#include "loom/serve/runtime/execution/scheduler.h"

#ifdef __cplusplus
extern "C" {
#endif

// Cold CLI policy storage, retained at a stable address until device-group
// construction copies policy. No flag strings or model registrations survive
// parsing. Devices retain the command line's zero-based ordinal; "host" denotes
// the shared application executor.
typedef struct loom_serve_execution_options_t {
  // Allocator owning the variable-length scope array.
  iree_allocator_t host_allocator;
  // Borrowed view over this object's limit and scope storage.
  loom_serve_execution_policy_t policy;
  // Accepted-unit ceilings in the scheduler's host-plus-devices namespace.
  uint32_t max_active[64];
  // Owned interference scopes, freed after group construction.
  loom_serve_execution_scope_t* scopes;
} loom_serve_execution_options_t;

// Resolves RESOURCE:COUNT limits and MODE:RESOURCE[,RESOURCE...] scopes, where
// MODE is exclude or overlap. Zero limits add no software ceiling. Omitted
// scopes select one exclusion scope over host and every device. Explicit scope
// coverage is checked by scheduler allocation, as for native policy callers.
// Failure releases partial storage and leaves a zero-initialized output.
iree_status_t loom_serve_execution_options_initialize(
    iree_host_size_t device_count, iree_string_view_list_t limits,
    iree_string_view_list_t scopes, iree_allocator_t host_allocator,
    loom_serve_execution_options_t* out_options);

void loom_serve_execution_options_deinitialize(
    loom_serve_execution_options_t* options);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLS_LOOM_SERVE_EXECUTION_OPTIONS_H_
