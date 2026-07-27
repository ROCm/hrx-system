// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Execution provider composition for Loom run, benchmark, and tuning tools.

#ifndef LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_
#define LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_

#include "iree/base/api.h"
#include "loom/target/provider.h"
#include "loom/tooling/execution/execution_backend.h"
#include "loom/tooling/execution/session.h"

#ifdef __cplusplus
extern "C" {
#endif

// Target-owned execution contribution linked into run/benchmark/tune tools.
typedef struct loom_run_execution_provider_t {
  // Stable provider name used in diagnostics and help text.
  iree_string_view_t name;
  // Optional target provider contribution reused by compile and check tools.
  const loom_target_provider_t* target_provider;
  // Optional execution backend table contributed by this provider.
  const loom_run_execution_backend_t* const* execution_backends;
  // Number of entries in |execution_backends|.
  iree_host_size_t execution_backend_count;
} loom_run_execution_provider_t;

// Static provider table linked into a run/benchmark/tune binary or embedding.
typedef struct loom_run_execution_provider_set_t {
  // Provider contribution table.
  const loom_run_execution_provider_t* const* providers;
  // Number of entries in |providers|.
  iree_host_size_t provider_count;
} loom_run_execution_provider_set_t;

enum {
  LOOM_RUN_EXECUTION_PROVIDER_TARGET_PROVIDER_CAPACITY = 64,
  LOOM_RUN_EXECUTION_PROVIDER_EXECUTION_BACKEND_CAPACITY = 64,
};

// Composed execution environment derived from a provider set.
typedef struct loom_run_execution_environment_t {
  // Provider table selected by the linked binary or embedding.
  const loom_run_execution_provider_set_t* provider_set;
  // Core target provider table assembled once for the environment.
  const loom_target_provider_t*
      target_providers[LOOM_RUN_EXECUTION_PROVIDER_TARGET_PROVIDER_CAPACITY];
  // Number of entries in |target_providers|.
  iree_host_size_t target_provider_count;
  // Provider-set view over |target_providers|.
  loom_target_provider_set_t target_provider_set;
  // Core target environment composed from |target_providers|.
  loom_target_environment_t target_environment;
  // Execution backend table assembled once for the environment.
  const loom_run_execution_backend_t* execution_backends
      [LOOM_RUN_EXECUTION_PROVIDER_EXECUTION_BACKEND_CAPACITY];
  // Number of entries in |execution_backends|.
  iree_host_size_t execution_backend_count;
  // Execution backend registry view over |execution_backends|.
  loom_run_execution_backend_registry_t execution_backend_registry;
} loom_run_execution_environment_t;

// Initializes |out_environment| from |provider_set|.
iree_status_t loom_run_execution_environment_initialize(
    const loom_run_execution_provider_set_t* provider_set,
    loom_run_execution_environment_t* out_environment);

// Resets |environment| to an empty state. No provider-owned storage is freed.
void loom_run_execution_environment_deinitialize(
    loom_run_execution_environment_t* environment);

// Returns a session context-registration callback backed by |environment|.
loom_run_register_context_callback_t
loom_run_execution_environment_register_context_callback(
    loom_run_execution_environment_t* environment);

// Returns a session descriptor-registry callback backed by |environment|. The
// returned registry view borrows immutable tables owned by |environment| and
// remains valid until |environment| deinitialization.
loom_run_initialize_low_descriptor_registry_callback_t
loom_run_execution_environment_low_descriptor_registry_callback(
    loom_run_execution_environment_t* environment);

// Returns the target environment composed from |environment|'s providers.
const loom_target_environment_t*
loom_run_execution_environment_target_environment(
    const loom_run_execution_environment_t* environment);

// Returns the execution backend registry composed from |environment|'s
// providers.
const loom_run_execution_backend_registry_t*
loom_run_execution_environment_execution_backend_registry(
    const loom_run_execution_environment_t* environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_EXECUTION_EXECUTION_PROVIDER_H_
