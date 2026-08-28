// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// VM function-call bridge for Loom check testbench actual execution.

#ifndef LOOM_TOOLING_TARGET_VM_TESTBENCH_ACTUAL_H_
#define LOOM_TOOLING_TARGET_VM_TESTBENCH_ACTUAL_H_

#include "iree/base/api.h"
#include "loom/target/provider.h"
#include "loom/tooling/config/config.h"
#include "loom/tooling/execution/session.h"
#include "loom/tooling/testbench/invocation.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct iree_vm_invocation_t iree_vm_invocation_t;
typedef struct iree_vm_program_t iree_vm_program_t;
typedef struct iree_vm_variant_t iree_vm_variant_t;

typedef struct loom_vm_testbench_actual_options_t {
  // Shared execution session used to clone and compile the private module.
  loom_run_session_t* session;
  // Composed target environment used by the source-to-low pipeline.
  const loom_target_environment_t* target_environment;
  // Canonical parsed module borrowed through actual deinitialization.
  const loom_run_module_t* run_module;
  // Complete module plan selecting the union of function-call roots.
  const loom_testbench_module_plan_t* module_plan;
  // User-selected compiler pipeline.
  iree_string_view_t pipeline;
  // Compile-time config bindings materialized into the private module.
  const loom_tooling_config_set_t* config_set;
  // Host allocator used for the compiled program and reusable scratch.
  iree_allocator_t host_allocator;
} loom_vm_testbench_actual_options_t;

// Module-lifetime VM actual shared by every prepared check case.
typedef struct loom_vm_testbench_actual_t {
  // Source module that owns invocation types, symbols, and export names.
  const loom_module_t* source_module;
  // Immutable compiled program shared by every callback process.
  iree_vm_program_t* program;
  // Reusable VM execution storage.
  iree_vm_invocation_t* invocation;
  // Allocation containing |arguments| followed by |results|.
  iree_vm_variant_t* variant_storage;
  // Reusable argument carriers within |variant_storage|.
  iree_vm_variant_t* arguments;
  // Reusable result carriers within |variant_storage|.
  iree_vm_variant_t* results;
  // Runtime module name used to resolve exported functions.
  iree_string_view_t module_name;
  // Maximum function argument count represented by |arguments|.
  iree_host_size_t argument_capacity;
  // Maximum function result count represented by |results|.
  iree_host_size_t result_capacity;
  // Host allocator owning runtime objects and |variant_storage|.
  iree_allocator_t host_allocator;
} loom_vm_testbench_actual_t;

// Compiles one private VM program and initializes reusable invocation scratch.
//
// The actual borrows source IR from |options| through deinitialization. Each
// callback execution receives a fresh process; a contiguous invocation
// sequence shares that mutable process state across all calls in the sequence.
iree_status_t loom_vm_testbench_actual_initialize(
    const loom_vm_testbench_actual_options_t* options,
    loom_vm_testbench_actual_t* out_actual);

// Releases all program and scratch storage owned by |actual|.
void loom_vm_testbench_actual_deinitialize(loom_vm_testbench_actual_t* actual);

// Returns the borrowed callback view shared by every prepared case.
loom_testbench_invocation_provider_t loom_vm_testbench_actual_provider(
    loom_vm_testbench_actual_t* actual);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_VM_TESTBENCH_ACTUAL_H_
