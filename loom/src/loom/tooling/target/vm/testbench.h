// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TOOLING_TARGET_VM_TESTBENCH_H_
#define LOOM_TOOLING_TARGET_VM_TESTBENCH_H_

#include "iree/vm/buffer.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "loom/tooling/testbench/compiled_provider.h"
#include "loom/tooling/testbench/scenario/executor.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Executes semantic functions through the VM. The direct-call provider compiles
// all selected function roots on its first call and reuses one process. The
// check.scenario profile instead uses this instance as compiler configuration
// and eagerly creates one bytecode module and process per unique target or
// oracle subject in a concrete scenario configuration. Trial domains calling
// the same subject share that product. The authored functions need no target
// binding.
// Buffer arguments and results share storage with testbench HAL bindings.
// Arguments require coherent persistent host mappings; results retain the VM
// storage until the final binding or alias is released.
typedef struct loom_vm_testbench_t {
  // Borrowed public compiler inputs, live through the final invocation.
  loom_testbench_compilation_t compilation;
  // Observer receiving each complete public compiler result.
  loom_testbench_compile_result_callback_t result_callback;
  // Owned VM core profile applied to every selected semantic function.
  loomc_target_profile_t* target_profile;
  // Borrowed selected calls identifying the functions crossing the host ABI.
  loom_testbench_invocation_plan_list_t invocations;
  // Allocator for bytecode and runtime objects.
  iree_allocator_t host_allocator;
  // Whether compilation semantically rejected the selected source module.
  bool compile_rejected;
  // Stable compilation stage that rejected the source module.
  iree_string_view_t compile_failure_stage;
  // Stable diagnostic or fallback rejection identifier.
  iree_string_view_t compile_failure_kind;
  // Stable human-facing summary of the compilation rejection.
  iree_string_view_t compile_failure_message;
  // Owned storage backing dynamic failure kind and message views.
  char* compile_failure_storage;
  // Owned process, or NULL until the first function call is prepared.
  iree_vm_process_t* process;
  // Owned reusable execution storage, never shared by concurrent calls.
  iree_vm_invocation_t* invocation;
  // Owned reusable argument variants; also the base of the combined IO slab.
  iree_vm_variant_t* arguments;
  // Result variants within the IO slab, sized from the immutable case plan.
  iree_vm_variant_t* results;
  // Retained imported buffer arguments used to trace returned references.
  iree_vm_buffer_t** argument_buffers;
  // Borrowed Core descriptors whose provider lives with the linked VM runtime.
  iree_vm_ref_types_t ref_types;
} loom_vm_testbench_t;

// Initializes a lazy function provider and selects the VM core profile.
iree_status_t loom_vm_testbench_initialize(
    loomc_target_environment_t* target_environment,
    iree_allocator_t host_allocator, loom_vm_testbench_t* out_testbench);

// Releases runtime objects. Safe for a zero-initialized or failed provider.
void loom_vm_testbench_deinitialize(loom_vm_testbench_t* testbench);

// Binds the runner-selected calls and returns a borrowed function-call
// callback. |user_data| points to an initialized loom_vm_testbench_t. The
// compilation inputs and invocation list remain live through the final call;
// deinitialization does not access them. Each compile uses a private clone of
// the canonical public module.
loom_testbench_invocation_provider_t loom_vm_testbench_invocation_provider(
    void* user_data, const loom_testbench_compilation_t* compilation,
    loom_testbench_invocation_plan_list_t invocations,
    loom_testbench_compile_result_callback_t result_callback);

// Binds compilation inputs and returns an eager VM execution profile. Every
// unique prepared product owns a bytecode module and process. Product
// preparation finishes before any trial-local values are materialized; product
// execution only marshals batches through that prepared process.
loom_testbench_execution_profile_t loom_vm_testbench_execution_profile(
    void* user_data, const loom_testbench_compilation_t* compilation,
    const loom_source_table_resolver_t* sources,
    loom_diagnostic_sink_t diagnostic_sink,
    loom_testbench_compile_result_callback_t result_callback);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TOOLING_TARGET_VM_TESTBENCH_H_
