// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Compiler-owned kernel launch-configuration program construction.

#ifndef LOOM_CODEGEN_LOW_LAUNCH_CONFIG_PROGRAM_H_
#define LOOM_CODEGEN_LOW_LAUNCH_CONFIG_PROGRAM_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/function_version.h"
#include "loom/ir/module.h"
#include "loom/pass/environment.h"
#include "loom/pass/types.h"
#include "loom/pass/value_facts.h"
#include "loom/target/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_kernel_launch_config_program_entry_t
    loom_kernel_launch_config_program_entry_t;

// Compiler-owned launch-config product accumulated while source kernels lower.
//
// The product owns a separate host module. Source-to-low captures each
// selected kernel's residual launch computation while exact specialization and
// target facts are available. Authored workload predicates become observable
// entry checks. Finalization joins the completed low kernel's workgroup-storage
// requirement and terminates each host function.
typedef struct loom_kernel_launch_config_program_t {
  // Pass capability embedded for source-to-low observation.
  loom_pass_environment_capability_t capability;

  // Host launch-config module under construction.
  loom_module_t* module;

  // Captured kernel entries.
  struct {
    // First entry in capture order.
    loom_kernel_launch_config_program_entry_t* head;

    // Last entry in capture order.
    loom_kernel_launch_config_program_entry_t* tail;

    // Number of captured entries.
    iree_host_size_t count;
  } entries;
} loom_kernel_launch_config_program_t;

// Initializes an empty product using |block_pool| for host-module storage.
iree_status_t loom_kernel_launch_config_program_initialize(
    loom_context_t* context, iree_arena_block_pool_t* block_pool,
    iree_allocator_t allocator,
    loom_kernel_launch_config_program_t* out_program);

// Releases the host module owned by |program|.
void loom_kernel_launch_config_program_deinitialize(
    loom_kernel_launch_config_program_t* program);

// Returns the capability to append to a pass invocation environment.
static inline const loom_pass_environment_capability_t*
loom_kernel_launch_config_program_capability(
    const loom_kernel_launch_config_program_t* program) {
  return &program->capability;
}

// Returns the launch-config product in |pass|, or NULL when not requested.
loom_kernel_launch_config_program_t*
loom_kernel_launch_config_program_from_pass(const loom_pass_t* pass);

// Captures one selected source kernel before source-to-low replaces it.
//
// Non-kernel source functions are ignored. |source_facts| must be the exact
// target-aware fact table acquired by source-to-low for |source_function|.
iree_status_t loom_kernel_launch_config_program_capture(
    loom_kernel_launch_config_program_t* program,
    const loom_module_t* source_module, loom_func_like_t source_function,
    iree_string_view_t source_function_name,
    loom_function_version_t* version_handle,
    const loom_target_facts_t* target_facts,
    const loom_value_fact_table_t* source_facts);

// Completes all captured functions from their final low kernel products.
//
// |lowered_module| is the module after the complete pass program. The returned
// host module is owned by |program| and remains valid until deinitialization.
iree_status_t loom_kernel_launch_config_program_finalize(
    loom_kernel_launch_config_program_t* program,
    const loom_module_t* lowered_module,
    iree_arena_block_pool_t* scratch_block_pool, loom_module_t** out_module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LAUNCH_CONFIG_PROGRAM_H_
