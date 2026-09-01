// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_SRC_COMPILE_H_
#define LOOM_BINDING_C_SRC_COMPILE_H_

#include "iree/base/internal/arena.h"
#include "loom/ir/function_version.h"
#include "loom/ir/ir.h"
#include "loom/target/provider.h"
#include "loomc/compile.h"
#include "loomc/target.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Explicit source roots participating in a product compilation.
typedef struct loomc_compile_root_set_t {
  // Module-local source symbol IDs in product root order.
  const loom_symbol_id_t* symbol_ids;

  // Number of entries in |symbol_ids|.
  iree_host_size_t count;

  // Root-set ordinals requiring launch-config materialization.
  const loomc_request_root_ordinal_t* launch_root_ordinals;

  // Number of entries in |launch_root_ordinals|.
  iree_host_size_t launch_root_count;
} loomc_compile_root_set_t;

// Transient state retained across one module compilation transaction.
typedef struct loomc_compile_operation_t {
  // Target launch compiler selected for this invocation, or NULL.
  const loom_target_launch_config_compiler_t* launch_config_compiler;

  // Target-owned launch artifact and transient export projections.
  loom_target_emit_artifact_t launch_config_artifact;

  // Scratch storage borrowed by root indexing and artifact projections.
  iree_arena_allocator_t scratch_arena;

  // Stable function-version ordinal for every explicit source root.
  const loom_function_version_ordinal_t* root_function_version_ordinals;

  // Number of entries in |root_function_version_ordinals|.
  iree_host_size_t root_count;

  // Number of function versions in the completed compilation.
  iree_host_size_t function_version_count;

  // Result artifact ordinal of the launch artifact, or host-size max.
  loomc_host_size_t launch_config_artifact_ordinal;

  // True when |scratch_arena| must be deinitialized.
  bool scratch_arena_initialized;
} loomc_compile_operation_t;

// Releases one completed or partially completed compilation operation.
LOOMC_API_PRIVATE void loomc_compile_operation_deinitialize(
    loomc_compile_operation_t* operation);

// Returns the context retained by |compiler|.
LOOMC_API_PRIVATE loomc_context_t* loomc_compiler_context(
    const loomc_compiler_t* compiler);

// Resolves and validates target specialization options for |compiler|.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_resolve_target_specialization(
    const loomc_compiler_t* compiler, const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t** out_target_specialization);

// Validates an optional config module against |compiler| and |program_module|.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_validate_config_module(
    const loomc_compiler_t* compiler, const loomc_module_t* program_module,
    const loomc_compile_options_t* options);

// Validates request source ordinals against a deserialized module.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_validate_request_roots(
    const loomc_module_t* module, const loomc_request_t* request);

// Compiles a validated mutable module into an existing succeeded result.
//
// A non-NULL |root_set| creates stable compiler identities before mutable
// passes run and optionally limits launch-config materialization to selected
// roots. The caller consumes projections while |out_operation| remains live.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_module_into_result(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_options_t* options,
    const loomc_target_specialization_options_t* target_specialization,
    const loomc_compile_root_set_t* root_set, loomc_result_t* result,
    loomc_compile_operation_t* out_operation);

// Adds the retained launch-config artifact to |result| once.
LOOMC_API_PRIVATE loomc_status_t loomc_compile_add_launch_config_artifact(
    loomc_result_t* result, const loomc_compile_options_t* options,
    loomc_compile_operation_t* operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_BINDING_C_SRC_COMPILE_H_
