// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_BINDING_C_SRC_EMIT_H_
#define LOOM_BINDING_C_SRC_EMIT_H_

#include "iree/base/internal/arena.h"
#include "loom/target/provider.h"
#include "loomc/emit.h"
#include "visibility.h"

#ifdef __cplusplus
extern "C" {
#endif

// Transient state retained across one target emission transaction.
//
// Artifact bytes are also retained by the result after successful emission.
// The target artifact and scratch arena remain live here so a product builder
// can consume compiler-only export projections before deinitialization.
typedef struct loomc_emit_operation_t {
  // Target-owned artifact and transient export projections.
  loom_target_emit_artifact_t target_artifact;

  // Scratch storage borrowed by |target_artifact| projection tables.
  iree_arena_allocator_t scratch_arena;

  // Product artifact ordinal of the primary target artifact, or host-size max.
  loomc_host_size_t primary_artifact_ordinal;

  // True when |scratch_arena| must be deinitialized.
  bool scratch_arena_initialized;
} loomc_emit_operation_t;

// Releases one completed or partially completed emission operation.
LOOMC_API_PRIVATE void loomc_emit_operation_deinitialize(
    loomc_emit_operation_t* operation);

// Emits |module| into an existing succeeded result.
//
// The caller must keep |out_operation| live while consuming target export
// projections and deinitialize it before releasing |workspace| or |module|.
LOOMC_API_PRIVATE loomc_status_t loomc_emit_module_into_result(
    loomc_target_environment_t* target_environment,
    loomc_workspace_t* workspace, loomc_module_t* module,
    const loomc_emit_options_t* options, loomc_result_t* result,
    loomc_emit_operation_t* out_operation);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_BINDING_C_SRC_EMIT_H_
