// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SPIR-V module compilation from prepared target-low IR.

#ifndef LOOM_TARGET_EMIT_SPIRV_MODULE_COMPILER_H_
#define LOOM_TARGET_EMIT_SPIRV_MODULE_COMPILER_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/codegen/low/descriptors.h"
#include "loom/error/emitter.h"
#include "loom/ir/function_version.h"
#include "loom/ir/ir.h"
#include "loom/target/emit/spirv/module_builder.h"
#include "loom/target/facts.h"
#include "loom/target/provider.h"

#ifdef __cplusplus
extern "C" {
#endif

// One compiler-selected SPIR-V entry and its retained target facts.
typedef struct loom_spirv_compile_entry_t {
  // Verified target-low function definition.
  loom_op_t* function_op;
  // Immutable target facts already selected for |function_op|.
  const loom_target_facts_t* target_facts;
} loom_spirv_compile_entry_t;

typedef struct loom_spirv_compile_options_t {
  // Optional compiler-owned function versions participating in planning.
  const loom_function_version_list_t* function_versions;
  // Optional compiler-owned selected entry table in emission order. Every
  // entry carries target facts. NULL selects retained function versions or
  // public/retained definitions, falling back to all definitions when neither
  // root contract is present. Reachable helpers remain internal functions.
  const loom_spirv_compile_entry_t* entries;
  // Number of entries in |entries|. Zero selects from the module.
  iree_host_size_t entry_count;
} loom_spirv_compile_options_t;

// Plans and emits one SPIR-V module through the production boundary.
// Structured semantic rejection returns OK with |out_emitted| false and no
// binary.
iree_status_t loom_spirv_compile_module_binary(
    loom_module_t* module,
    const loom_low_descriptor_registry_t* descriptor_registry,
    iree_diagnostic_emitter_t diagnostic_emitter, iree_arena_allocator_t* arena,
    const loom_spirv_compile_options_t* options, iree_allocator_t allocator,
    bool* out_emitted, loom_spirv_module_binary_t* out_module);

// Plans and emits one SPIR-V artifact through the production boundary.
// |options| may restrict emission to compiler-selected entries. Structured
// semantic rejection returns OK with |out_emitted| false and no artifact. The
// caller releases a successfully emitted artifact with
// loom_target_emit_artifact_release.
iree_status_t loom_spirv_compile_module_artifact(
    const loom_target_emit_request_t* request,
    const loom_spirv_compile_options_t* options, bool* out_emitted,
    loom_target_emit_artifact_t* out_artifact);

// SPIR-V binary module emitter composed by target-owned provider joins.
extern const loom_target_emitter_t loom_spirv_module_emitter;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_SPIRV_MODULE_COMPILER_H_
