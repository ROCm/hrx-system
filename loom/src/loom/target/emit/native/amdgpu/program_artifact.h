// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AMDGPU artifact-wide retained-call planning and helper compilation.

#ifndef LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_ARTIFACT_H_
#define LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_ARTIFACT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/analysis/symbol_references.h"
#include "loom/codegen/low/target_binding.h"
#include "loom/target/emit/native/amdgpu/program_call.h"
#include "loom/target/function_version.h"
#include "loom/target/reporting/report.h"

#ifdef __cplusplus
extern "C" {
#endif

// One loader-visible root supplied to artifact planning.
typedef struct loom_amdgpu_program_artifact_root_input_t {
  // Prepared low.kernel.def operation.
  loom_op_t* function_op;
  // Concrete target selected for |function_op|.
  const loom_low_resolved_target_t* target;
} loom_amdgpu_program_artifact_root_input_t;

// Indexed direct-leaf closure retained for one loader-visible root.
typedef struct loom_amdgpu_program_artifact_root_t {
  // Prepared low.kernel.def operation.
  loom_op_t* function_op;
  // Concrete target selected for |function_op|.
  const loom_low_resolved_target_t* target;
  // Unique directly called helper symbols in ascending symbol order.
  const loom_symbol_id_t* helper_symbols;
  // Number of entries in |helper_symbols|.
  iree_host_size_t helper_count;
  // Artifact-owned call contracts for |target|'s descriptor set. Null when
  // |helper_count| is zero.
  loom_amdgpu_program_call_table_t* call_table;
} loom_amdgpu_program_artifact_root_t;

// Immutable inputs shared by every root in one native artifact.
typedef struct loom_amdgpu_program_artifact_options_t {
  // Canonical module symbol-use snapshot.
  const loom_symbol_reference_table_t* references;
  // Concrete target versions indexed by the same module snapshot.
  const loom_target_function_version_snapshot_t* function_versions;
  // Registry resolving helper representation contracts.
  const loom_low_descriptor_registry_t* descriptor_registry;
  // Optional report fixing helper evidence retention and detail policy.
  const loom_target_compile_report_t* report_template;
  // Diagnostic destination shared by artifact planning and compilation.
  iree_diagnostic_emitter_t emitter;
} loom_amdgpu_program_artifact_options_t;

// Compiled direct-leaf closure shared by every root in one native artifact.
typedef struct loom_amdgpu_program_artifact_t {
  // Module owning every root and helper symbol.
  const loom_module_t* module;
  // Root closures in caller-provided order.
  loom_amdgpu_program_artifact_root_t* roots;
  // Number of entries in |roots|.
  iree_host_size_t root_count;
  // Unique helper compilations in ascending symbol order. Each helper ordinal
  // is its direct index in this array and in final HSACO function order.
  loom_amdgpu_program_callable_t* helpers;
  // Number of entries in |helpers|.
  iree_host_size_t helper_count;
  // Descriptor-partitioned call tables borrowed by |roots|.
  loom_amdgpu_program_call_table_t* call_tables;
  // Number of entries in |call_tables|.
  iree_host_size_t call_table_count;
} loom_amdgpu_program_artifact_t;

// Builds and compiles one artifact-wide direct-leaf closure.
//
// The canonical symbol-reference table is traversed once per root to retain
// exact helper symbol lists. A helper is resolved and compiled once even when
// several roots call it. Roots with different descriptor sets receive separate
// call tables while sharing the artifact's unique helper array.
//
// User-dependent rejection emits diagnostics and leaves |out_built| false;
// infrastructure failures return a status.
iree_status_t loom_amdgpu_program_artifact_build(
    loom_module_t* module,
    const loom_amdgpu_program_artifact_root_input_t* root_inputs,
    iree_host_size_t root_count,
    const loom_amdgpu_program_artifact_options_t* options,
    iree_arena_allocator_t* arena, bool* out_built,
    loom_amdgpu_program_artifact_t* out_artifact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_NATIVE_AMDGPU_PROGRAM_ARTIFACT_H_
