// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_PREPARATION_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_PREPARATION_H_

#include "iree/vm/environment.h"
#include "iree/vm/program.h"
#include "iree/vm/variant.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/config.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-selected checkpoint files and preparation policy for one fixed
// binding.
typedef struct loom_serve_preparation_parameter_t {
  // Fixed-buffer ordinal in the compiled command's parameter reflection.
  uint32_t binding;
  // Owned resolved paths in registration order, independent of the bootstrap
  // VM.
  iree_string_view_list_t paths;
  // Owned preparation-policy path, relative to the bootstrap source directory.
  iree_string_view_t policy;
} loom_serve_preparation_parameter_t;

// Compares checkpoint domains by their ordered file lists and preparation
// policy. Binding ordinals are local to each command and do not affect domain
// identity.
bool loom_serve_preparation_parameter_equal(
    const loom_serve_preparation_parameter_t* lhs,
    const loom_serve_preparation_parameter_t* rhs);

// Compiler product selected by the model's cold source program.
typedef enum loom_serve_preparation_stage_kind_e {
  LOOM_SERVE_PREPARATION_STAGE_COMMAND = 0,
  LOOM_SERVE_PREPARATION_STAGE_DISPATCH = 1,
} loom_serve_preparation_stage_kind_t;

// Immutable stage declaration produced by a model's cold source program.
typedef struct loom_serve_preparation_stage_t {
  // Exported command or kernel root in the model source catalog.
  iree_string_view_t root;
  // Product contract selected explicitly by stage or dispatch registration.
  loom_serve_preparation_stage_kind_t kind;
  // Model-defined selection tag. The preparation service does not interpret it.
  int64_t tag;
  // Maximum reflected transient workspace in bytes, or zero if unrestricted.
  uint64_t workspace_limit_bytes;
  // Owned compiler bindings; compilation requires all configuration resolved.
  loomc_config_options_t config;
  // Number of fixed parameter bindings declared by the model.
  iree_host_size_t parameter_count;
  // Owned checkpoint declarations in source registration order.
  const loom_serve_preparation_parameter_t* parameters;
} loom_serve_preparation_stage_t;

// Cold declarations with no device or VM ownership after construction.
typedef struct loom_serve_preparation_t loom_serve_preparation_t;

// Invokes |entry| from |source_path| in |sources| once with borrowed
// |arguments|. The source imports runner-private "prepare" capabilities:
//   stage(buffer root, i64 tag) -> i32 stage
//   dispatch(buffer root, i64 tag) -> i32 stage
//   config(i32 stage, buffer key, buffer value)
//   config_i64(i32 stage, buffer key, i64 value)
//   workspace_limit(i32 stage, i64 bytes)
//   parameter(i32 stage, i32 binding, buffer directory, buffer path,
//             buffer policy)
// All string buffers exclude NUL bytes. Config values use the public loomc
// textual value syntax; config_i64 formats signed decimal values. Parameter
// paths join directory and path in the server filesystem namespace. Repeated
// parameter calls for the same binding append files to one checkpoint domain
// and must name the same policy. File order is preserved in the parameter
// index. Policy paths resolve through the source environment. Declarations
// copy their strings and retain no references to arguments, source mappings,
// or the VM environment.
// Additional caller libraries are linked beside the prepare module. Results
// are ordinary VM variants owned by the caller; the environment must outlive
// any returned references. The source program is destroyed before return.
// dispatch declares a buffer-only kernel with a source-specialized launch
// configuration. stage declares a command program with parameter reflection.
// No device compilation, allocation or parameter IO occurs here.
// On failure all partial declarations are released and *out_preparation is
// NULL.
// report receives copied compiler diagnostics independently of this outcome.
// target_environment supplies VM support and the complete source catalog
// grammar and is borrowed only through this call.
iree_status_t loom_serve_preparation_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_string_view_t entry,
    iree_vm_variant_span_t arguments, iree_vm_variant_span_t results,
    iree_vm_module_span_t libraries, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator,
    loom_serve_preparation_t** out_preparation);

void loom_serve_preparation_free(loom_serve_preparation_t* preparation);

iree_host_size_t loom_serve_preparation_stage_count(
    const loom_serve_preparation_t* preparation);

// Borrows an immutable declaration; |index| is below stage_count.
const loom_serve_preparation_stage_t* loom_serve_preparation_stage(
    const loom_serve_preparation_t* preparation, iree_host_size_t index);

// Checks the compiler-reflected transient workspace against the source's
// optional stage limit. A reflected requirement above the limit fails with
// RESOURCE_EXHAUSTED before execution storage is allocated.
iree_status_t loom_serve_preparation_stage_check_workspace(
    const loom_serve_preparation_stage_t* stage, uint64_t required_byte_length);

// Checks one source declaration against compiler-reflected checkpoint storage
// shared with the canonical stage. Ordered checkpoint paths, preparation policy
// and byte placement must agree before any domain is loaded or shared. A NULL
// program denotes a direct stage without fixed parameter domains. The caller
// checks the canonical stage against itself before checking subsequent stages.
// Dynamic buffer bindings and modality-specific geometry remain with the
// caller.
iree_status_t loom_serve_preparation_stage_check_parameters(
    const loom_serve_preparation_stage_t* stage,
    const loom_cmd_program_t* program,
    const loom_serve_preparation_stage_t* canonical_stage,
    const loom_cmd_program_t* canonical_program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_PREPARATION_H_
