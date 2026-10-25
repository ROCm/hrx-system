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

// Source-selected inputs and preparation policy for one parameter bank, or a
// fixed binding naming an arena without selecting one bank as its contents.
typedef struct loom_serve_preparation_parameter_t {
  // Fixed-buffer ordinal in the compiled command's parameter reflection.
  uint32_t binding;
  // Number of independently indexed input checkpoints, in source order.
  uint32_t source_count;
  // Finalized typed shard views in policy-local input order, borrowing the
  // capture's shared source catalog. Checkpoint owners copy these when
  // retaining identity.
  const loom_model_file_list_t* sources;
  // Owned preparation-policy path, relative to the bootstrap source directory.
  iree_string_view_t policy;
  // Shared source ordinals corresponding to the policy-local input order.
  const uint32_t* source_ordinals;
  // One-based cold arena ordinal, or zero for an ordinary whole-root bank.
  // An arena binding has no sources or policy; selected banks name that same
  // arena and carry their own complete sources and policy.
  uint32_t arena;
} loom_serve_preparation_parameter_t;

// Compares checkpoint domains by their ordered source file lists and
// preparation policy. Binding ordinals are local to each command and do not
// affect domain identity.
bool loom_serve_preparation_parameter_equal(
    const loom_serve_preparation_parameter_t* lhs,
    const loom_serve_preparation_parameter_t* rhs);

// Compiler product selected by the model's source program.
typedef enum loom_serve_preparation_stage_kind_e {
  LOOM_SERVE_PREPARATION_STAGE_COMMAND = 0,
  LOOM_SERVE_PREPARATION_STAGE_DISPATCH = 1,
} loom_serve_preparation_stage_kind_t;

// Immutable stage declaration produced by a model's source program.
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

// Declarations with no device or VM ownership after construction.
typedef struct loom_serve_preparation_t loom_serve_preparation_t;

// Borrowed selected model facts retained independently by these declarations.
// NULL for direct program callers without a card.
loom_model_card_t* loom_serve_preparation_card(
    const loom_serve_preparation_t* preparation);

// Borrowed component role retained through the card, or NULL for the primary.
const loom_model_component_t* loom_serve_preparation_component(
    const loom_serve_preparation_t* preparation);

// Creates the source-facing "prepare" declaration capability. The caller
// serializes begin/take with VM invocation and retains the module while linked.
// No declaration storage is allocated until the first source declaration.
// An optional retained card supplies named parameter scopes and fixed
// JIT configuration; captured declarations retain it and the selected component
// independently of the module.
iree_status_t loom_serve_preparation_module_create(
    iree_vm_environment_t* environment, loom_model_card_t* card,
    const loom_model_component_t* component, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

enum loom_serve_preparation_flag_bits_e {
  // Allows checkpoint declarations when the caller owns parameter resolution.
  // Cold construction prepares the declared roots; request choices reuse or
  // prepare compatible domains before recording.
  LOOM_SERVE_PREPARATION_FLAG_PARAMETERS = 1u << 0,
  // Allows the service to capture a host memory-forecast entry point. The
  // service owns that entry's signature and invokes it before scheduling work.
  LOOM_SERVE_PREPARATION_FLAG_MEMORY_FORECAST = 1u << 1,
  // Allows cold arena capacities and the host parameter-binding entry. Request
  // captures may select banks only from this immutable cold arena catalog.
  LOOM_SERVE_PREPARATION_FLAG_PARAMETER_ARENAS = 1u << 2,
};
typedef uint32_t loom_serve_preparation_flags_t;

// Begins one declaration capture. The previous capture has been taken. Stage
// ordinals start at first_stage, preserving a caller's existing cold table.
// first_stage is at most INT32_MAX. base supplies immutable cold sources;
// request sources follow that prefix and cannot modify it. The caller keeps
// base alive through this capture and every consumer of its result. NULL means
// no prefix. Calls to prepare outside a capture fail.
void loom_serve_preparation_module_begin(iree_vm_module_t* module,
                                         uint32_t first_stage,
                                         const loom_serve_preparation_t* base,
                                         loom_serve_preparation_flags_t flags);

// Ends the capture and transfers its declarations, including partial results
// after source failure. Returns NULL if no declaration was made. The caller
// frees or retains the result independently of the module and VM lifetime.
loom_serve_preparation_t* loom_serve_preparation_module_take(
    iree_vm_module_t* module);

// Invokes |entry| from |source_path| in |sources| once with borrowed
// |arguments|. The source imports runner-private "prepare" capabilities:
//   stage(buffer root, i64 tag) -> i32 stage
//   dispatch(buffer root, i64 tag) -> i32 stage
//   config(i32 stage, buffer key, buffer value)
//   config_i64(i32 stage, buffer key, i64 value)
//   workspace_limit(i32 stage, i64 bytes)
//   source(buffer directory, buffer path) -> i32 source
//   source_append(i32 source, buffer directory, buffer path)
//   source_scope(buffer scope) -> i32 source
//   parameter(i32 stage, i32 binding, i32 input, i32 source, buffer policy)
//   parameter_arena(i64 capacity_bytes) -> i32 arena
//   parameter_bank(i32 stage, i32 binding, i32 arena, buffer sources,
//                  buffer policy) -> i32 bank
//   bind_parameters(buffer root)
//   memory_forecast(buffer root)
// All string buffers exclude NUL bytes. Config values use the public loomc
// textual value syntax; config_i64 formats signed decimal values. Parameter
// sources join directory and path in the server filesystem namespace. source
// declares the first shard of an independent index; source_append adds shards
// in order during the declaring capture. Card-backed programs use source_scope
// to select one named parameter scope's complete typed shard list. Explicit
// deployment inputs, such as request-selectable adapters, use source and
// source_append alongside card scopes. Neither a card scope nor an inherited
// cold source can be extended. Sources need no resident binding and may contain
// identical keys.
// parameter maps a dense policy-local input ordinal to a declared source. Each
// input is bound once; all inputs for a binding name one policy. Several stages
// and bindings may share a source. Policy paths resolve through the source
// environment, under programs.directory for cards. A card supplies stage config
// defaults; explicit declarations override individual keys. Declarations copy
// their strings and retain no references to arguments, source mappings,
// or the VM environment.
// parameter_arena declares a finite cold VA-byte budget, not physical storage
// or an adapter-count specialization. parameter_bank is request-only: sources
// is a nonempty little-endian i32 array in policy-local input order. It binds
// the full arena at the command root and returns a capture-local bank ordinal.
// Repeated selections may resolve to one prepared bank while retaining distinct
// ordinals. bind_parameters selects a host entry in the control program; its
// caller owns the signature and publishes offsets only after complete capture.
// Additional caller libraries are linked beside the prepare module. Results
// are ordinary VM variants owned by the caller; the environment must outlive
// any returned references. The source program is destroyed before return.
// dispatch declares a buffer-only kernel with a source-specialized launch
// configuration. stage declares a command program with parameter reflection.
// No device compilation, allocation or parameter IO occurs here.
// flags selects declaration capabilities; unsupported declarations fail at
// their source call. memory_forecast selects one host entry in the control
// program, not a command/kernel root or a compiler-enforced workspace limit.
// On failure all partial declarations are released and *out_preparation is
// NULL.
// report receives copied compiler diagnostics independently of this outcome.
// target_environment supplies VM support and the complete source catalog
// grammar and is borrowed only through this call.
iree_status_t loom_serve_preparation_allocate(
    loomc_target_environment_t* target_environment,
    iree_vm_environment_t* environment, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_string_view_t entry,
    loom_serve_preparation_flags_t flags, iree_vm_variant_span_t arguments,
    iree_vm_variant_span_t results, iree_vm_module_span_t libraries,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    loom_serve_preparation_t** out_preparation);

void loom_serve_preparation_free(loom_serve_preparation_t* preparation);

iree_host_size_t loom_serve_preparation_stage_count(
    const loom_serve_preparation_t* preparation);

// Number of shared sources, including the immutable cold prefix. NULL has no
// sources. Source ordinals are scoped to this capture, never process-global.
uint32_t loom_serve_preparation_source_count(
    const loom_serve_preparation_t* preparation);

// Borrows ordered typed shards for a previously declared source ordinal.
loom_model_file_list_t loom_serve_preparation_source(
    const loom_serve_preparation_t* preparation, uint32_t ordinal);

// Borrows this capture's new shard lists, excluding its immutable cold prefix.
// Returns their count; their ordinals follow the prefix's source_count. NULL
// has no sources. The finalized lists outlive cold indexing consumers.
uint32_t loom_serve_preparation_local_sources(
    const loom_serve_preparation_t* preparation,
    const loom_model_file_list_t** out_sources);

// Requested owned bytes, including spare array capacity and copied strings.
// NULL has no storage. This query does not traverse declarations.
iree_host_size_t loom_serve_preparation_storage_size(
    const loom_serve_preparation_t* preparation);

// Borrowed source-selected host entry, or empty when none was declared.
iree_string_view_t loom_serve_preparation_memory_forecast(
    const loom_serve_preparation_t* preparation);

// Cold arena catalog. Request captures borrow their base's immutable catalog.
// NULL has no arenas; index is below the returned count.
uint32_t loom_serve_preparation_arena_count(
    const loom_serve_preparation_t* preparation);
uint64_t loom_serve_preparation_arena_capacity(
    const loom_serve_preparation_t* preparation, uint32_t index);

// Invocation bank declarations in the source-visible returned ordinal order.
// These own no device storage or VM references. NULL has no bank declarations.
uint32_t loom_serve_preparation_bank_count(
    const loom_serve_preparation_t* preparation);
const loom_serve_preparation_parameter_t* loom_serve_preparation_bank(
    const loom_serve_preparation_t* preparation, uint32_t index);

// Cold-selected host entry, or empty if parameter arenas are not declared.
iree_string_view_t loom_serve_preparation_bind_parameters(
    const loom_serve_preparation_t* preparation);

// Exact ordered declaration identity, including configuration spelling,
// workspace policy, and checkpoint domains. Capture ordinal offsets are not
// part of identity. NULL compares equal only to NULL.
bool loom_serve_preparation_equal(const loom_serve_preparation_t* lhs,
                                  const loom_serve_preparation_t* rhs);

// Exact compiler inputs in declaration order: product kind, root, and config
// spelling. Checkpoint identity, selection tags, and workspace limits do not
// specialize code. A consumer reusing these products still checks the new
// declaration's workspace limit and binds its selected checkpoint storage.
bool loom_serve_preparation_code_equal(const loom_serve_preparation_t* lhs,
                                       const loom_serve_preparation_t* rhs);

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
