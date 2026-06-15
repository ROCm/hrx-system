// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_COMPILE_H_
#define LOOMC_COMPILE_H_

#include "loomc/config.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Prepared compilers and module compilation.
///
/// A prepared compiler owns validated compile-time configuration and provider
/// state. Compile invocations compose that compiler with a caller-selected
/// prepared pass program, caller-owned workspace scratch, and a mutable module.
/// Results return in-memory diagnostics, reports, and artifacts.
///
/// The core compile primitive accepts an already-formed `loomc_module_t`.
/// Sources are deserialized into modules and indexes are linked into modules
/// before compilation so each stage has independent options, diagnostics, and
/// lifetime boundaries.
///
/// @par Example
/// Share prepared compiler state and keep scratch local to each worker:
///
/// @code{.c}
/// void worker_main(loomc_compiler_t* compiler,
///                  loomc_pass_program_t* pass_program) {
///   loomc_compiler_retain(compiler);
///   loomc_pass_program_retain(pass_program);
///
///   loomc_workspace_t* workspace = NULL;
///   loomc_status_t status =
///       loomc_workspace_create(NULL, loomc_allocator_system(), &workspace);
///   if (!loomc_status_is_ok(status)) {
///     loomc_pass_program_release(pass_program);
///     loomc_compiler_release(compiler);
///     return;
///   }
///
///   loomc_config_binding_t bindings[] = {
///       {
///           .key = loomc_make_cstring_view("tile_m"),
///           .value = loomc_make_cstring_view("128"),
///       },
///   };
///   loomc_compile_options_t compile_options = {
///       .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
///       .structure_size = sizeof(loomc_compile_options_t),
///       .artifact_flags = LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE |
///                         LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON,
///       .config =
///           {
///               .bindings = bindings,
///               .binding_count = 1,
///           },
///   };
///
///   // `module` is produced by deserialization, linking, or another module
///   // operation. The compile invocation borrows it and may rewrite its IR, so
///   // this worker owns exclusive access for the duration of the call. Reuse
///   // compiler, pass program, and workspace for many independent module
///   // invocations on this worker.
///   loomc_result_t* result = NULL;
///   status = loomc_compile_module(compiler, workspace, pass_program, module,
///                                 &compile_options,
///                                 loomc_allocator_system(), &result);
///   if (loomc_status_is_ok(status)) {
///     // Inspect diagnostics and artifacts through result.
///     loomc_result_release(result);
///   }
///
///   loomc_workspace_release(workspace);
///   loomc_pass_program_release(pass_program);
///   loomc_compiler_release(compiler);
/// }
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Prepared immutable compiler.
///
/// A compiler is intended to be created once and reused across many
/// invocations. It must not require repeated filesystem access on the hot path.
/// Pass pipelines are prepared as `loomc_pass_program_t` handles and selected
/// per invocation.
///
/// @thread_safety
/// Prepared compilers are immutable after creation and may be shared across
/// threads. Invocation-local scratch belongs in `loomc_workspace_t`.
typedef struct loomc_compiler_t loomc_compiler_t;

/// Compile artifact request bits.
typedef enum loomc_compile_artifact_flag_bits_e {
  /// Return textual Loom module IR after successful compilation.
  LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_TEXT = 1u << 0,

  /// Return binary Loom bytecode after successful compilation.
  LOOMC_COMPILE_ARTIFACT_FLAG_MODULE_BYTECODE = 1u << 1,

  /// Return a JSON compile report artifact. Reports are emitted for completed
  /// invocations, including failed results that still returned diagnostics.
  LOOMC_COMPILE_ARTIFACT_FLAG_REPORT_JSON = 1u << 2,
} loomc_compile_artifact_flag_bits_t;

/// Bitmask of `loomc_compile_artifact_flag_bits_t`.
typedef uint32_t loomc_compile_artifact_flags_t;

/// Compiler creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_compiler_options_t)`, and fill the requested fields.
typedef struct loomc_compiler_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_COMPILER_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future compiler options.
  const void* next;
} loomc_compiler_options_t;

/// Compile invocation options.
///
/// A compile invocation borrows a mutable module and returns a result with
/// diagnostics and artifacts. Per-kernel configuration values materialize into
/// the module before the selected pass program runs. These values live here
/// rather than on the prepared compiler so autotuning and JIT sweeps can vary
/// config without constructing many compiler handles. Target selections are
/// supplied through `loomc_target_selection_options_t` on `next` so the same
/// compiler can be reused across source-selected, partial-target, and concrete
/// target invocations.
typedef struct loomc_compile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for compile invocation options such as
  /// `loomc_target_selection_options_t`.
  const void* next;

  /// Runtime artifact module name for this invocation. Empty uses the
  /// compiler's default.
  loomc_string_view_t module_name;

  /// Requested result artifacts. Zero avoids artifact serialization and keeps
  /// the hot path focused on diagnostics and in-place module transformation.
  loomc_compile_artifact_flags_t artifact_flags;

  /// Per-invocation config dialect bindings and optional JSON object.
  loomc_config_options_t config;
} loomc_compile_options_t;

/// Creates a prepared immutable compiler.
///
/// @param context Context shared with modules compiled by this compiler.
/// @param options Compiler configuration, or `NULL` for defaults.
/// @param allocator Host allocator used for compiler-owned storage.
/// @param out_compiler Receives one retained compiler on success.
/// @return OK when the compiler was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_compiler_release`.
///
/// @thread_safety
/// The returned compiler is immutable and may be shared across worker threads.
LOOMC_API_EXPORT loomc_status_t loomc_compiler_create(
    loomc_context_t* context, const loomc_compiler_options_t* options,
    loomc_allocator_t allocator, loomc_compiler_t** out_compiler);

/// Compiles a mutable module into in-memory artifacts.
///
/// @param compiler Prepared compiler.
/// @param workspace Invocation-local scratch workspace.
/// @param pass_program Prepared pass program selected for this invocation.
/// @param module Module produced by deserialization, linking, or another module
/// operation. The invocation borrows the module for the duration of the call
/// and may rewrite its IR in place.
/// @param options Compile invocation options, or `NULL` for defaults.
/// @param allocator Host allocator used for result-owned storage.
/// @param out_result Receives a retained result for the operation.
/// @return OK when the invocation ran to a result. Non-OK statuses represent
/// API misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`.
///
/// @par Postcondition
/// The caller retains ownership of `module`, and the module handle remains
/// valid after the call. Its IR contents may have been transformed by the
/// selected pass program. Callers needing independent later invocations of
/// the original IR provide independent module storage before compilation.
///
/// @lifetime
/// Returned results and artifacts do not borrow from `workspace` and remain
/// valid after `loomc_workspace_trim`.
///
/// @par Artifact Requests
/// Artifact emission is opt-in through `loomc_compile_options_t`. Requesting
/// module text or bytecode serializes the transformed module into result-owned
/// bytes. Requesting a report returns a JSON artifact with invocation metadata,
/// result state, diagnostic count, and the count of other artifacts emitted by
/// the invocation.
///
/// @thread_safety
/// Calls using the same compiler and pass program may run concurrently when
/// each call uses a distinct workspace and distinct module, or when access to
/// shared workspaces and modules is synchronized externally. The compiler and
/// pass program are immutable after creation.
///
/// @par Target Selection
/// `loomc_target_selection_options_t` may be attached to
/// `loomc_compile_options_t::next`. The selected profile must be compatible
/// with the compiler context's target environment. Omitting the extension or
/// passing an explicit empty selection runs the invocation without a concrete
/// target overlay.
LOOMC_API_EXPORT loomc_status_t loomc_compile_module(
    loomc_compiler_t* compiler, loomc_workspace_t* workspace,
    const loomc_pass_program_t* pass_program, loomc_module_t* module,
    const loomc_compile_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result);

/// Retains a prepared compiler for another owner.
///
/// @param compiler Compiler to retain.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads.
LOOMC_API_EXPORT void loomc_compiler_retain(loomc_compiler_t* compiler);

/// Releases a prepared compiler from one owner.
///
/// @param compiler Compiler to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads. The
/// compiler is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_compiler_release(loomc_compiler_t* compiler);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_COMPILE_H_
