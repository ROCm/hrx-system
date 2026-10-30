// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PASS_H_
#define LOOMC_PASS_H_

#include "loomc/context.h"
#include "loomc/module.h"
#include "loomc/result.h"

/// @file
/// Prepared pass programs.
///
/// A pass program is the reusable, immutable representation of a Loom compile
/// pipeline. Constructor helpers prepare this handle from cold configuration
/// forms such as a textual flag-like pass list; compilation and direct
/// execution borrow the prepared object. This keeps the hot JIT path from
/// reparsing pipeline strings or rebuilding pass descriptor state for every
/// kernel.
///
/// @par Example
/// Prepare a textual pass list once and use it from many execution calls:
///
/// @code{.c}
/// loomc_pass_program_t* pass_program = NULL;
/// loomc_result_t* pass_result = NULL;
/// loomc_status_t status = loomc_pass_program_create_from_pipeline_text(
///     context, loomc_make_cstring_view("combine,cse,dce,location-dce"),
///     NULL, loomc_allocator_system(), &pass_program, &pass_result);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
/// if (!loomc_result_succeeded(pass_result)) {
///   // Inspect pass_result diagnostics. No pass program was produced.
///   loomc_result_release(pass_result);
///   return loomc_ok_status();
/// }
/// loomc_result_release(pass_result);
///
/// // Each worker provides its own workspace and mutable module. The result
/// // contains diagnostics; the transformed module remains caller-owned.
/// loomc_result_t* run_result = NULL;
/// status = loomc_pass_program_run(pass_program, workspace, module, NULL,
///                                  loomc_allocator_system(), &run_result);
/// loomc_result_release(run_result);
/// loomc_pass_program_release(pass_program);
/// @endcode
///
/// @par Example
/// Prepare a named pipeline from a loaded module:
///
/// @code{.c}
/// loomc_pass_program_t* pass_program = NULL;
/// loomc_result_t* pass_result = NULL;
/// loomc_status_t status = loomc_pass_program_create_from_module_symbol(
///     pipeline_module, loomc_make_cstring_view("@cleanup"),
///     NULL, loomc_allocator_system(), &pass_program, &pass_result);
/// if (!loomc_status_is_ok(status)) {
///   return status;
/// }
/// if (!loomc_result_succeeded(pass_result)) {
///   // Inspect pass_result diagnostics. No pass program was produced.
/// }
/// loomc_result_release(pass_result);
/// // `pass_program` owns a prepared snapshot of @cleanup and any pipelines it
/// // calls. The source module may now be released, reused, or mutated.
/// loomc_module_release(pipeline_module);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Prepared immutable pass program.
///
/// Pass programs are separate from compilers so JITs and autotuners can cache
/// common pipelines, choose among several prepared programs per invocation, or
/// sweep pipeline configurations without constructing many compiler handles.
/// Pass programs remain target-independent. Target-aware predicates resolve
/// each function's durable target when the program executes.
///
/// @thread_safety
/// Pass programs are immutable after creation and may be shared across worker
/// threads. Retain/release operations are safe from multiple threads.
typedef struct loomc_pass_program_t loomc_pass_program_t;

/// Structured pass invocation report artifact format.
#define LOOMC_ARTIFACT_FORMAT_PASS_REPORT_JSON "loom-pass-report-json"

/// Pass report observation flags.
enum loomc_pass_report_flag_bits_e {
  /// Measure elapsed time for each invocation. Without this flag, duration_ns
  /// is zero and collection performs no clock reads.
  LOOMC_PASS_REPORT_FLAG_TIMING = 1u << 0,
};

/// Bitmask of `loomc_pass_report_flag_bits_e`.
typedef uint32_t loomc_pass_report_flags_t;

/// Requests a result-owned JSON report of executed passes.
///
/// Attach through `loomc_pass_run_options_t::next`,
/// `loomc_compile_options_t::next`, or
/// `loomc_compile_artifact_options_t::next`. Each attempted program produces
/// one REPORT artifact with pass identity, status, change flags, statistics,
/// and details. A diagnosed pass failure retains completed observations in its
/// failed result. Rejection before the program starts produces no pass report;
/// a non-OK infrastructure status returns no result and stops further report
/// serialization.
///
/// Omitting this extension disables collection. Attaching it with zero flags
/// collects deterministic statistics without timing. Reports neither select
/// passes nor retain modules, programs, or target providers after returning.
/// This extension and native pass report options are mutually exclusive.
typedef struct loomc_pass_report_options_t {
  /// Structure type. Must be LOOMC_STRUCTURE_TYPE_PASS_REPORT_OPTIONS.
  loomc_structure_type_t type;
  /// Size of this structure in bytes, or zero.
  loomc_host_size_t structure_size;
  /// Next invocation option extension.
  const void* next;
  /// Additional observations. Zero collects statistics and details only.
  loomc_pass_report_flags_t flags;
  /// Result artifact identifier. Empty leaves the identifier empty.
  loomc_string_view_t identifier;
} loomc_pass_report_options_t;

/// Pass program creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_pass_program_options_t)`, and fill the requested fields.
typedef struct loomc_pass_program_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_PASS_PROGRAM_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain reserved for future pass-program options.
  const void* next;

  /// Stable identifier used in diagnostics and reports. Empty selects a
  /// library-defined identifier.
  loomc_string_view_t identifier;
} loomc_pass_program_options_t;

/// Options for executing one prepared pass program.
///
/// Zero initialization selects defaults. Set `type` and `structure_size` when
/// supplying a versioned descriptor. Preparation and execution are separate:
/// these options neither parse a pipeline nor select passes.
typedef struct loomc_pass_run_options_t {
  /// Structure type; LOOMC_STRUCTURE_TYPE_PASS_RUN_OPTIONS when nonzero.
  loomc_structure_type_t type;
  /// Size of this structure in bytes, or zero.
  loomc_host_size_t structure_size;
  /// Invocation extensions, borrowed until execution returns. Supports
  /// loomc_pass_trace_options_t, loomc_pass_report_options_t,
  /// loomc_compile_report_options_t, and native report options from
  /// loomc/interop.h.
  const void* next;
} loomc_pass_run_options_t;

/// Creates a prepared no-op pass program.
///
/// @param context Context shared with modules compiled using this pass program.
/// @param options Pass program options, or `NULL` for defaults.
/// @param allocator Host allocator used for pass-program-owned storage.
/// @param out_pass_program Receives one retained pass program on success.
/// @return OK when the pass program was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_pass_program_release`.
///
/// @thread_safety
/// The returned pass program is immutable and may be shared across worker
/// threads.
///
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_create_empty(
    loomc_context_t* context, const loomc_pass_program_options_t* options,
    loomc_allocator_t allocator, loomc_pass_program_t** out_pass_program);

/// Creates a prepared pass program from a shallow textual pass list.
///
/// `pipeline_text` is the same cold-path pass-list shape used by command-line
/// flags, for example `canonicalize{max-iterations=20},dce`. The textual form
/// is intended for configuration and examples; hot paths should create a pass
/// program once and pass the prepared handle to every compile invocation.
///
/// @param context Context shared with modules compiled using this pass program.
/// @param pipeline_text Borrowed, non-NUL-terminated textual pass list.
/// @param options Pass program options, or `NULL` for defaults.
/// @param allocator Host allocator used for pass-program and result storage.
/// @param out_pass_program Receives one retained pass program when the result
/// succeeds. Receives `NULL` when the pass list is invalid.
/// @param out_result Receives a retained result containing pass-list
/// diagnostics and operation state.
/// @return OK when the creation operation completed far enough to report a
/// result. Non-OK statuses represent API misuse or infrastructure failures
/// before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. When a pass program is produced, the caller owns the
/// returned reference and releases it with `loomc_pass_program_release`.
///
/// @lifetime
/// The returned pass program copies or owns all storage it needs and does not
/// borrow from `pipeline_text`.
///
/// @error_contract
/// Unknown pass names, invalid pass options, unavailable passes, and invalid
/// textual pass-list structure return OK status with a failed result.
/// Allocation failure, API misuse, and infrastructure failures return non-OK
/// status and no result.
///
/// @thread_safety
/// The returned pass program is immutable and may be shared across worker
/// threads.
///
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_create_from_pipeline_text(
    loomc_context_t* context, loomc_string_view_t pipeline_text,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result);

/// Creates a prepared pass program from a named `pass.pipeline` symbol.
///
/// `pipeline_symbol` names a symbol in `module`; both `cleanup` and `@cleanup`
/// are accepted. The symbol must resolve to a `pass.pipeline` op. Any pipeline
/// symbols reached through `call @other_pipeline` are included in the prepared
/// program snapshot so the returned handle is independent of the source module.
///
/// @param module Module containing the selected pass pipeline.
/// @param pipeline_symbol Borrowed, non-NUL-terminated pipeline symbol name.
/// @param options Pass program options, or `NULL` for defaults.
/// @param allocator Host allocator used for pass-program and result storage.
/// @param out_pass_program Receives one retained pass program when the result
/// succeeds. Receives `NULL` when the symbol does not resolve to a valid
/// pipeline program.
/// @param out_result Receives a retained result containing pipeline selection,
/// verification, and preparation diagnostics.
/// @return OK when the creation operation completed far enough to report a
/// result. Non-OK statuses represent API misuse or infrastructure failures
/// before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. When a pass program is produced, the caller owns the
/// returned reference and releases it with `loomc_pass_program_release`.
///
/// @lifetime
/// The returned pass program owns the prepared pipeline snapshot and does not
/// borrow from `module` or `pipeline_symbol`. The caller may release, reuse, or
/// mutate `module` after this function returns.
///
/// @error_contract
/// Empty symbols, missing symbols, non-pipeline symbols, invalid pass options,
/// unavailable passes, and invalid pipeline structure return OK status with a
/// failed result. Allocation failure, API misuse, and infrastructure failures
/// return non-OK status and no result.
///
/// @thread_safety
/// The returned pass program is immutable and may be shared across worker
/// threads. The source module is read during creation and requires the caller
/// to prevent concurrent mutation for the duration of this call.
///
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_create_from_module_symbol(
    const loomc_module_t* module, loomc_string_view_t pipeline_symbol,
    const loomc_pass_program_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result);

/// Retains a pass program for another owner.
///
/// @param pass_program Pass program to retain.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_pass_program_retain(
    loomc_pass_program_t* pass_program);

/// Releases a pass program from one owner.
///
/// @param pass_program Pass program to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. The pass program
/// is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_pass_program_release(
    loomc_pass_program_t* pass_program);

/// Prints the prepared program's pipeline IR as textual `.loom`.
///
/// The listing contains the pipeline selected at creation and its called
/// pipelines, preserving source-level control flow and pass options. It does
/// not expand calls into the flattened instruction program. Programs created
/// from a module symbol print their owned snapshot, independently of later
/// changes to the original module.
///
/// Printing is optional: preparing and running programs do not select the
/// text printer. Link-time optimization can omit it from nonprinting clients.
///
/// @param pass_program Immutable prepared program to inspect.
/// @param options Module text presentation, location, and source identifier
/// options. `NULL` selects defaults. `format` must be unknown or text.
/// @param allocator Host allocator used for source-owned storage.
/// @param out_source Receives one retained text source on success, or NULL on
/// failure.
/// @return OK when printing succeeded.
///
/// @ownership
/// The caller releases the returned source with `loomc_source_release`. Its
/// bytes remain valid independently of the pass program and context.
///
/// @thread_safety
/// Printing is read-only and may run concurrently with other uses of the
/// immutable program.
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_print_text_to_source(
    const loomc_pass_program_t* pass_program,
    const loomc_module_serialize_options_t* options,
    loomc_allocator_t allocator, loomc_source_t** out_source);

/// Executes a prepared pipeline, transforming a caller-owned module in place.
///
/// Structural and target-Low input invariants are established before execution,
/// reusing verification already owned by the module. Configuration application,
/// target specialization, and artifact emission remain separate operations.
/// A function-root pipeline executes over each bodyful function in the module.
///
/// @param pass_program Immutable program created in the module's context.
/// @param workspace Invocation-local scratch workspace.
/// @param module Mutable module borrowed for this invocation.
/// @param options Execution options, or NULL for defaults.
/// @param allocator Host allocator used for result-owned storage.
/// @param out_result Receives one owned execution result on an OK return.
/// @return OK for successful execution or diagnosed input/pipeline rejection.
/// API misuse, allocation, and callback failures return non-OK with NULL
/// result.
///
/// @ownership
/// The caller retains the same module and releases the returned result with
/// loomc_result_release. Results do not borrow workspace scratch. The caller
/// may continue transforming, clone, serialize as text or bytecode, or emit the
/// module. Successful execution preserves live target function-version facts
/// and applied configuration for these subsequent operations.
///
/// @error_contract
/// Failure does not roll back IR. A failed execution invalidates cached input
/// verification and compilation products. Callers needing independent trials
/// clone the original module before execution. Explicit loomc_module_verify
/// checks transformed IR when a caller requests post-pass verification.
///
/// @thread_safety
/// Programs may be shared by concurrent invocations. Each invocation requires
/// exclusive access to its module, workspace, and observation callback state.
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_run(
    const loomc_pass_program_t* pass_program, loomc_workspace_t* workspace,
    loomc_module_t* module, const loomc_pass_run_options_t* options,
    loomc_allocator_t allocator, loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PASS_H_
