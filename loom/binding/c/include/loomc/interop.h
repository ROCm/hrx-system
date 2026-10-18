// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_INTEROP_H_
#define LOOMC_INTEROP_H_

#include "loom/error/source.h"
#include "loom/ir/module.h"
#include "loom/pass/registry.h"
#include "loom/pass/report.h"
#include "loom/target/pipeline.h"
#include "loom/target/provider.h"
#include "loom/target/reporting/report.h"
#include "loomc/compile.h"
#include "loomc/module.h"
#include "loomc/pass.h"
#include "loomc/result.h"
#include "loomc/target.h"

/// @file
/// Exact-version interop between public LoomC handles and native Loom APIs.
///
/// This optional adapter exposes compiler-native types and is intentionally not
/// part of `loomc/loomc.h` or the stable LoomC ABI. The application, LoomC, and
/// native Loom consumers must come from the same source revision and build
/// configuration. Mixing versions or configurations is undefined behavior.

#ifdef __cplusplus
extern "C" {
#endif

/// Read-only native projection of a structurally verified LoomC module.
///
/// @lifetime
/// Both pointers borrow from the public module that produced the view and
/// remain valid until that module is mutated or released. Native consumers may
/// inspect but must not mutate the projected module or source table.
typedef struct loomc_module_interop_view_t {
  /// Native module owned by the public module handle.
  const loom_module_t* module;

  /// Exact source snapshots indexed by native module source ID.
  const loom_source_table_resolver_t* source_table;
} loomc_module_interop_view_t;

/// Mutable native projection of a LoomC module for exact-version producers.
///
/// Requesting this view invalidates cached verification and compiler products
/// before exposing the native module. The caller may mutate the module using
/// native Loom APIs and must leave it structurally valid before the next public
/// LoomC operation. That operation re-establishes the required invariants.
///
/// The source table is read-only: native mutations may reuse existing source
/// IDs but must not invent source snapshots or retain either pointer beyond the
/// public module lifetime.
typedef struct loomc_module_mutable_interop_view_t {
  /// Mutable native module owned by the public module handle.
  loom_module_t* module;

  /// Exact source snapshots indexed by native module source ID.
  const loom_source_table_resolver_t* source_table;
} loomc_module_mutable_interop_view_t;

/// Creates a public target environment from exact-version native providers.
///
/// This lets a final embedding binary compose its linked native target
/// providers once while leaving environment construction and ownership with
/// LoomC. Native consumers can borrow the resulting environment with
/// `loomc_target_environment_get_interop_view`.
///
/// @param provider_set Native target providers to compose.
/// @param allocator Host allocator used for public handle storage.
/// @param out_target_environment Receives one retained public target handle.
/// @return OK when the environment and pass capabilities were prepared.
///
/// @ownership
/// The returned handle owns the composed target environment. The caller owns
/// the handle and releases it with `loomc_target_environment_release`.
///
/// @lifetime
/// `provider_set` and every provider it references must remain valid until the
/// returned handle and every object derived from it have been released.
LOOMC_API_EXPORT loomc_status_t
loomc_target_environment_create_from_provider_set(
    const loom_target_provider_set_t* provider_set, loomc_allocator_t allocator,
    loomc_target_environment_t** out_target_environment);

/// Composes builtin and target pass descriptors for exact-version inspection.
///
/// This is the same registry used to prepare pass programs. It lets clients
/// present pass names, descriptions, and option schemas without linking the
/// builtin pass implementations. No context or heap allocation is required.
///
/// @param target_environment Environment supplying target-specific passes.
/// May be `NULL` to inspect builtin passes alone.
/// @param out_storage Caller-owned storage receiving the composed registry.
/// @return OK on success; non-OK if the provider registries cannot be composed.
///
/// @lifetime
/// The storage must remain at a stable address while its registry is used.
/// Descriptors borrow from the target providers; retain the environment until
/// inspection is complete. The registry has the exact-version native contract
/// of this header, including native callback types, and is not a stable ABI.
LOOMC_API_EXPORT loomc_status_t
loomc_target_environment_get_native_pass_registry(
    const loomc_target_environment_t* target_environment,
    loom_pass_registry_storage_t* out_storage);

/// Creates a public pass program from a native source-Low pipeline builder.
///
/// This bridge lets an exact-version embedding select a native source-Low
/// inspection boundary without serializing pass IR or adding presentation-only
/// pipeline variants to the stable LoomC target-pipeline enum. The builder is
/// invoked once while preparing the immutable pass program.
///
/// @param context Public context whose target environment supplies native
/// target and pass capabilities.
/// @param build_pipeline Native source-Low pipeline builder to invoke.
/// @param identifier Stable pipeline symbol and diagnostic identifier.
/// @param options Native target pipeline options borrowed during preparation.
/// May be `NULL` for builder defaults.
/// @param allocator Host allocator used for pass-program and result storage.
/// @param out_pass_program Receives one retained pass program when the result
/// succeeds. Receives `NULL` when preparation fails.
/// @param out_result Receives one retained preparation result.
/// @return OK when preparation ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failure.
///
/// @ownership
/// The caller owns both returned handles and releases them with
/// `loomc_pass_program_release` and `loomc_result_release`.
LOOMC_API_EXPORT loomc_status_t
loomc_pass_program_create_from_native_source_low_pipeline(
    loomc_context_t* context, loom_target_pipeline_build_fn_t build_pipeline,
    loomc_string_view_t identifier,
    const loom_target_pipeline_options_t* options, loomc_allocator_t allocator,
    loomc_pass_program_t** out_pass_program, loomc_result_t** out_result);

/// Collects native pass invocation records during prepared execution.
///
/// Attach to `loomc_pass_run_options_t::next` or
/// `loomc_compile_options_t::next`. The caller initializes the collector with
/// `loom_pass_report_initialize` and releases its storage with
/// `loom_pass_report_deinitialize`. Execution appends completed records,
/// including partial progress when the operation fails or returns a non-OK
/// status with no result. Failure to allocate the last record leaves earlier
/// records available.
///
/// @lifetime
/// The collector has exclusive access during the synchronous invocation. Its
/// descriptor keys and statistic names borrow from the native providers, which
/// must outlive report consumption. Pipeline/symbol names, values, and detail
/// rows are owned by the collector. Independent executions use distinct
/// reports.
typedef struct loomc_native_pass_report_options_t {
  /// Must be `LOOMC_STRUCTURE_TYPE_NATIVE_PASS_REPORT_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Initialized caller-owned collector. Must not be NULL.
  loom_pass_report_t* report;
} loomc_native_pass_report_options_t;

/// Collects target compilation details during prepared execution.
///
/// Attach to `loomc_pass_run_options_t::next` or
/// `loomc_compile_options_t::next` for `loomc_compile_module`.
/// `loomc_compile_request` does not accept this extension because its internal
/// module is released before returning. The caller initializes the collector
/// with `loom_target_compile_report_initialize`, selects its requested detail
/// flags, and releases its storage with
/// `loom_target_compile_report_deinitialize`. Existing rows are preserved and
/// passes append the requested observations. This collector does not request
/// target specialization or artifact emission.
///
/// @lifetime
/// The collector has exclusive access during the synchronous invocation. Native
/// report identity and descriptor data follow the native report lifetime
/// contract: consume the report before mutating or releasing its module,
/// invocation profiles, or target providers.
typedef struct loomc_native_compile_report_options_t {
  /// Must be `LOOMC_STRUCTURE_TYPE_NATIVE_COMPILE_REPORT_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Initialized caller-owned collector. Must not be NULL.
  loom_target_compile_report_t* report;
} loomc_native_compile_report_options_t;

/// Verifies a public module and projects its native read-only view.
///
/// Structural invariants are always verified. When the module's context owns a
/// target environment, target-Low legality is also verified against that
/// environment. This makes the projected view safe for exact-version native
/// consumers without duplicating the compiler's target verification contract.
///
/// @param module Module to verify and inspect.
/// @param allocator Host allocator used for the returned result.
/// @param out_view Receives a borrowed view when the result succeeds. Receives
/// a zeroed view when verification fails.
/// @param out_result Receives a retained verification result.
/// @return OK when verification ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller retains `module` and owns `out_result` on an OK return, releasing
/// it with `loomc_result_release`.
///
/// @thread_safety
/// This operation may establish cached verification state on `module` and
/// requires exclusive access. The returned view may subsequently be read
/// concurrently while no operation mutates the module.
LOOMC_API_EXPORT loomc_status_t loomc_module_get_interop_view(
    loomc_module_t* module, loomc_allocator_t allocator,
    loomc_module_interop_view_t* out_view, loomc_result_t** out_result);

/// Projects a mutable native view after invalidating derived module state.
///
/// This is an exact-version escape hatch for producers that construct or adapt
/// Loom IR with native builder APIs before returning to public LoomC
/// compilation. It performs no verification and returns a zero view when
/// `module` is NULL or does not contain native IR.
///
/// @param module Module whose native IR will be mutated.
/// @return Borrowed mutable module and read-only source-table pointers.
///
/// @lifetime
/// The returned pointers remain valid until `module` is replaced, mutated by a
/// public operation, or released. Concurrent access to `module` is invalid.
LOOMC_API_EXPORT loomc_module_mutable_interop_view_t
loomc_module_get_mutable_interop_view(loomc_module_t* module);

/// Returns the immutable native environment owned by a public target handle.
///
/// @param target_environment Target environment to inspect. May be `NULL`.
/// @return Borrowed native environment, or `NULL` when the input is `NULL`.
///
/// @lifetime
/// The returned pointer remains valid until `target_environment` is released.
LOOMC_API_EXPORT const loom_target_environment_t*
loomc_target_environment_get_interop_view(
    const loomc_target_environment_t* target_environment);

/// Returns the immutable native profile owned by a public target handle.
///
/// @param target_profile Target profile to inspect. May be `NULL`.
/// @return Borrowed native profile, or `NULL` when the input is `NULL`.
///
/// @lifetime
/// The returned pointer remains valid until `target_profile` is released.
LOOMC_API_EXPORT const loom_target_profile_t*
loomc_target_profile_get_interop_view(
    const loomc_target_profile_t* target_profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_INTEROP_H_
