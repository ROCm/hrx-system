// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_H_
#define LOOMC_TARGET_H_

#include "loomc/context.h"
#include "loomc/pass.h"
#include "loomc/result.h"

/// @file
/// Prepared target environments and target lowering pipelines.
///
/// A target environment is the reusable compiler capability package selected
/// by the embedding binary. It describes target dialects, target-low descriptor
/// registries, source-to-low policy tables, target math policies, pass
/// registries, and target-aware pass predicates. It is not a device handle and
/// does not require hardware to exist.
///
/// Target environments are prepared before contexts because target dialects
/// must be registered while the context is still mutable. A context created
/// with a target environment can parse and build IR for those target dialects.
/// Prepared compilers and target pipelines created from that context reuse the
/// same environment-derived tables.
///
/// A target profile is a reusable, immutable set of facts for a concrete,
/// partial, saved, or synthetic target. A target selection is the cheap
/// invocation object that chooses one profile and optional overlay policy for a
/// compile, link, pipeline, or emission operation. This split lets a JIT share
/// one prepared profile across many worker threads while varying ordinary
/// per-kernel compile options independently.
///
/// @par Example
/// Create a context linked with a target environment:
///
/// @code{.c}
/// #include "loomc/target/spirv/base.h"
///
/// loomc_target_environment_t* target_environment = NULL;
/// loomc_status_t status = loomc_target_environment_create_spirv(
///     loomc_allocator_system(), &target_environment);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_context_target_options_t target_options = {
///     .type = LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS,
///     .structure_size = sizeof(loomc_context_target_options_t),
///     .target_environment = target_environment,
/// };
/// loomc_context_options_t context_options = {
///     .type = LOOMC_STRUCTURE_TYPE_CONTEXT_OPTIONS,
///     .structure_size = sizeof(loomc_context_options_t),
///     .next = &target_options,
/// };
///
/// loomc_context_t* context = NULL;
/// status = loomc_context_create(&context_options, loomc_allocator_system(),
///                               &context);
/// loomc_target_environment_release(target_environment);
/// if (!loomc_status_is_ok(status)) return status;
///
/// // `context` can now parse SPIR-V target records and create target
/// // pipelines backed by the prepared target environment.
/// @endcode
///
/// @par Example
/// Pass a target selection into an invocation option chain:
///
/// @code{.c}
/// loomc_target_profile_options_t profile_options = {
///     .type = LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_target_profile_options_t),
///     .identifier = loomc_make_cstring_view("offline-vulkan13"),
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_status_t status = loomc_target_profile_create_empty(
///     target_environment, &profile_options, loomc_allocator_system(),
///     &profile);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_target_selection_t* target_selection = NULL;
/// status = loomc_target_selection_create_from_profile(
///     profile, loomc_allocator_system(), &target_selection);
/// loomc_target_profile_release(profile);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_target_selection_options_t target_options = {
///     .type = LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS,
///     .structure_size = sizeof(loomc_target_selection_options_t),
///     .target_selection = target_selection,
/// };
/// loomc_compile_options_t compile_options = {
///     .type = LOOMC_STRUCTURE_TYPE_COMPILE_OPTIONS,
///     .structure_size = sizeof(loomc_compile_options_t),
///     .next = &target_options,
/// };
/// // Pass compile_options to loomc_compile_module.
/// loomc_target_selection_release(target_selection);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Prepared immutable target environment.
///
/// A target environment owns target provider composition selected by the
/// linked binary. It should be created once per target provider set and reused
/// across contexts, compilers, pass programs, and worker threads.
///
/// @thread_safety
/// Target environments are immutable after creation and may be shared across
/// threads. Retain/release operations are safe from multiple threads.
typedef struct loomc_target_environment_t loomc_target_environment_t;

/// Prepared immutable target profile.
///
/// A target profile owns normalized target facts and any target-family payload
/// needed by compiler passes or emitters. Profiles are separate from contexts
/// and compilers so embedders can cache one profile per device, saved hardware
/// profile, or cross-compilation bucket.
///
/// @thread_safety
/// Target profiles are immutable after creation and may be shared across
/// threads. Retain/release operations are safe from multiple threads.
typedef struct loomc_target_profile_t loomc_target_profile_t;

/// Prepared immutable target selection.
///
/// A target selection is an invocation-ready view of a target profile. Compile,
/// link, pipeline, and emission option chains borrow selections for the
/// duration of the call. Selections retain their profile, so callers may
/// release the profile after creating the selection.
///
/// @thread_safety
/// Target selections are immutable after creation and may be shared across
/// threads. Retain/release operations are safe from multiple threads.
typedef struct loomc_target_selection_t loomc_target_selection_t;

/// Three-valued target fact state.
///
/// Target profiles distinguish unavailable, available, and not-yet-known facts
/// because partial targets are valid inputs. Unknown is not false: a later live
/// probe, saved profile, or user override may refine it.
typedef enum loomc_target_fact_state_e {
  /// No fact has been supplied yet.
  LOOMC_TARGET_FACT_STATE_UNKNOWN = 0,

  /// The fact is known to be unavailable or false.
  LOOMC_TARGET_FACT_STATE_FALSE = 1,

  /// The fact is known to be available or true.
  LOOMC_TARGET_FACT_STATE_TRUE = 2,
} loomc_target_fact_state_t;

/// Context option extension that registers a target environment.
///
/// Put this descriptor on `loomc_context_options_t::next` so target dialects
/// are registered before the context is finalized. The context retains the
/// target environment on success; the caller may release its reference after
/// `loomc_context_create` returns.
typedef struct loomc_context_target_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_CONTEXT_TARGET_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next context option extension.
  const void* next;

  /// Target environment to register with the context.
  loomc_target_environment_t* target_environment;
} loomc_context_target_options_t;

/// Target profile creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_target_profile_options_t)`, and fill the requested fields.
/// Target-family APIs may define richer profile constructors for concrete fact
/// rows; this descriptor is for the target-neutral empty/unknown profile.
typedef struct loomc_target_profile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_TARGET_PROFILE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future target profile options.
  const void* next;

  /// Stable identifier used in diagnostics and reports. Empty selects a
  /// library-defined identifier.
  loomc_string_view_t identifier;
} loomc_target_profile_options_t;

/// Invocation option extension that selects a target profile.
///
/// Put this descriptor on the `next` chain of compile, link, target-pipeline,
/// or emission options when an invocation should use a prepared target
/// selection. The invocation borrows `target_selection` and does not retain it.
typedef struct loomc_target_selection_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_TARGET_SELECTION_OPTIONS`.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Next invocation option extension.
  const void* next;

  /// Target selection borrowed for the invocation.
  loomc_target_selection_t* target_selection;
} loomc_target_selection_options_t;

/// Target lowering pipeline boundary.
typedef enum loomc_target_pipeline_kind_e {
  /// Lower source/kernel IR to target-low IR prepared for target emission.
  LOOMC_TARGET_PIPELINE_KIND_PREPARED_LOW = 0,

  /// Lower source/kernel IR to target-low IR before target ABI/resource
  /// materialization and packetization preparation.
  LOOMC_TARGET_PIPELINE_KIND_SOURCE_LOW = 1,
} loomc_target_pipeline_kind_t;

/// Control-flow shape selected for source-to-low lowering.
typedef enum loomc_target_control_flow_lowering_e {
  /// Lower source structured control flow to explicit CFG before source-to-low.
  LOOMC_TARGET_CONTROL_FLOW_LOWERING_CFG = 0,

  /// Preserve supported structured control flow into target-low IR.
  LOOMC_TARGET_CONTROL_FLOW_LOWERING_STRUCTURED_LOW = 1,
} loomc_target_control_flow_lowering_t;

/// Target pipeline creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_target_pipeline_options_t)`, and fill the requested fields.
/// A zero-initialized descriptor selects the prepared target-low pipeline with
/// CFG control-flow lowering.
typedef struct loomc_target_pipeline_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_TARGET_PIPELINE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for target pipeline options such as
  /// `loomc_target_selection_options_t`.
  const void* next;

  /// Stable identifier used for the synthetic pipeline module and diagnostics.
  /// Empty selects a library-defined identifier.
  loomc_string_view_t identifier;

  /// Target pipeline boundary to build.
  loomc_target_pipeline_kind_t kind;

  /// Control-flow shape selected for the source-to-low boundary.
  loomc_target_control_flow_lowering_t control_flow_lowering;

  /// Maximum source-to-low diagnostics. Zero uses source-to-low's default.
  uint32_t source_to_low_max_errors;
} loomc_target_pipeline_options_t;

/// Creates an empty target profile scoped to a target environment.
///
/// Empty profiles contain no concrete device facts. They are useful for
/// partial-target flows where source IR target records, later overlays, or
/// family-specific APIs will provide more detail. Family-specific constructors
/// should return the same `loomc_target_profile_t` handle type after
/// normalizing their richer fact rows.
///
/// @param target_environment Target environment whose provider package
/// understands the profile. The profile retains it on success.
/// @param options Profile options, or `NULL` for defaults.
/// @param allocator Host allocator used for profile-owned storage.
/// @param out_profile Receives one retained profile on success.
/// @return OK when the profile was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_profile_release`.
///
/// @thread_safety
/// The returned profile is immutable and may be shared across worker threads.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_empty(
    loomc_target_environment_t* target_environment,
    const loomc_target_profile_options_t* options, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile);

/// Creates an explicit empty target selection.
///
/// Empty selections behave the same as omitting target-selection options. They
/// are provided so bindings and frameworks can construct uniform option chains
/// without treating source-selected compilation as a special case.
///
/// @param allocator Host allocator used for selection-owned storage.
/// @param out_target_selection Receives one retained target selection on
/// success.
/// @return OK when the selection was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_selection_release`.
LOOMC_API_EXPORT loomc_status_t loomc_target_selection_create_empty(
    loomc_allocator_t allocator,
    loomc_target_selection_t** out_target_selection);

/// Creates a target selection from a prepared target profile.
///
/// @param profile Profile to select for invocations. The selection retains the
/// profile on success.
/// @param allocator Host allocator used for selection-owned storage.
/// @param out_target_selection Receives one retained target selection on
/// success.
/// @return OK when the selection was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_target_selection_release`.
///
/// @lifetime
/// The caller may release `profile` after this function returns. The returned
/// selection keeps the profile alive until the selection is released.
LOOMC_API_EXPORT loomc_status_t loomc_target_selection_create_from_profile(
    loomc_target_profile_t* profile, loomc_allocator_t allocator,
    loomc_target_selection_t** out_target_selection);

/// Creates a prepared target lowering pass program.
///
/// The context must have been created with a target environment using
/// `loomc_context_target_options_t`. The returned pass program is an ordinary
/// `loomc_pass_program_t` and is passed to `loomc_compile_module` like any
/// other prepared pipeline.
///
/// @param context Context whose target environment selects the target package.
/// @param options Target pipeline options, or `NULL` for defaults.
/// @param allocator Host allocator used for pass-program and result storage.
/// @param out_pass_program Receives one retained pass program when the result
/// succeeds. Receives `NULL` when target pipeline preparation fails.
/// @param out_result Receives a retained result containing pipeline
/// preparation diagnostics and operation state.
/// @return OK when the creation operation completed far enough to report a
/// result. Non-OK statuses represent API misuse or infrastructure failures
/// before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. When a pass program is produced, the caller owns the
/// returned reference and releases it with `loomc_pass_program_release`.
///
/// @thread_safety
/// The returned pass program is immutable and may be shared across worker
/// threads.
///
/// @par Target Selection
/// `loomc_target_selection_options_t` may be attached to
/// `loomc_target_pipeline_options_t::next`. The selected profile must be
/// compatible with the context's target environment. Omitting the extension or
/// passing an explicit empty selection prepares the target pipeline without a
/// concrete target overlay.
LOOMC_API_EXPORT loomc_status_t loomc_pass_program_create_from_target_pipeline(
    loomc_context_t* context, const loomc_target_pipeline_options_t* options,
    loomc_allocator_t allocator, loomc_pass_program_t** out_pass_program,
    loomc_result_t** out_result);

/// Retains a target profile for another owner.
///
/// @param profile Target profile to retain.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_target_profile_retain(
    loomc_target_profile_t* profile);

/// Releases a target profile from one owner.
///
/// @param profile Target profile to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. The profile is
/// destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_target_profile_release(
    loomc_target_profile_t* profile);

/// Retains a target selection for another owner.
///
/// @param target_selection Target selection to retain.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_target_selection_retain(
    loomc_target_selection_t* target_selection);

/// Releases a target selection from one owner.
///
/// @param target_selection Target selection to release. Passing `NULL` is
/// allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. The selection is
/// destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_target_selection_release(
    loomc_target_selection_t* target_selection);

/// Retains a target environment for another owner.
///
/// @param target_environment Target environment to retain.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads.
LOOMC_API_EXPORT void loomc_target_environment_retain(
    loomc_target_environment_t* target_environment);

/// Releases a target environment from one owner.
///
/// @param target_environment Target environment to release. Passing `NULL` is
/// allowed.
///
/// @thread_safety
/// Retain/release operations are safe from multiple threads. The environment
/// is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_target_environment_release(
    loomc_target_environment_t* target_environment);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_H_
