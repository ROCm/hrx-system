// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_PROGRAM_PLAN_H_
#define LOOMC_PROGRAM_PLAN_H_

#include "loomc/compile.h"

/// @file
/// Root-selected heterogeneous program production plans.
///
/// A plan partitions the closed dependency graph of one or more explicitly
/// selected roots into independently compilable hermetic units. A unit may be
/// a host program, command program, device executable, or another
/// provider-defined product; no product class is privileged by this API.
///
/// The application inspects each unit module, selects an appropriate prepared
/// pass program, and compiles cache misses independently. Cache hits bypass
/// Loom entirely. Compilation returns ordinary result artifacts: plans never
/// load cached artifacts, retain native runtime objects, or perform a later
/// universal finalization step.
///
/// @par Caller Flow
/// A framework prepares one plan for its selected entry points, then fans unit
/// work out through its own scheduler and cache:
///
/// @code{.c}
/// const loomc_string_view_t roots[] = {
///     loomc_make_cstring_view("prefill"),
///     loomc_make_cstring_view("decode"),
/// };
/// loomc_program_plan_t* plan = NULL;
/// loomc_result_t* prepare_result = NULL;
/// loomc_status_t status = loomc_program_plan_prepare(
///     workspace, linked_module, roots, 2, NULL, allocator, &plan,
///     &prepare_result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(prepare_result)) {
///   // Inspect preparation diagnostics.
///   loomc_result_release(prepare_result);
///   loomc_program_plan_release(plan);
///   return loomc_ok_status();
/// }
/// loomc_result_release(prepare_result);
///
/// for (loomc_host_size_t i = 0; i < loomc_program_plan_unit_count(plan);
///      ++i) {
///   loomc_program_plan_unit_t unit = loomc_program_plan_unit_at(plan, i);
///   loomc_program_plan_unit_info_t info;
///   status = loomc_program_plan_unit_info(plan, unit, &info);
///   if (!loomc_status_is_ok(status)) break;
///
///   // Inspect or serialize info.module and combine its identity with the
///   // selected pass program and options in the framework's cache key. A hit
///   // needs no further Loom call.
///   bool framework_cache_hit = framework_cache_lookup(
///       info.module, selected_pass_program, /*compile_options=*/NULL);
///   if (framework_cache_hit) continue;
///
///   loomc_result_t* compile_result = NULL;
///   status = loomc_program_plan_compile_unit(
///       plan, compiler, worker_workspace, unit, selected_pass_program, NULL,
///       allocator, &compile_result);
///   if (!loomc_status_is_ok(status)) break;
///   // Persist or load the ordinary artifacts in compile_result using the
///   // framework's native deployment path.
///   loomc_result_release(compile_result);
/// }
/// loomc_program_plan_release(plan);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Immutable root-selected production plan.
///
/// @thread_safety
/// Plans are immutable after creation. Retained handles may be queried and
/// used to compile units concurrently when each compilation has its own
/// workspace.
typedef struct loomc_program_plan_t loomc_program_plan_t;

/// Plan-local selected-root token.
typedef struct loomc_program_plan_root_t {
  /// Dense value in the owning plan's root table.
  uint64_t value;
} loomc_program_plan_root_t;

/// Plan-local independently compilable unit token.
typedef struct loomc_program_plan_unit_t {
  /// Dense value in the owning plan's unit table.
  uint64_t value;
} loomc_program_plan_unit_t;

/// Invalid program-plan root token value.
#define LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE UINT64_MAX

/// Invalid program-plan unit token value.
#define LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE UINT64_MAX

/// Returns an invalid program-plan root token.
static inline loomc_program_plan_root_t loomc_program_plan_root_invalid(void) {
  loomc_program_plan_root_t root = {
      LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE,
  };
  return root;
}

/// Returns true when `root` is not the invalid token value.
///
/// This does not prove that the token belongs to a particular plan.
static inline bool loomc_program_plan_root_is_valid(
    loomc_program_plan_root_t root) {
  return root.value != LOOMC_PROGRAM_PLAN_ROOT_INVALID_VALUE;
}

/// Returns an invalid program-plan unit token.
static inline loomc_program_plan_unit_t loomc_program_plan_unit_invalid(void) {
  loomc_program_plan_unit_t unit = {
      LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE,
  };
  return unit;
}

/// Returns true when `unit` is not the invalid token value.
///
/// This does not prove that the token belongs to a particular plan.
static inline bool loomc_program_plan_unit_is_valid(
    loomc_program_plan_unit_t unit) {
  return unit.value != LOOMC_PROGRAM_PLAN_UNIT_INVALID_VALUE;
}

/// Program-plan preparation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_OPTIONS`, and set `structure_size` to
/// `sizeof(loomc_program_plan_options_t)`.
typedef struct loomc_program_plan_options_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Optional `loomc_compile_report_options_t` extension.
  const void* next;
} loomc_program_plan_options_t;

/// Immutable metadata for one selected root.
///
/// @lifetime
/// All views borrow from the plan and remain valid until its final release.
typedef struct loomc_program_plan_root_info_t {
  /// Canonical public root name without a leading `@`.
  loomc_string_view_t name;

  /// Complete transitive unit closure required by this root.
  ///
  /// Each unit occurs at most once. Unit ordering is compiler-defined and
  /// stable for the lifetime of the exact plan.
  const loomc_program_plan_unit_t* required_units;

  /// Number of entries in `required_units`.
  loomc_host_size_t required_unit_count;
} loomc_program_plan_root_info_t;

/// Immutable metadata for one independently compilable unit.
///
/// @lifetime
/// `module` borrows from the plan and remains valid until its final release.
typedef struct loomc_program_plan_unit_info_t {
  /// Exact hermetic module compiled by this unit.
  ///
  /// The module contains all definitions and target facts required for unit
  /// compilation. It is read-only; `loomc_program_plan_compile_unit` clones it
  /// before running the caller-selected pass program.
  const loomc_module_t* module;
} loomc_program_plan_unit_info_t;

/// Unit compilation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_PROGRAM_PLAN_UNIT_COMPILE_OPTIONS`, and set
/// `structure_size` to
/// `sizeof(loomc_program_plan_unit_compile_options_t)`.
typedef struct loomc_program_plan_unit_compile_options_t {
  /// Structure type. May be `LOOMC_STRUCTURE_TYPE_NONE` when zero-initialized.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Optional `loomc_compile_report_options_t` extension.
  ///
  /// Extensions may request observations but cannot change the product fixed
  /// by the plan.
  const void* next;
} loomc_program_plan_unit_compile_options_t;

/// Prepares an exact production plan for explicitly selected roots.
///
/// `module` is the final link-closed program universe. Configuration, target
/// selection, and transformations that may change root reachability or ABI
/// have already completed. Preparation seals transient compiler facts into
/// ordinary IR, projects the selected union closure, and partitions it into
/// hermetic units. It does not search libraries or compile any unit.
///
/// Root names are ordered, unique, non-empty, and may optionally begin with
/// `@`. Their order defines the plan root table and is preserved by providers.
/// Every root must belong to one common linked program family.
///
/// @param workspace Caller-owned scratch workspace used during preparation.
/// @param module Final linked and specialized program universe.
/// @param root_names Ordered selected root names.
/// @param root_count Number of entries in `root_names`; must be nonzero.
/// @param options Preparation options, or `NULL` for defaults.
/// @param allocator Host allocator used for returned objects.
/// @param out_program_plan Receives a retained plan when preparation succeeds.
/// @param out_result Receives the retained preparation result.
/// @return OK when preparation ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller always owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. A successful result also returns a plan owned by the
/// caller and released with `loomc_program_plan_release`. Neither returned
/// object borrows the input module, root-name array, or caller's workspace
/// reference.
///
/// @thread_safety
/// One call has exclusive access to `workspace`. Independent calls may read
/// the same module concurrently while no caller mutates it.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_prepare(
    loomc_workspace_t* workspace, const loomc_module_t* module,
    const loomc_string_view_t* root_names, loomc_host_size_t root_count,
    const loomc_program_plan_options_t* options, loomc_allocator_t allocator,
    loomc_program_plan_t** out_program_plan, loomc_result_t** out_result);

/// Retains `program_plan` for another owner.
LOOMC_API_EXPORT void loomc_program_plan_retain(
    loomc_program_plan_t* program_plan);

/// Releases `program_plan` from one owner. Passing `NULL` is allowed.
LOOMC_API_EXPORT void loomc_program_plan_release(
    loomc_program_plan_t* program_plan);

/// Returns the number of selected roots in `program_plan`, or zero for NULL.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_plan_root_count(const loomc_program_plan_t* program_plan);

/// Returns the token at `index`, or an invalid token when out of range.
LOOMC_API_EXPORT loomc_program_plan_root_t loomc_program_plan_root_at(
    const loomc_program_plan_t* program_plan, loomc_host_size_t index);

/// Looks up a selected root by canonical name.
///
/// A leading `@` is accepted. `out_root` receives an invalid token on failure.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_lookup_root(
    const loomc_program_plan_t* program_plan, loomc_string_view_t name,
    loomc_program_plan_root_t* out_root);

/// Returns metadata for one selected root.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_root_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_root_t root,
    loomc_program_plan_root_info_t* out_info);

/// Returns the number of independently compilable units, or zero for NULL.
LOOMC_API_EXPORT loomc_host_size_t
loomc_program_plan_unit_count(const loomc_program_plan_t* program_plan);

/// Returns the token at `index`, or an invalid token when out of range.
LOOMC_API_EXPORT loomc_program_plan_unit_t loomc_program_plan_unit_at(
    const loomc_program_plan_t* program_plan, loomc_host_size_t index);

/// Returns the hermetic module for one independently compilable unit.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_unit_info(
    const loomc_program_plan_t* program_plan, loomc_program_plan_unit_t unit,
    loomc_program_plan_unit_info_t* out_info);

/// Compiles one plan unit without consuming another unit's products.
///
/// The caller selects `pass_program` after inspecting the unit module. Distinct
/// units may use different pass programs and compile concurrently with
/// distinct workspaces. The invocation clones the plan-owned module, runs the
/// selected program, and emits unit-defined artifacts into `out_result`.
/// A cache hit makes this call unnecessary.
///
/// @param program_plan Plan owning `unit`.
/// @param compiler Prepared compiler selected for this unit.
/// @param workspace Caller-owned compilation workspace.
/// @param unit Plan-local unit token.
/// @param pass_program Prepared pass program selected for this unit.
/// @param options Unit compilation options, or `NULL` for defaults.
/// @param allocator Host allocator used for result-owned storage.
/// @param out_result Receives the retained compilation result.
/// @return OK when compilation ran to a result. Non-OK statuses represent API
/// misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. The result and its artifacts do not borrow the plan,
/// workspace, or cloned unit module.
LOOMC_API_EXPORT loomc_status_t loomc_program_plan_compile_unit(
    const loomc_program_plan_t* program_plan, loomc_compiler_t* compiler,
    loomc_workspace_t* workspace, loomc_program_plan_unit_t unit,
    const loomc_pass_program_t* pass_program,
    const loomc_program_plan_unit_compile_options_t* options,
    loomc_allocator_t allocator, loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_PROGRAM_PLAN_H_
