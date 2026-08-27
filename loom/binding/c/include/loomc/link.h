// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_H_
#define LOOMC_LINK_H_

#include "loomc/config.h"
#include "loomc/link_index.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Linker and opaque linked-module handles.
///
/// Linking resolves primary sources/modules against reusable frozen indexes and
/// produces an opaque module handle that can be consumed by later API
/// operations without printing and reparsing text. The optimized embedding path
/// prepares linkers and indexes once, then invokes them many times with
/// per-worker workspaces.
///
/// @par Example
/// Link a reusable index with an invocation-local workspace:
///
/// @code{.c}
/// loomc_linker_t* linker = NULL;
/// loomc_status_t status = loomc_linker_create(
///     context, NULL, loomc_allocator_system(), &linker);
/// if (!loomc_status_is_ok(status)) return status;
///
/// loomc_workspace_t* workspace = NULL;
/// status = loomc_workspace_create(NULL, loomc_allocator_system(), &workspace);
/// if (!loomc_status_is_ok(status)) {
///   loomc_linker_release(linker);
///   return status;
/// }
///
/// const loomc_string_view_t roots[] = {
///     loomc_make_cstring_view("@entry"),
/// };
/// loomc_link_options_t link_options = {
///     .type = LOOMC_STRUCTURE_TYPE_LINK_OPTIONS,
///     .structure_size = sizeof(loomc_link_options_t),
///     .link_index = library_index,
///     .mode = LOOMC_LINK_MODE_LINK,
///     .root_symbols = roots,
///     .root_symbol_count = 1,
/// };
///
/// loomc_module_t* module = NULL;
/// loomc_result_t* result = NULL;
/// status = loomc_link_module(
///     linker, workspace, &link_options, &module, &result);
/// if (loomc_status_is_ok(status) && loomc_result_succeeded(result)) {
///   // Pass module to compile without a text round trip.
/// }
/// loomc_result_release(result);
/// loomc_module_release(module);
/// loomc_workspace_release(workspace);
/// loomc_linker_release(linker);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Prepared immutable linker.
///
/// A linker owns validated link-time configuration. Frozen indexes carry
/// reusable provider state and are supplied per invocation.
///
/// @thread_safety
/// Prepared linkers are immutable after creation and may be shared across
/// threads. Invocation-local scratch belongs in `loomc_workspace_t`.
typedef struct loomc_linker_t loomc_linker_t;

/// Linker creation options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LINKER_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_linker_options_t)`, and fill the requested fields.
typedef struct loomc_linker_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LINKER_OPTIONS` when
  /// nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future linker options.
  const void* next;

  /// Default output module name for invocations that do not override it.
  loomc_string_view_t module_name;
} loomc_linker_options_t;

/// Link operation flags.
typedef enum loomc_link_flag_bits_e {
  /// Select exported INPUT-provider symbols as roots and compute their closure.
  LOOMC_LINK_FLAG_INCLUDE_INPUT_EXPORTS = 1u << 0,

  /// Preserve reachable unresolved references for a later link or
  /// specialization step.
  LOOMC_LINK_FLAG_ALLOW_UNRESOLVED_SYMBOLS = 1u << 1,

  /// Strip symbols used only by test or benchmark tooling.
  LOOMC_LINK_FLAG_STRIP_TEST_SYMBOLS = 1u << 2,
} loomc_link_flag_bits_t;

/// Bitmask of `loomc_link_flag_bits_t` values.
typedef uint32_t loomc_link_flags_t;

/// Link product selection mode.
typedef enum loomc_link_mode_e {
  /// Merge every primary-input symbol into one relocatable module. Library
  /// providers are not materialized.
  LOOMC_LINK_MODE_MERGE = 0,

  /// Retain explicit roots or exported primary-input roots and their reachable
  /// closure through the complete supplied library universe. Selected root
  /// providers contribute all of their exported symbols as roots without
  /// changing provider visibility.
  LOOMC_LINK_MODE_LINK = 1,
} loomc_link_mode_t;

/// Link invocation options.
///
/// A link invocation consumes a frozen index and returns either a retained
/// module or a failed result with diagnostics. Merge mode materializes only
/// primary INPUT providers and preserves unresolved contracts. Link mode
/// selects a dependency closure from explicit roots or exported primary roots
/// plus the exports of selected root providers across the complete supplied
/// library universe. Linking preserves every target carried by the selected
/// symbols. Target-specialization extensions participate in template-provider
/// selection and project exact target facts into the standalone linked output.
/// Config options materialize on the linked output for this invocation; frozen
/// indexes and reusable input/library modules are never mutated by link-time
/// specialization.
typedef struct loomc_link_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_LINK_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Optional invocation extensions such as
  /// `loomc_target_specialization_options_t`.
  const void* next;

  /// Frozen provider index to link.
  loomc_link_index_t* link_index;

  /// Output module name for this invocation. Empty uses the linker's default.
  loomc_string_view_t module_name;

  /// Product selection mode. Zero defaults to `LOOMC_LINK_MODE_MERGE`.
  loomc_link_mode_t mode;

  /// Root symbol names for link mode. Function-like roots are retained as
  /// module-boundary entries in the linked output.
  const loomc_string_view_t* root_symbols;

  /// Number of entries in `root_symbols`.
  loomc_host_size_t root_symbol_count;

  /// Link operation flags.
  loomc_link_flags_t flags;

  /// Per-invocation config bindings materialized into the linked output module.
  loomc_config_options_t config;

  /// Provider ordinals whose exported symbols are roots in link mode. This
  /// selects public API surfaces without changing provider linkage roles or
  /// making private symbols visible across provider boundaries.
  const loomc_host_size_t* root_provider_ordinals;

  /// Number of entries in `root_provider_ordinals`.
  loomc_host_size_t root_provider_count;
} loomc_link_options_t;

/// Creates a prepared immutable linker.
///
/// @param context Context shared with all link indexes used by this linker.
/// @param options Linker configuration, or `NULL` for defaults.
/// @param allocator Host allocator used for linker-owned storage.
/// @param out_linker Receives one retained linker on success.
/// @return OK when the linker was created.
///
/// @ownership
/// The caller owns the returned reference and releases it with
/// `loomc_linker_release`.
///
/// @thread_safety
/// The returned linker is immutable and may be shared across worker threads.
LOOMC_API_EXPORT loomc_status_t loomc_linker_create(
    loomc_context_t* context, const loomc_linker_options_t* options,
    loomc_allocator_t allocator, loomc_linker_t** out_linker);

/// Links a frozen index into an opaque module.
///
/// @param linker Prepared linker.
/// @param workspace Invocation-local scratch workspace.
/// @param options Link invocation options.
/// @param out_module Receives a retained linked module when the result
/// succeeds. Receives `NULL` when the result fails.
/// @param out_result Receives a retained result for the operation.
/// @return OK when the invocation ran to a result. Non-OK statuses represent
/// API misuse or infrastructure failures before a result could be produced.
///
/// @ownership
/// The caller owns `out_module` when non-NULL and releases it with
/// `loomc_module_release`. The caller always owns `out_result` on an OK return
/// and releases it with `loomc_result_release`.
///
/// @lifetime
/// Returned results do not borrow from `workspace` and remain valid after
/// `loomc_workspace_trim`. Returned modules retain `workspace` and keep their
/// arena blocks live until released.
///
/// @thread_safety
/// Calls using the same linker may run concurrently when each call uses a
/// distinct workspace. The same workspace requires external synchronization.
///
LOOMC_API_EXPORT loomc_status_t
loomc_link_module(loomc_linker_t* linker, loomc_workspace_t* workspace,
                  const loomc_link_options_t* options,
                  loomc_module_t** out_module, loomc_result_t** out_result);

/// Retains a prepared linker for another owner.
///
/// @param linker Linker to retain.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads.
LOOMC_API_EXPORT void loomc_linker_retain(loomc_linker_t* linker);

/// Releases a prepared linker from one owner.
///
/// @param linker Linker to release. Passing `NULL` is allowed.
///
/// @thread_safety
/// Retain/release operations are intended to be safe from multiple threads. The
/// linker is destroyed when the final reference is released.
LOOMC_API_EXPORT void loomc_linker_release(loomc_linker_t* linker);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_H_
