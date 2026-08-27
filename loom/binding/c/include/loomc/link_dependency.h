// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_LINK_DEPENDENCY_H_
#define LOOMC_LINK_DEPENDENCY_H_

#include "loomc/link_index.h"
#include "loomc/result.h"
#include "loomc/workspace.h"

/// @file
/// Strict direct-dependency analysis over frozen link indexes.
///
/// Dependency analysis checks the authored boundary of the `INPUT` providers
/// in a link index. Exact symbol requirements must be owned by those inputs or
/// exported by a declared direct `LIBRARY` provider. Other library providers
/// remain visible as the transitive audit universe so diagnostics can
/// distinguish a missing direct dependency from an unsatisfied requirement.
///
/// Analysis is metadata-only. It does not materialize function bodies, select
/// roots, specialize templates, alter link resolution, or write files. Open
/// template families remain valid when no provider is available so relocatable
/// modules can be specialized against a target library in a later link.
///
/// @par Example
/// Analyze a component whose indexed library provider is a declared direct
/// dependency and request a machine-readable report:
///
/// @code{.c}
/// const loomc_host_size_t direct_dependencies[] = {
///     library_provider_ordinal,
/// };
/// loomc_link_dependency_analysis_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS,
///     .structure_size = sizeof(options),
///     .direct_provider_ordinals = direct_dependencies,
///     .direct_provider_count = 1,
///     .component_name = loomc_make_cstring_view("//model:layers"),
///     .artifact_flags =
///         LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON,
/// };
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_link_analyze_dependencies(
///     link_index, workspace, &options, &result);
/// if (loomc_status_is_ok(status) && !loomc_result_succeeded(result)) {
///   // Inspect structured diagnostics for missing, inaccessible,
///   // incompatible, unsatisfied, or ambiguous requirements.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Loom link dependency report JSON artifact format.
#define LOOMC_ARTIFACT_FORMAT_LINK_DEPENDENCY_REPORT_JSON \
  "loom-link-dependency-report-json"

/// Dependency-analysis artifact flag bit values.
typedef enum loomc_link_dependency_artifact_flag_bits_e {
  /// Emits a schema-versioned JSON dependency report on success or failure.
  LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON = 1u << 0,
} loomc_link_dependency_artifact_flag_bits_t;

/// Bitmask of `loomc_link_dependency_artifact_flag_bits_t` values.
typedef uint32_t loomc_link_dependency_artifact_flags_t;

/// Strict dependency analysis options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS`, set
/// `structure_size` to `sizeof(loomc_link_dependency_analysis_options_t)`, and
/// fill the requested fields.
typedef struct loomc_link_dependency_analysis_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_LINK_DEPENDENCY_ANALYSIS_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for future analysis options.
  const void* next;

  /// `LIBRARY` provider ordinals declared as direct dependencies.
  ///
  /// Other `LIBRARY` providers in the index remain visible as the transitive
  /// audit universe but cannot directly satisfy dependency ownership.
  const loomc_host_size_t* direct_provider_ordinals;

  /// Number of entries in `direct_provider_ordinals`.
  loomc_host_size_t direct_provider_count;

  /// Caller-owned component name used only as diagnostic/report provenance.
  ///
  /// This may be a build label, logical package name, or empty. It never
  /// participates in symbol identity or resolution.
  loomc_string_view_t component_name;

  /// Optional report artifacts to attach to the returned result.
  loomc_link_dependency_artifact_flags_t artifact_flags;

  /// Identifier for the JSON report artifact. Empty uses
  /// `dependency-report.json`.
  ///
  /// This must be empty unless
  /// `LOOMC_LINK_DEPENDENCY_ARTIFACT_FLAG_REPORT_JSON` is set.
  loomc_string_view_t report_identifier;
} loomc_link_dependency_analysis_options_t;

/// Analyzes strict direct dependencies over a frozen link index.
///
/// @param link_index Frozen index containing `INPUT` providers being analyzed,
/// declared direct `LIBRARY` providers, and any transitive library audit
/// universe.
/// @param workspace Invocation-local scratch workspace.
/// @param options Analysis options, or `NULL` for no direct dependencies and
/// no report artifact.
/// @param out_result Receives a retained result for the completed analysis.
/// @return OK when analysis ran to a semantic result. A failed result contains
/// structured dependency diagnostics. Non-OK statuses represent API misuse,
/// malformed indexed metadata, or infrastructure failures.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`.
///
/// @lifetime
/// Returned diagnostics and artifacts own their storage and remain valid after
/// the index, workspace, or source handles supplied to the index are released.
///
/// @thread_safety
/// The same frozen index may be analyzed concurrently when each call uses a
/// distinct workspace. The same workspace requires external synchronization.
LOOMC_API_EXPORT loomc_status_t loomc_link_analyze_dependencies(
    const loomc_link_index_t* link_index, loomc_workspace_t* workspace,
    const loomc_link_dependency_analysis_options_t* options,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_LINK_DEPENDENCY_H_
