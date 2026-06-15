// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_EMIT_H_
#define LOOMC_EMIT_H_

#include "loomc/artifact.h"
#include "loomc/artifact_manifest.h"
#include "loomc/compile_report.h"
#include "loomc/module.h"
#include "loomc/result.h"
#include "loomc/target.h"
#include "loomc/workspace.h"

/// @file
/// Target artifact emission from prepared Loom modules.
///
/// Emission is the boundary that turns already-prepared Loom IR into
/// target-native artifacts such as SPIR-V binaries, object files, executable
/// containers, reports, and other sidecar outputs. It does not deserialize
/// source, compile source IR, link modules, or choose filesystem paths. Those
/// steps are explicit API operations that callers compose around emission.
///
/// The target environment selects which emitters are linked into the process.
/// A minimal embedder can link only the target emitter it needs, while a
/// packaged distribution can link many emitters and use `artifact_format` to
/// select one at runtime.
///
/// @par Example
/// Emit a prepared target-low module to an in-memory artifact:
///
/// @code{.c}
/// loomc_emit_options_t emit_options = {
///     .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
///     .structure_size = sizeof(loomc_emit_options_t),
///     .identifier = loomc_make_cstring_view("kernel.spv"),
/// };
///
/// loomc_result_t* emit_result = NULL;
/// loomc_status_t status = loomc_emit_module(
///     target_environment, workspace, module, &emit_options,
///     loomc_allocator_system(), &emit_result);
/// if (!loomc_status_is_ok(status)) return status;
///
/// if (loomc_result_succeeded(emit_result)) {
///   const loomc_artifact_t* artifact =
///       loomc_result_artifact_at(emit_result, 0);
///   // Load, cache, or write artifact->contents.
/// }
/// loomc_result_release(emit_result);
/// @endcode
///
/// @par Example
/// Pass a framework-owned option dictionary while preserving override order:
///
/// @code{.c}
/// const loomc_option_entry_t entries[] = {
///     {loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
///      loomc_make_cstring_view("default.spv")},
///     {loomc_make_cstring_view(LOOMC_EMIT_OPTION_KEY_IDENTIFIER),
///      loomc_make_cstring_view("override.spv")},
/// };
/// loomc_option_dict_t dict = {
///     .type = LOOMC_STRUCTURE_TYPE_OPTION_DICT,
///     .structure_size = sizeof(loomc_option_dict_t),
///     .entries = entries,
///     .entry_count = 2,
/// };
/// loomc_emit_options_t emit_options = {
///     .type = LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS,
///     .structure_size = sizeof(loomc_emit_options_t),
///     .next = &dict,
/// };
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// Loose option key overriding `loomc_emit_options_t::identifier`.
#define LOOMC_EMIT_OPTION_KEY_IDENTIFIER "emit.identifier"

/// Loose option key overriding artifact manifest mode.
#define LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_MODE \
  "emit.artifact_manifest.mode"

/// Loose option key overriding artifact manifest result identifier.
#define LOOMC_EMIT_OPTION_KEY_ARTIFACT_MANIFEST_IDENTIFIER \
  "emit.artifact_manifest.identifier"

/// Loose option key overriding compile report mode.
#define LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_MODE "emit.compile_report.mode"

/// Loose option key overriding compile report result identifier.
#define LOOMC_EMIT_OPTION_KEY_COMPILE_REPORT_IDENTIFIER \
  "emit.compile_report.identifier"

/// Emit artifact request flags.
typedef uint32_t loomc_emit_artifact_flags_t;

/// Emit artifact request flag bits.
typedef enum loomc_emit_artifact_flag_bits_e {
  /// Request the primary loadable artifact. A zero flag value also requests
  /// the primary artifact.
  LOOMC_EMIT_ARTIFACT_FLAG_PRIMARY = 1u << 0,
} loomc_emit_artifact_flag_bits_t;

/// Target emission options.
///
/// Callers zero-initialize this descriptor, set `type` to
/// `LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS`, set `structure_size` to
/// `sizeof(loomc_emit_options_t)`, and fill the requested fields. Attach
/// `loomc_target_selection_options_t`,
/// `loomc_artifact_manifest_options_t`, `loomc_compile_report_options_t`,
/// `loomc_option_dict_t`, and target-specific descriptors through `next`.
typedef struct loomc_emit_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_EMIT_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Extension chain for target selection and emission options.
  const void* next;

  /// Artifact format to emit. Empty selects the only linked emitter and fails
  /// if the target environment has zero or multiple emitters.
  loomc_string_view_t artifact_format;

  /// Artifact identifier for the primary output. Empty selects the emitter's
  /// default identifier.
  loomc_string_view_t identifier;

  /// Requested artifact classes. Zero requests the primary artifact.
  loomc_emit_artifact_flags_t artifact_flags;
} loomc_emit_options_t;

/// Emits target artifacts from a prepared module.
///
/// The module must already be in the target-low form expected by the selected
/// emitter. This function does not run compile passes or link additional
/// modules. The operation may mutate invocation-local scratch state inside
/// `module`; callers that need to emit the same module concurrently should
/// pass distinct module handles.
///
/// @param target_environment Target environment that contains one or more
/// linked emitters.
/// @param workspace Invocation-local scratch workspace.
/// @param module Mutable module containing prepared target IR.
/// @param options Emission options, or `NULL` for defaults.
/// @param allocator Host allocator used for result-owned storage.
/// @param out_result Receives a retained result containing diagnostics and
/// artifacts when the operation completes far enough to report them.
/// @return OK when the emission operation produced a result. Non-OK statuses
/// represent API misuse or infrastructure failures before a result could be
/// produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`.
///
/// @lifetime
/// Returned artifacts and diagnostics are owned by the result and remain valid
/// until the result is released. They do not borrow from `workspace` or
/// `module`.
///
/// @thread_safety
/// The target environment is immutable and may be shared across threads. The
/// workspace and module are invocation-local mutable objects and must not be
/// used concurrently by multiple emission calls.
LOOMC_API_EXPORT loomc_status_t
loomc_emit_module(loomc_target_environment_t* target_environment,
                  loomc_workspace_t* workspace, loomc_module_t* module,
                  const loomc_emit_options_t* options,
                  loomc_allocator_t allocator, loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_EMIT_H_
