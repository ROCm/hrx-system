// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_MODULE_CONFIG_H_
#define LOOMC_MODULE_CONFIG_H_

#include "loomc/config.h"
#include "loomc/module.h"

/// @file
/// Applying and inspecting configuration on in-memory modules.

#ifdef __cplusplus
extern "C" {
#endif

/// Configuration bindings and resolution policy for a module operation.
typedef struct loomc_module_config_options_t {
  /// Structure type; LOOMC_STRUCTURE_TYPE_MODULE_CONFIG_OPTIONS when nonzero.
  loomc_structure_type_t type;
  /// Size of this structure in bytes, or zero.
  loomc_host_size_t structure_size;
  /// Reserved extension chain. Must be NULL.
  const void* next;
  /// Bindings borrowed for the call and final resolution policy.
  loomc_config_options_t config;
} loomc_module_config_options_t;

/// Applies text/JSON bindings to the module's existing configuration symbols.
///
/// Explicit bindings override JSON entries, and unmatched keys are ignored.
/// Applied values become module-owned; no borrowed binding storage escapes the
/// call. Successful application preserves continuing compilation facts and
/// records the applied values alongside them. Configuration changes affect the
/// current IR; they do not undo transformations that have already consumed it.
///
/// With no bindings, REQUIRE_RESOLVED checks that no unresolved declarations
/// remain after pipeline execution. This check does not mutate IR or discard
/// retained compilation facts, including when unresolved declarations remain.
///
/// @param module Module to configure; its IR must satisfy input invariants.
/// @param workspace Workspace providing transient configuration scratch.
/// @param options Bindings and policy, or NULL for an admission-only check.
/// @param allocator Host allocator used for result and transient storage.
/// @param out_result Receives one owned result on an OK return.
/// @return OK for successful application or diagnosed invalid IR/configuration.
/// API misuse and infrastructure failures return non-OK with NULL result.
///
/// @ownership
/// The caller releases the result with loomc_result_release. A failed binding
/// application may leave partially modified IR and invalidates compilation
/// facts. Discard that module, or clone before applying to support rollback.
///
/// @thread_safety
/// Requires exclusive access to the module and both its owning workspace and
/// the supplied scratch workspace for the duration of the call.
LOOMC_API_EXPORT loomc_status_t loomc_module_apply_config(
    loomc_module_t* module, loomc_workspace_t* workspace,
    const loomc_module_config_options_t* options, loomc_allocator_t allocator,
    loomc_result_t** out_result);

/// Returns the module's configuration schema as a JSON REPORT artifact.
///
/// A successful result contains one artifact named `config-schema.json` with
/// format LOOMC_ARTIFACT_FORMAT_JSON. It describes unresolved declarations and
/// resolved definitions, including their types, values, and constraints, using
/// the same JSON schema as `loom-opt --print-config-schema`.
///
/// @param module Module to inspect; admission may populate verification caches.
/// @param allocator Host allocator used for result and report storage.
/// @param out_result Receives one owned result on an OK return.
/// @return OK for a schema report or diagnosed invalid IR. API misuse and
/// infrastructure failures return non-OK with NULL result.
///
/// @ownership
/// The result owns the report bytes independently of the module, context, and
/// workspace. The caller releases the result with loomc_result_release.
///
/// @thread_safety
/// Requires exclusive access to the module and its workspace. The module's IR
/// and retained compilation facts are unchanged.
LOOMC_API_EXPORT loomc_status_t loomc_module_query_config_schema(
    loomc_module_t* module, loomc_allocator_t allocator,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_MODULE_CONFIG_H_
