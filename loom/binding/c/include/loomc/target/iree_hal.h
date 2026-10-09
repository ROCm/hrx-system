// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_IREE_HAL_H_
#define LOOMC_TARGET_IREE_HAL_H_

#include <stdbool.h>

#include "iree/hal/api.h"
#include "loomc/module.h"
#include "loomc/sanitizer.h"
#include "loomc/target.h"

/// @file
/// Optional compiler integration for IREE HAL devices.
///
/// This leaf is for hosts that compile Loom modules for IREE HAL devices. It
/// projects source-module requirements into device creation and selects the
/// compiler target and HAL loader target as one operation. Core Loom C API
/// headers stay free of IREE HAL types; embedders opt in by linking this leaf
/// and one or more target-family provider leaves.
///
/// A successful selection preserves the exact executable-target row used to
/// derive or validate the Loom profile. Callers compile with `target_profile`
/// and load the resulting executable with `executable_target`; they never need
/// to repeat target selection after compilation.
///
/// Routing is explicit and ordered. Callers pass the provider table selected by
/// the linked binary. A provider that cannot handle the requested profile or
/// device returns a route miss. A provider that recognizes the request returns
/// a normal Loom result so unsupported device capabilities and incompatible
/// targets are reported as structured diagnostics.

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loomc_iree_hal_target_options_t loomc_iree_hal_target_options_t;

/// IREE HAL target provider descriptor.
typedef struct loomc_iree_hal_target_provider_t
    loomc_iree_hal_target_provider_t;

/// Queries HAL runtime services required by a module and sanitizer policy.
///
/// Call this after parsing and linking the source module and before creating
/// the HAL device that will execute it. `sanitizer_options` must be the same
/// descriptor, if any, supplied to compilation. The query accounts for both
/// authored sanitizer operations and sanitizer instrumentation requested by
/// the caller.
///
/// @param module Source module that will be compiled and executed.
/// @param sanitizer_options Sanitizer compilation options, or `NULL` when no
/// instrumentation is requested.
/// @param out_runtime_features Receives the required HAL runtime feature bits.
/// Receives `IREE_HAL_DEVICE_RUNTIME_FEATURE_FLAG_NONE` on failure.
/// @return OK when the requirements were queried. Non-OK statuses represent
/// API misuse or allocation failures.
///
/// @thread_safety
/// The query holds no mutable process-global state. It may run concurrently
/// with other read-only module operations when no mutation is active.
LOOMC_API_EXPORT loomc_status_t loomc_iree_hal_module_query_runtime_features(
    const loomc_module_t* module,
    const loomc_sanitizer_options_t* sanitizer_options,
    iree_hal_device_runtime_feature_flags_t* out_runtime_features);

/// One compiler-and-loader target selected from an IREE HAL device.
typedef struct loomc_iree_hal_target_selection_t {
  /// Complete Loom target profile retained for the caller.
  ///
  /// The caller releases this reference with `loomc_target_profile_release`.
  loomc_target_profile_t* target_profile;

  /// Exact HAL executable target that can load artifacts for `target_profile`.
  ///
  /// This pointer is borrowed from the device specification and remains valid
  /// while the device is alive.
  const iree_hal_executable_target_t* executable_target;
} loomc_iree_hal_target_selection_t;

/// Attempts to select a compiler-and-loader target from an IREE HAL device.
///
/// @param user_data Provider-owned pointer from
/// `loomc_iree_hal_target_provider_t::user_data`.
/// @param target_environment Target environment that owns profile semantics.
/// @param options Router options borrowed for the duration of the call.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_supported Receives true when this provider handled the request.
/// @param out_selection Receives a complete selection when supported and the
/// provider's result succeeds. Receives zero on route miss or failed result.
/// @param out_result Receives the provider result when supported. Receives
/// `NULL` on route miss.
/// @return OK when the provider completed far enough to report whether it
/// supports the request. Non-OK statuses represent API misuse or
/// infrastructure failures before a result could be produced.
///
/// @ownership
/// Providers transfer one retained result through `out_result` only when
/// `out_supported` is true. A successful result transfers one retained target
/// profile through `out_selection`; the executable target remains borrowed
/// from `options->device`.
///
/// @thread_safety
/// Provider callbacks must be thread-compatible. They may be called
/// concurrently for unrelated invocations. Any shared provider state reachable
/// through `user_data` must be immutable or internally synchronized.
typedef loomc_status_t(LOOMC_API_PTR* loomc_iree_hal_target_provider_fn_t)(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    bool* out_supported, loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result);

/// One linked IREE HAL target provider.
struct loomc_iree_hal_target_provider_t {
  /// Stable provider name used in diagnostics and reports.
  loomc_string_view_t name;

  /// Provider-owned callback state.
  void* user_data;

  /// Provider callback that attempts target selection.
  loomc_iree_hal_target_provider_fn_t select_target;
};

/// IREE HAL target-routing options.
struct loomc_iree_hal_target_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_IREE_HAL_TARGET_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Provider-specific option descriptors.
  const void* next;

  /// Stable identifier for a profile derived from the device.
  loomc_string_view_t identifier;

  /// IREE HAL device borrowed for the duration of the call.
  iree_hal_device_t* device;

  /// Optional physical-device set the selected target must fully cover.
  ///
  /// Zero selects the unique highest-priority target for the logical device.
  /// A heterogeneous logical device may require an explicit affinity.
  iree_hal_physical_device_affinity_t physical_device_affinity;

  /// Optional caller-selected profile that the HAL target must load exactly.
  ///
  /// When NULL, the selected provider derives the best compatible profile from
  /// immutable device facts. When non-NULL, the provider validates that an
  /// exact compatible executable target exists and retains this profile in the
  /// returned selection.
  loomc_target_profile_t* target_profile;

  /// Ordered borrowed array of provider descriptors.
  const loomc_iree_hal_target_provider_t* const* providers;

  /// Number of entries in `providers`.
  loomc_host_size_t provider_count;
};

/// Selects a compiler-and-loader target from an IREE HAL device.
///
/// @param target_environment Target environment whose provider package
/// understands the returned profile.
/// @param options Routing options.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_selection Receives a complete selection when routing succeeds and
/// the selected provider result succeeds. Receives zero on failed result.
/// @param out_result Receives a retained result for the routing operation.
/// @return OK when routing completed far enough to report a result. Non-OK
/// statuses represent API misuse or infrastructure failures before a result
/// could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. The caller releases a returned target profile with
/// `loomc_target_profile_release`. The executable target remains borrowed from
/// `options->device`.
///
/// @thread_safety
/// The router holds no mutable process-global state. It may be called from many
/// threads when the supplied providers meet the callback contract.
LOOMC_API_EXPORT loomc_status_t loomc_target_select_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_target_options_t* options, loomc_allocator_t allocator,
    loomc_iree_hal_target_selection_t* out_selection,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_IREE_HAL_H_
