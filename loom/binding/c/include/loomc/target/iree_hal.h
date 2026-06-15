// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_IREE_HAL_H_
#define LOOMC_TARGET_IREE_HAL_H_

#include <stdbool.h>

#include "iree/hal/api.h"
#include "loomc/target.h"

/// @file
/// Optional target-profile router for IREE HAL devices.
///
/// This leaf is for hosts that already own an `iree_hal_device_t` and want a
/// Loom target profile without coupling callsites to a specific HAL backend.
/// Core Loom C API headers stay free of IREE HAL types; embedders opt in by
/// linking this leaf and one or more target-family provider leaves.
///
/// Routing is explicit and ordered. Callers pass the provider table selected by
/// the linked binary, and the router asks each provider whether it can describe
/// the device. A provider that does not recognize the device returns a route
/// miss, not an error. A provider that recognizes the device returns the normal
/// Loom operation result so unsupported device capabilities, missing executable
/// formats, or incomplete facts are reported as structured diagnostics.
///
/// @par Example
/// Route an IREE HAL Vulkan device through the linked SPIR-V provider:
///
/// @code{.c}
/// #include "loomc/target/iree_hal.h"
/// #include "loomc/target/spirv/iree_hal.h"
///
/// const loomc_iree_hal_profile_provider_t* providers[] = {
///     loomc_spirv_iree_hal_profile_provider(),
/// };
/// loomc_iree_hal_profile_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_iree_hal_profile_options_t),
///     .identifier = loomc_make_cstring_view("jit-device"),
///     .device = device,
///     .executable_cache = executable_cache,
///     .providers = providers,
///     .provider_count = 1,
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_target_profile_create_iree_hal(
///     target_environment, &options, loomc_allocator_system(), &profile,
///     &result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect diagnostics. `profile` is NULL when no route succeeded.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loomc_iree_hal_profile_options_t
    loomc_iree_hal_profile_options_t;

/// IREE HAL target-profile provider descriptor.
typedef struct loomc_iree_hal_profile_provider_t
    loomc_iree_hal_profile_provider_t;

/// Attempts to create a target profile from an IREE HAL device.
///
/// @param user_data Provider-owned pointer from
/// `loomc_iree_hal_profile_provider_t::user_data`.
/// @param target_environment Target environment that will own the profile.
/// @param options Router options borrowed for the duration of the call.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_supported Receives true when this provider handled the device.
/// @param out_profile Receives one retained profile when supported and the
/// provider's result succeeds. Receives `NULL` on route miss or failed result.
/// @param out_result Receives the provider result when supported. Receives
/// `NULL` on route miss.
/// @return OK when the provider completed far enough to report whether it
/// supports the device. Non-OK statuses represent API misuse or
/// infrastructure failures before a result could be produced.
///
/// @ownership
/// Providers transfer one retained result through `out_result` only when
/// `out_supported` is true. Providers transfer one retained profile through
/// `out_profile` only when the result succeeds.
///
/// @thread_safety
/// Provider callbacks must be thread-compatible. They may be called
/// concurrently for unrelated invocations. Any shared provider state reachable
/// through `user_data` must be immutable or internally synchronized.
typedef loomc_status_t(LOOMC_API_PTR* loomc_iree_hal_profile_provider_fn_t)(
    void* user_data, loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, bool* out_supported,
    loomc_target_profile_t** out_profile, loomc_result_t** out_result);

/// One linked IREE HAL profile provider.
struct loomc_iree_hal_profile_provider_t {
  /// Stable provider name used in diagnostics and reports.
  loomc_string_view_t name;

  /// Provider-owned callback state.
  void* user_data;

  /// Provider callback that attempts profile creation.
  loomc_iree_hal_profile_provider_fn_t create_profile;
};

/// IREE HAL target-profile routing options.
struct loomc_iree_hal_profile_options_t {
  /// Structure type. Must be `LOOMC_STRUCTURE_TYPE_IREE_HAL_PROFILE_OPTIONS`
  /// when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Provider-specific option descriptors.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  loomc_string_view_t identifier;

  /// IREE HAL device borrowed for the duration of the call.
  iree_hal_device_t* device;

  /// IREE HAL executable cache borrowed for the duration of the call.
  iree_hal_executable_cache_t* executable_cache;

  /// Ordered borrowed array of provider descriptors.
  const loomc_iree_hal_profile_provider_t* const* providers;

  /// Number of entries in `providers`.
  loomc_host_size_t provider_count;
};

/// Routes an IREE HAL device through the linked target-profile providers.
///
/// @param target_environment Target environment whose provider package
/// understands the returned profile.
/// @param options Routing options.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_profile Receives one retained profile when routing succeeds and
/// the selected provider result succeeds. Receives `NULL` on failed result.
/// @param out_result Receives a retained result for the routing operation.
/// @return OK when routing completed far enough to report a result. Non-OK
/// statuses represent API misuse or infrastructure failures before a result
/// could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. When a profile is produced, the caller owns the
/// returned reference and releases it with `loomc_target_profile_release`.
///
/// @thread_safety
/// The router holds no mutable process-global state. It may be called from many
/// threads when the supplied providers meet the callback contract.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_IREE_HAL_H_
