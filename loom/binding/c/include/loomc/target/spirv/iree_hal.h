// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_SPIRV_IREE_HAL_H_
#define LOOMC_TARGET_SPIRV_IREE_HAL_H_

#include "loomc/target/iree_hal.h"
#include "loomc/target/spirv/profile.h"

/// @file
/// SPIR-V target profiles from IREE HAL devices.
///
/// This optional leaf adapts an IREE HAL Vulkan device into the public SPIR-V
/// target profile fact model. It does not expose Vulkan headers or driver
/// private types. The adapter uses stable HAL device queries such as
/// `vulkan.device :: api_version` and `vulkan.feature ::
/// buffer_device_address`, then creates an ordinary `loomc_target_profile_t`
/// with SPIR-V facts and diagnostics.
///
/// The initial execution profile targets IREE HAL's raw Vulkan BDA SPIR-V
/// executable format. Devices or executable caches that cannot support that
/// execution mode return a failed result with structured diagnostics instead
/// of forcing callers to infer support from status codes.
///
/// @par Example
/// Create a SPIR-V profile directly from an IREE HAL Vulkan device:
///
/// @code{.c}
/// loomc_spirv_iree_hal_profile_options_t options = {
///     .type = LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS,
///     .structure_size = sizeof(loomc_spirv_iree_hal_profile_options_t),
///     .identifier = loomc_make_cstring_view("jit-vulkan"),
///     .device = device,
///     .executable_cache = executable_cache,
/// };
/// loomc_target_profile_t* profile = NULL;
/// loomc_result_t* result = NULL;
/// loomc_status_t status = loomc_target_profile_create_spirv_iree_hal(
///     target_environment, &options, loomc_allocator_system(), &profile,
///     &result);
/// if (!loomc_status_is_ok(status)) return status;
/// if (!loomc_result_succeeded(result)) {
///   // Inspect diagnostics before deciding whether to fall back or skip.
/// }
/// loomc_result_release(result);
/// @endcode

#ifdef __cplusplus
extern "C" {
#endif

/// SPIR-V profile options for an IREE HAL Vulkan device.
typedef struct loomc_spirv_iree_hal_profile_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_SPIRV_IREE_HAL_PROFILE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  loomc_string_view_t identifier;

  /// IREE HAL device borrowed for the duration of the call.
  iree_hal_device_t* device;

  /// IREE HAL executable cache borrowed for the duration of the call.
  iree_hal_executable_cache_t* executable_cache;
} loomc_spirv_iree_hal_profile_options_t;

/// Creates a SPIR-V target profile from an IREE HAL Vulkan device.
///
/// @param target_environment SPIR-V target environment that will own the
/// profile.
/// @param options SPIR-V IREE HAL profile options.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_profile Receives one retained profile when the result succeeds.
/// Receives `NULL` on failed result.
/// @param out_result Receives a retained result containing adapter or SPIR-V
/// profile diagnostics.
/// @return OK when profile creation completed far enough to report a result.
/// Non-OK statuses represent API misuse or infrastructure failures before a
/// result could be produced.
///
/// @ownership
/// The caller owns `out_result` on an OK return and releases it with
/// `loomc_result_release`. When a profile is produced, the caller owns the
/// returned reference and releases it with `loomc_target_profile_release`.
///
/// @thread_safety
/// The adapter holds no mutable process-global state. It may be called
/// concurrently for different invocations. The supplied HAL device and
/// executable cache must satisfy their own thread-safety contracts.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_spirv_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_spirv_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result);

/// Returns the generic IREE HAL router provider for SPIR-V/Vulkan devices.
///
/// @return Process-lifetime provider descriptor. The returned pointer is
/// immutable and may be placed directly in a
/// `loomc_iree_hal_profile_options_t::providers` array.
LOOMC_API_EXPORT const loomc_iree_hal_profile_provider_t*
loomc_spirv_iree_hal_profile_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_SPIRV_IREE_HAL_H_
