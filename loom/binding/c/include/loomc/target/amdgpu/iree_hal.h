// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMDGPU_IREE_HAL_H_
#define LOOMC_TARGET_AMDGPU_IREE_HAL_H_

#include "loomc/target/amdgpu/profile.h"
#include "loomc/target/iree_hal.h"

/// @file
/// AMDGPU target profiles from IREE HAL devices.
///
/// This optional leaf projects the canonical exact AMDGPU executable target
/// advertised by an IREE HAL device into Loom's structured AMDGPU target
/// profile. Target selectors and AMDHSA feature modes are parsed at the HAL
/// family boundary and remain structured in the resulting profile.
///
/// A logical device containing heterogeneous physical devices must identify
/// the physical-device set being compiled for. Omitting affinity is valid only
/// when the logical device advertises one unambiguous highest-priority exact
/// AMDGPU target.

#ifdef __cplusplus
extern "C" {
#endif

/// AMDGPU profile options for an IREE HAL device.
typedef struct loomc_amdgpu_iree_hal_profile_options_t {
  /// Structure type. Must be
  /// `LOOMC_STRUCTURE_TYPE_AMDGPU_IREE_HAL_PROFILE_OPTIONS` when nonzero.
  loomc_structure_type_t type;

  /// Size of this structure in bytes.
  loomc_host_size_t structure_size;

  /// Reserved extension chain. Must be `NULL`.
  const void* next;

  /// Stable profile identifier used in diagnostics.
  ///
  /// Empty uses the exact AMDGPU target selector.
  loomc_string_view_t identifier;

  /// IREE HAL device borrowed for the duration of the call.
  iree_hal_device_t* device;

  /// Optional physical-device set the exact target must fully cover.
  iree_hal_physical_device_affinity_t physical_device_affinity;
} loomc_amdgpu_iree_hal_profile_options_t;

/// Creates an exact AMDGPU target profile from an IREE HAL device.
///
/// The adapter selects one exact `amdgpu` executable target from the device
/// specification, parses its canonical target key into target identity and
/// AMDHSA feature states, and creates an ordinary AMDGPU profile.
///
/// @param target_environment AMDGPU target environment that will own the
/// profile.
/// @param options AMDGPU IREE HAL profile options.
/// @param allocator Host allocator used for result and profile storage.
/// @param out_profile Receives one retained profile when the result succeeds.
/// Receives `NULL` on failed result.
/// @param out_result Receives a retained result containing adapter diagnostics.
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
/// concurrently for different invocations. The supplied HAL device must
/// satisfy its own thread-safety contract.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_amdgpu_iree_hal(
    loomc_target_environment_t* target_environment,
    const loomc_amdgpu_iree_hal_profile_options_t* options,
    loomc_allocator_t allocator, loomc_target_profile_t** out_profile,
    loomc_result_t** out_result);

/// Returns the generic IREE HAL router provider for AMDGPU devices.
///
/// @return Process-lifetime provider descriptor. The returned pointer is
/// immutable and may be placed directly in a
/// `loomc_iree_hal_profile_options_t::providers` array.
LOOMC_API_EXPORT const loomc_iree_hal_profile_provider_t*
loomc_amdgpu_iree_hal_profile_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMDGPU_IREE_HAL_H_
