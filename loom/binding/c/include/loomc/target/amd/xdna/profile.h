// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMD_XDNA_PROFILE_H_
#define LOOMC_TARGET_AMD_XDNA_PROFILE_H_

#include "loomc/target/amd/xdna/base.h"

/// @file
/// Exact XDNA deployment profiles for target specialization.

#ifdef __cplusplus
extern "C" {
#endif

/// Creates an immutable profile for a supported AIE2P device key.
///
/// The key identifies the array geometry and register/configuration ABI, for
/// example `amd.xdna.strix_halo.17f0_11`. This is an offline lookup; it does
/// not probe hardware. Unknown keys fail instead of assuming a compatible
/// device. Supply the profile through ordinary
/// `loomc_target_specialization_options_t` function or target-declaration
/// bindings. The compiled function versions retain their device facts through
/// emission, so no separate emit profile is needed.
///
/// @param target_environment Environment containing the AIE2P target package.
/// @param device_key Exact device key; the view need not be NUL-terminated.
/// @param allocator Host allocator used for the profile handle.
/// @param out_profile Receives one retained profile on success, NULL on
/// failure.
/// @return OK on success; invalid argument for an unsupported key or
/// environment.
///
/// @ownership
/// The caller releases the returned reference with
/// `loomc_target_profile_release`. The profile copies the device key and
/// retains the target environment. The profile is immutable and may be reused
/// across threads and compilations.
LOOMC_API_EXPORT loomc_status_t loomc_target_profile_create_xdna(
    loomc_target_environment_t* target_environment,
    loomc_string_view_t device_key, loomc_allocator_t allocator,
    loomc_target_profile_t** out_profile);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMD_XDNA_PROFILE_H_
