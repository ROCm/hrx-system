// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOMC_TARGET_AMD_XDNA_IREE_HAL_H_
#define LOOMC_TARGET_AMD_XDNA_IREE_HAL_H_

#include "loomc/target/iree_hal.h"

/// @file
/// AMD XDNA target selection for IREE HAL devices.
///
/// The provider pairs an exact XDNA executable target advertised by an IREE
/// HAL device with Loom's immutable AIE2P array profile for the same device
/// key. Compilation and loading therefore share one physical deployment
/// contract.

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the generic IREE HAL router provider for AMD XDNA devices.
///
/// @return Process-lifetime provider descriptor. The returned pointer is
/// immutable and may be placed directly in a
/// `loomc_iree_hal_target_options_t::providers` array.
LOOMC_API_EXPORT const loomc_iree_hal_target_provider_t*
loomc_xdna_iree_hal_target_provider(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOMC_TARGET_AMD_XDNA_IREE_HAL_H_
