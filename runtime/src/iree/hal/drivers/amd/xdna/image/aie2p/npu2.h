// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// NPU2-array XDNA executable-image target qualification.

#ifndef IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_NPU2_H_
#define IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_NPU2_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/drivers/amd/xdna/image/aie2p/target.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Initializes the exact device target contract for an NPU2 logical context.
//
// |target_id| is the canonical enumerated device key, currently
// amd.xdna.strix.17f0_10 or amd.xdna.strix_halo.17f0_11. Unknown keys return
// UNIMPLEMENTED; sharing the array architecture does not imply image identity.
// |context_column_count| must be in the hardware-supported range [1, 8]. The
// returned descriptor contains no borrowed state and may be copied or shared
// between concurrent image construction calls. Failure leaves |out_target|
// unchanged.
iree_status_t iree_hal_amd_xdna_aie2p_npu2_target_initialize(
    iree_string_view_t target_id, uint16_t context_column_count,
    iree_hal_amd_xdna_aie2p_target_t* out_target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMD_XDNA_IMAGE_AIE2P_NPU2_H_
