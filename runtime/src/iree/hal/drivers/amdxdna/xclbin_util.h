// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_
#define IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Locates one PDI image in an AXLF/xclbin2 AIE_PARTITION section. The
// returned span aliases `xclbin` and is valid only while it is.
iree_status_t iree_hal_amdxdna_xclbin_find_pdi(iree_const_byte_span_t xclbin,
                                               uint32_t pdi_index,
                                               iree_const_byte_span_t* out_pdi);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDXDNA_XCLBIN_UTIL_H_
