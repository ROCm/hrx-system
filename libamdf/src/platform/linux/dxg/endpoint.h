// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_
#define AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_

#include "libamdf/src/platform/linux/endpoint_state.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

amdf_status_t amdf_linux_dxg_endpoint_enumerate(
    amdf_platform_instance_t* instance, uint32_t capacity,
    amdf_endpoint_summary_t* summaries, uint32_t* out_count);

amdf_status_t amdf_linux_dxg_endpoint_open(
    amdf_platform_instance_t* instance, const amdf_endpoint_id_t* id,
    amdf_platform_endpoint_t** out_endpoint, amdf_endpoint_info_t* out_info);

amdf_status_t amdf_linux_dxg_endpoint_close(amdf_platform_endpoint_t* endpoint);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_LINUX_DXG_ENDPOINT_H_
