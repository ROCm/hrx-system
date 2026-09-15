// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_CONTEXT_H_
#define AMDF_SRC_XDNA_UMD_CONTEXT_H_

#include "amdf/xdna.h"
#include "libamdf/src/xdna/endpoint_profile.h"
#include "libamdf/src/xdna/umd/device.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct amdf_xdna_umd_context_t amdf_xdna_umd_context_t;

// Native result used to publish one successfully admitted XDNA context.
typedef struct amdf_xdna_umd_context_result_t {
  // Opaque identity of the live native scheduling context.
  amdf_xdna_context_id_t id;
  // Single scheduling mode selected for this context.
  amdf_xdna_scheduling_modes_t scheduling_mode;
  // Origin of the fixed backing, present only with a device placement mode.
  uint32_t physical_column_origin;
  // Width of the fixed backing, present only with a device placement mode.
  uint32_t physical_column_count;
} amdf_xdna_umd_context_result_t;

// Creates one program-independent native XDNA scheduling context.
amdf_status_t amdf_xdna_umd_context_create(
    amdf_xdna_umd_device_t* device,
    const amdf_xdna_context_create_info_t* create_info,
    amdf_xdna_umd_context_t** out_context,
    amdf_xdna_umd_context_result_t* out_result);

// Releases native XDNA context state in reverse ownership order.
amdf_status_t amdf_xdna_umd_context_destroy(amdf_xdna_umd_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_CONTEXT_H_
