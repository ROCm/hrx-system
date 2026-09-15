// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_LEGACY_CONTEXT_H_
#define AMDF_SRC_XDNA_UMD_MCDM_LEGACY_CONTEXT_H_

#include <stdint.h>

#include "amdf/amdf.h"
#include "libamdf/src/xdna/bootstrap.h"
#include "libamdf/src/xdna/umd/mcdm/native_abi.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Builds the admitted context representation for the native ABI and bootstrap.
// Metadata-only drivers receive no xclbin or application kernel description.
// The retained xclbin ABI requires a matching embedded bootstrap image.
// Partition inputs never guarantee achieved placement. Outputs are unchanged on
// failure.
amdf_status_t amdf_windows_xdna_legacy_context_build(
    const amdf_windows_xdna_native_abi_t* abi,
    const amdf_xdna_bootstrap_t* bootstrap, uint32_t partition_column_count,
    uint32_t first_start_column, amdf_allocator_t host_allocator,
    uint8_t** out_data, uint32_t* out_data_size);

// Reads the uint32 command-aperture cookie after successful native creation.
// The record was built with this ABI; zero is a valid native context ID.
uint32_t amdf_windows_xdna_legacy_context_query_command_aperture_cookie(
    const amdf_windows_xdna_native_abi_t* abi, const uint8_t* data);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_XDNA_UMD_MCDM_LEGACY_CONTEXT_H_
