// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_NATIVE_ABI_H_
#define AMDF_SRC_XDNA_UMD_MCDM_NATIVE_ABI_H_

#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Context metadata encoding selected by the installed driver, not the device.
typedef enum amdf_windows_xdna_context_encoding_e {
  AMDF_WINDOWS_XDNA_CONTEXT_ENCODING_XCLBIN = 0,
  AMDF_WINDOWS_XDNA_CONTEXT_ENCODING_METADATA = 1,
} amdf_windows_xdna_context_encoding_t;

// Immutable native wire facts resolved before context construction.
typedef struct amdf_windows_xdna_native_abi_t {
  // Context-private payload representation consumed by the miniport.
  amdf_windows_xdna_context_encoding_t context_encoding;
  // Byte offset of the uint32 context ID in the context-private reply.
  uint32_t context_cookie_byte_offset;
  // Fixed prefix preceding the 512-byte command copy in submission records.
  uint32_t submission_header_byte_length;
} amdf_windows_xdna_native_abi_t;

// Queries the adapter's driver identity and resolves an inspected native ABI.
// No device or context is created. Success publishes a process-lifetime table;
// failure leaves the output unchanged. Private-query success without a written
// response is not an ABI version.
amdf_status_t amdf_windows_xdna_native_abi_query(
    const amdf_kmt_api_t* kmt, D3DKMT_HANDLE adapter,
    const amdf_windows_xdna_native_abi_t** out_abi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_UMD_MCDM_NATIVE_ABI_H_
