// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_XDNA_UMD_MCDM_ADAPTER_INFO_H_
#define AMDF_SRC_XDNA_UMD_MCDM_ADAPTER_INFO_H_

#include "libamdf/src/platform/windows/kmt_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Coupled context, allocation and submission contracts exposed by the private
// adapter query. These are native interfaces, not driver release identities.
typedef enum amdf_windows_xdna_protocol_e {
  // Explicit partition width and a retained native kernel buffer.
  AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT = 0,
  // Expanded partition metadata and hardware information from the basic query.
  AMDF_WINDOWS_XDNA_PROTOCOL_METADATA = 1,
  // Baseline partition metadata with no private adapter-query output.
  AMDF_WINDOWS_XDNA_PROTOCOL_METADATA_COMPACT = 2,
} amdf_windows_xdna_protocol_t;

// Native execution interface and its allocation policy.
typedef struct amdf_windows_xdna_adapter_info_t {
  // Complete wire protocol established before preparing a native context.
  amdf_windows_xdna_protocol_t protocol;
  // Whether native context/response buffers require shared KMT resources.
  bool shared_kernel_buffers;
} amdf_windows_xdna_adapter_info_t;

// Resolves the native interface using only an adapter query on an XDNA
// endpoint. The baseline provider leaves the basic reply empty; its successor
// supplies hardware information. An extended provider requires a larger reply
// with kernel-buffer policy. Reply presence and required size distinguish the
// coupled protocols, independently of driver releases and specific hardware
// kinds. Failure leaves the output unchanged.
amdf_status_t amdf_windows_xdna_adapter_info_query(
    const amdf_kmt_api_t* kmt, D3DKMT_HANDLE adapter,
    amdf_windows_xdna_adapter_info_t* out_info);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_XDNA_UMD_MCDM_ADAPTER_INFO_H_
