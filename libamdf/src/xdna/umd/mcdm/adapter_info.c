// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/adapter_info.h"

amdf_status_t amdf_windows_xdna_adapter_info_query(
    const amdf_kmt_api_t* kmt, D3DKMT_HANDLE adapter,
    amdf_windows_xdna_adapter_info_t* out_info) {
  struct {
    // Reserved native word, not a protocol version.
    uint32_t reserved;
    // Zero remains when the baseline provider supplies no private information.
    uint32_t hardware_kind;
  } basic_info = {0};
  amdf_status_t status =
      amdf_kmt_query_adapter_info(kmt, adapter, KMTQAITYPE_UMDRIVERPRIVATE,
                                  &basic_info, sizeof(basic_info));
  if (amdf_status_is_ok(status)) {
    // The baseline metadata provider implements this query as a successful
    // no-op. Zero initialization also handles native marshalling that clears
    // unwritten output. A populated reply selects the expanded metadata
    // contract; no particular hardware identity selects its layout.
    *out_info = (amdf_windows_xdna_adapter_info_t){
        .protocol = basic_info.hardware_kind == 0
                        ? AMDF_WINDOWS_XDNA_PROTOCOL_METADATA_COMPACT
                        : AMDF_WINDOWS_XDNA_PROTOCOL_METADATA,
        .shared_kernel_buffers = true,
    };
    return AMDF_STATUS_OK;
  }
  if (status != amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, 0xC0000023u)) {
    return status;
  }
  // The direct protocol requires this extended reply. Starting with the basic
  // size matters: a basic provider can accept oversized storage without writing
  // the policy byte, and native marshalling can zero unwritten output bytes.
  struct {
    // Reserved native word, not consumed by this provider.
    uint32_t reserved;
    // Hardware kind, not a protocol version or an admission requirement.
    uint32_t hardware_kind;
    // One permits kernel buffers without a shared KMT resource.
    uint8_t unshared_kernel_buffers;
    // Native alignment padding, not part of the returned information.
    uint8_t padding[3];
  } info = {0, 0, UINT8_MAX, {0}};
  status = amdf_kmt_query_adapter_info(kmt, adapter, KMTQAITYPE_UMDRIVERPRIVATE,
                                       &info, sizeof(info));
  if (!amdf_status_is_ok(status)) return status;
  if (info.unshared_kernel_buffers > 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_info = (amdf_windows_xdna_adapter_info_t){
      .protocol = AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT,
      .shared_kernel_buffers = info.unshared_kernel_buffers == 0,
  };
  return AMDF_STATUS_OK;
}
