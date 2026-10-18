// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_export.h"

void amdf_platform_host_export_initialize(
    amdf_platform_host_export_t* service) {
  *service = (amdf_platform_host_export_t){
      .availability = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED),
  };
}

amdf_status_t amdf_platform_host_export_deinitialize(
    amdf_platform_host_export_t* service) {
  (void)service;
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_platform_host_export_memory(
    const amdf_platform_host_export_t* service,
    const amdf_platform_host_backing_t* backing, uint64_t byte_length,
    amdf_external_memory_t* out_memory) {
  (void)service;
  (void)backing;
  (void)byte_length;
  (void)out_memory;
  return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
}
