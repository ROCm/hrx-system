// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_PLATFORM_DXG_ENDPOINT_IDENTITY_H_
#define AMDF_SRC_PLATFORM_DXG_ENDPOINT_IDENTITY_H_

#include "amdf/amdf.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Stable device identifiers reported by KMT for one physical adapter.
typedef struct amdf_dxg_device_ids_t {
  // PCI vendor identifier.
  uint32_t vendor_id;
  // PCI device identifier.
  uint32_t device_id;
  // PCI subsystem vendor identifier.
  uint32_t subsystem_vendor_id;
  // PCI subsystem device identifier.
  uint32_t subsystem_device_id;
  // PCI revision identifier.
  uint32_t revision_id;
  // Native KMT bus-type identifier.
  uint32_t bus_type;
} amdf_dxg_device_ids_t;

// Encodes one contemporaneous DXG adapter identity as a public endpoint id.
amdf_endpoint_id_t amdf_dxg_endpoint_id_encode(
    uint64_t adapter_luid, uint32_t physical_adapter_index,
    const amdf_dxg_device_ids_t* device_ids);

// Decodes the native adapter coordinates carried by an endpoint id. Callers
// must re-query device identifiers and compare the complete re-encoded id.
void amdf_dxg_endpoint_id_decode(const amdf_endpoint_id_t* id,
                                 uint64_t* out_adapter_luid,
                                 uint32_t* out_physical_adapter_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // AMDF_SRC_PLATFORM_DXG_ENDPOINT_IDENTITY_H_
