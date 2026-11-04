// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/dxg/endpoint_identity.h"

static uint32_t amdf_dxg_hash_word(uint32_t hash, uint32_t value) {
  for (uint32_t i = 0; i < 4; ++i) {
    hash ^= value & 0xFFu;
    hash *= 16777619u;
    value >>= 8;
  }
  return hash;
}

static uint32_t amdf_dxg_hash_device_ids(
    const amdf_dxg_device_ids_t* device_ids) {
  uint32_t hash = 2166136261u;
  hash = amdf_dxg_hash_word(hash, device_ids->vendor_id);
  hash = amdf_dxg_hash_word(hash, device_ids->device_id);
  hash = amdf_dxg_hash_word(hash, device_ids->subsystem_vendor_id);
  hash = amdf_dxg_hash_word(hash, device_ids->subsystem_device_id);
  hash = amdf_dxg_hash_word(hash, device_ids->revision_id);
  return amdf_dxg_hash_word(hash, device_ids->bus_type);
}

amdf_endpoint_id_t amdf_dxg_endpoint_id_encode(
    uint64_t adapter_luid, uint32_t physical_adapter_index,
    const amdf_dxg_device_ids_t* device_ids) {
  return (amdf_endpoint_id_t){
      .words = {adapter_luid,
                ((uint64_t)amdf_dxg_hash_device_ids(device_ids) << 32) |
                    physical_adapter_index},
  };
}

void amdf_dxg_endpoint_id_decode(const amdf_endpoint_id_t* id,
                                 uint64_t* out_adapter_luid,
                                 uint32_t* out_physical_adapter_index) {
  *out_adapter_luid = id->words[0];
  *out_physical_adapter_index = (uint32_t)id->words[1];
}
