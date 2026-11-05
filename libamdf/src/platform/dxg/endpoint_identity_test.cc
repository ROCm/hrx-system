// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/dxg/endpoint_identity.h"

#include "gtest/gtest.h"

namespace {

constexpr amdf_dxg_device_ids_t kDeviceIds = {
    .vendor_id = 0x1002,
    .device_id = 0x744C,
    .subsystem_vendor_id = 0x1DA2,
    .subsystem_device_id = 0xE471,
    .revision_id = 0xC8,
    .bus_type = 0x12,
};

TEST(DxgEndpointIdentityTest, PreservesNativeCoordinatesAndDeviceHash) {
  constexpr uint64_t kAdapterLuid = UINT64_C(0x89ABCDEF01234567);
  const amdf_endpoint_id_t id =
      amdf_dxg_endpoint_id_encode(kAdapterLuid, 3, &kDeviceIds);
  EXPECT_EQ(id.words[0], kAdapterLuid);
  EXPECT_EQ(id.words[1], UINT64_C(0xBCD97AA100000003));

  uint64_t adapter_luid = 0;
  uint32_t physical_adapter_index = 0;
  amdf_dxg_endpoint_id_decode(&id, &adapter_luid, &physical_adapter_index);
  EXPECT_EQ(adapter_luid, kAdapterLuid);
  EXPECT_EQ(physical_adapter_index, 3u);
}

TEST(DxgEndpointIdentityTest, IncludesEveryDeviceIdentifier) {
  const amdf_endpoint_id_t reference =
      amdf_dxg_endpoint_id_encode(1, 2, &kDeviceIds);
  using DeviceIdField = uint32_t amdf_dxg_device_ids_t::*;
  constexpr DeviceIdField fields[] = {
      &amdf_dxg_device_ids_t::vendor_id,
      &amdf_dxg_device_ids_t::device_id,
      &amdf_dxg_device_ids_t::subsystem_vendor_id,
      &amdf_dxg_device_ids_t::subsystem_device_id,
      &amdf_dxg_device_ids_t::revision_id,
      &amdf_dxg_device_ids_t::bus_type,
  };
  for (DeviceIdField field : fields) {
    amdf_dxg_device_ids_t changed = kDeviceIds;
    changed.*field ^= 1;
    const amdf_endpoint_id_t id = amdf_dxg_endpoint_id_encode(1, 2, &changed);
    EXPECT_NE(id.words[1], reference.words[1]);
  }
}

}  // namespace
