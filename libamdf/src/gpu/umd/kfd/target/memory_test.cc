// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/kfd/target/memory.h"

#include <cstring>

#include "gtest/gtest.h"

namespace {

TEST(KfdTargetMemoryTest, DescribesOrdinaryArchitectureAndPackagePlacement) {
  struct TestCase {
    // KFD ISA identity, not raw GC IP identity.
    uint32_t gfx_target_version;
    // Package identity supplied by cached PCI discovery.
    uint32_t pci_device_id;
    // Ordinary low GPU address width.
    uint32_t address_bit_count;
    // Whether physical device-local memory can be requested.
    bool local;
  };
  const TestCase cases[] = {
      {70000, 0x1304, 40, false},  {80003, 0x67df, 40, true},
      {90010, 0x740f, 47, true},   {90402, 0x74a0, 47, false},
      {90402, 0x74a1, 47, true},   {90402, 0x74b5, 47, true},
      {90402, 0x74b6, 47, true},   {90500, 0x75a0, 47, true},
      {100303, 0x163f, 47, false}, {110001, 0x747e, 47, true},
      {110501, 0x1586, 47, false}, {120001, 0x7550, 47, true},
      {120500, 0, 56, true},
  };
  for (const auto& test : cases) {
    SCOPED_TRACE(test.gfx_target_version);
    SCOPED_TRACE(test.pci_device_id);
    amdf_gpu_kfd_topology_t topology = {};
    topology.properties.gfx_ip = {test.gfx_target_version / 10000,
                                  (test.gfx_target_version / 100) % 100,
                                  test.gfx_target_version % 100};
    topology.vram.total_byte_length = UINT64_C(8) << 30;
    topology.vram.visible_byte_length = topology.vram.total_byte_length;
    ASSERT_TRUE(amdf_gpu_kfd_target_memory_initialize(test.pci_device_id, 4096,
                                                      &topology));
    EXPECT_EQ(topology.virtual_address.begin, UINT64_C(0x10000));
    EXPECT_EQ(topology.virtual_address.end, UINT64_C(1)
                                                << test.address_bit_count);
    EXPECT_EQ(topology.virtual_address.alignment, 4096u);
    EXPECT_EQ(topology.memory_features,
              test.local ? AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY |
                               AMDF_GPU_DEVICE_FEATURE_HOST_VISIBLE_LOCAL_MEMORY
                         : 0u);
  }
}

TEST(KfdTargetMemoryTest, CachedHeapVisibilityDeterminesHostLocalAccess) {
  amdf_gpu_kfd_topology_t topology = {};
  topology.properties.gfx_ip = {11, 0, 0};
  topology.vram.total_byte_length = UINT64_C(8) << 30;
  topology.vram.visible_byte_length = UINT64_C(256) << 20;
  ASSERT_TRUE(amdf_gpu_kfd_target_memory_initialize(0, 4096, &topology));
  EXPECT_EQ(topology.memory_features, AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY);
  topology.vram.total_byte_length = 0;
  topology.vram.visible_byte_length = 0;
  ASSERT_TRUE(amdf_gpu_kfd_target_memory_initialize(0, 4096, &topology));
  EXPECT_EQ(topology.memory_features, 0u);
}

TEST(KfdTargetMemoryTest, UnknownTargetsAndPackagesDoNotPublishGuesses) {
  for (uint32_t version : {90402u, 110505u, 120100u, 130000u}) {
    amdf_gpu_kfd_topology_t topology = {};
    topology.properties.gfx_ip = {version / 10000, (version / 100) % 100,
                                  version % 100};
    topology.virtual_address.begin = 0x20000;
    topology.memory_features = AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY;
    const amdf_gpu_kfd_topology_t original = topology;
    EXPECT_FALSE(
        amdf_gpu_kfd_target_memory_initialize(0xffff, 4096, &topology));
    EXPECT_EQ(std::memcmp(&topology, &original, sizeof(topology)), 0);
  }
}

}  // namespace
