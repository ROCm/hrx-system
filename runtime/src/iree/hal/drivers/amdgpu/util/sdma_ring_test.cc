// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/util/sdma_ring.h"

#include <array>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

struct Ring {
  alignas(64) std::array<uint32_t, 64> data;
  alignas(8) uint64_t read = 0, write = 0, bell = 0;
  iree_hal_amdgpu_sdma_ring_t ring{};
  Ring() { data.fill(0xdeadbeef); }
  void Initialize() {
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_initialize(data.data(), 256, &read,
                                                        &write, &bell, &ring));
  }
};

TEST(SdmaRingTest, PublicationAndCancellation) {
  Ring r;
  r.Initialize();
  uint32_t* p = nullptr;
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 28, &p));
  EXPECT_EQ(p, r.data.data());
  EXPECT_EQ(r.write, 0u);
  EXPECT_EQ(r.bell, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 4, &p));
  iree_hal_amdgpu_sdma_ring_cancel(&r.ring);
  EXPECT_EQ(r.write, 0u);
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 28, &p));
  for (int i = 0; i < 7; ++i) p[i] = 0;
  iree_hal_amdgpu_sdma_ring_commit(&r.ring);
  EXPECT_EQ(r.write, 28u);
  EXPECT_EQ(r.bell, 28u);
}

TEST(SdmaRingTest, FullDoesNotMutate) {
  Ring r;
  r.write = 252;
  r.Initialize();
  auto before = r.data;
  uint32_t* p = r.data.data() + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 4, &p));
  EXPECT_EQ(p, r.data.data() + 1);
  EXPECT_EQ(r.data, before);
  EXPECT_EQ(r.write, 252u);
  EXPECT_EQ(r.bell, 0u);
  r.read = 4;
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 4, &p));
  EXPECT_EQ(p, r.data.data() + 63);
}

TEST(SdmaRingTest, WrapPaddingCanDrainBeforeReservation) {
  Ring r;
  r.write = 240;
  r.read = 64;
  r.Initialize();
  uint32_t* p = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 128, &p));
  EXPECT_EQ(p, nullptr);
  EXPECT_EQ(r.write, 256u);
  EXPECT_EQ(r.bell, 256u);
  for (int i = 60; i < 64; ++i) EXPECT_EQ(r.data[i], 0u);
  EXPECT_EQ(r.data[0], 0xdeadbeefu);
  r.read = 256;
  IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 128, &p));
  EXPECT_EQ(p, r.data.data());
  iree_hal_amdgpu_sdma_ring_cancel(&r.ring);
  EXPECT_EQ(r.write, 256u);
}

TEST(SdmaRingTest, RejectsCorruptionAndOverflow) {
  Ring r;
  r.Initialize();
  uint32_t* p = nullptr;
  r.read = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS,
                        iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 4, &p));
  r.read = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 256, &p));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 3, &p));
  r.write = r.read = UINT64_MAX - 3;
  r.Initialize();
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 4, &p));
}

TEST(SdmaRingTest, RepeatedWrapAndRetirement) {
  Ring r;
  r.Initialize();
  for (int i = 0; i < 1000; ++i) {
    uint32_t* p = nullptr;
    IREE_ASSERT_OK(iree_hal_amdgpu_sdma_ring_try_reserve(&r.ring, 76, &p));
    for (int j = 0; j < 19; ++j) p[j] = 0;
    iree_hal_amdgpu_sdma_ring_commit(&r.ring);
    EXPECT_EQ(r.bell, r.write);
    r.read = r.write;
  }
  EXPECT_GT(r.write, 76000u);
}
}  // namespace
