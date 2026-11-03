// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_reservation.h"

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>

#include <cstdint>

#include "gtest/gtest.h"

namespace {

class WindowsHostReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(amdf_platform_host_reservation_create(3 * kAlignment, kAlignment,
                                                    0, UINTPTR_MAX, &address_),
              AMDF_STATUS_OK);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(address_) % kAlignment, 0u);
  }

  void TearDown() override {
    if (address_ != nullptr) {
      EXPECT_EQ(amdf_platform_host_reservation_destroy(address_, byte_length_),
                AMDF_STATUS_OK);
    }
  }

  // Stronger-than-system alignment shared by the bounded native witnesses.
  static constexpr uint64_t kAlignment = UINT64_C(2) << 20;
  // Owned native placeholder or NULL after checked release.
  void* address_ = nullptr;
  // Exact current placeholder extent required by terminal removal.
  uint64_t byte_length_ = 3 * kAlignment;
};

TEST_F(WindowsHostReservationTest, AlignsTheLowerBoundInsideTheRequiredWindow) {
  const uint64_t base = reinterpret_cast<uintptr_t>(address_);
  ASSERT_EQ(amdf_platform_host_reservation_destroy(address_, byte_length_),
            AMDF_STATUS_OK);
  address_ = nullptr;
  byte_length_ = 65536;
  ASSERT_EQ(amdf_platform_host_reservation_create(
                byte_length_, kAlignment, base + 65536,
                base + 2 * kAlignment - 1, &address_),
            AMDF_STATUS_OK);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(address_), base + kAlignment);
}

TEST_F(WindowsHostReservationTest, OccupiedIntervalPreservesMappingsAndOutput) {
  const uint64_t base = reinterpret_cast<uintptr_t>(address_);
  void* output = address_;
  const amdf_status_t status = amdf_platform_host_reservation_create(
      65536, kAlignment, base, base + 65535, &output);
  EXPECT_FALSE(amdf_status_is_ok(status));
  EXPECT_EQ(output, address_);
  MEMORY_BASIC_INFORMATION information = {};
  ASSERT_NE(VirtualQuery(address_, &information, sizeof(information)), 0u);
  EXPECT_EQ(information.State, MEM_RESERVE);
  EXPECT_EQ(information.AllocationBase, address_);
  EXPECT_EQ(information.RegionSize, byte_length_);
}

}  // namespace
