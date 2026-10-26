// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/host_reservation.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cstdint>

#include "gtest/gtest.h"

namespace {

class LinuxHostReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const long page_size = sysconf(_SC_PAGESIZE);
    ASSERT_GT(page_size, 0);
    page_size_ = static_cast<size_t>(page_size);
    address_ = static_cast<uint8_t*>(mmap(nullptr, 3 * page_size_,
                                          PROT_READ | PROT_WRITE,
                                          MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
    ASSERT_NE(address_, MAP_FAILED);
    address_[0] = 0x45;
    address_[2 * page_size_] = 0x67;
  }

  void TearDown() override {
    if (address_ != nullptr && address_ != MAP_FAILED) {
      EXPECT_EQ(munmap(address_, 3 * page_size_), 0);
    }
  }

  // Native page size used by the reservation envelope.
  size_t page_size_ = 0;
  // Three owned pages bracketing the only address the constructor may claim.
  uint8_t* address_ = nullptr;
};

TEST_F(LinuxHostReservationTest, ClaimsTheBoundedGapWithoutReplacingNeighbors) {
  void* gap = address_ + page_size_;
  ASSERT_EQ(munmap(gap, page_size_), 0);
  const uint64_t minimum = reinterpret_cast<uintptr_t>(gap);
  void* reservation = nullptr;
  ASSERT_EQ(amdf_platform_host_reservation_create(
                page_size_, page_size_, minimum, minimum + page_size_ - 1,
                &reservation),
            AMDF_STATUS_OK);
  EXPECT_EQ(reservation, gap);
  EXPECT_EQ(address_[0], 0x45);
  EXPECT_EQ(address_[2 * page_size_], 0x67);
  ASSERT_EQ(mprotect(reservation, page_size_, PROT_READ | PROT_WRITE), 0);
  *static_cast<uint8_t*>(reservation) = 0x89;
  EXPECT_EQ(address_[page_size_], 0x89);
}

TEST_F(LinuxHostReservationTest, OccupiedIntervalPreservesMappingsAndOutput) {
  void* occupied = address_ + page_size_;
  const uint64_t minimum = reinterpret_cast<uintptr_t>(occupied);
  void* reservation = address_;
  const amdf_status_t status = amdf_platform_host_reservation_create(
      page_size_, page_size_, minimum, minimum + page_size_ - 1, &reservation);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  EXPECT_EQ(reservation, address_);
  EXPECT_EQ(address_[0], 0x45);
  EXPECT_EQ(address_[2 * page_size_], 0x67);
}

}  // namespace
