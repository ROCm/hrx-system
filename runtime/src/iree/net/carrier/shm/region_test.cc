// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/region.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(ShmRegionLayoutTest, CalculatesMinimumCapacity) {
  iree_net_shm_region_layout_t layout;
  IREE_ASSERT_OK(iree_net_shm_region_layout_calculate(
      IREE_MPSC_QUEUE_MIN_CAPACITY, /*mapping_alignment=*/4096, &layout));

  EXPECT_EQ(layout.ring_capacity, IREE_MPSC_QUEUE_MIN_CAPACITY);
  EXPECT_EQ(layout.mapping_alignment, 4096u);
  EXPECT_EQ(layout.ring_size, 320u);
  EXPECT_EQ(layout.ring_a_offset, 512u);
  EXPECT_EQ(layout.ring_b_offset, 832u);
  EXPECT_EQ(layout.region_size, 4096u);
}

TEST(ShmRegionLayoutTest, RoundsExtentToCreatorAlignment) {
  iree_net_shm_region_layout_t layout;
  IREE_ASSERT_OK(iree_net_shm_region_layout_calculate(
      /*ring_capacity=*/4096, /*mapping_alignment=*/16384, &layout));

  EXPECT_EQ(layout.ring_size, 4352u);
  EXPECT_EQ(layout.ring_b_offset, 4864u);
  EXPECT_EQ(layout.region_size, 16384u);
}

TEST(ShmRegionLayoutTest, AcceptsSupportedMappingAlignments) {
  const iree_host_size_t alignments[] = {4096, 8192, 16384, 32768, 65536};
  for (iree_host_size_t alignment : alignments) {
    iree_net_shm_region_layout_t layout;
    IREE_EXPECT_OK(iree_net_shm_region_layout_calculate(
        IREE_MPSC_QUEUE_MIN_CAPACITY, alignment, &layout));
  }
}

TEST(ShmRegionLayoutTest, RejectsInvalidCapacity) {
  iree_net_shm_region_layout_t layout;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_region_layout_calculate(
                            IREE_MPSC_QUEUE_MIN_CAPACITY / 2,
                            /*mapping_alignment=*/4096, &layout));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_net_shm_region_layout_calculate(
                            IREE_MPSC_QUEUE_MIN_CAPACITY + 1,
                            /*mapping_alignment=*/4096, &layout));
}

TEST(ShmRegionLayoutTest, RejectsInvalidMappingAlignment) {
  const iree_host_size_t alignments[] = {0, 2048, 12288, 131072};
  for (iree_host_size_t alignment : alignments) {
    iree_net_shm_region_layout_t layout;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        iree_net_shm_region_layout_calculate(IREE_MPSC_QUEUE_MIN_CAPACITY,
                                             alignment, &layout));
  }
}

TEST(ShmRegionLayoutTest, ChecksHostSizeOverflow) {
  iree_net_shm_region_layout_t layout;
#if defined(IREE_PTR_SIZE_64)
  IREE_ASSERT_OK(iree_net_shm_region_layout_calculate(
      UINT32_C(0x80000000), /*mapping_alignment=*/4096, &layout));
  EXPECT_GT(layout.region_size, (iree_host_size_t)UINT32_MAX);
#else
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_layout_calculate(
          UINT32_C(0x80000000), /*mapping_alignment=*/4096, &layout));
#endif  // IREE_PTR_SIZE_64
}

class ShmRegionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    mapping_.handle = IREE_SHM_HANDLE_INVALID;
    IREE_ASSERT_OK(iree_net_shm_region_layout_calculate(
        /*ring_capacity=*/4096, iree_shm_required_size(0), &layout_));
    IREE_ASSERT_OK(
        iree_shm_create(/*options=*/nullptr, layout_.region_size, &mapping_));
    IREE_ASSERT_OK(iree_net_shm_region_initialize(&layout_, &mapping_, &ring_a_,
                                                  &ring_b_));
  }

  void TearDown() override { iree_shm_close(&mapping_); }

  iree_net_shm_region_header_t* header() {
    return reinterpret_cast<iree_net_shm_region_header_t*>(
        static_cast<uint8_t*>(mapping_.base) +
        IREE_NET_SHM_REGION_OFFSET_HEADER);
  }

  iree_mpsc_queue_header_t* ring_a_header() {
    return reinterpret_cast<iree_mpsc_queue_header_t*>(
        static_cast<uint8_t*>(mapping_.base) + layout_.ring_a_offset);
  }

  iree_net_shm_region_layout_t layout_ = {};
  iree_shm_mapping_t mapping_ = {};
  iree_mpsc_queue_t ring_a_ = {};
  iree_mpsc_queue_t ring_b_ = {};
};

TEST_F(ShmRegionTest, InitializesAndOpensCanonicalRegion) {
  EXPECT_EQ(mapping_.size, layout_.region_size);
  EXPECT_EQ(header()->magic, IREE_NET_SHM_REGION_MAGIC);
  EXPECT_EQ(header()->version, IREE_NET_SHM_REGION_VERSION);
  EXPECT_EQ(header()->ring_capacity, layout_.ring_capacity);
  for (uint8_t value : header()->reserved) EXPECT_EQ(value, 0u);
  EXPECT_EQ(ring_a_.capacity, layout_.ring_capacity);
  EXPECT_EQ(ring_b_.capacity, layout_.ring_capacity);

  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_ASSERT_OK(iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                                          &opened_ring_b));
  EXPECT_EQ(opened_ring_a.capacity, layout_.ring_capacity);
  EXPECT_EQ(opened_ring_b.capacity, layout_.ring_capacity);
}

TEST_F(ShmRegionTest, RejectsTruncatedMapping) {
  iree_shm_mapping_t truncated_mapping = mapping_;
  --truncated_mapping.size;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &truncated_mapping, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsOversizedMapping) {
  iree_shm_mapping_t oversized_mapping = mapping_;
  ++oversized_mapping.size;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &oversized_mapping, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsOuterMagic) {
  header()->magic ^= 1u;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsOuterVersion) {
  ++header()->version;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsOuterCapacity) {
  header()->ring_capacity /= 2;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsOuterReservedBytes) {
  header()->reserved[sizeof(header()->reserved) - 1] = 1;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, DelegatesInnerHeaderValidation) {
  ring_a_header()->magic ^= 1u;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

TEST_F(ShmRegionTest, RejectsInnerCapacityMismatch) {
  ring_a_header()->capacity /= 2;
  iree_mpsc_queue_t opened_ring_a;
  iree_mpsc_queue_t opened_ring_b;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_net_shm_region_open(&layout_, &mapping_, &opened_ring_a,
                               &opened_ring_b));
}

}  // namespace
