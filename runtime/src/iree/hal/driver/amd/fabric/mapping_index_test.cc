// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/fabric/mapping_index.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <random>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class MappingIndexTest : public ::testing::Test {
 protected:
  void TearDown() override {
    iree_hal_amd_fabric_mapping_index_deinitialize(&index_, allocator());
    EXPECT_EQ(allocations_, frees_);
  }

  iree_allocator_t allocator() {
    return {this,
            +[](void* self, iree_allocator_command_t command,
                const void* params, void** pointer) -> iree_status_t {
              auto* test = static_cast<MappingIndexTest*>(self);
              if (command == IREE_ALLOCATOR_COMMAND_FREE) {
                ++test->frees_;
              } else {
                ++test->attempts_;
                if (!test->remaining_allocations_) {
                  return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED);
                }
                --test->remaining_allocations_;
                ++test->allocations_;
                test->allocated_bytes_ +=
                    static_cast<const iree_allocator_alloc_params_t*>(params)
                        ->byte_length;
              }
              const auto system = iree_allocator_system();
              return system.ctl(system.self, command, params, pointer);
            }};
  }

  iree_hal_amd_fabric_virtual_mapping_t Mapping(
      uint64_t position, uint64_t length, uint64_t offset, int source = 0,
      amdf_memory_access_t access = AMDF_MEMORY_ACCESS_READ) {
    iree_hal_amd_fabric_virtual_mapping_t mapping = {};
    mapping.native.target_byte_offset = position;
    mapping.native.byte_length = length;
    mapping.native.source_byte_offset = offset;
    mapping.native.access = access;
    if (source >= 0) {
      mapping.source = &sources_[source];
      mapping.native.memory =
          reinterpret_cast<amdf_memory_t*>(&sources_[source]);
    }
    return mapping;
  }

  iree_status_t Update(
      const std::vector<iree_hal_amd_fabric_virtual_mapping_t>& mappings) {
    std::vector<iree_hal_memory_mapping_update_t> updates(mappings.size());
    std::vector<amdf_memory_mapping_update_t> native(mappings.size());
    for (size_t i = 0; i < mappings.size(); ++i) {
      updates[i].source.mapping_source.descriptor = mappings[i].source;
      native[i] = mappings[i].native;
    }
    IREE_RETURN_IF_ERROR(iree_hal_amd_fabric_mapping_index_prepare(
        &index_, native.size(), native.data(), allocator()));
    const size_t attempts = attempts_;
    iree_hal_amd_fabric_mapping_index_apply(&index_, updates.size(),
                                            updates.data(), native.data());
    EXPECT_EQ(attempts_, attempts);
    return iree_ok_status();
  }

  // Independent byte assignment oracle, deliberately not another interval tree.
  void Check(const std::vector<iree_hal_amd_fabric_virtual_mapping_t>& bytes) {
    for (size_t position = 0; position < bytes.size(); ++position) {
      SCOPED_TRACE(position);
      const auto& expected = bytes[position];
      iree_hal_amd_fabric_virtual_mapping_t actual = {};
      const bool mapped = iree_hal_amd_fabric_mapping_index_query(
          &index_, position, bytes.size() - position, &actual);
      ASSERT_EQ(mapped, expected.native.memory != nullptr);
      EXPECT_EQ(actual.native.memory, expected.native.memory);
      if (!mapped) {
        EXPECT_EQ(actual.source, nullptr);
        EXPECT_EQ(actual.native.byte_length, 0u);
        continue;
      }
      EXPECT_EQ(actual.source, expected.source);
      EXPECT_EQ(actual.native.access, expected.native.access);
      EXPECT_EQ(actual.native.target_byte_offset, position);
      EXPECT_EQ(actual.native.source_byte_offset,
                expected.native.source_byte_offset);
      size_t end = position + 1;
      while (end < bytes.size() && bytes[end].source == expected.source &&
             bytes[end].native.memory == expected.native.memory &&
             bytes[end].native.access == expected.native.access &&
             bytes[end].native.source_byte_offset ==
                 expected.native.source_byte_offset + end - position) {
        ++end;
      }
      EXPECT_EQ(actual.native.byte_length, end - position);
    }
  }

  // Subject owns metadata only; source identities are borrowed opaque values.
  iree_hal_amd_fabric_mapping_index_t index_ = {};
  // Stable backing identities; the index never reads their contents.
  std::array<iree_hal_amd_fabric_mapping_source_t, 4> sources_ = {};
  // Allocation attempts, including rejected requests.
  size_t attempts_ = 0;
  // Successful allocation calls.
  size_t allocations_ = 0;
  // Successful allocations returned to the allocator.
  size_t frees_ = 0;
  // Cumulative requested bytes, excluding failed requests.
  size_t allocated_bytes_ = 0;
  // Failure-injection budget; freeing never requires an allocation.
  size_t remaining_allocations_ = SIZE_MAX;
};

TEST_F(MappingIndexTest, OrderedAndShuffledIncrementalInsertion) {
  std::mt19937 random(42);
  for (int order = 0; order < 3; ++order) {
    SCOPED_TRACE(order);
    constexpr size_t kCount = 16384;
    std::vector<uint64_t> positions(kCount);
    std::iota(positions.begin(), positions.end(), 0);
    if (order == 1) {
      std::reverse(positions.begin(), positions.end());
    }
    if (order == 2) {
      std::shuffle(positions.begin(), positions.end(), random);
    }
    const size_t initial_bytes = allocated_bytes_;
    for (auto position : positions) {
      IREE_ASSERT_OK(Update({Mapping(position * 4096, 4096, 0)}));
    }
    // Storage and allocations scale with live runs, not the cumulative edits.
    EXPECT_LT(allocations_ - frees_, kCount / 32);
    EXPECT_LT(allocated_bytes_ - initial_bytes, kCount * 128);
    for (size_t i = 0; i < kCount; ++i) {
      iree_hal_amd_fabric_virtual_mapping_t actual;
      ASSERT_TRUE(iree_hal_amd_fabric_mapping_index_query(
          &index_, i * 4096 + 123, 8192, &actual));
      EXPECT_EQ(actual.native.source_byte_offset, 123u);
      EXPECT_EQ(actual.native.byte_length, 4096u - 123u);
    }
    iree_hal_amd_fabric_mapping_index_deinitialize(&index_, allocator());
  }
}

TEST_F(MappingIndexTest, MixedBatchesMatchByteAssignments) {
  std::mt19937 random(12345);
  std::vector<iree_hal_amd_fabric_virtual_mapping_t> bytes(128);
  for (size_t iteration = 0; iteration < 10000; ++iteration) {
    SCOPED_TRACE(iteration);
    std::vector<iree_hal_amd_fabric_virtual_mapping_t> changes;
    for (size_t position = random() % bytes.size(); position < bytes.size();) {
      const size_t length =
          1 + random() % std::min(size_t(16), bytes.size() - position);
      auto mapping =
          Mapping(position, length, random() % 128,
                  static_cast<int>(random() % 5) - 1, 1 + random() % 3);
      changes.push_back(mapping);
      for (size_t i = 0; i < length; ++i) {
        auto& byte = bytes[position + i];
        byte = mapping;
        byte.native.source_byte_offset += i;
      }
      position += length + random() % 8;
    }
    IREE_ASSERT_OK(Update(changes));
    Check(bytes);
  }
}

TEST_F(MappingIndexTest, ExhaustedMetadataPreservesExistingIndexAndCapacity) {
  IREE_ASSERT_OK(Update({Mapping(0, 4096, 8192)}));
  const auto original = index_;
  const size_t live_allocations = allocations_ - frees_;
  std::vector<iree_hal_amd_fabric_virtual_mapping_t> changes;
  for (size_t i = 1; i <= 128; ++i) {
    changes.push_back(Mapping(i * 4096, 4096, 0));
  }
  // Reject after two successful growth blocks, not just the first allocation.
  remaining_allocations_ = 2;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Update(changes));
  EXPECT_EQ(index_.root, original.root);
  EXPECT_EQ(index_.blocks, original.blocks);
  EXPECT_EQ(index_.available, original.available);
  EXPECT_EQ(index_.available_count, original.available_count);
  EXPECT_EQ(allocations_ - frees_, live_allocations);
  iree_hal_amd_fabric_virtual_mapping_t mapping;
  ASSERT_TRUE(
      iree_hal_amd_fabric_mapping_index_query(&index_, 123, 8192, &mapping));
  EXPECT_EQ(mapping.native.source_byte_offset, 8192u + 123u);
  EXPECT_EQ(mapping.native.byte_length, 4096u - 123u);
  EXPECT_FALSE(
      iree_hal_amd_fabric_mapping_index_query(&index_, 4096, 4096, &mapping));
  remaining_allocations_ = SIZE_MAX;
  IREE_ASSERT_OK(Update(changes));
  EXPECT_TRUE(iree_hal_amd_fabric_mapping_index_query(&index_, 128 * 4096, 1,
                                                      &mapping));
}

TEST_F(MappingIndexTest, ReplacementClearAndReuseNeedNoAllocation) {
  std::vector<iree_hal_amd_fabric_virtual_mapping_t> changes;
  for (size_t i = 0; i < 128; ++i) {
    changes.push_back(Mapping(i * 4096, 4096, 0));
  }
  IREE_ASSERT_OK(Update(changes));
  remaining_allocations_ = 0;
  const size_t attempts = attempts_;
  for (size_t iteration = 0; iteration < 128; ++iteration) {
    for (auto& change : changes) {
      change.native.source_byte_offset = iteration;
    }
    IREE_ASSERT_OK(Update(changes));
  }
  // Whole-run clears recycle their nodes, including when no allocation is
  // possible. Rebinding reuses that storage at different ordered positions.
  for (auto& change : changes) {
    change.native.memory = nullptr;
  }
  IREE_ASSERT_OK(Update(changes));
  EXPECT_EQ(index_.root, nullptr);
  for (size_t i = 0; i < changes.size(); ++i) {
    changes[i] = Mapping(i * 8192, 4096, 0);
  }
  IREE_ASSERT_OK(Update(changes));
  EXPECT_EQ(attempts_, attempts);
}

TEST_F(MappingIndexTest, CoalescingEarlierUpdatesPreservesLaterSplitCapacity) {
  // Fill the first block. Replacing the first range coalesces across the second
  // update's start; its recycled node must cover the resulting split.
  IREE_ASSERT_OK(Update({Mapping(0, 4, 0, 1), Mapping(4, 4, 4, 0)}));
  remaining_allocations_ = 0;
  const size_t attempts = attempts_;
  IREE_ASSERT_OK(Update({Mapping(0, 4, 0, 0), Mapping(4, 4, 0, 1)}));
  EXPECT_EQ(attempts_, attempts);
  std::vector<iree_hal_amd_fabric_virtual_mapping_t> bytes(8);
  for (size_t i = 0; i < bytes.size(); ++i) {
    bytes[i] = Mapping(i, 1, i % 4, i < 4 ? 0 : 1);
  }
  Check(bytes);
}

TEST_F(MappingIndexTest, EmptyClearAndFinalReleaseNeverAllocate) {
  remaining_allocations_ = 0;
  IREE_ASSERT_OK(Update({Mapping(0, 65536, 0, -1)}));
  EXPECT_EQ(attempts_, 0u);
  remaining_allocations_ = SIZE_MAX;
  IREE_ASSERT_OK(Update({Mapping(0, 65536, 0)}));
  remaining_allocations_ = 0;
  const size_t attempts = attempts_;
  IREE_ASSERT_OK(Update({Mapping(0, 65536, 0, -1)}));
  iree_hal_amd_fabric_mapping_index_deinitialize(&index_, allocator());
  EXPECT_EQ(index_.root, nullptr);
  EXPECT_EQ(index_.blocks, nullptr);
  EXPECT_EQ(index_.available, nullptr);
  EXPECT_EQ(index_.available_count, 0u);
  EXPECT_EQ(attempts_, attempts);
}

}  // namespace
