// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/array/facts.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

TEST(XdnaArrayFactsTest, ExposesCompleteNpu2Topology) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  EXPECT_STREQ(family->key, "amd.xdna.npu2");
  EXPECT_EQ(family->architecture, LOOM_XDNA_ARCHITECTURE_AIE2P);
  EXPECT_EQ(family->column_count, 8u);
  EXPECT_EQ(family->row_count, 6u);
  EXPECT_EQ(family->column_shift, 25u);
  EXPECT_EQ(family->row_shift, 20u);

  const loom_xdna_tile_facts_t* tile =
      loom_xdna_array_tile_facts(family, {0, 0});
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  EXPECT_EQ(tile->dma.buffer_descriptor_count, 16u);
  EXPECT_EQ(tile->dma.maximum_task_repeat_count, 256u);
  EXPECT_EQ(tile->dma.address_encoding_shift, 0u);
  EXPECT_EQ(tile->dma.maximum_encoded_transfer_length, UINT32_MAX);
  EXPECT_EQ(tile->dma.transfer_length_granularity, 4u);
  EXPECT_EQ(tile->dma.transfer_length_offset, 0u);
  EXPECT_EQ(tile->dma.memory_to_stream_port_base, 3u);
  EXPECT_EQ(tile->dma.memory_to_stream_port_stride, 4u);
  EXPECT_EQ(tile->dma.stream_to_memory_port_base, 2u);
  EXPECT_EQ(tile->dma.stream_to_memory_port_stride, 1u);
  tile = loom_xdna_array_tile_facts(family, {7, 1});
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_MEMORY);
  EXPECT_EQ(tile->memory.local_capacity, 512u * 1024u);
  EXPECT_EQ(tile->memory.local_load_base, 0x80000u);
  EXPECT_EQ(tile->dma.buffer_descriptor_count, 48u);
  EXPECT_EQ(tile->dma.maximum_encoded_transfer_length, 0x1FFFFu);
  tile = loom_xdna_array_tile_facts(family, {3, 5});
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_COMPUTE);
  EXPECT_EQ(tile->memory.local_capacity, 64u * 1024u);
  EXPECT_EQ(tile->memory.local_load_base, 0x70000u);
  EXPECT_EQ(tile->memory.program_capacity, 16u * 1024u);
  EXPECT_EQ(tile->dma.address_encoding_shift, 2u);
  EXPECT_EQ(tile->dma.maximum_encoded_transfer_length, 0x3FFFu);
  EXPECT_EQ(tile->dma.memory_to_stream_port_base, 0u);
  EXPECT_EQ(tile->dma.memory_to_stream_port_stride, 1u);
  EXPECT_EQ(tile->dma.stream_to_memory_port_base, 0u);
  EXPECT_EQ(tile->dma.stream_to_memory_port_stride, 1u);

  tile = loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  EXPECT_EQ(tile->first_row, 0u);
  EXPECT_EQ(tile->array_resources.physical_tile_count, 8u);
  EXPECT_EQ(tile->array_resources.lock_count, 128u);
  EXPECT_EQ(tile->array_resources.dma_channel_count_per_direction, 16u);
  EXPECT_EQ(tile->array_resources.dma_buffer_descriptor_count, 128u);
  tile = loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_MEMORY);
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_MEMORY);
  EXPECT_EQ(tile->first_row, 1u);
  tile = loom_xdna_array_tile_kind_facts(family, LOOM_XDNA_TILE_KIND_COMPUTE);
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_COMPUTE);
  EXPECT_EQ(tile->first_row, 2u);
  EXPECT_EQ(tile->array_resources.physical_tile_count, 32u);
  EXPECT_EQ(tile->array_resources.lock_count, 512u);
  EXPECT_EQ(tile->array_resources.dma_channel_count_per_direction, 64u);
  EXPECT_EQ(tile->array_resources.dma_buffer_descriptor_count, 512u);
}

TEST(XdnaArrayFactsTest, ExposesTraceInputsForEachTileKind) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  const loom_xdna_stream_port_range_t* range =
      loom_xdna_array_stream_port_range(family, LOOM_XDNA_TILE_KIND_COMPUTE,
                                        LOOM_XDNA_STREAM_DIRECTION_SLAVE,
                                        LOOM_XDNA_STREAM_PORT_TRACE);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->ordinal, 23u);
  EXPECT_EQ(range->count, 2u);
  range = loom_xdna_array_stream_port_range(family, LOOM_XDNA_TILE_KIND_MEMORY,
                                            LOOM_XDNA_STREAM_DIRECTION_SLAVE,
                                            LOOM_XDNA_STREAM_PORT_TRACE);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->ordinal, 17u);
  EXPECT_EQ(range->count, 1u);
  range = loom_xdna_array_stream_port_range(
      family, LOOM_XDNA_TILE_KIND_SHIM_NOC, LOOM_XDNA_STREAM_DIRECTION_SLAVE,
      LOOM_XDNA_STREAM_PORT_TRACE);
  ASSERT_NE(range, nullptr);
  EXPECT_EQ(range->ordinal, 22u);
  EXPECT_EQ(range->count, 1u);
}

TEST(XdnaArrayFactsTest, CanonicalizesComputeNeighborAliases) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  loom_xdna_memory_placement_t west = {};
  loom_xdna_memory_placement_t owner_self = {};
  IREE_ASSERT_OK(loom_xdna_array_resolve_load_memory(
      family, {1, 3}, LOOM_XDNA_MEMORY_SPACE_DATA, 0x50040, 64, &west));
  IREE_ASSERT_OK(loom_xdna_array_resolve_load_memory(
      family, {0, 3}, LOOM_XDNA_MEMORY_SPACE_DATA, 0x70040, 64, &owner_self));

  EXPECT_EQ(west.owner.column, 0u);
  EXPECT_EQ(west.owner.row, 3u);
  EXPECT_EQ(west.owner_offset, 0x40u);
  EXPECT_EQ(owner_self.owner.column, west.owner.column);
  EXPECT_EQ(owner_self.owner.row, west.owner.row);
  EXPECT_EQ(owner_self.owner_offset, west.owner_offset);
}

TEST(XdnaArrayFactsTest, SelectsLoadWindowsWithMemoryAndLockBases) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  struct TestCase {
    // Tile whose instruction or DMA accesses the owner.
    loom_xdna_tile_coordinate_t accessor;
    // Physical tile whose memory and locks are mapped.
    loom_xdna_tile_coordinate_t owner;
    // Accessor-relative data-memory base.
    uint32_t base;
    // Number of bytes mapped from the owner.
    uint32_t capacity;
    // Accessor-relative first lock selector.
    uint16_t lock_selector_base;
  };
  const TestCase cases[] = {
      {{1, 3}, {1, 2}, 0x40000, 64 * 1024, 0},
      {{1, 3}, {0, 3}, 0x50000, 64 * 1024, 16},
      {{1, 3}, {1, 4}, 0x60000, 64 * 1024, 32},
      {{1, 3}, {1, 3}, 0x70000, 64 * 1024, 48},
      {{1, 1}, {0, 1}, 0x00000, 512 * 1024, 0},
      {{1, 1}, {1, 1}, 0x80000, 512 * 1024, 64},
      {{1, 1}, {2, 1}, 0x100000, 512 * 1024, 128},
  };
  for (const TestCase& test_case : cases) {
    SCOPED_TRACE(test_case.base);
    const loom_xdna_address_window_t* window = loom_xdna_array_load_window(
        family, test_case.accessor, test_case.owner);
    ASSERT_NE(window, nullptr);
    EXPECT_EQ(window->base, test_case.base);
    EXPECT_EQ(window->capacity, test_case.capacity);
    EXPECT_EQ(window->lock_selector_base, test_case.lock_selector_base);
    EXPECT_EQ(window->owner_kind,
              loom_xdna_array_tile_facts(family, test_case.owner)->kind);
  }
}

TEST(XdnaArrayFactsTest, OmitsUnavailableLoadWindows) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  EXPECT_EQ(loom_xdna_array_load_window(family, {0, 3}, {1, 3}), nullptr);
  EXPECT_EQ(loom_xdna_array_load_window(family, {2, 3}, {0, 3}), nullptr);
  EXPECT_EQ(loom_xdna_array_load_window(family, {0, 2}, {0, 1}), nullptr);
  EXPECT_EQ(loom_xdna_array_load_window(family, {0, 1}, {0, 2}), nullptr);
  EXPECT_EQ(loom_xdna_array_load_window(family, {0, 1}, {2, 1}), nullptr);
  EXPECT_EQ(loom_xdna_array_load_window(family, {0, 0}, {0, 0}), nullptr);
}

TEST(XdnaArrayFactsTest, RejectsMissingOrCrossKindNeighbors) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  loom_xdna_memory_placement_t placement = {};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_xdna_array_resolve_load_memory(
                            family, {0, 3}, LOOM_XDNA_MEMORY_SPACE_DATA,
                            0x50000, 64, &placement));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_xdna_array_resolve_load_memory(
                            family, {0, 2}, LOOM_XDNA_MEMORY_SPACE_DATA,
                            0x40000, 64, &placement));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_xdna_array_resolve_load_memory(
                            family, {0, 3}, LOOM_XDNA_MEMORY_SPACE_DATA,
                            0x4FFF0, 32, &placement));
}

TEST(XdnaArrayFactsTest, ResolvesMemoryTileAndProgramStorage) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  loom_xdna_memory_placement_t placement = {};
  IREE_ASSERT_OK(loom_xdna_array_resolve_load_memory(
      family, {1, 1}, LOOM_XDNA_MEMORY_SPACE_DATA, 0x00100, 64, &placement));
  EXPECT_EQ(placement.owner.column, 0u);
  EXPECT_EQ(placement.owner.row, 1u);
  EXPECT_EQ(placement.owner_offset, 0x100u);

  IREE_ASSERT_OK(loom_xdna_array_resolve_load_memory(
      family, {4, 2}, LOOM_XDNA_MEMORY_SPACE_PROGRAM, 0, 2352, &placement));
  EXPECT_EQ(placement.owner.column, 4u);
  EXPECT_EQ(placement.owner.row, 2u);
  EXPECT_EQ(placement.owner_offset, 0u);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_xdna_array_resolve_load_memory(
                            family, {4, 2}, LOOM_XDNA_MEMORY_SPACE_PROGRAM, 0,
                            16 * 1024 + 1, &placement));
}

TEST(XdnaArrayFactsTest, ValidatesRegisterModuleAgainstTile) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  uint64_t address = 0;
  IREE_ASSERT_OK(loom_xdna_array_register_address(
      family, {2, 3}, LOOM_XDNA_REGISTER_MODULE_COMPUTE_MEMORY, 0x1D014,
      &address));
  EXPECT_EQ(address, (UINT64_C(2) << 25) | (UINT64_C(3) << 20) | 0x1D014);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_xdna_array_register_address(
                            family, {2, 3}, LOOM_XDNA_REGISTER_MODULE_SHIM_NOC,
                            0x1D014, &address));
}

}  // namespace
