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

  const loom_xdna_tile_facts_t* tile = nullptr;
  IREE_ASSERT_OK(loom_xdna_array_tile_facts(family, {0, 0}, &tile));
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_SHIM_NOC);
  EXPECT_EQ(tile->dma.buffer_descriptor_count, 16u);
  IREE_ASSERT_OK(loom_xdna_array_tile_facts(family, {7, 1}, &tile));
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_MEMORY);
  EXPECT_EQ(tile->memory.local_capacity, 512u * 1024u);
  EXPECT_EQ(tile->dma.buffer_descriptor_count, 48u);
  IREE_ASSERT_OK(loom_xdna_array_tile_facts(family, {3, 5}, &tile));
  EXPECT_EQ(tile->kind, LOOM_XDNA_TILE_KIND_COMPUTE);
  EXPECT_EQ(tile->memory.local_capacity, 64u * 1024u);
  EXPECT_EQ(tile->memory.program_capacity, 16u * 1024u);
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

TEST(XdnaArrayFactsTest, FormsLoadAddressesFromCanonicalStorage) {
  const loom_xdna_array_family_t* family = loom_xdna_npu2_array_family();
  uint32_t address = 0;
  IREE_ASSERT_OK(loom_xdna_array_form_load_address(
      family, {1, 3}, LOOM_XDNA_MEMORY_SPACE_DATA, {0, 3}, 0x40, 64, &address));
  EXPECT_EQ(address, 0x50040u);
  IREE_ASSERT_OK(loom_xdna_array_form_load_address(
      family, {0, 3}, LOOM_XDNA_MEMORY_SPACE_DATA, {0, 3}, 0x40, 64, &address));
  EXPECT_EQ(address, 0x70040u);
  IREE_ASSERT_OK(loom_xdna_array_form_load_address(
      family, {4, 2}, LOOM_XDNA_MEMORY_SPACE_PROGRAM, {4, 2}, 0x80, 64,
      &address));
  EXPECT_EQ(address, 0x80u);

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_xdna_array_form_load_address(
                            family, {0, 3}, LOOM_XDNA_MEMORY_SPACE_DATA, {1, 3},
                            0, 64, &address));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_xdna_array_form_load_address(
                            family, {4, 2}, LOOM_XDNA_MEMORY_SPACE_PROGRAM,
                            {4, 3}, 0, 64, &address));
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
