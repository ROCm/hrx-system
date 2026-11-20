// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/worker_upload.h"

#include <array>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/dma_plan.h"

namespace {

class WorkerUploadTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    iree_arena_initialize(&pool_, &arena_);
    family_ = loom_xdna_npu2_array_family();
    const size_t places = loom_xdna_array_physical_tile_count(family_);
    occupancy_.resize(places);
    route_counts_.resize(places);
    plan_.routing.tile_route_counts = route_counts_.data();
    plan_.tiles = occupancy_.data();
    plan_.routing.family = family_;
    // The surrounding application owns the full interval between three
    // workers. Supply their linked programs in a different order from routing.
    for (uint16_t column = 1; column <= 5; ++column) {
      for (uint16_t row = 0; row <= 2; ++row) {
        const loom_xdna_tile_coordinate_t coordinate = {column, row};
        route_counts_[loom_xdna_array_physical_tile_index(family_,
                                                          coordinate)] = 1;
      }
    }
    const uint16_t columns[] = {5, 1, 3};
    for (size_t i = 0; i < workers_.size(); ++i) {
      const auto column = columns[i];
      auto& shim =
          occupancy_[loom_xdna_array_physical_tile_index(family_, {column, 0})];
      shim.mm2s_count = shim.descriptor_counts[0] = 1;
      const loom_xdna_tile_coordinate_t coordinate = {column, 2};
      occupancy_[loom_xdna_array_physical_tile_index(family_, coordinate)]
          .lock_count = 1;
      bytes_[i].resize(8192, 0x5a);
      sections_[i].storage = LOOM_NATIVE_SECTION_STORAGE_CONTENTS;
      sections_[i].access =
          LOOM_NATIVE_SECTION_ACCESS_READ | LOOM_NATIVE_SECTION_ACCESS_EXECUTE;
      sections_[i].contents =
          iree_make_const_byte_span(bytes_[i].data(), bytes_[i].size());
      linked_[i].assembly.sections = &sections_[i];
      linked_[i].assembly.section_count = 1;
      linked_[i].section_placements = &placement_;
      linked_[i].section_placement_count = 1;
      workers_[i].coordinate = coordinate;
      workers_[i].linked_tile = &linked_[i];
      records_[i].type = LOOM_AIE2P_PROGRAM_RECORD_TILE_PROGRAM_LOAD;
      records_[i].value.tile_program_load.tile_program_index = i;
    }
    placement_.memory_space = LOOM_XDNA_MEMORY_SPACE_PROGRAM;
    initial_.array_records = records_.data();
    initial_.array_record_count = records_.size();
  }

  void TearDown() override {
    iree_arena_deinitialize(&arena_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  void Install() {
    IREE_ASSERT_OK(loom_aie2p_worker_upload_install(
        &plan_, workers_.data(), workers_.size(), workers_.size(), &arena_,
        &initial_));
  }

  iree_arena_block_pool_t pool_;
  iree_arena_allocator_t arena_;
  const loom_xdna_array_family_t* family_ = nullptr;
  loom_aie2p_execution_plan_t plan_ = {};
  std::vector<loom_aie2p_execution_tile_t> occupancy_;
  std::vector<iree_host_size_t> route_counts_;
  loom_aie2p_linked_section_placement_t placement_ = {};
  std::array<std::vector<uint8_t>, 3> bytes_;
  std::array<loom_native_section_t, 3> sections_ = {};
  std::array<loom_aie2p_linked_tile_t, 3> linked_ = {};
  std::array<loom_aie2p_xdna_tile_t, 3> workers_ = {};
  std::array<loom_aie2p_program_record_t, 3> records_ = {};
  loom_aie2p_array_program_t initial_ = {};
};

TEST_F(WorkerUploadTest, EqualSparseWorkersShareOneDescriptor) {
  Install();
  ASSERT_NE(initial_.upload, nullptr);
  const auto& upload = *initial_.upload;
  ASSERT_EQ(upload.patch_count, 1u);
  const auto& patch = upload.patches[0];
  EXPECT_EQ(patch.source_offset, 0u);
  // Four data words and two headers per 16-byte block, then the arrival credit.
  EXPECT_EQ(patch.byte_length, (bytes_[0].size() / 16 * 6 + 3) * 4);
  const auto& record = upload.records[patch.record_index];
  ASSERT_EQ(record.type, LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32);
  EXPECT_EQ(
      record.value.register_block_write32.address >> family_->column_shift, 1u);
  EXPECT_EQ(record.value.register_block_write32.words[0],
            patch.byte_length / 4);
  EXPECT_EQ(
      iree_unaligned_load_le_u32(upload.contents.data + patch.byte_length - 4),
      1u);
  for (size_t i = 0; i < workers_.size(); ++i) {
    EXPECT_TRUE(upload.worker_entries[i]);
  }
  EXPECT_EQ(initial_.array_records, records_.data());
  EXPECT_EQ(initial_.array_record_count, records_.size());
}

TEST_F(WorkerUploadTest, DrainsBroadcastBeforeRepairAndPreservesCatalogSource) {
  bytes_[0][4] ^= 0xff;
  // A second immutable source will use the same shim descriptor after the
  // worker upload. Its absolute byte offset and command order must survive.
  std::array<loom_aie2p_program_record_t, 4> original = {
      records_[0], records_[1], records_[2]};
  const uint32_t descriptor[8] = {1, 0, 0, 0, 0, 0, 0, 1};
  const uint16_t index = 0;
  const auto shim_descriptor =
      LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_DMA_BD_WORD0_BUFFER_LENGTH;
  const uint32_t address =
      static_cast<uint32_t>(loom_xdna_register_field_address(
          family_, shim_descriptor, {1, 0}, &index));
  original[3].type = LOOM_AIE2P_PROGRAM_RECORD_REGISTER_BLOCK_WRITE32;
  original[3].value.register_block_write32 = {address, descriptor, 8};
  const uint8_t contents[] = {1, 2, 3, 4};
  const loom_aie2p_program_upload_patch_t old_patch = {3, 0, sizeof(contents)};
  const loom_aie2p_program_upload_t catalog = {
      .records = original.data(),
      .record_count = original.size(),
      .contents = iree_make_const_byte_span(contents, sizeof(contents)),
      .patches = &old_patch,
      .patch_count = 1,
  };
  initial_.upload = &catalog;
  Install();
  ASSERT_NE(initial_.upload, &catalog);
  const auto& upload = *initial_.upload;
  ASSERT_EQ(upload.patch_count, 3u);
  EXPECT_EQ(upload.patches[0].source_offset,
            LOOM_AIE2P_PROGRAM_UPLOAD_ALIGNMENT);
  const auto& repair = upload.patches[1];
  EXPECT_EQ(repair.byte_length, (2 + 4 + 3) * 4u);
  const uint8_t* patch_data = upload.contents.data + repair.source_offset;
  EXPECT_EQ(patch_data[8 + 4], bytes_[0][4]);
  EXPECT_EQ(iree_unaligned_load_le_u32(patch_data + repair.byte_length - 4),
            2u);
  const auto& successor = upload.patches[2];
  EXPECT_EQ(successor.record_index, upload.record_count - 1);
  EXPECT_EQ(successor.source_offset, 0u);
  EXPECT_EQ(
      upload.records[successor.record_index].value.register_block_write32.words,
      descriptor);
  EXPECT_EQ(iree_unaligned_load_le_u32(upload.contents.data), 0x04030201u);

  std::vector<uint32_t> waited;
  for (size_t i = upload.patches[0].record_index; i < repair.record_index;
       ++i) {
    const auto& record = upload.records[i];
    if (record.type == LOOM_AIE2P_PROGRAM_RECORD_REGISTER_MASK_WAIT32) {
      const auto& wait = record.value.register_mask_wait32;
      waited.push_back(wait.address);
      EXPECT_EQ(wait.value, waited.size() <= workers_.size() ? 1u : 0u);
    }
  }
  std::vector<uint32_t> expected;
  for (uint16_t column : {1, 3, 5}) {
    expected.push_back(static_cast<uint32_t>(loom_xdna_register_field_address(
        family_, loom_aie2p_compute_memory.lock_value, {column, 2}, &index)));
  }
  expected.push_back(static_cast<uint32_t>(
      loom_aie2p_native_queue_status(family_, {1, 0}, nullptr, 0,
                                     LOOM_XDNA_DMA_DIRECTION_MEMORY_TO_STREAM)
          .address));
  EXPECT_EQ(waited, expected);
}

TEST_F(WorkerUploadTest, KeepsInlineLoadsWhenTransferCostsMore) {
  for (auto& section : sections_) {
    section.contents.data_length = 16;
  }
  Install();
  EXPECT_EQ(initial_.upload, nullptr);
  EXPECT_EQ(initial_.array_records, records_.data());
}

TEST_F(WorkerUploadTest, KeepsInlineLoadsWhenIntermediateSwitchIsUnowned) {
  route_counts_[loom_xdna_array_physical_tile_index(family_, {2, 2})] = 0;
  Install();
  EXPECT_EQ(initial_.upload, nullptr);
  EXPECT_EQ(initial_.array_records, records_.data());
}

}  // namespace
