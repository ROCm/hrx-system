// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/hevc_fixture.h"

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/media/hevc_inputs.h"

namespace amdf::cts::media {
namespace {

// HEVC's default 8x8 coefficient grids, in raster order. The nonuniform grids
// exercise scaling-list delivery against independently reconstructed pixels.
constexpr uint8_t kDefaultScaling[2][64] = {
    {16, 16, 16, 16, 17, 18, 21, 24, 16, 16, 16, 16, 17, 19, 22, 25,
     16, 16, 17, 18, 20, 22, 25, 29, 16, 16, 18, 21, 24, 27, 31, 36,
     17, 17, 20, 24, 30, 35, 41, 47, 18, 19, 22, 27, 35, 44, 54, 65,
     21, 22, 25, 31, 41, 54, 70, 88, 24, 25, 29, 36, 47, 65, 88, 115},
    {16, 16, 16, 16, 17, 18, 20, 24, 16, 16, 16, 17, 18, 20, 24, 25,
     16, 16, 17, 18, 20, 24, 25, 28, 16, 17, 18, 20, 24, 25, 28, 33,
     17, 18, 20, 24, 25, 28, 33, 41, 18, 20, 24, 25, 28, 33, 41, 54,
     20, 24, 25, 28, 33, 41, 54, 71, 24, 25, 28, 33, 41, 54, 71, 91},
};

}  // namespace

void LoadHevcFixture(uint32_t depth, HevcFixture* fixture) {
  const char* filename = depth == 8 ? "hevc8.packet" : "hevc10.packet";
  const iree_file_toc_t* selected = nullptr;
  for (const auto& entry : std::span(amdf_cts_media_hevc_inputs_create(),
                                     amdf_cts_media_hevc_inputs_size())) {
    if (std::strcmp(entry.name, filename) == 0) {
      selected = &entry;
    }
  }
  ASSERT_NE(selected, nullptr);
  const auto& file = *selected;
  const auto* cursor = reinterpret_cast<const uint8_t*>(file.data);
  const auto* end = cursor + file.size;
  ASSERT_GE(file.size, 24u);
  ASSERT_EQ(std::memcmp(cursor, "H265CTSF", 8), 0);
  cursor += 8;
  auto word = [&]() {
    uint32_t value;
    std::memcpy(&value, cursor, 4);
    cursor += 4;
    return value;
  };
  ASSERT_EQ(word(), depth);
  ASSERT_EQ(word(), 256u);
  ASSERT_EQ(word(), 200u);
  ASSERT_EQ(word(), 12u);
  auto& matrices = fixture->matrices;
  std::memset(&matrices, 16, sizeof(matrices));
  for (uint32_t index = 0; index < 6; ++index) {
    std::memcpy(matrices.scaling_8x8[index], kDefaultScaling[index / 3], 64);
    std::memcpy(matrices.scaling_16x16[index], kDefaultScaling[index / 3], 64);
  }
  for (uint32_t index = 0; index < 2; ++index) {
    std::memcpy(matrices.scaling_32x32[index], kDefaultScaling[index], 64);
  }

  for (uint32_t index = 0; index < 12; ++index) {
    ASSERT_GE(end - cursor, 104);
    auto& entry = fixture->pictures[index];
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.sequence.maximum_decoded_picture_buffer_count_minus1 = 4;
    picture.sequence.log2_max_picture_order_count_lsb_minus4 = 4;
    picture.sequence.log2_diff_max_min_luma_coding_block_size = 3;
    picture.sequence.log2_diff_max_min_luma_transform_block_size = 3;
    picture.sequence.scaling_list_enabled = 1;
    picture.sequence.sample_adaptive_offset_enabled = 1;
    picture.sequence.temporal_motion_vector_prediction_enabled = 1;
    picture.sequence.strong_intra_smoothing_enabled = 1;
    picture.picture.sign_data_hiding_enabled = 1;
    picture.picture.coding_unit_qp_delta_enabled = 1;
    picture.picture.coding_unit_qp_delta_depth = 1;
    picture.picture.weighted_prediction = 1;
    picture.picture.loop_filter_across_slices_enabled = 1;
    picture.scaling_matrices = &matrices;
    picture.picture_order_count = static_cast<int32_t>(word());
    const uint32_t slot = word();
    ASSERT_LT(slot, 5u);
    const uint32_t flags = word();
    picture.is_random_access = flags & 1u;
    picture.is_idr = (flags >> 1) & 1u;
    picture.is_intra = (flags >> 2) & 1u;
    picture.short_term_reference_set_bit_count = word();
    picture.reference_count = word();
    picture.short_term_before.count = word();
    picture.short_term_after.count = word();
    const uint32_t length = word();
    ASSERT_LE(picture.reference_count, 4u);
    ASSERT_LE(picture.short_term_before.count, 4u);
    ASSERT_LE(picture.short_term_after.count, 4u);
    ASSERT_LE(length, 65536u);
    for (uint32_t reference = 0; reference < 4; ++reference) {
      const uint32_t reference_slot = word();
      ASSERT_LT(reference_slot, 5u);
      entry.references[reference].binding.slot_index = reference_slot;

      entry.references[reference].picture_order_count =
          static_cast<int32_t>(word());
    }
    for (auto* set : {&picture.short_term_before, &picture.short_term_after}) {
      for (uint32_t reference = 0; reference < 4; ++reference) {
        set->indices[reference] = static_cast<uint8_t>(word());
      }
    }
    const uint64_t low = word(), high = word();
    ASSERT_GE(end - cursor, length);
    entry.slice = {0, length};
    picture.slice_count = 1;
    picture.slices = &entry.slice;
    picture.references = entry.references;
    entry.bytes = cursor;
    entry.byte_length = length;
    entry.reference_slot = slot;
    entry.pixel_hash = low | (high << 32);
    cursor += length;
  }
  ASSERT_EQ(cursor, end);
}

}  // namespace amdf::cts::media
