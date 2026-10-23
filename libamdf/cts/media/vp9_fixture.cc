// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/vp9_fixture.h"

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/media/vp9_inputs.h"

namespace amdf::cts::media {

void LoadVp9Fixture(const char* filename, Vp9Fixture* fixture) {
  const iree_file_toc_t* selected = nullptr;
  for (const auto& entry : std::span(amdf_cts_media_vp9_inputs_create(),
                                     amdf_cts_media_vp9_inputs_size())) {
    if (std::strcmp(entry.name, filename) == 0) {
      selected = &entry;
    }
  }
  ASSERT_NE(selected, nullptr);
  const auto* cursor = reinterpret_cast<const uint8_t*>(selected->data);
  const auto* end = cursor + selected->size;
  ASSERT_GE(end - cursor, 24);
  ASSERT_EQ(std::memcmp(cursor, "VP9CTSF1", 8), 0);
  cursor += 8;
  auto word = [&]() {
    uint32_t value;
    std::memcpy(&value, cursor, 4);
    cursor += 4;
    return value;
  };
  auto copy = [&](void* destination, size_t size) {
    std::memcpy(destination, cursor, size);
    cursor += size;
  };
  fixture->depth = word();
  fixture->width = word();
  fixture->height = word();
  ASSERT_EQ(word(), 12u);
  for (auto& entry : fixture->pictures) {
    // Fixed parsed metadata precedes the independently encoded payload.
    ASSERT_GE(end - cursor, 242);
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_VP9_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.frame_type = word();
    picture.show_frame = word();
    picture.error_resilient = word();
    picture.intra_only = word();
    picture.high_precision_motion_vectors = word();
    picture.refresh_frame_context = word();
    picture.parallel_decoding = word();
    picture.frame_context_index = word();
    picture.reset_frame_context = word();
    picture.base_quantizer_index = word();
    picture.y_dc_quantizer_delta = static_cast<int32_t>(word());
    picture.uv_dc_quantizer_delta = static_cast<int32_t>(word());
    picture.uv_ac_quantizer_delta = static_cast<int32_t>(word());
    picture.log2_tile_columns = word();
    picture.log2_tile_rows = word();
    picture.loop_filter.level = word();
    picture.loop_filter.sharpness = word();
    picture.loop_filter.deltas_enabled = word();
    picture.loop_filter.deltas_updated = word();
    picture.segmentation.enabled = word();
    picture.segmentation.update_map = word();
    picture.segmentation.temporal_update = word();
    picture.segmentation.update_data = word();
    picture.use_previous_frame_motion_vectors = word();
    picture.interpolation_filter = word();
    picture.uncompressed_header_byte_length = word();
    picture.compressed_header_byte_length = word();
    picture.segmentation.absolute_values = word();
    copy(picture.loop_filter.reference_deltas, 4);
    copy(picture.loop_filter.mode_deltas, 2);
    copy(picture.segmentation.tree_probabilities, 7);
    copy(picture.segmentation.prediction_probabilities, 3);
    copy(picture.segmentation.feature_masks, 8);
    copy(picture.segmentation.feature_values, 64);
    if (!picture.segmentation.enabled) {
      // Disabled segmentation may retain parser state. Nonzero filter features
      // must not change the independently decoded pixels while it is disabled.
      for (uint32_t segment = 0; segment < 8; ++segment) {
        picture.segmentation.feature_masks[segment] = 2;
        picture.segmentation.feature_values[segment][1] = 63;
      }
    }
    entry.reference_slot = word();
    picture.reference_count = word();
    ASSERT_LT(entry.reference_slot, 9u);
    ASSERT_LE(picture.reference_count, 8u);
    for (uint32_t i = 0; i < 8; ++i) {
      entry.references[i].slot_index = *cursor++;
    }
    picture.references = entry.references;
    copy(picture.reference_map, 8);
    copy(picture.active_reference_indices, 3);
    copy(picture.reference_sign_bias, 3);
    entry.byte_length = word();
    const uint64_t low = word(), high = word();
    entry.pixel_hash = low | (high << 32);
    ASSERT_GE(end - cursor, entry.byte_length);
    ASSERT_LE(entry.byte_length, 65536u);
    entry.bytes = cursor;
    cursor += entry.byte_length;
  }
  ASSERT_EQ(cursor, end);
}

}  // namespace amdf::cts::media
