// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/av1_fixture.h"

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/media/av1_inputs.h"

namespace amdf::cts::media {

void LoadAv1Fixture(const char* filename, Av1Fixture* fixture) {
  const iree_file_toc_t* selected = nullptr;
  for (const auto& entry : std::span(amdf_cts_media_av1_inputs_create(),
                                     amdf_cts_media_av1_inputs_size())) {
    if (std::strcmp(entry.name, filename) == 0) {
      selected = &entry;
    }
  }
  ASSERT_NE(selected, nullptr);
  const auto* cursor = reinterpret_cast<const uint8_t*>(selected->data);
  const auto* end = cursor + selected->size;
  ASSERT_GE(end - cursor, 24);
  ASSERT_EQ(std::memcmp(cursor, "AV1CTSF1", 8), 0);
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
    // Immutable offline syntax is serialized field-by-field, independent of
    // host structure packing, pointers, and native provider representations.
    ASSERT_GE(end - cursor, 859);
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.sequence.use_128x128_superblock = word();
    picture.sequence.filter_intra = word();
    picture.sequence.intra_edge_filter = word();
    picture.sequence.interintra_compound = word();
    picture.sequence.masked_compound = word();
    picture.sequence.dual_filter = word();
    picture.sequence.joint_compound = word();
    picture.sequence.reference_frame_motion_vectors = word();
    picture.sequence.cdef = word();
    picture.sequence.restoration = word();
    picture.sequence.film_grain = word();
    picture.sequence.order_hint_bit_count = word();
    picture.frame_type = word();
    picture.show_frame = word();
    picture.disable_cdf_update = word();
    picture.disable_frame_end_cdf_update = word();
    picture.warped_motion = word();
    picture.screen_content_tools = word();
    picture.intra_block_copy = word();
    picture.high_precision_motion_vectors = word();
    picture.switchable_motion_mode = word();
    picture.reference_select = word();
    picture.skip_mode = word();
    picture.reduced_transform_set = word();
    picture.transform_mode = word();
    picture.use_reference_motion_vectors = word();
    picture.loop_filter.sharpness = word();
    picture.loop_filter.deltas_enabled = word();
    picture.loop_filter.deltas_updated = word();
    picture.loop_filter.delta_present = word();
    picture.loop_filter.delta_resolution = word();
    picture.loop_filter.delta_multi = word();
    picture.quantization.base_index = word();
    picture.quantization.y_dc_delta = static_cast<int32_t>(word());
    picture.quantization.delta_present = word();
    picture.quantization.delta_resolution = word();
    picture.cdef.damping_minus3 = word();
    picture.cdef.bits = word();
    picture.segmentation.enabled = word();
    picture.segmentation.update_map = word();
    picture.segmentation.temporal_update = word();
    picture.segmentation.update_data = word();
    picture.film_grain.apply = word();
    picture.film_grain.seed = word();
    picture.film_grain.luma_point_count = word();
    picture.film_grain.u_point_count = word();
    picture.film_grain.v_point_count = word();
    picture.film_grain.scaling_shift_minus8 = word();
    picture.film_grain.chroma_scaling_from_luma = word();
    picture.film_grain.coefficient_lag = word();
    picture.film_grain.coefficient_shift_minus6 = word();
    picture.film_grain.scale_shift = word();
    picture.film_grain.overlap = word();
    picture.film_grain.clip_to_restricted_range = word();
    picture.showable_frame = word();
    picture.primary_reference = word();
    picture.order_hint = word();
    picture.integer_motion_vectors = word();
    picture.interpolation_filter = word();
    picture.superres_denominator = word();
    picture.tile_column_count = word();
    picture.tile_row_count = word();
    picture.context_update_tile = word();
    copy(picture.loop_filter.levels, sizeof(picture.loop_filter.levels));
    copy(picture.loop_filter.reference_deltas,
         sizeof(picture.loop_filter.reference_deltas));
    copy(picture.loop_filter.mode_deltas,
         sizeof(picture.loop_filter.mode_deltas));
    copy(picture.loop_filter.restoration_types,
         sizeof(picture.loop_filter.restoration_types));
    copy(picture.loop_filter.log2_restoration_unit_sizes,
         sizeof(picture.loop_filter.log2_restoration_unit_sizes));
    copy(picture.quantization.chroma_dc_deltas,
         sizeof(picture.quantization.chroma_dc_deltas));
    copy(picture.quantization.chroma_ac_deltas,
         sizeof(picture.quantization.chroma_ac_deltas));
    copy(picture.quantization.matrix_levels,
         sizeof(picture.quantization.matrix_levels));
    copy(picture.cdef.luma_primary, sizeof(picture.cdef.luma_primary));
    copy(picture.cdef.luma_secondary, sizeof(picture.cdef.luma_secondary));
    copy(picture.cdef.chroma_primary, sizeof(picture.cdef.chroma_primary));
    copy(picture.cdef.chroma_secondary, sizeof(picture.cdef.chroma_secondary));
    copy(picture.segmentation.feature_masks,
         sizeof(picture.segmentation.feature_masks));
    copy(picture.segmentation.feature_values,
         sizeof(picture.segmentation.feature_values));
    copy(picture.film_grain.luma_points,
         sizeof(picture.film_grain.luma_points));
    copy(picture.film_grain.u_points, sizeof(picture.film_grain.u_points));
    copy(picture.film_grain.v_points, sizeof(picture.film_grain.v_points));
    copy(picture.film_grain.luma_coefficients,
         sizeof(picture.film_grain.luma_coefficients));
    copy(picture.film_grain.u_coefficients,
         sizeof(picture.film_grain.u_coefficients));
    copy(picture.film_grain.v_coefficients,
         sizeof(picture.film_grain.v_coefficients));
    copy(picture.film_grain.chroma_multipliers,
         sizeof(picture.film_grain.chroma_multipliers));
    copy(picture.film_grain.chroma_luma_multipliers,
         sizeof(picture.film_grain.chroma_luma_multipliers));
    copy(picture.film_grain.chroma_offsets,
         sizeof(picture.film_grain.chroma_offsets));
    entry.reference_slot = word();
    picture.reference_count = word();
    ASSERT_LT(entry.reference_slot, 9u);
    ASSERT_LE(picture.reference_count, 8u);
    for (auto& reference : entry.references) {
      reference.slot_index = *cursor++;
    }
    picture.references = entry.references;
    copy(picture.reference_map, 8);
    copy(picture.active_reference_indices, 7);
    copy(picture.reference_sign_bias, 7);
    for (auto& motion : picture.global_motion) {
      motion.type = word();
      copy(motion.parameters, sizeof(motion.parameters));
      motion.invalid = word();
    }
    ASSERT_LE(picture.tile_column_count, 64u);
    ASSERT_LE(picture.tile_row_count, 64u);
    const uint32_t tile_count =
        picture.tile_column_count * picture.tile_row_count;
    ASSERT_GE(tile_count, 1u);
    ASSERT_LE(tile_count, 4u);
    ASSERT_GE(end - cursor,
              2 * (picture.tile_column_count + picture.tile_row_count + 2) +
                  tile_count * 8 + 12);
    copy(picture.tile_column_starts, (picture.tile_column_count + 1) * 2);
    copy(picture.tile_row_starts, (picture.tile_row_count + 1) * 2);
    for (auto& tile : std::span(entry.tiles, tile_count)) {
      tile.byte_offset = word();
      tile.byte_length = word();
    }
    picture.tiles = entry.tiles;
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
