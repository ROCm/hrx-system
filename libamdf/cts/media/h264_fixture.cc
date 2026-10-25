// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/h264_fixture.h"

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/media/h264_geometry_inputs.h"
#include "libamdf/cts/media/reference_main.h"

namespace amdf::cts::media {

void LoadH264Fixture(std::array<H264FixturePicture, 12>* pictures) {
  const auto& file = amdf_cts_media_reference_main_create()[0];
  const auto* cursor = reinterpret_cast<const uint8_t*>(file.data);
  ASSERT_GE(file.size, 68u);
  ASSERT_EQ(std::memcmp(cursor, "H264REFS", 8), 0);
  cursor += 8;
  auto word = [&]() {
    uint32_t value;
    std::memcpy(&value, cursor, 4);
    cursor += 4;
    return value;
  };
  ASSERT_EQ(word(), pictures->size());
  ASSERT_EQ(word(), 256u);
  ASSERT_EQ(word(), 256u);
  const uint32_t frame_bits = word(), direct8 = word(), deblock = word();
  const uint32_t constrained = word(), level = word();
  const int32_t qp = static_cast<int32_t>(word());
  const int32_t qs = static_cast<int32_t>(word());
  const int32_t chroma = static_cast<int32_t>(word());
  const uint32_t poc_bits = word(), maximum_references = word();
  const uint32_t l0 = word(), l1 = word();
  ASSERT_EQ(level, 13u);
  for (uint32_t index = 0; index < pictures->size(); ++index) {
    ASSERT_LE(cursor + 72,
              reinterpret_cast<const uint8_t*>(file.data) + file.size);
    auto& entry = (*pictures)[index];
    ASSERT_EQ(word(), index);
    entry.byte_length = word();
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.frame_number = word();
    picture.top_field_order_count = static_cast<int32_t>(word());
    picture.bottom_field_order_count = picture.top_field_order_count;
    entry.reference_slot = word();
    const uint32_t display_index = word();
    ASSERT_EQ(display_index * 2, picture.top_field_order_count);
    picture.is_reference = word();
    picture.is_intra = word();
    picture.reference_count = word();
    ASSERT_LE(picture.reference_count, 3u);
    for (uint32_t reference = 0; reference < 3; ++reference) {
      entry.reference_slots[reference] = word();
      entry.references[reference].frame_number = word();
      entry.references[reference].top_field_order_count =
          static_cast<int32_t>(word());
      entry.references[reference].bottom_field_order_count =
          entry.references[reference].top_field_order_count;
    }
    picture.sequence.maximum_reference_frame_count = maximum_references;
    picture.sequence.log2_max_frame_num_minus4 = frame_bits;
    picture.sequence.log2_max_picture_order_count_lsb_minus4 = poc_bits;
    picture.sequence.direct_8x8_inference = direct8;
    picture.picture.default_reference_l0_count_minus1 = l0;
    picture.picture.default_reference_l1_count_minus1 = l1;
    picture.picture.initial_qp_minus26 = qp;
    picture.picture.initial_qs_minus26 = qs;
    picture.picture.chroma_qp_index_offset = chroma;
    picture.picture.second_chroma_qp_index_offset = chroma;
    picture.picture.deblocking_filter_control_present = deblock;
    picture.picture.constrained_intra_prediction = constrained;
    entry.slice = {0, entry.byte_length};
    picture.slice_count = 1;
    picture.slices = &entry.slice;
    picture.references = entry.references;
    entry.pixel_hash = kH264PixelHashes[index];
    ASSERT_LE(entry.byte_length, 65536u);
    ASSERT_LE(cursor + entry.byte_length,
              reinterpret_cast<const uint8_t*>(file.data) + file.size);
    entry.bytes = cursor;
    cursor += entry.byte_length;
  }
  ASSERT_EQ(cursor, reinterpret_cast<const uint8_t*>(file.data) + file.size);
}

void LoadH264GeometryFixture(std::array<H264FixturePicture, 2>* pictures,
                             amdf_media_h264_configuration_t* configuration,
                             amdf_media_h264_scaling_matrices_t* matrices) {
  const iree_file_toc_t* inputs[2] = {};
  for (const auto& entry :
       std::span(amdf_cts_media_h264_geometry_inputs_create(),
                 amdf_cts_media_h264_geometry_inputs_size())) {
    if (std::strcmp(entry.name, "h264_geometry_idr.slice") == 0) {
      inputs[0] = &entry;
    } else if (std::strcmp(entry.name, "h264_geometry_p.slice") == 0) {
      inputs[1] = &entry;
    }
  }
  ASSERT_NE(inputs[0], nullptr);
  ASSERT_NE(inputs[1], nullptr);

  *configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
      .structure_size = sizeof(*configuration),
      .profile = AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE,
      .level_idc = 11,
  };
  std::memset(matrices, 16, sizeof(*matrices));
  constexpr uint64_t kPixelHashes[] = {UINT64_C(0xa653e999eb11e8cb),
                                       UINT64_C(0x44b0ab355631c6c2)};
  for (uint32_t index = 0; index < pictures->size(); ++index) {
    ASSERT_GT(inputs[index]->size, 0u);
    ASSERT_LE(inputs[index]->size, 65536u);
    ASSERT_EQ(inputs[index]->size % 128, 0u);
    auto& entry = (*pictures)[index];
    entry.bytes = reinterpret_cast<const uint8_t*>(inputs[index]->data);
    entry.byte_length = static_cast<uint32_t>(inputs[index]->size);
    entry.reference_slot = index;
    entry.slice = {0, entry.byte_length};
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.sequence.maximum_reference_frame_count = 1;
    picture.sequence.log2_max_frame_num_minus4 = 11;
    picture.sequence.picture_order_count_type = 2;
    picture.picture.deblocking_filter_control_present = 1;
    picture.scaling_matrices = matrices;
    picture.frame_number = index;
    picture.is_reference = 1;
    picture.is_intra = index == 0;
    picture.top_field_order_count = static_cast<int32_t>(index * 2);
    picture.bottom_field_order_count = picture.top_field_order_count;
    picture.slice_count = 1;
    picture.slices = &entry.slice;
    picture.reference_count = index;
    if (index != 0) {
      entry.reference_slots[0] = 0;
      entry.references[0].frame_number = 0;
      entry.references[0].top_field_order_count = 0;
      entry.references[0].bottom_field_order_count = 0;
      picture.references = entry.references;
    }
    entry.pixel_hash = kPixelHashes[index];
  }
}

}  // namespace amdf::cts::media
