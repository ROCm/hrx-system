// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/h264_profile_fixture.h"

#include <cstring>
#include <span>

#include "gtest/gtest.h"
#include "libamdf/cts/media/h264_profile_inputs.h"

namespace amdf::cts::media {

void LoadH264ProfileFixture(amdf_media_h264_profile_t profile,
                            H264ProfileFixture* fixture) {
  ASSERT_TRUE(profile == AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE ||
              profile == AMDF_MEDIA_H264_PROFILE_HIGH);
  const bool high = profile == AMDF_MEDIA_H264_PROFILE_HIGH;
  const char* filename =
      high ? "h264_high.slice" : "h264_constrained_baseline.slice";
  const iree_file_toc_t* selected = nullptr;
  for (const auto& entry :
       std::span(amdf_cts_media_h264_profile_inputs_create(),
                 amdf_cts_media_h264_profile_inputs_size())) {
    if (std::strcmp(entry.name, filename) == 0) {
      selected = &entry;
    }
  }
  ASSERT_NE(selected, nullptr);
  ASSERT_GT(selected->size, 0u);
  ASSERT_LE(selected->size, 65536u);
  ASSERT_EQ(selected->size % 128, 0u);

  *fixture = {};
  fixture->configuration.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION;
  fixture->configuration.structure_size = sizeof(fixture->configuration);
  fixture->configuration.profile = profile;
  fixture->configuration.level_idc = 13;
  std::memset(&fixture->matrices, 16, sizeof(fixture->matrices));

  auto& picture = fixture->picture;
  picture.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO;
  picture.structure_size = sizeof(picture);
  picture.sequence.picture_order_count_type = 2;
  picture.sequence.direct_8x8_inference = 1;
  picture.picture.entropy_coding_mode = high ? 1u : 0u;
  picture.picture.initial_qp_minus26 = -6;
  picture.picture.chroma_qp_index_offset = -2;
  picture.picture.second_chroma_qp_index_offset = -2;
  picture.picture.deblocking_filter_control_present = 1;
  picture.picture.transform_8x8_mode = high ? 1u : 0u;
  picture.scaling_matrices = &fixture->matrices;
  picture.is_reference = 1;
  picture.is_intra = 1;
  fixture->slice = {0, static_cast<uint32_t>(selected->size)};
  picture.slice_count = 1;
  picture.slices = &fixture->slice;

  fixture->bytes = reinterpret_cast<const uint8_t*>(selected->data);
  fixture->byte_length = static_cast<uint32_t>(selected->size);
  fixture->pixel_hash =
      high ? UINT64_C(0x6df67afa646b57df) : UINT64_C(0x7081551a20d97854);
}

}  // namespace amdf::cts::media
