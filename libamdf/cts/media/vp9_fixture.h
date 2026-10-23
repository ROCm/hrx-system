// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_MEDIA_VP9_FIXTURE_H_
#define AMDF_CTS_MEDIA_VP9_FIXTURE_H_

#include <array>

#include "amdf/media/vp9.h"

namespace amdf::cts::media {

struct Vp9FixturePicture {
  // Complete picture bytes borrowed from the embedded fixture.
  const uint8_t* bytes;
  // Compressed extent including native alignment padding.
  uint32_t byte_length;
  // Stable application-selected reconstructed output slot.
  uint32_t reference_slot;
  // Independent FNV-1a over each raster pixel's Y/U/V storage bytes.
  uint64_t pixel_hash;
  // Resolved header and retained segmentation/filter state.
  amdf_media_vp9_decode_picture_info_t picture;
  // Unique physical reference slots; callers bind the frame pointers.
  amdf_media_reference_t references[8];
};

struct Vp9Fixture {
  // Encoded sample precision, eight or ten bits.
  uint32_t depth;
  // Exact coded width in pixels.
  uint32_t width;
  // Exact coded height in pixels.
  uint32_t height;
  // Two keyframes and ten interframes with adaptive contexts and segmentation.
  std::array<Vp9FixturePicture, 12> pictures;
};

// Loads resolved syntax and independent sample fingerprints by embedded name.
// Pointers borrow embedded data and |fixture|; reload after moving it.
void LoadVp9Fixture(const char* filename, Vp9Fixture* fixture);

}  // namespace amdf::cts::media

#endif  // AMDF_CTS_MEDIA_VP9_FIXTURE_H_
