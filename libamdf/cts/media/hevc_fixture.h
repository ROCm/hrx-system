// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_MEDIA_HEVC_FIXTURE_H_
#define AMDF_CTS_MEDIA_HEVC_FIXTURE_H_

#include <array>

#include "amdf/media/hevc.h"

namespace amdf::cts::media {

struct HevcFixturePicture {
  // Padded Annex B slice bytes borrowed from the embedded fixture.
  const uint8_t* bytes;
  // Compressed extent including native alignment padding.
  uint32_t byte_length;
  // Stable application-selected reconstructed output slot.
  uint32_t reference_slot;
  // Independent FNV-1a over each raster pixel's Y/U/V component bytes; P010
  // includes both little-endian bytes with samples in the upper ten bits.
  uint64_t pixel_hash;
  // Complete parsed picture metadata, borrowing this fixture's arrays.
  amdf_media_hevc_decode_picture_info_t picture;
  // Stable slots and POCs; callers bind the corresponding frame pointers.
  amdf_media_hevc_reference_t references[4];
  // One complete padded slice segment.
  amdf_media_hevc_slice_t slice;
};

struct HevcFixture {
  // Resolved nonuniform raster grids shared by all twelve pictures.
  amdf_media_hevc_scaling_matrices_t matrices;
  // Reordered pictures with independently decoded sample fingerprints.
  std::array<HevcFixturePicture, 12> pictures;
};

// Loads twelve reordered 256x200 Main/Main10 pictures. The partial coding
// block at the bottom exercises distinct visible, output, and reference
// extents. Pointers borrow embedded bytes and |fixture|; moving the fixture
// requires reloading those pointers.
void LoadHevcFixture(uint32_t depth, HevcFixture* fixture);

}  // namespace amdf::cts::media

#endif  // AMDF_CTS_MEDIA_HEVC_FIXTURE_H_
