// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_MEDIA_H264_PROFILE_FIXTURE_H_
#define AMDF_CTS_MEDIA_H264_PROFILE_FIXTURE_H_

#include "amdf/media/h264.h"

namespace amdf::cts::media {

struct H264ProfileFixture {
  // Exact profile and level admitted by the native session.
  amdf_media_h264_configuration_t configuration;
  // Flat resolved scaling grids borrowed by picture.
  amdf_media_h264_scaling_matrices_t matrices;
  // Complete parsed IDR metadata, borrowing matrices and slice.
  amdf_media_h264_decode_picture_info_t picture;
  // One complete padded Annex B IDR slice.
  amdf_media_h264_slice_t slice;
  // Padded slice bytes borrowed from the embedded fixture.
  const uint8_t* bytes;
  // Compressed extent including native alignment padding.
  uint32_t byte_length;
  // Independent FNV-1a over every raster pixel's Y/U/V storage bytes.
  uint64_t pixel_hash;
};

// Loads a constrained-baseline or high 256x256 IDR picture. Pointers borrow
// embedded data and |fixture|; moving the fixture requires reloading them.
void LoadH264ProfileFixture(amdf_media_h264_profile_t profile,
                            H264ProfileFixture* fixture);

}  // namespace amdf::cts::media

#endif  // AMDF_CTS_MEDIA_H264_PROFILE_FIXTURE_H_
