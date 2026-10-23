// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_MEDIA_H264_FIXTURE_H_
#define AMDF_CTS_MEDIA_H264_FIXTURE_H_

#include <array>

#include "amdf/media/h264.h"

namespace amdf::cts::media {

// FNV-1a over each raster pixel's Y/U/V bytes from independent FFmpeg decode,
// verified against the encoder reconstruction. Callers separately clear reused
// output or check generation tags so stale pixels cannot satisfy the oracle.
inline constexpr uint64_t kH264PixelHashes[12] = {
    0x4f93364832b3f2d7, 0xdea0cbf06a7e24cd, 0x7e2ee8aa35fe6ce0,
    0xb160a8118523b73f, 0x5945a85843a5a234, 0xde9a0b9c42fd47bc,
    0x8193763875ecb1e9, 0x42798e3de4cf9fbc, 0x62edb9ad1a92162e,
    0x3330931192bf43b3, 0x17c4311a4e05c771, 0x1c443a7a7c988909};

struct H264FixturePicture {
  // Complete padded Annex B slice bytes retained by the embedded fixture.
  const uint8_t* bytes;
  // Native compressed extent, including alignment padding.
  uint32_t byte_length;
  // Application-selected reconstructed output slot.
  uint32_t reference_slot;
  // Complete public parsed picture metadata, with borrowed arrays below.
  amdf_media_h264_decode_picture_info_t picture;
  // Explicit physical reference-slot choices, independent of native indices.
  uint32_t reference_slots[3];
  // Parsed reference markings and POCs; frames are bound by the caller.
  amdf_media_h264_reference_t references[3];
  // One complete slice, already packed for the native byte framing.
  amdf_media_h264_slice_t slice;
};

// Loads the fixed parsed stream; pointers borrow embedded bytes and |pictures|.
void LoadH264Fixture(std::array<H264FixturePicture, 12>* pictures);

}  // namespace amdf::cts::media

#endif  // AMDF_CTS_MEDIA_H264_FIXTURE_H_
