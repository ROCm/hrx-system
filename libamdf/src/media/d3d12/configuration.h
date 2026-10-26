// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_D3D12_CONFIGURATION_H_
#define AMDF_SRC_MEDIA_D3D12_CONFIGURATION_H_

#include <d3d12video.h>

#include "amdf/media/device.h"
#include "libamdf/src/media/encode.h"

namespace amdf::media::d3d12 {

struct Device;

// Self-contained native rate-control state for one temporal layer. The
// descriptor points to exactly one sibling payload selected by its mode.
struct EncodeRateControl {
  // Constant-QP payload used when descriptor selects CQP.
  D3D12_VIDEO_ENCODER_RATE_CONTROL_CQP cqp;
  // Constant-bit-rate payload used when descriptor selects CBR.
  D3D12_VIDEO_ENCODER_RATE_CONTROL_CBR cbr;
  // Variable-bit-rate payload used when descriptor selects VBR.
  D3D12_VIDEO_ENCODER_RATE_CONTROL_VBR vbr;
  // Type-erased descriptor referencing one payload above.
  D3D12_VIDEO_ENCODER_RATE_CONTROL descriptor;
};

// Self-contained native encode descriptors. Pointer-bearing descriptors refer
// to sibling fields and remain valid only while this owner stays in place.
struct EncodeConfiguration {
  // Native codec selector shared by capability queries and construction.
  D3D12_VIDEO_ENCODER_CODEC codec;
  // H.264 profile storage referenced by profile when that codec is selected.
  D3D12_VIDEO_ENCODER_PROFILE_H264 h264_profile;
  // HEVC profile storage referenced by profile when that codec is selected.
  D3D12_VIDEO_ENCODER_PROFILE_HEVC hevc_profile;
  // AV1 profile storage referenced by profile when that codec is selected.
  D3D12_VIDEO_ENCODER_AV1_PROFILE av1_profile;
  // Type-erased profile descriptor referencing one profile field above.
  D3D12_VIDEO_ENCODER_PROFILE_DESC profile;
  // H.264 tool configuration referenced by codec_configuration.
  D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_H264 h264;
  // HEVC tool configuration referenced by codec_configuration.
  D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC hevc;
  // AV1 tool configuration referenced by codec_configuration.
  D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION av1;
  // Type-erased codec descriptor referencing one codec field above.
  D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION codec_configuration;
  // H.264 GOP storage referenced by gop when that codec is selected.
  D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_H264 h264_gop;
  // HEVC GOP storage referenced by gop when that codec is selected.
  D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE_HEVC hevc_gop;
  // AV1 GOP storage referenced by gop when that codec is selected.
  D3D12_VIDEO_ENCODER_AV1_SEQUENCE_STRUCTURE av1_gop;
  // Type-erased GOP descriptor referencing one GOP field above.
  D3D12_VIDEO_ENCODER_SEQUENCE_GOP_STRUCTURE gop;
  // H.264 level storage referenced by level when that codec is selected.
  D3D12_VIDEO_ENCODER_LEVELS_H264 h264_level;
  // HEVC level/tier storage referenced by level when that codec is selected.
  D3D12_VIDEO_ENCODER_LEVEL_TIER_CONSTRAINTS_HEVC hevc_level;
  // AV1 level/tier storage referenced by level when that codec is selected.
  D3D12_VIDEO_ENCODER_AV1_LEVEL_TIER_CONSTRAINTS av1_level;
  // Type-erased level descriptor referencing one level field above.
  D3D12_VIDEO_ENCODER_LEVEL_SETTING level;
  // Independent native rate-control state indexed by temporal layer.
  EncodeRateControl temporal_layer_rate_controls
      [AMDF_MEDIA_ENCODE_TEMPORAL_LAYER_STORAGE_CAPACITY];
  // Maximum temporal-layer count admitted for this codec and device.
  uint32_t maximum_temporal_layer_count;
  // Exact coded resolution used by support queries and construction.
  D3D12_VIDEO_ENCODER_PICTURE_RESOLUTION_DESC resolution;
  // Single-tile AV1 partition referenced by extended support queries.
  D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES tiles;
  // Exact native subregion mode admitted with the configuration.
  D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE subregion_mode;
  // Private per-recording metadata extent required by EncodeFrame.
  uint32_t metadata_byte_length;
  // Native alignment for raw and resolved metadata offsets.
  uint32_t metadata_alignment;
};

// Copies a codec envelope after validating its public schema and native tuple.
// Unsupported configurations are answered queries; native failures are
// statuses. Native outputs are populated only for the matching direction.
amdf_status_t QueryConfiguration(
    Device* device, uint32_t family_ordinal,
    const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    D3D12_VIDEO_DECODE_CONFIGURATION* out_native_decode,
    EncodeConfiguration* out_native_encode);

// Native resource description for an already-admitted immutable image shape.
D3D12_RESOURCE_DESC ImageDescription(
    const amdf_media_image_description_t& description);

// Exact native placement requirements; no backing allocation is performed.
amdf_status_t ImageRequirements(
    Device* device, const amdf_media_image_description_t& description,
    amdf_media_image_requirements_t* out_requirements);

}  // namespace amdf::media::d3d12

#endif  // AMDF_SRC_MEDIA_D3D12_CONFIGURATION_H_
