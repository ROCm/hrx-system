// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_ENCODE_H_
#define AMDF_MEDIA_ENCODE_H_

#include "amdf/media/session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION \
  ((amdf_structure_type_t)0x00040801u)
#define AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040802u)
#define AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO \
  ((amdf_structure_type_t)0x00040803u)

typedef uint32_t amdf_media_encode_rate_control_mode_t;
enum amdf_media_encode_rate_control_mode_e {
  /// Fixed quantization parameters selected by the application.
  AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP = 1,
  /// Constant target bit rate with an explicit virtual buffer.
  AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR = 2,
  /// Variable bit rate bounded by an explicit peak and virtual buffer.
  AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR = 3,
};

/// Quantization controls for CQP. The codec configuration defines the numeric
/// range: [0,51] for H264/HEVC and [0,255] for AV1.
typedef struct amdf_media_encode_cqp_t {
  /// Quantization parameter used for intra-coded pictures.
  uint32_t intra_quantization_parameter;
  /// Quantization parameter used for forward-predicted pictures.
  uint32_t predicted_quantization_parameter;
} amdf_media_encode_cqp_t;

/// Bit-rate controls shared by CBR and VBR.
typedef struct amdf_media_encode_bit_rate_t {
  /// Minimum quantization parameter the native controller may select.
  uint32_t minimum_quantization_parameter;
  /// Maximum quantization parameter the native controller may select.
  uint32_t maximum_quantization_parameter;
  /// Optional maximum encoded bits for one picture, or zero for no limit.
  uint64_t maximum_frame_bit_count;
  /// CBR target or VBR average rate, in bits per second.
  uint64_t target_bit_rate;
  /// VBR peak rate in bits per second; equal to target for CBR.
  uint64_t peak_bit_rate;
  /// Virtual-buffer capacity in bits.
  uint64_t virtual_buffer_capacity;
  /// Initial virtual-buffer fullness in bits.
  uint64_t initial_virtual_buffer_fullness;
} amdf_media_encode_bit_rate_t;

/// Exact rate-control mode admitted for one encode session.
typedef struct amdf_media_encode_rate_control_t {
  /// Selects the active control member.
  amdf_media_encode_rate_control_mode_t mode;
  /// Target frame-rate numerator.
  uint32_t frame_rate_numerator;
  /// Target frame-rate denominator.
  uint32_t frame_rate_denominator;
  /// Mode-specific controls.
  union {
    /// Controls selected when mode is CQP.
    amdf_media_encode_cqp_t cqp;
    /// Controls selected when mode is CBR or VBR.
    amdf_media_encode_bit_rate_t bit_rate;
  } control;
} amdf_media_encode_rate_control_t;

/// Shared IP-only encode envelope chained after the codec configuration.
/// Codec-specific encode configuration, when required, is chained from next.
typedef struct amdf_media_encode_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Codec-specific encode configuration, or NULL.
  const void* next;
  /// Pictures between intra refreshes, including the intra picture.
  uint32_t intra_picture_period;
  /// Distance between forward-predicted pictures; one excludes B pictures.
  uint32_t inter_picture_period;
  /// Session rate-control contract.
  amdf_media_encode_rate_control_t rate_control;
} amdf_media_encode_configuration_t;

/// Shared operands of one encoded picture. Codec-specific ordering and
/// reference syntax is chained from next. Preparation neither maps nor copies
/// the input or compressed output.
typedef struct amdf_media_encode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Codec-specific picture description.
  const void* next;
  /// Borrowed uncompressed input produced before this command executes.
  const amdf_media_frame_t* input;
  /// Borrowed reconstruction target for this picture.
  const amdf_media_frame_t* reconstruction;
  /// Stable application-owned reconstruction slot.
  uint32_t reconstruction_slot_index;
  /// Caller-owned compressed output range. Its accessible extent satisfies the
  /// complete minimum reported by configuration admission.
  amdf_media_buffer_range_t compressed;
} amdf_media_encode_picture_info_t;

typedef uint32_t amdf_media_encoded_region_kind_t;
enum amdf_media_encoded_region_kind_e {
  /// One complete H264 Annex B slice NAL unit.
  AMDF_MEDIA_ENCODED_REGION_H264_ANNEX_B_SLICE = 1,
  /// One complete HEVC Annex B slice-segment NAL unit.
  AMDF_MEDIA_ENCODED_REGION_HEVC_ANNEX_B_SLICE = 2,
  /// One complete AV1 frame OBU produced by the native encoder.
  AMDF_MEDIA_ENCODED_REGION_AV1_FRAME_OBU = 3,
  /// AV1 tile-group payload requiring application-authored OBU framing.
  AMDF_MEDIA_ENCODED_REGION_AV1_TILE_GROUP_PAYLOAD = 4,
};

/// One valid region relative to the picture's supplied compressed range.
typedef struct amdf_media_encoded_region_t {
  /// Codec unit represented by this region.
  amdf_media_encoded_region_kind_t kind;
  /// Region start relative to compressed.byte_offset.
  uint64_t byte_offset;
  /// Exact valid byte extent, excluding native tail padding.
  uint64_t byte_length;
} amdf_media_encoded_region_t;

/// Optional output chained from amdf_media_command_result_info_t for an encode
/// picture. The caller supplies regions and its capacity before the query.
typedef struct amdf_media_encode_result_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Caller-provided codec-specific result output, or NULL.
  void* next;
  /// Total native-written extent beginning at compressed.byte_offset.
  uint64_t written_byte_length;
  /// Number of records available in the caller-owned regions array.
  uint32_t region_capacity;
  /// Number of valid encoded regions, including when capacity is insufficient.
  uint32_t region_count;
  /// Caller-owned output array, or NULL when region_capacity is zero.
  amdf_media_encoded_region_t* regions;
} amdf_media_encode_result_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_ENCODE_H_
