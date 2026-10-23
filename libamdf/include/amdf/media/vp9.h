// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_VP9_H_
#define AMDF_MEDIA_VP9_H_

#include "amdf/media/session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION \
  ((amdf_structure_type_t)0x00040601u)
#define AMDF_STRUCTURE_TYPE_MEDIA_VP9_DECODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040602u)

typedef uint32_t amdf_media_vp9_profile_t;
enum amdf_media_vp9_profile_e {
  AMDF_MEDIA_VP9_PROFILE_0 = 0,
  AMDF_MEDIA_VP9_PROFILE_2 = 2,
};

/// 4:2:0 profile 0/eight-bit or profile 2/ten-bit syntax envelope. The exact
/// coded extent is fixed by the session; render-size scaling belongs to the
/// image consumer. Profile 2/twelve-bit and other chroma formats need separate
/// native admission and are not implied by profile 2/ten-bit support.
typedef struct amdf_media_vp9_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// Bitstream profile number.
  amdf_media_vp9_profile_t profile;
  /// Equal luma/chroma precision, eight or ten.
  uint32_t sample_bit_depth;
} amdf_media_vp9_configuration_t;

/// Resolved loop-filter state, including deltas retained from prior pictures.
typedef struct amdf_media_vp9_loop_filter_t {
  /// loop_filter_level, in [0, 63].
  uint32_t level;
  /// loop_filter_sharpness, in [0, 7].
  uint32_t sharpness;
  /// mode_ref_delta_enabled, zero or one.
  uint32_t deltas_enabled;
  /// mode_ref_delta_update for this picture, zero or one.
  uint32_t deltas_updated;
  /// Resolved INTRA/LAST/GOLDEN/ALTREF deltas, in [-63, 63].
  int8_t reference_deltas[4];
  /// Resolved zero/other inter-motion mode deltas, in [-63, 63].
  int8_t mode_deltas[2];
} amdf_media_vp9_loop_filter_t;

/// Resolved segmentation state. Update flags describe this picture's syntax;
/// feature values and probabilities include retained or inferred values.
/// When disabled, feature values and probabilities do not affect decoding.
typedef struct amdf_media_vp9_segmentation_t {
  /// segmentation_enabled, zero or one.
  uint32_t enabled;
  /// segmentation_update_map, zero or one.
  uint32_t update_map;
  /// segmentation_temporal_update, zero or one.
  uint32_t temporal_update;
  /// segmentation_update_data, zero or one.
  uint32_t update_data;
  /// One for absolute feature values, zero for deltas.
  uint32_t absolute_values;
  /// Segment-tree probabilities; 255 denotes an uncoded node.
  uint8_t tree_probabilities[7];
  /// Temporal prediction probabilities; 255 denotes an uncoded node.
  uint8_t prediction_probabilities[3];
  /// Bits zero through three enable ALT_Q, ALT_LF, REF_FRAME, and SKIP.
  uint8_t feature_masks[8];
  /// Resolved ALT_Q, ALT_LF, REF_FRAME, and SKIP values for each segment.
  /// Ranges are [-255, 255], [-63, 63], [0, 3], and zero respectively.
  int16_t feature_values[8][4];
} amdf_media_vp9_segmentation_t;

/// A complete VP9 picture beginning at the compressed range's first byte.
/// Header sizes locate the compressed partition without host payload access.
/// libamdf retains native entropy state but no image map: the caller supplies
/// every live reference, all inferred syntax, and stable physical slots. A
/// show_existing_frame header only selects an existing application image and
/// does not require a decode command.
typedef struct amdf_media_vp9_decode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_VP9_DECODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// frame_type: zero for a keyframe, one for an interframe.
  uint32_t frame_type;
  /// show_frame, zero or one.
  uint32_t show_frame;
  /// error_resilient_mode, zero or one.
  uint32_t error_resilient;
  /// intra_only for a non-key picture, zero or one.
  uint32_t intra_only;
  /// allow_high_precision_mv, zero or one.
  uint32_t high_precision_motion_vectors;
  /// refresh_frame_context, including its inferred value.
  uint32_t refresh_frame_context;
  /// frame_parallel_decoding_mode, including its inferred value.
  uint32_t parallel_decoding;
  /// Selected entropy context, in [0, 3].
  uint32_t frame_context_index;
  /// reset_frame_context, including its inferred value, in [0, 3].
  uint32_t reset_frame_context;
  /// Resolved use_prev_frame_mvs, zero or one.
  uint32_t use_previous_frame_motion_vectors;
  /// Interpolation filter: EIGHTTAP=0, SMOOTH=1, SHARP=2, BILINEAR=3,
  /// SWITCHABLE=4. Intra pictures carry zero.
  uint32_t interpolation_filter;
  /// base_q_idx, in [0, 255].
  uint32_t base_quantizer_index;
  /// Y DC quantizer delta, in [-15, 15].
  int32_t y_dc_quantizer_delta;
  /// UV DC quantizer delta, in [-15, 15].
  int32_t uv_dc_quantizer_delta;
  /// UV AC quantizer delta, in [-15, 15].
  int32_t uv_ac_quantizer_delta;
  /// log2_tile_cols, with VP9's 256-to-4096-pixel column limits.
  uint32_t log2_tile_columns;
  /// log2_tile_rows, in [0, 2].
  uint32_t log2_tile_rows;
  /// Byte-aligned uncompressed header extent, including its size field.
  uint32_t uncompressed_header_byte_length;
  /// Compressed header extent from header_size_in_bytes.
  uint32_t compressed_header_byte_length;
  /// Fully resolved loop-filter state.
  amdf_media_vp9_loop_filter_t loop_filter;
  /// Fully resolved segment state.
  amdf_media_vp9_segmentation_t segmentation;
  /// Number of unique live physical references, at most eight.
  uint32_t reference_count;
  /// Borrowed unique frame/slot bindings from this session's coded geometry.
  const amdf_media_reference_t* references;
  /// Eight logical map entries indexing references; 255 means absent. Multiple
  /// map entries may name the same reference. A keyframe supplies absent
  /// entries.
  uint8_t reference_map[8];
  /// LAST/GOLDEN/ALTREF indices into reference_map, or 255 for intra pictures.
  uint8_t active_reference_indices[3];
  /// LAST/GOLDEN/ALTREF sign biases, each zero or one.
  uint8_t reference_sign_bias[3];
} amdf_media_vp9_decode_picture_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_VP9_H_
