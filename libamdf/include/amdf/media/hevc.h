// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_HEVC_H_
#define AMDF_MEDIA_HEVC_H_

#include "amdf/media/session.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION \
  ((amdf_structure_type_t)0x00040501u)
#define AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040502u)

typedef uint32_t amdf_media_hevc_profile_t;
enum amdf_media_hevc_profile_e {
  AMDF_MEDIA_HEVC_PROFILE_MAIN = 1,
  AMDF_MEDIA_HEVC_PROFILE_MAIN10 = 2,
};

/// Main/Main10 4:2:0 syntax envelope. Coded dimensions are the SPS luma
/// dimensions; conformance-window cropping remains with the image consumer.
/// Range, screen-content, multilayer, and multiview extensions are distinct
/// profiles and cannot be implied by this envelope.
typedef struct amdf_media_hevc_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// Profile whose syntax the caller supplies.
  amdf_media_hevc_profile_t profile;
  /// Equal luma/chroma precision: eight for Main, eight or ten for Main10.
  uint32_t sample_bit_depth;
} amdf_media_hevc_configuration_t;

/// Parsed SPS facts for native slice parsing. Flags are zero or one; absent
/// optional syntax carries its HEVC-inferred value. No SPS or payload parsing
/// occurs in libamdf.
typedef struct amdf_media_hevc_sequence_parameters_t {
  /// sps_max_dec_pic_buffering_minus1 for the decoded temporal sublayer.
  uint32_t maximum_decoded_picture_buffer_count_minus1;
  /// log2_max_pic_order_cnt_lsb_minus4.
  uint32_t log2_max_picture_order_count_lsb_minus4;
  /// log2_min_luma_coding_block_size_minus3.
  uint32_t log2_min_luma_coding_block_size_minus3;
  /// log2_diff_max_min_luma_coding_block_size.
  uint32_t log2_diff_max_min_luma_coding_block_size;
  /// log2_min_luma_transform_block_size_minus2.
  uint32_t log2_min_luma_transform_block_size_minus2;
  /// log2_diff_max_min_luma_transform_block_size.
  uint32_t log2_diff_max_min_luma_transform_block_size;
  /// max_transform_hierarchy_depth_inter.
  uint32_t maximum_transform_hierarchy_depth_inter;
  /// max_transform_hierarchy_depth_intra.
  uint32_t maximum_transform_hierarchy_depth_intra;
  /// num_short_term_ref_pic_sets in the SPS, at most 64.
  uint32_t short_term_reference_set_count;
  /// num_long_term_ref_pics_sps, at most 32.
  uint32_t long_term_reference_count;
  /// scaling_list_enabled_flag.
  uint32_t scaling_list_enabled;
  /// amp_enabled_flag.
  uint32_t asymmetric_motion_partitions_enabled;
  /// sample_adaptive_offset_enabled_flag.
  uint32_t sample_adaptive_offset_enabled;
  /// pcm_enabled_flag.
  uint32_t pcm_enabled;
  /// pcm_sample_bit_depth_luma_minus1 when PCM is enabled, otherwise zero.
  uint32_t pcm_luma_bit_depth_minus1;
  /// pcm_sample_bit_depth_chroma_minus1 when PCM is enabled, otherwise zero.
  uint32_t pcm_chroma_bit_depth_minus1;
  /// log2_min_pcm_luma_coding_block_size_minus3, otherwise zero.
  uint32_t log2_min_pcm_luma_coding_block_size_minus3;
  /// log2_diff_max_min_pcm_luma_coding_block_size, otherwise zero.
  uint32_t log2_diff_max_min_pcm_luma_coding_block_size;
  /// pcm_loop_filter_disabled_flag.
  uint32_t pcm_loop_filter_disabled;
  /// long_term_ref_pics_present_flag.
  uint32_t long_term_references_present;
  /// sps_temporal_mvp_enabled_flag.
  uint32_t temporal_motion_vector_prediction_enabled;
  /// strong_intra_smoothing_enabled_flag.
  uint32_t strong_intra_smoothing_enabled;
} amdf_media_hevc_sequence_parameters_t;

/// Parsed PPS facts, including inferred values. Nonuniform tile extents exclude
/// the last row/column, whose extent follows from the coded picture size.
typedef struct amdf_media_hevc_picture_parameters_t {
  /// num_ref_idx_l0_default_active_minus1.
  uint32_t default_reference_l0_count_minus1;
  /// num_ref_idx_l1_default_active_minus1.
  uint32_t default_reference_l1_count_minus1;
  /// init_qp_minus26.
  int32_t initial_qp_minus26;
  /// dependent_slice_segments_enabled_flag.
  uint32_t dependent_slice_segments_enabled;
  /// output_flag_present_flag.
  uint32_t output_flag_present;
  /// num_extra_slice_header_bits.
  uint32_t extra_slice_header_bit_count;
  /// sign_data_hiding_enabled_flag.
  uint32_t sign_data_hiding_enabled;
  /// cabac_init_present_flag.
  uint32_t cabac_initialization_present;
  /// constrained_intra_pred_flag.
  uint32_t constrained_intra_prediction;
  /// transform_skip_enabled_flag.
  uint32_t transform_skip_enabled;
  /// cu_qp_delta_enabled_flag.
  uint32_t coding_unit_qp_delta_enabled;
  /// diff_cu_qp_delta_depth, otherwise zero.
  uint32_t coding_unit_qp_delta_depth;
  /// pps_slice_chroma_qp_offsets_present_flag.
  uint32_t slice_chroma_qp_offsets_present;
  /// weighted_pred_flag.
  uint32_t weighted_prediction;
  /// weighted_bipred_flag.
  uint32_t weighted_biprediction;
  /// transquant_bypass_enabled_flag.
  uint32_t transform_quantization_bypass_enabled;
  /// tiles_enabled_flag.
  uint32_t tiles_enabled;
  /// entropy_coding_sync_enabled_flag (wavefront parallel processing).
  uint32_t entropy_coding_synchronization_enabled;
  /// uniform_spacing_flag; zero when tiles are disabled.
  uint32_t uniform_tile_spacing;
  /// loop_filter_across_tiles_enabled_flag; zero when tiles are disabled.
  uint32_t loop_filter_across_tiles_enabled;
  /// pps_loop_filter_across_slices_enabled_flag.
  uint32_t loop_filter_across_slices_enabled;
  /// deblocking_filter_override_enabled_flag.
  uint32_t deblocking_filter_override_enabled;
  /// pps_deblocking_filter_disabled_flag.
  uint32_t deblocking_filter_disabled;
  /// lists_modification_present_flag.
  uint32_t reference_lists_modification_present;
  /// slice_segment_header_extension_present_flag.
  uint32_t slice_segment_header_extension_present;
  /// pps_cb_qp_offset.
  int32_t cb_qp_offset;
  /// pps_cr_qp_offset.
  int32_t cr_qp_offset;
  /// pps_beta_offset_div2, including its inferred value.
  int32_t beta_offset_div2;
  /// pps_tc_offset_div2, including its inferred value.
  int32_t tc_offset_div2;
  /// log2_parallel_merge_level_minus2.
  uint32_t log2_parallel_merge_level_minus2;
  /// num_tile_columns_minus1, otherwise zero; at most 19.
  uint32_t tile_column_count_minus1;
  /// num_tile_rows_minus1, otherwise zero; at most 21.
  uint32_t tile_row_count_minus1;
  /// Nonuniform column_width_minus1 values in coding tree blocks.
  uint32_t tile_column_width_minus1[19];
  /// Nonuniform row_height_minus1 values in coding tree blocks.
  uint32_t tile_row_height_minus1[21];
} amdf_media_hevc_picture_parameters_t;

/// Fully resolved scaling coefficients in raster order, including inherited
/// and default lists. Larger transforms use their 8x8 coefficient grid plus a
/// separate DC coefficient; callers do not expand that grid to 16x16/32x32.
typedef struct amdf_media_hevc_scaling_matrices_t {
  /// Six 4x4 coefficient grids in HEVC matrix-ID order.
  uint8_t scaling_4x4[6][16];
  /// Six 8x8 coefficient grids in HEVC matrix-ID order.
  uint8_t scaling_8x8[6][64];
  /// Six 8x8 grids for 16x16 transforms.
  uint8_t scaling_16x16[6][64];
  /// Two 8x8 grids for 32x32 transforms, matrix IDs zero and three.
  uint8_t scaling_32x32[2][64];
  /// Six resolved DC coefficients for 16x16 transforms.
  uint8_t dc_16x16[6];
  /// Two resolved DC coefficients for 32x32 transforms.
  uint8_t dc_32x32[2];
} amdf_media_hevc_scaling_matrices_t;

/// One complete Annex B slice segment, including a three-byte start code.
typedef struct amdf_media_hevc_slice_t {
  /// Offset relative to the supplied compressed byte range.
  uint32_t byte_offset;
  /// Extent including required padding after the final slice segment.
  uint32_t byte_length;
} amdf_media_hevc_slice_t;

/// Explicit decoded-picture identity and marking. Display order, eviction,
/// random-access state, and physical slot reuse belong to the application.
typedef struct amdf_media_hevc_reference_t {
  /// Borrowed reconstructed picture and its stable application-owned slot.
  amdf_media_reference_t binding;
  /// Fully computed PicOrderCntVal, not the truncated slice-header POC.
  int32_t picture_order_count;
  /// One for a long-term reference, zero for a short-term reference.
  uint32_t long_term;
} amdf_media_hevc_reference_t;

/// One ordered current-reference set. Indices address the supplied references
/// array, not physical image slots. The three current sets are disjoint.
typedef struct amdf_media_hevc_reference_set_t {
  /// Number of active indices, at most eight.
  uint32_t count;
  /// Ordered reference-array indices; entries beyond count are ignored.
  uint8_t indices[8];
} amdf_media_hevc_reference_set_t;

/// Parsed Main/Main10 picture input consumed during preparation. The caller
/// supplies resolved POCs, reference sets, scaling lists, and slice extents;
/// libamdf neither reads compressed device bytes nor manages a DPB.
typedef struct amdf_media_hevc_decode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// Parsed SPS facts for this picture.
  amdf_media_hevc_sequence_parameters_t sequence;
  /// Parsed PPS facts for this picture.
  amdf_media_hevc_picture_parameters_t picture;
  /// Resolved scaling data when enabled, otherwise NULL.
  const amdf_media_hevc_scaling_matrices_t* scaling_matrices;
  /// Current full PicOrderCntVal.
  int32_t picture_order_count;
  /// One for an IRAP NAL unit, zero otherwise.
  uint32_t is_random_access;
  /// One for an IDR NAL unit, zero otherwise; implies is_random_access.
  uint32_t is_idr;
  /// One when all slices are intra-coded; zero otherwise.
  uint32_t is_intra;
  /// NumDeltaPocs of the RPS used for inter-RPS prediction, otherwise zero.
  uint32_t predicted_reference_set_delta_count;
  /// Bits occupied by short_term_ref_pic_set() in the slice, otherwise zero.
  uint32_t short_term_reference_set_bit_count;
  /// Number of complete slice-segment records.
  uint32_t slice_count;
  /// Borrowed slice layout, consumed during preparation.
  const amdf_media_hevc_slice_t* slices;
  /// Number of live reference identities, at most fifteen.
  uint32_t reference_count;
  /// Borrowed reference array, consumed during preparation.
  const amdf_media_hevc_reference_t* references;
  /// RefPicSetStCurrBefore, in descending POC order.
  amdf_media_hevc_reference_set_t short_term_before;
  /// RefPicSetStCurrAfter, in ascending POC order.
  amdf_media_hevc_reference_set_t short_term_after;
  /// RefPicSetLtCurr in the application-resolved HEVC order.
  amdf_media_hevc_reference_set_t long_term;
} amdf_media_hevc_decode_picture_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_HEVC_H_
