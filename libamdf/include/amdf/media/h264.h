// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_H264_H_
#define AMDF_MEDIA_H264_H_

#include "amdf/media/encode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION \
  ((amdf_structure_type_t)0x00040401u)
#define AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040402u)
#define AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION \
  ((amdf_structure_type_t)0x00040403u)
#define AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040404u)

typedef uint32_t amdf_media_h264_profile_t;
enum amdf_media_h264_profile_e {
  AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE = 1,
  AMDF_MEDIA_H264_PROFILE_MAIN = 2,
  AMDF_MEDIA_H264_PROFILE_HIGH = 3,
};

/// Parsed syntax envelope on a configuration query or session constructor.
/// This schema describes progressive eight-bit 4:2:0 frame coding. Field/MBAFF,
/// flexible macroblock ordering, and other bit depths require a distinct
/// admitted syntax envelope; they cannot be implied by the profile name.
typedef struct amdf_media_h264_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// Profile whose syntax the caller supplies.
  amdf_media_h264_profile_t profile;
  /// Exact SPS level_idc: 10-13, 20-22, 30-32, 40-42, 50-52, or 60-62.
  /// Level 1b is not represented because its constraint_set3_flag is outside
  /// this syntax envelope.
  /// Configuration admission does not rewrite the stream.
  uint32_t level_idc;
} amdf_media_h264_configuration_t;

/// SPS facts needed by native off-host slice parsing. The application computes
/// picture order counts and maintains reference markings; no DPB is hidden
/// here.
typedef struct amdf_media_h264_sequence_parameters_t {
  /// Maximum number of short- and long-term reference frames.
  uint32_t maximum_reference_frame_count;
  /// SPS log2_max_frame_num_minus4, in [0,12].
  uint32_t log2_max_frame_num_minus4;
  /// SPS pic_order_cnt_type, in [0,2].
  uint32_t picture_order_count_type;
  /// SPS log2_max_pic_order_cnt_lsb_minus4 for type zero, otherwise zero.
  uint32_t log2_max_picture_order_count_lsb_minus4;
  /// SPS delta_pic_order_always_zero_flag for type one, otherwise zero.
  uint32_t delta_picture_order_always_zero;
  /// SPS direct_8x8_inference_flag.
  uint32_t direct_8x8_inference;
} amdf_media_h264_sequence_parameters_t;

/// PPS facts for one picture. Variable slice headers remain in the compressed
/// bytes; preparation does not read those bytes from device memory.
typedef struct amdf_media_h264_picture_parameters_t {
  /// Nonzero for CABAC, zero for CAVLC.
  uint32_t entropy_coding_mode;
  /// PPS bottom_field_pic_order_in_frame_present_flag.
  uint32_t bottom_field_picture_order_present;
  /// Default L0 active reference count minus one, before slice overrides.
  uint32_t default_reference_l0_count_minus1;
  /// Default L1 active reference count minus one, before slice overrides.
  uint32_t default_reference_l1_count_minus1;
  /// PPS weighted_pred_flag.
  uint32_t weighted_prediction;
  /// PPS weighted_bipred_idc, in [0,2].
  uint32_t weighted_biprediction;
  /// PPS pic_init_qp_minus26.
  int32_t initial_qp_minus26;
  /// PPS pic_init_qs_minus26.
  int32_t initial_qs_minus26;
  /// First chroma QP index offset.
  int32_t chroma_qp_index_offset;
  /// Second chroma QP index offset, including its inferred value.
  int32_t second_chroma_qp_index_offset;
  /// PPS deblocking_filter_control_present_flag.
  uint32_t deblocking_filter_control_present;
  /// PPS constrained_intra_pred_flag.
  uint32_t constrained_intra_prediction;
  /// PPS redundant_pic_cnt_present_flag.
  uint32_t redundant_picture_count_present;
  /// PPS transform_8x8_mode_flag.
  uint32_t transform_8x8_mode;
} amdf_media_h264_picture_parameters_t;

/// Fully resolved scaling lists in raster order, including inferred lists.
typedef struct amdf_media_h264_scaling_matrices_t {
  /// Six 4x4 matrices in H264 component/intra-inter order.
  uint8_t scaling_4x4[6][16];
  /// Luma intra and inter 8x8 matrices, respectively.
  uint8_t scaling_8x8[2][64];
} amdf_media_h264_scaling_matrices_t;

/// One complete Annex B slice, relative to the supplied compressed byte range.
typedef struct amdf_media_h264_slice_t {
  /// Offset including this slice's three-byte Annex B start code.
  uint32_t byte_offset;
  /// Complete slice extent; the final slice may include required input padding.
  uint32_t byte_length;
} amdf_media_h264_slice_t;

/// An explicitly marked reference, independent of any output-image ring.
typedef struct amdf_media_h264_reference_t {
  /// Borrowed reconstructed picture and its stable application-owned slot.
  amdf_media_reference_t binding;
  /// Short-term frame_num, or long-term frame index when long_term is nonzero.
  uint32_t frame_number;
  /// Nonzero for a long-term reference; zero for a short-term reference.
  uint32_t long_term;
  /// Fully computed top-field picture order count.
  int32_t top_field_order_count;
  /// Fully computed bottom-field picture order count.
  int32_t bottom_field_order_count;
} amdf_media_h264_reference_t;

/// Codec-specific input extension on amdf_media_decode_picture_info_t.
/// Preparation consumes these CPU arrays; all image/backing owners remain
/// borrowed through their accepted uses. Reference eviction and display order
/// belong to the application and are not inferred from this record.
typedef struct amdf_media_h264_decode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// Parsed SPS facts for the current picture.
  amdf_media_h264_sequence_parameters_t sequence;
  /// Parsed PPS facts for the current picture.
  amdf_media_h264_picture_parameters_t picture;
  /// Fully resolved scaling data, consumed during preparation.
  const amdf_media_h264_scaling_matrices_t* scaling_matrices;
  /// Current frame_num from the slice header.
  uint32_t frame_number;
  /// Nonzero when nal_ref_idc marks this picture as a reference.
  uint32_t is_reference;
  /// Nonzero only when every slice in this picture is intra-coded.
  uint32_t is_intra;
  /// Fully computed top-field picture order count.
  int32_t top_field_order_count;
  /// Fully computed bottom-field picture order count.
  int32_t bottom_field_order_count;
  /// Number of complete slice records in slices.
  uint32_t slice_count;
  /// Borrowed slice layout; no compressed-payload read occurs in preparation.
  const amdf_media_h264_slice_t* slices;
  /// Number of live reference records, at most sixteen.
  uint32_t reference_count;
  /// Explicit reference identities and markings, consumed during preparation.
  const amdf_media_h264_reference_t* references;
} amdf_media_h264_decode_picture_info_t;

/// H264 syntax controls chained from amdf_media_encode_configuration_t.
typedef struct amdf_media_h264_encode_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// pic_order_cnt_type, currently two for the admitted IP-only envelope.
  uint32_t picture_order_count_type;
  /// log2_max_frame_num_minus4, in [0,12].
  uint32_t log2_max_frame_number_minus4;
  /// log2_max_pic_order_cnt_lsb_minus4 for type zero, otherwise zero.
  uint32_t log2_max_picture_order_count_lsb_minus4;
} amdf_media_h264_encode_configuration_t;

typedef uint32_t amdf_media_h264_encode_frame_type_t;
enum amdf_media_h264_encode_frame_type_e {
  AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR = 1,
  AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P = 2,
};

/// One H264 encode reference and its application-owned reconstruction slot.
typedef struct amdf_media_h264_encode_reference_t {
  /// Borrowed reconstructed picture and stable physical slot.
  amdf_media_reference_t binding;
  /// Temporal layer in which this reconstructed picture was encoded.
  uint32_t temporal_layer_index;
  /// IDR or forward-predicted type used when this reference was encoded.
  amdf_media_h264_encode_frame_type_t frame_type;
  /// frame_num used when this reference was encoded.
  uint32_t frame_number;
  /// Full picture order count used when this reference was encoded.
  int32_t picture_order_count;
  /// Monotonic decoding-order number supplied for native bookkeeping.
  uint32_t frame_decoding_order_number;
} amdf_media_h264_encode_reference_t;

/// Parsed H264 picture controls chained from amdf_media_encode_picture_info_t.
/// Parameter sets and container framing remain application-owned.
typedef struct amdf_media_h264_encode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// IDR or forward-predicted picture type.
  amdf_media_h264_encode_frame_type_t frame_type;
  /// Current frame_num in the configured modulus.
  uint32_t frame_number;
  /// Full picture order count.
  int32_t picture_order_count;
  /// Monotonic decoding-order number.
  uint32_t frame_decoding_order_number;
  /// Nonzero when this picture remains available as a later reference.
  uint32_t used_as_reference;
  /// Number of supplied references: zero for IDR and one for P.
  uint32_t reference_count;
  /// Borrowed reference records consumed during preparation.
  const amdf_media_h264_encode_reference_t* references;
} amdf_media_h264_encode_picture_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_H264_H_
