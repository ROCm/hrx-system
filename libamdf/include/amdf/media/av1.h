// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_MEDIA_AV1_H_
#define AMDF_MEDIA_AV1_H_

#include "amdf/media/encode.h"

#ifdef __cplusplus
extern "C" {
#endif

#define AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION \
  ((amdf_structure_type_t)0x00040701u)
#define AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040702u)
#define AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION \
  ((amdf_structure_type_t)0x00040703u)
#define AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_PICTURE_INFO \
  ((amdf_structure_type_t)0x00040704u)
#define AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO \
  ((amdf_structure_type_t)0x00040705u)

/// Profile 0, 4:2:0 eight/ten-bit decoding. The session's coded extent is the
/// upscaled picture extent; super-resolution changes the coded sample count,
/// not the output allocation. References use this same upscaled geometry.
typedef struct amdf_media_av1_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// seq_profile, zero for the admitted 4:2:0 envelope.
  uint32_t profile;
  /// Equal luma/chroma precision, eight or ten.
  uint32_t sample_bit_depth;
} amdf_media_av1_configuration_t;

/// Sequence tools resolved from the sequence header. Boolean fields are zero
/// or one. The caller retains and supplies these facts for every picture.
typedef struct amdf_media_av1_sequence_parameters_t {
  /// use_128x128_superblock; zero selects 64x64.
  uint32_t use_128x128_superblock;
  /// enable_filter_intra.
  uint32_t filter_intra;
  /// enable_intra_edge_filter.
  uint32_t intra_edge_filter;
  /// enable_interintra_compound.
  uint32_t interintra_compound;
  /// enable_masked_compound.
  uint32_t masked_compound;
  /// enable_dual_filter.
  uint32_t dual_filter;
  /// enable_jnt_comp.
  uint32_t joint_compound;
  /// enable_ref_frame_mvs.
  uint32_t reference_frame_motion_vectors;
  /// enable_cdef.
  uint32_t cdef;
  /// enable_restoration.
  uint32_t restoration;
  /// film_grain_params_present.
  uint32_t film_grain;
  /// OrderHintBits in [0, 8]; zero means enable_order_hint is false.
  uint32_t order_hint_bit_count;
} amdf_media_av1_sequence_parameters_t;

/// Resolved loop filter and restoration state, including retained deltas.
typedef struct amdf_media_av1_loop_filter_t {
  /// Vertical Y, horizontal Y, U, and V filter levels, each in [0, 63].
  uint8_t levels[4];
  /// loop_filter_sharpness in [0, 7].
  uint32_t sharpness;
  /// loop_filter_delta_enabled, zero or one.
  uint32_t deltas_enabled;
  /// loop_filter_delta_update, zero or one.
  uint32_t deltas_updated;
  /// Resolved INTRA/LAST/LAST2/LAST3/GOLDEN/BWDREF/ALTREF2/ALTREF deltas.
  int8_t reference_deltas[8];
  /// Resolved zero/other motion mode deltas, each in [-64, 63].
  int8_t mode_deltas[2];
  /// delta_lf_present, zero or one.
  uint32_t delta_present;
  /// delta_lf_res exponent in [0, 3], including its inferred value.
  uint32_t delta_resolution;
  /// delta_lf_multi, zero or one.
  uint32_t delta_multi;
  /// Y/U/V FrameRestorationType: NONE=0, WIENER=1, SGRPROJ=2, SWITCHABLE=3.
  uint8_t restoration_types[3];
  /// Resolved log2(LoopRestorationSize) for Y/U/V, in [5, 8].
  uint8_t log2_restoration_unit_sizes[3];
} amdf_media_av1_loop_filter_t;

typedef struct amdf_media_av1_quantization_t {
  /// base_q_idx in [0, 255].
  uint32_t base_index;
  /// Y DC delta, in [-64, 63].
  int32_t y_dc_delta;
  /// U/V DC deltas, each in [-64, 63].
  int8_t chroma_dc_deltas[2];
  /// U/V AC deltas, each in [-64, 63].
  int8_t chroma_ac_deltas[2];
  /// Y/U/V quantization matrix levels in [0, 15], or all 255 when disabled.
  uint8_t matrix_levels[3];
  /// delta_q_present, zero or one.
  uint32_t delta_present;
  /// delta_q_res exponent in [0, 3], including its inferred value.
  uint32_t delta_resolution;
} amdf_media_av1_quantization_t;

typedef struct amdf_media_av1_cdef_t {
  /// cdef_damping_minus_3 in [0, 3].
  uint32_t damping_minus3;
  /// cdef_bits in [0, 3]; selects the first 2^bits strengths.
  uint32_t bits;
  /// Luma primary strengths in [0, 15].
  uint8_t luma_primary[8];
  /// Luma secondary strength syntax in [0, 3]; syntax 3 represents strength 4.
  uint8_t luma_secondary[8];
  /// Chroma primary strengths in [0, 15].
  uint8_t chroma_primary[8];
  /// Chroma secondary strength syntax in [0, 3].
  uint8_t chroma_secondary[8];
} amdf_media_av1_cdef_t;

/// Resolved segment features; disabled segmentation ignores retained features.
typedef struct amdf_media_av1_segmentation_t {
  /// segmentation_enabled, zero or one.
  uint32_t enabled;
  /// segmentation_update_map, zero or one.
  uint32_t update_map;
  /// segmentation_temporal_update, zero or one.
  uint32_t temporal_update;
  /// segmentation_update_data, zero or one.
  uint32_t update_data;
  /// Eight bits select ALT_Q, ALT_LF_Y_V/Y_H/U/V, REF_FRAME, SKIP, GLOBALMV.
  uint8_t feature_masks[8];
  /// Resolved values: ALT_Q in [-255, 255], filters in [-63, 63], REF_FRAME
  /// in [0, 7], and zero for SKIP/GLOBALMV.
  int16_t feature_values[8][8];
} amdf_media_av1_segmentation_t;

/// Resolved film grain, including values inherited through
/// film_grain_params_ref_idx. Grain affects output only; reconstructed
/// references remain grain-free.
typedef struct amdf_media_av1_film_grain_t {
  /// apply_grain, zero or one. Other fields are ignored when zero.
  uint32_t apply;
  /// grain_seed in [0, 65535].
  uint32_t seed;
  /// Number of ordered luma scaling points, at most fourteen.
  uint32_t luma_point_count;
  /// Luma intensity/scaling pairs; intensities are strictly increasing.
  uint8_t luma_points[14][2];
  /// Number of ordered U scaling points, at most ten.
  uint32_t u_point_count;
  /// U intensity/scaling pairs; intensities are strictly increasing.
  uint8_t u_points[10][2];
  /// Number of ordered V scaling points, at most ten.
  uint32_t v_point_count;
  /// V intensity/scaling pairs; intensities are strictly increasing.
  uint8_t v_points[10][2];
  /// grain_scaling_minus_8 in [0, 3].
  uint32_t scaling_shift_minus8;
  /// chroma_scaling_from_luma, zero or one.
  uint32_t chroma_scaling_from_luma;
  /// ar_coeff_lag in [0, 3].
  uint32_t coefficient_lag;
  /// Signed luma autoregression coefficients, with the +128 syntax bias
  /// removed.
  int8_t luma_coefficients[24];
  /// Signed U autoregression coefficients, including its optional luma term.
  int8_t u_coefficients[25];
  /// Signed V autoregression coefficients, including its optional luma term.
  int8_t v_coefficients[25];
  /// ar_coeff_shift_minus_6 in [0, 3].
  uint32_t coefficient_shift_minus6;
  /// grain_scale_shift in [0, 3].
  uint32_t scale_shift;
  /// cb_mult/cr_mult unsigned syntax values, before subtracting 128.
  uint8_t chroma_multipliers[2];
  /// cb_luma_mult/cr_luma_mult unsigned syntax values.
  uint8_t chroma_luma_multipliers[2];
  /// cb_offset/cr_offset unsigned syntax values, each in [0, 511].
  uint16_t chroma_offsets[2];
  /// overlap_flag, zero or one.
  uint32_t overlap;
  /// clip_to_restricted_range, zero or one.
  uint32_t clip_to_restricted_range;
} amdf_media_av1_film_grain_t;

typedef struct amdf_media_av1_global_motion_t {
  /// IDENTITY=0, TRANSLATION=1, ROTZOOM=2, AFFINE=3.
  uint32_t type;
  /// Resolved gm_params[0..5], in AV1's fixed-point representation.
  int32_t parameters[6];
  /// Invalid shear parameters disable this warp model, zero or one.
  uint32_t invalid;
} amdf_media_av1_global_motion_t;

typedef struct amdf_media_av1_tile_t {
  /// First tile byte relative to the compressed range, excluding its size
  /// field.
  uint32_t byte_offset;
  /// Complete tile payload length, excluding its size field and native padding.
  uint32_t byte_length;
} amdf_media_av1_tile_t;

/// Complete picture metadata and row-major tiles. The caller resolves inferred
/// syntax, reference maps, global motion, and inherited grain before
/// preparation. Payloads contain complete low-overhead OBUs; tile offsets
/// locate entropy data without host reads. show_existing_frame selects an
/// existing application image and requires no decode operation. This schema has
/// no large-scale tile anchors.
typedef struct amdf_media_av1_decode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// Resolved sequence tools for this picture.
  amdf_media_av1_sequence_parameters_t sequence;
  /// KEY=0, INTER=1, INTRA_ONLY=2, SWITCH=3.
  uint32_t frame_type;
  /// show_frame, zero or one.
  uint32_t show_frame;
  /// showable_frame, zero or one.
  uint32_t showable_frame;
  /// primary_ref_frame in [0, 7]; seven means PRIMARY_REF_NONE.
  uint32_t primary_reference;
  /// order_hint in the sequence's order-hint bit width.
  uint32_t order_hint;
  /// disable_cdf_update, zero or one.
  uint32_t disable_cdf_update;
  /// disable_frame_end_update_cdf, zero or one.
  uint32_t disable_frame_end_cdf_update;
  /// allow_warped_motion, zero or one.
  uint32_t warped_motion;
  /// allow_screen_content_tools, zero or one.
  uint32_t screen_content_tools;
  /// force_integer_mv, including its inferred value, zero or one.
  uint32_t integer_motion_vectors;
  /// allow_intrabc, zero or one.
  uint32_t intra_block_copy;
  /// allow_high_precision_mv, zero or one.
  uint32_t high_precision_motion_vectors;
  /// EIGHTTAP=0, SMOOTH=1, SHARP=2, BILINEAR=3, SWITCHABLE=4.
  uint32_t interpolation_filter;
  /// is_motion_mode_switchable, zero or one.
  uint32_t switchable_motion_mode;
  /// reference_select, zero or one.
  uint32_t reference_select;
  /// skip_mode_present, zero or one.
  uint32_t skip_mode;
  /// reduced_tx_set, zero or one.
  uint32_t reduced_transform_set;
  /// TxMode: ONLY_4X4=0, TX_MODE_LARGEST=1, TX_MODE_SELECT=2.
  uint32_t transform_mode;
  /// use_ref_frame_mvs, zero or one.
  uint32_t use_reference_motion_vectors;
  /// SuperresDenom in [8, 16]; eight disables super-resolution.
  uint32_t superres_denominator;
  /// Resolved loop-filter/restoration state.
  amdf_media_av1_loop_filter_t loop_filter;
  /// Resolved quantization state.
  amdf_media_av1_quantization_t quantization;
  /// Resolved constrained directional enhancement filter state.
  amdf_media_av1_cdef_t cdef;
  /// Resolved segmentation state.
  amdf_media_av1_segmentation_t segmentation;
  /// Resolved grain parameters; output and reference storage remain separate.
  amdf_media_av1_film_grain_t film_grain;
  /// Number of unique live reconstructed references, at most eight.
  uint32_t reference_count;
  /// Borrowed stable physical frame/slot bindings.
  const amdf_media_reference_t* references;
  /// Logical reference-map indices into references; 255 means absent.
  uint8_t reference_map[8];
  /// LAST through ALTREF logical map indices, or 255 for intra pictures.
  uint8_t active_reference_indices[7];
  /// Resolved LAST through ALTREF RefFrameSignBias values, each zero or one.
  uint8_t reference_sign_bias[7];
  /// Resolved LAST through ALTREF global motion, including identity matrices.
  amdf_media_av1_global_motion_t global_motion[7];
  /// Number of tile columns, in [1, 64].
  uint32_t tile_column_count;
  /// Number of tile rows, in [1, 64]; total tiles cannot exceed 256.
  uint32_t tile_row_count;
  /// context_update_tile_id, less than the complete tile count.
  uint32_t context_update_tile;
  /// Tile column boundaries in superblocks, from zero through coded width.
  uint16_t tile_column_starts[65];
  /// Tile row boundaries in superblocks, from zero through coded height.
  uint16_t tile_row_starts[65];
  /// Borrowed row-major tile_column_count * tile_row_count payload ranges.
  const amdf_media_av1_tile_t* tiles;
} amdf_media_av1_decode_picture_info_t;

typedef uint32_t amdf_media_av1_encode_feature_flags_t;
enum amdf_media_av1_encode_feature_flag_bits_e {
  /// Enables order hints and reference-order tools.
  AMDF_MEDIA_AV1_ENCODE_FEATURE_ORDER_HINT = 1u << 0,
  /// Enables constrained directional enhancement filtering.
  AMDF_MEDIA_AV1_ENCODE_FEATURE_CDEF = 1u << 1,
};

typedef uint32_t amdf_media_av1_tier_t;
enum amdf_media_av1_tier_e {
  AMDF_MEDIA_AV1_TIER_MAIN = 1,
  AMDF_MEDIA_AV1_TIER_HIGH = 2,
};

/// AV1 encoder tools chained from amdf_media_encode_configuration_t.
typedef struct amdf_media_av1_encode_configuration_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed configuration, or NULL.
  const void* next;
  /// AV1 level index selected by the application.
  uint32_t level;
  /// Main or high tier at the selected level.
  amdf_media_av1_tier_t tier;
  /// Explicitly enabled encode features.
  amdf_media_av1_encode_feature_flags_t feature_flags;
  /// OrderHintBits in [1,8] when order hints are enabled, otherwise zero.
  uint32_t order_hint_bit_count;
} amdf_media_av1_encode_configuration_t;

/// One AV1 encode reference and its application-owned reconstruction slot.
typedef struct amdf_media_av1_encode_reference_t {
  /// Borrowed reconstructed picture and stable physical slot.
  amdf_media_reference_t binding;
  /// KEY=0, INTER=1, INTRA_ONLY=2, or SWITCH=3.
  uint32_t frame_type;
  /// Reference order hint in the configured bit width.
  uint32_t order_hint;
  /// Application picture index carried through native metadata.
  uint32_t picture_index;
} amdf_media_av1_encode_reference_t;

/// Parsed AV1 picture controls chained from amdf_media_encode_picture_info_t.
/// The initial envelope admits one full-frame tile and no temporal/spatial
/// layers, segmentation map, restoration, super-resolution, or B pictures.
typedef struct amdf_media_av1_encode_picture_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_PICTURE_INFO.
  amdf_structure_type_t type;
  /// Size of this caller-provided structure.
  uint32_t structure_size;
  /// Further typed picture input, or NULL.
  const void* next;
  /// KEY=0, INTER=1, INTRA_ONLY=2, or SWITCH=3.
  uint32_t frame_type;
  /// Order hint in the configured bit width.
  uint32_t order_hint;
  /// Application picture index carried through native metadata.
  uint32_t picture_index;
  /// primary_ref_frame in [0,7]; seven means PRIMARY_REF_NONE.
  uint32_t primary_reference;
  /// Eight refresh-frame bits for the logical AV1 reference map.
  uint32_t refresh_frame_flags;
  /// EIGHTTAP=0, SMOOTH=1, SHARP=2, BILINEAR=3, SWITCHABLE=4.
  uint32_t interpolation_filter;
  /// TxMode: ONLY_4X4=0, TX_MODE_LARGEST=1, TX_MODE_SELECT=2.
  uint32_t transform_mode;
  /// Initial loop-filter state; native post values are reported separately.
  amdf_media_av1_loop_filter_t loop_filter;
  /// Initial quantization state; native post values are reported separately.
  amdf_media_av1_quantization_t quantization;
  /// Initial CDEF state; native post values are reported separately.
  amdf_media_av1_cdef_t cdef;
  /// Number of unique live reconstructed references, at most eight.
  uint32_t reference_count;
  /// Borrowed reference records consumed during preparation.
  const amdf_media_av1_encode_reference_t* references;
  /// LAST through ALTREF indices into references; 255 means absent.
  uint8_t reference_indices[7];
} amdf_media_av1_encode_picture_info_t;

/// Optional AV1 post-encode output chained from
/// amdf_media_encode_result_info_t. Fields are valid when the returned encoded
/// region is tile payload requiring application-authored OBU framing.
typedef struct amdf_media_av1_encode_result_info_t {
  /// AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO.
  amdf_structure_type_t type;
  /// Size of caller-provided result storage.
  uint32_t structure_size;
  /// Further typed result output, or NULL.
  void* next;
  /// Native-selected compound prediction type.
  uint32_t compound_prediction_type;
  /// Native-selected loop-filter state.
  amdf_media_av1_loop_filter_t loop_filter;
  /// Native-selected quantization state.
  amdf_media_av1_quantization_t quantization;
  /// Native-selected CDEF state.
  amdf_media_av1_cdef_t cdef;
  /// Native-selected segmentation state.
  amdf_media_av1_segmentation_t segmentation;
  /// Native-selected primary reference.
  uint32_t primary_reference;
  /// Native-selected LAST through ALTREF logical reference indices.
  uint8_t reference_indices[7];
} amdf_media_av1_encode_result_info_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_MEDIA_AV1_H_
