// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/hevc.h"

#include "libamdf/src/media/encode.h"
#include "libamdf/src/structure.h"

static bool amdf_media_hevc_reference_set_valid(
    const amdf_media_hevc_decode_picture_info_t* picture,
    const amdf_media_hevc_reference_set_t* set, uint32_t kind, uint32_t* used) {
  if (set->count > 8) {
    return false;
  }
  int32_t previous = picture->picture_order_count;
  for (uint32_t index = 0; index < set->count; ++index) {
    const uint32_t reference_index = set->indices[index];
    if (reference_index >= picture->reference_count ||
        (*used & (1u << reference_index)) != 0) {
      return false;
    }
    const amdf_media_hevc_reference_t* reference =
        &picture->references[reference_index];
    if ((reference->long_term != 0) != (kind == 2) ||
        (kind == 0 && reference->picture_order_count >= previous) ||
        (kind == 1 && reference->picture_order_count <= previous)) {
      return false;
    }
    previous = reference->picture_order_count;
    *used |= 1u << reference_index;
  }
  return true;
}

amdf_status_t amdf_media_hevc_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_hevc_configuration_t* hevc,
    const amdf_media_hevc_decode_picture_info_t* picture,
    uint64_t compressed_byte_length) {
  const amdf_media_hevc_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_hevc_picture_parameters_t* parameters = &picture->picture;
  const uint32_t flags =
      sequence->scaling_list_enabled |
      sequence->asymmetric_motion_partitions_enabled |
      sequence->sample_adaptive_offset_enabled | sequence->pcm_enabled |
      sequence->pcm_loop_filter_disabled |
      sequence->long_term_references_present |
      sequence->temporal_motion_vector_prediction_enabled |
      sequence->strong_intra_smoothing_enabled |
      parameters->dependent_slice_segments_enabled |
      parameters->output_flag_present | parameters->sign_data_hiding_enabled |
      parameters->cabac_initialization_present |
      parameters->constrained_intra_prediction |
      parameters->transform_skip_enabled |
      parameters->coding_unit_qp_delta_enabled |
      parameters->slice_chroma_qp_offsets_present |
      parameters->weighted_prediction | parameters->weighted_biprediction |
      parameters->transform_quantization_bypass_enabled |
      parameters->tiles_enabled |
      parameters->entropy_coding_synchronization_enabled |
      parameters->uniform_tile_spacing |
      parameters->loop_filter_across_tiles_enabled |
      parameters->loop_filter_across_slices_enabled |
      parameters->deblocking_filter_override_enabled |
      parameters->deblocking_filter_disabled |
      parameters->reference_lists_modification_present |
      parameters->slice_segment_header_extension_present |
      picture->is_random_access | picture->is_idr | picture->is_intra;
  if (flags > 1 || picture->slice_count == 0 || picture->slices == NULL ||
      picture->reference_count > 15 ||
      (picture->reference_count != 0 && picture->references == NULL) ||
      sequence->maximum_decoded_picture_buffer_count_minus1 > 15 ||
      sequence->maximum_decoded_picture_buffer_count_minus1 + 1 >
          configuration->reference_image_count ||
      picture->reference_count >
          sequence->maximum_decoded_picture_buffer_count_minus1 ||
      sequence->log2_max_picture_order_count_lsb_minus4 > 12 ||
      sequence->log2_min_luma_coding_block_size_minus3 > 3 ||
      sequence->log2_diff_max_min_luma_coding_block_size > 3 ||
      sequence->log2_min_luma_transform_block_size_minus2 > 3 ||
      sequence->log2_diff_max_min_luma_transform_block_size > 3 ||
      sequence->short_term_reference_set_count > 64 ||
      sequence->long_term_reference_count > 32 ||
      (!sequence->long_term_references_present &&
       sequence->long_term_reference_count != 0) ||
      parameters->default_reference_l0_count_minus1 > 14 ||
      parameters->default_reference_l1_count_minus1 > 14 ||
      parameters->initial_qp_minus26 <
          -26 - 6 * (int32_t)(hevc->sample_bit_depth - 8) ||
      parameters->initial_qp_minus26 > 25 ||
      parameters->extra_slice_header_bit_count > 7 ||
      parameters->cb_qp_offset < -12 || parameters->cb_qp_offset > 12 ||
      parameters->cr_qp_offset < -12 || parameters->cr_qp_offset > 12 ||
      parameters->beta_offset_div2 < -6 || parameters->beta_offset_div2 > 6 ||
      parameters->tc_offset_div2 < -6 || parameters->tc_offset_div2 > 6 ||
      parameters->tile_column_count_minus1 > 19 ||
      parameters->tile_row_count_minus1 > 21 ||
      picture->predicted_reference_set_delta_count > 15 ||
      picture->short_term_reference_set_bit_count > UINT16_MAX ||
      (picture->is_random_access && !picture->is_intra) ||
      (picture->is_idr &&
       (!picture->is_random_access || picture->picture_order_count != 0)) ||
      (sequence->scaling_list_enabled && picture->scaling_matrices == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t minimum_block_log2 =
      sequence->log2_min_luma_coding_block_size_minus3 + 3;
  const uint32_t maximum_block_log2 =
      minimum_block_log2 + sequence->log2_diff_max_min_luma_coding_block_size;
  const uint32_t minimum_transform_log2 =
      sequence->log2_min_luma_transform_block_size_minus2 + 2;
  const uint32_t maximum_transform_log2 =
      minimum_transform_log2 +
      sequence->log2_diff_max_min_luma_transform_block_size;
  if (maximum_block_log2 < 4 || maximum_block_log2 > 6 ||
      minimum_transform_log2 >= minimum_block_log2 ||
      maximum_transform_log2 > 5 ||
      maximum_transform_log2 > maximum_block_log2 ||
      sequence->maximum_transform_hierarchy_depth_inter >
          maximum_block_log2 - minimum_transform_log2 ||
      sequence->maximum_transform_hierarchy_depth_intra >
          maximum_block_log2 - minimum_transform_log2 ||
      ((configuration->coded_width | configuration->coded_height) &
       ((1u << minimum_block_log2) - 1)) != 0 ||
      parameters->coding_unit_qp_delta_depth >
          sequence->log2_diff_max_min_luma_coding_block_size ||
      (!parameters->coding_unit_qp_delta_enabled &&
       parameters->coding_unit_qp_delta_depth != 0) ||
      parameters->log2_parallel_merge_level_minus2 > maximum_block_log2 - 2) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (sequence->pcm_enabled &&
      (sequence->pcm_luma_bit_depth_minus1 >= hevc->sample_bit_depth ||
       sequence->pcm_chroma_bit_depth_minus1 >= hevc->sample_bit_depth ||
       sequence->log2_min_pcm_luma_coding_block_size_minus3 > 2 ||
       sequence->log2_diff_max_min_pcm_luma_coding_block_size > 2 ||
       sequence->log2_min_pcm_luma_coding_block_size_minus3 + 3 <
           minimum_block_log2 ||
       sequence->log2_min_pcm_luma_coding_block_size_minus3 + 3 +
               sequence->log2_diff_max_min_pcm_luma_coding_block_size >
           maximum_block_log2)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t block_size = 1u << maximum_block_log2;
  const uint32_t columns =
      (configuration->coded_width + block_size - 1) / block_size;
  const uint32_t rows =
      (configuration->coded_height + block_size - 1) / block_size;
  if (parameters->tile_column_count_minus1 >= columns ||
      parameters->tile_row_count_minus1 >= rows ||
      (!parameters->tiles_enabled &&
       (parameters->tile_column_count_minus1 != 0 ||
        parameters->tile_row_count_minus1 != 0 ||
        parameters->uniform_tile_spacing != 0 ||
        parameters->loop_filter_across_tiles_enabled != 0))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (parameters->tiles_enabled && !parameters->uniform_tile_spacing) {
    uint64_t width = 0, height = 0;
    for (uint32_t index = 0; index < parameters->tile_column_count_minus1;
         ++index) {
      width += (uint64_t)parameters->tile_column_width_minus1[index] + 1;
    }
    for (uint32_t index = 0; index < parameters->tile_row_count_minus1;
         ++index) {
      height += (uint64_t)parameters->tile_row_height_minus1[index] + 1;
    }
    if (width >= columns || height >= rows) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  for (uint32_t index = 0; index < picture->reference_count; ++index) {
    const amdf_media_hevc_reference_t* reference = &picture->references[index];
    if (reference->binding.frame == NULL || reference->long_term > 1 ||
        (reference->long_term && !sequence->long_term_references_present)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  uint32_t used = 0;
  if (!amdf_media_hevc_reference_set_valid(picture, &picture->short_term_before,
                                           0, &used) ||
      !amdf_media_hevc_reference_set_valid(picture, &picture->short_term_after,
                                           1, &used) ||
      !amdf_media_hevc_reference_set_valid(picture, &picture->long_term, 2,
                                           &used) ||
      (!picture->is_intra && used == 0) || (picture->is_idr && used != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  uint64_t end = 0;
  for (uint32_t index = 0; index < picture->slice_count; ++index) {
    const amdf_media_hevc_slice_t* slice = &picture->slices[index];
    if (slice->byte_length < 6 || slice->byte_offset < end ||
        slice->byte_offset > compressed_byte_length ||
        slice->byte_length > compressed_byte_length - slice->byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    end = (uint64_t)slice->byte_offset + slice->byte_length;
  }
  if (sequence->scaling_list_enabled) {
    for (uint32_t i = 0; i < 6; ++i) {
      if (picture->scaling_matrices->dc_16x16[i] == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (picture->scaling_matrices->dc_32x32[i] == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  return AMDF_STATUS_OK;
}

void amdf_media_hevc_scan_scaling_list(const uint8_t* source, uint32_t width,
                                       uint8_t* target) {
  // Both native interfaces consume the same diagonal permutation. Fixed maps
  // avoid expanding repeated diagonal bounds tests at each inlined callsite.
  static const uint8_t scan4x4[16] = {0, 4, 1,  8,  5, 2,  12, 9,
                                      6, 3, 13, 10, 7, 14, 11, 15};
  static const uint8_t scan8x8[64] = {
      0,  8,  1,  16, 9,  2,  24, 17, 10, 3,  32, 25, 18, 11, 4,  40,
      33, 26, 19, 12, 5,  48, 41, 34, 27, 20, 13, 6,  56, 49, 42, 35,
      28, 21, 14, 7,  57, 50, 43, 36, 29, 22, 15, 58, 51, 44, 37, 30,
      23, 59, 52, 45, 38, 31, 60, 53, 46, 39, 61, 54, 47, 62, 55, 63};
  const uint8_t* scan = width == 4 ? scan4x4 : scan8x8;
  for (uint32_t index = 0; index < width * width; ++index) {
    target[index] = source[scan[index]];
  }
}

amdf_status_t amdf_media_hevc_validate_encode_configuration(
    const amdf_media_configuration_t* configuration,
    const amdf_media_hevc_configuration_t* hevc,
    const amdf_media_encode_configuration_t* encode,
    const amdf_media_hevc_encode_configuration_t* hevc_encode) {
  amdf_status_t status = amdf_structure_validate_header(
      hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION, sizeof(*hevc));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_media_encode_validate_configuration(configuration, encode);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      hevc_encode, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_CONFIGURATION,
      sizeof(*hevc_encode));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (hevc->next != encode || encode->next != hevc_encode ||
      (hevc->profile != AMDF_MEDIA_HEVC_PROFILE_MAIN &&
       hevc->profile != AMDF_MEDIA_HEVC_PROFILE_MAIN10) ||
      (hevc->sample_bit_depth != 8 && hevc->sample_bit_depth != 10) ||
      (hevc->profile == AMDF_MEDIA_HEVC_PROFILE_MAIN &&
       hevc->sample_bit_depth != 8) ||
      configuration->format != (hevc->sample_bit_depth == 8
                                    ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                    : AMDF_MEDIA_PIXEL_FORMAT_P010) ||
      hevc_encode->level_idc == 0 || hevc_encode->level_idc > 255 ||
      (hevc_encode->tier != AMDF_MEDIA_HEVC_TIER_MAIN &&
       hevc_encode->tier != AMDF_MEDIA_HEVC_TIER_HIGH) ||
      hevc_encode->log2_max_picture_order_count_lsb_minus4 > 12 ||
      hevc_encode->minimum_luma_coding_unit_log2 < 3 ||
      hevc_encode->maximum_luma_coding_unit_log2 > 6 ||
      hevc_encode->minimum_luma_coding_unit_log2 >
          hevc_encode->maximum_luma_coding_unit_log2 ||
      hevc_encode->minimum_luma_transform_unit_log2 < 2 ||
      hevc_encode->maximum_luma_transform_unit_log2 > 5 ||
      hevc_encode->minimum_luma_transform_unit_log2 >
          hevc_encode->maximum_luma_transform_unit_log2 ||
      hevc_encode->maximum_luma_transform_unit_log2 >
          hevc_encode->maximum_luma_coding_unit_log2 ||
      hevc_encode->maximum_transform_hierarchy_depth_inter >
          hevc_encode->maximum_luma_coding_unit_log2 -
              hevc_encode->minimum_luma_transform_unit_log2 ||
      hevc_encode->maximum_transform_hierarchy_depth_intra >
          hevc_encode->maximum_luma_coding_unit_log2 -
              hevc_encode->minimum_luma_transform_unit_log2) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_hevc_validate_encode_picture(
    const amdf_media_configuration_info_t* requirements,
    const amdf_media_encode_picture_info_t* picture,
    const amdf_media_hevc_encode_picture_info_t* hevc_picture) {
  amdf_status_t status =
      amdf_media_encode_validate_picture(requirements, picture);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      hevc_picture, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_PICTURE_INFO,
      sizeof(*hevc_picture));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (picture->next != hevc_picture || hevc_picture->used_as_reference > 1 ||
      (hevc_picture->reference_count != 0 &&
       hevc_picture->references == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (hevc_picture->frame_type == AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR) {
    if (hevc_picture->picture_order_count != 0 ||
        hevc_picture->reference_count != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  } else if (hevc_picture->frame_type == AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_P) {
    if (hevc_picture->reference_count != 1) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  } else {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t index = 0; index < hevc_picture->reference_count; ++index) {
    const amdf_media_hevc_encode_reference_t* reference =
        &hevc_picture->references[index];
    if (reference->binding.frame == NULL || reference->long_term != 0 ||
        reference->binding.slot_index >=
            requirements->configuration.reference_image_count ||
        reference->binding.slot_index == picture->reconstruction_slot_index) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  return AMDF_STATUS_OK;
}
