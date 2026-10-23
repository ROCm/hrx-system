// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/av1.h"

static bool amdf_media_av1_grain_points_valid(const uint8_t (*points)[2],
                                              uint32_t count) {
  for (uint32_t i = 1; i < count; ++i) {
    if (points[i - 1][0] >= points[i][0]) {
      return false;
    }
  }
  return true;
}

amdf_status_t amdf_media_av1_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_av1_decode_picture_info_t* picture,
    uint64_t compressed_byte_length) {
  const amdf_media_av1_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_av1_loop_filter_t* filter = &picture->loop_filter;
  const amdf_media_av1_quantization_t* quantization = &picture->quantization;
  const amdf_media_av1_segmentation_t* segments = &picture->segmentation;
  const amdf_media_av1_film_grain_t* grain = &picture->film_grain;
  const uint32_t flags =
      sequence->use_128x128_superblock | sequence->filter_intra |
      sequence->intra_edge_filter | sequence->interintra_compound |
      sequence->masked_compound | sequence->dual_filter |
      sequence->joint_compound | sequence->reference_frame_motion_vectors |
      sequence->cdef | sequence->restoration | sequence->film_grain |
      picture->show_frame | picture->showable_frame |
      picture->disable_cdf_update | picture->disable_frame_end_cdf_update |
      picture->warped_motion | picture->screen_content_tools |
      picture->integer_motion_vectors | picture->intra_block_copy |
      picture->high_precision_motion_vectors | picture->switchable_motion_mode |
      picture->reference_select | picture->skip_mode |
      picture->reduced_transform_set | picture->use_reference_motion_vectors |
      filter->deltas_enabled | filter->deltas_updated | filter->delta_present |
      filter->delta_multi | quantization->delta_present | segments->enabled |
      segments->update_map | segments->temporal_update | segments->update_data |
      grain->apply;
  const bool intra = picture->frame_type == 0 || picture->frame_type == 2;
  if (flags > 1 || sequence->order_hint_bit_count > 8 ||
      picture->frame_type > 3 || picture->primary_reference > 7 ||
      picture->order_hint >= (1u << sequence->order_hint_bit_count) ||
      picture->transform_mode > 2 || picture->interpolation_filter > 4 ||
      picture->superres_denominator < 8 || picture->superres_denominator > 16 ||
      filter->sharpness > 7 || filter->delta_resolution > 3 ||
      quantization->base_index > 255 || quantization->y_dc_delta < -64 ||
      quantization->y_dc_delta > 63 || quantization->delta_resolution > 3 ||
      picture->cdef.damping_minus3 > 3 || picture->cdef.bits > 3 ||
      picture->reference_count > 8 ||
      (picture->reference_count != 0 && picture->references == NULL) ||
      (intra && picture->primary_reference != 7) ||
      (picture->intra_block_copy && (!intra || !picture->screen_content_tools ||
                                     picture->superres_denominator != 8)) ||
      (picture->high_precision_motion_vectors &&
       picture->integer_motion_vectors) ||
      (picture->use_reference_motion_vectors &&
       (intra || !sequence->reference_frame_motion_vectors)) ||
      (sequence->order_hint_bit_count == 0 &&
       (sequence->joint_compound ||
        sequence->reference_frame_motion_vectors)) ||
      (filter->deltas_updated && !filter->deltas_enabled) ||
      (filter->delta_present && !quantization->delta_present) ||
      (filter->delta_multi && !filter->delta_present) ||
      (segments->temporal_update && !segments->update_map) ||
      ((segments->update_map || segments->update_data) && !segments->enabled) ||
      (grain->apply && !sequence->film_grain)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    if (filter->levels[i] > 63) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 3; ++i) {
    if (filter->restoration_types[i] > 3 ||
        filter->log2_restoration_unit_sizes[i] < 5 ||
        filter->log2_restoration_unit_sizes[i] > 8 ||
        (filter->restoration_types[i] != 0 && !sequence->restoration) ||
        (quantization->matrix_levels[0] == 255
             ? quantization->matrix_levels[i] != 255
             : quantization->matrix_levels[i] > 15)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 2; ++i) {
    if (filter->mode_deltas[i] < -64 || filter->mode_deltas[i] > 63 ||
        quantization->chroma_dc_deltas[i] < -64 ||
        quantization->chroma_dc_deltas[i] > 63 ||
        quantization->chroma_ac_deltas[i] < -64 ||
        quantization->chroma_ac_deltas[i] > 63) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 8; ++i) {
    if (filter->reference_deltas[i] < -64 || filter->reference_deltas[i] > 63 ||
        picture->cdef.luma_primary[i] > 15 ||
        picture->cdef.luma_secondary[i] > 3 ||
        picture->cdef.chroma_primary[i] > 15 ||
        picture->cdef.chroma_secondary[i] > 3 ||
        (picture->reference_map[i] != 255 &&
         picture->reference_map[i] >= picture->reference_count)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    const int16_t* values = segments->feature_values[i];
    if (values[0] < -255 || values[0] > 255 || values[5] < 0 || values[5] > 7 ||
        values[6] != 0 || values[7] != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    for (uint32_t j = 1; j < 5; ++j) {
      if (values[j] < -63 || values[j] > 63) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  for (uint32_t i = 0; i < 7; ++i) {
    const uint32_t index = picture->active_reference_indices[i];
    if (picture->reference_sign_bias[i] > 1 ||
        picture->global_motion[i].type > 3 ||
        picture->global_motion[i].invalid > 1 ||
        (intra ? index != 255
               : index >= 8 || picture->reference_map[index] == 255)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < picture->reference_count; ++i) {
    if (picture->references[i].frame == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  if (picture->tile_column_count == 0 || picture->tile_column_count > 64 ||
      picture->tile_row_count == 0 || picture->tile_row_count > 64 ||
      picture->tiles == NULL || picture->tile_column_starts[0] != 0 ||
      picture->tile_row_starts[0] != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t tile_count =
      picture->tile_column_count * picture->tile_row_count;
  if (tile_count > 256) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (picture->context_update_tile >= tile_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint32_t superblock_size = sequence->use_128x128_superblock ? 128 : 64;
  const uint32_t coded_width =
      (configuration->coded_width * 8 + picture->superres_denominator / 2) /
      picture->superres_denominator;
  if (picture->tile_column_starts[picture->tile_column_count] !=
          (coded_width + superblock_size - 1) / superblock_size ||
      picture->tile_row_starts[picture->tile_row_count] !=
          (configuration->coded_height + superblock_size - 1) /
              superblock_size) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  for (uint32_t i = 0; i < picture->tile_column_count; ++i) {
    if (picture->tile_column_starts[i] >= picture->tile_column_starts[i + 1]) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < picture->tile_row_count; ++i) {
    if (picture->tile_row_starts[i] >= picture->tile_row_starts[i + 1]) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  uint64_t end = 0;
  for (uint32_t i = 0; i < tile_count; ++i) {
    const amdf_media_av1_tile_t* tile = &picture->tiles[i];
    if (tile->byte_length == 0 || tile->byte_offset < end ||
        tile->byte_offset > compressed_byte_length ||
        tile->byte_length > compressed_byte_length - tile->byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    end = (uint64_t)tile->byte_offset + tile->byte_length;
  }
  if (grain->apply &&
      (grain->seed > UINT16_MAX || grain->luma_point_count > 14 ||
       grain->u_point_count > 10 || grain->v_point_count > 10 ||
       grain->scaling_shift_minus8 > 3 || grain->coefficient_lag > 3 ||
       grain->coefficient_shift_minus6 > 3 || grain->scale_shift > 3 ||
       grain->chroma_offsets[0] > 511 || grain->chroma_offsets[1] > 511 ||
       (grain->chroma_scaling_from_luma | grain->overlap |
        grain->clip_to_restricted_range) > 1 ||
       ((grain->chroma_scaling_from_luma || grain->luma_point_count == 0) &&
        (grain->u_point_count != 0 || grain->v_point_count != 0)) ||
       !amdf_media_av1_grain_points_valid(grain->luma_points,
                                          grain->luma_point_count) ||
       !amdf_media_av1_grain_points_valid(grain->u_points,
                                          grain->u_point_count) ||
       !amdf_media_av1_grain_points_valid(grain->v_points,
                                          grain->v_point_count))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}
