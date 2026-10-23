// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/vp9.h"

amdf_status_t amdf_media_vp9_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_vp9_decode_picture_info_t* picture,
    uint64_t compressed_byte_length) {
  const amdf_media_vp9_loop_filter_t* filter = &picture->loop_filter;
  const amdf_media_vp9_segmentation_t* segments = &picture->segmentation;
  const uint32_t flags =
      picture->frame_type | picture->show_frame | picture->error_resilient |
      picture->intra_only | picture->high_precision_motion_vectors |
      picture->refresh_frame_context | picture->parallel_decoding |
      picture->use_previous_frame_motion_vectors | filter->deltas_enabled |
      filter->deltas_updated | segments->enabled | segments->update_map |
      segments->temporal_update | segments->update_data |
      segments->absolute_values;
  const bool intra = picture->frame_type == 0 || picture->intra_only;
  if (flags > 1 || picture->frame_context_index > 3 ||
      picture->reset_frame_context > 3 || picture->interpolation_filter > 4 ||
      picture->base_quantizer_index > 255 ||
      picture->y_dc_quantizer_delta < -15 ||
      picture->y_dc_quantizer_delta > 15 ||
      picture->uv_dc_quantizer_delta < -15 ||
      picture->uv_dc_quantizer_delta > 15 ||
      picture->uv_ac_quantizer_delta < -15 ||
      picture->uv_ac_quantizer_delta > 15 || picture->log2_tile_columns > 6 ||
      picture->log2_tile_rows > 2 || filter->level > 63 ||
      filter->sharpness > 7 ||
      (filter->deltas_updated && !filter->deltas_enabled) ||
      (segments->temporal_update && !segments->update_map) ||
      ((segments->update_map || segments->update_data) && !segments->enabled) ||
      (picture->intra_only &&
       (picture->frame_type == 0 || picture->show_frame)) ||
      (intra && picture->use_previous_frame_motion_vectors) ||
      (picture->error_resilient &&
       (picture->refresh_frame_context || !picture->parallel_decoding)) ||
      picture->reference_count > 8 ||
      (picture->reference_count != 0 && picture->references == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (picture->uncompressed_header_byte_length == 0 ||
      picture->uncompressed_header_byte_length > UINT16_MAX ||
      picture->compressed_header_byte_length == 0 ||
      picture->compressed_header_byte_length > UINT16_MAX ||
      (uint64_t)picture->uncompressed_header_byte_length +
              picture->compressed_header_byte_length >=
          compressed_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint32_t superblock_columns = (configuration->coded_width + 63) / 64;
  const uint32_t columns = 1u << picture->log2_tile_columns;
  // A frame narrower than four superblocks still has its single tile. VP9's
  // maximum log2 permits a shorter final column when the width is uneven.
  if (superblock_columns > columns * 64 ||
      (columns > 1 && columns * 4 > superblock_columns)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  for (uint32_t i = 0; i < 4; ++i) {
    if (filter->reference_deltas[i] < -63 || filter->reference_deltas[i] > 63) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 2; ++i) {
    if (filter->mode_deltas[i] < -63 || filter->mode_deltas[i] > 63) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 8; ++i) {
    const int16_t* values = segments->feature_values[i];
    if (segments->feature_masks[i] > 15 || values[0] < -255 ||
        values[0] > 255 || values[1] < -63 || values[1] > 63 || values[2] < 0 ||
        values[2] > 3 || values[3] != 0 ||
        (picture->reference_map[i] != 255 &&
         (picture->reference_map[i] >= picture->reference_count ||
          picture->frame_type == 0))) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t index = picture->active_reference_indices[i];
    if (picture->reference_sign_bias[i] > 1 ||
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
  return AMDF_STATUS_OK;
}
