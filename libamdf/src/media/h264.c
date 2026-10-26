// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/h264.h"

#include "libamdf/src/media/encode.h"
#include "libamdf/src/structure.h"

bool amdf_media_h264_level_idc_valid(uint32_t level_idc) {
  switch (level_idc) {
    case 10:
    case 11:
    case 12:
    case 13:
    case 20:
    case 21:
    case 22:
    case 30:
    case 31:
    case 32:
    case 40:
    case 41:
    case 42:
    case 50:
    case 51:
    case 52:
    case 60:
    case 61:
    case 62:
      return true;
    default:
      return false;
  }
}

amdf_status_t amdf_media_h264_validate(
    const amdf_media_configuration_t* configuration,
    const amdf_media_h264_configuration_t* h264,
    const amdf_media_h264_decode_picture_info_t* picture,
    uint64_t compressed_byte_length) {
  const amdf_media_h264_sequence_parameters_t* sequence = &picture->sequence;
  const amdf_media_h264_picture_parameters_t* parameters = &picture->picture;
  if (picture->scaling_matrices == NULL || picture->slice_count == 0 ||
      picture->slices == NULL || picture->reference_count > 16 ||
      (picture->reference_count != 0 && picture->references == NULL) ||
      sequence->maximum_reference_frame_count > 16 ||
      sequence->maximum_reference_frame_count + 1 >
          configuration->reference_image_count ||
      picture->reference_count > sequence->maximum_reference_frame_count ||
      sequence->log2_max_frame_num_minus4 > 12 ||
      sequence->picture_order_count_type > 2 ||
      sequence->log2_max_picture_order_count_lsb_minus4 > 12 ||
      (sequence->picture_order_count_type != 0 &&
       sequence->log2_max_picture_order_count_lsb_minus4 != 0) ||
      (sequence->picture_order_count_type != 1 &&
       sequence->delta_picture_order_always_zero != 0) ||
      sequence->delta_picture_order_always_zero > 1 ||
      sequence->direct_8x8_inference > 1 || picture->is_reference > 1 ||
      picture->is_intra > 1 || parameters->entropy_coding_mode > 1 ||
      parameters->bottom_field_picture_order_present > 1 ||
      parameters->default_reference_l0_count_minus1 > 31 ||
      parameters->default_reference_l1_count_minus1 > 31 ||
      parameters->weighted_prediction > 1 ||
      parameters->weighted_biprediction > 2 ||
      parameters->initial_qp_minus26 < -26 ||
      parameters->initial_qp_minus26 > 25 ||
      parameters->initial_qs_minus26 < -26 ||
      parameters->initial_qs_minus26 > 25 ||
      parameters->chroma_qp_index_offset < -12 ||
      parameters->chroma_qp_index_offset > 12 ||
      parameters->second_chroma_qp_index_offset < -12 ||
      parameters->second_chroma_qp_index_offset > 12 ||
      parameters->deblocking_filter_control_present > 1 ||
      parameters->constrained_intra_prediction > 1 ||
      parameters->redundant_picture_count_present > 1 ||
      parameters->transform_8x8_mode > 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((h264->profile != AMDF_MEDIA_H264_PROFILE_HIGH &&
       parameters->transform_8x8_mode != 0) ||
      (h264->profile == AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE &&
       (parameters->entropy_coding_mode != 0 ||
        parameters->weighted_prediction != 0 ||
        parameters->weighted_biprediction != 0))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t maximum_frame_number =
      1u << (sequence->log2_max_frame_num_minus4 + 4);
  if (picture->frame_number >= maximum_frame_number) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  for (uint32_t index = 0; index < picture->reference_count; ++index) {
    const amdf_media_h264_reference_t* reference = &picture->references[index];
    if (reference->binding.frame == NULL || reference->long_term > 1 ||
        reference->frame_number > UINT16_MAX ||
        (reference->long_term == 0 &&
         reference->frame_number >= maximum_frame_number)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  uint64_t end = 0;
  for (uint32_t index = 0; index < picture->slice_count; ++index) {
    const amdf_media_h264_slice_t* slice = &picture->slices[index];
    if (slice->byte_length < 4 || slice->byte_offset < end ||
        slice->byte_offset > compressed_byte_length ||
        slice->byte_length > compressed_byte_length - slice->byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    end = (uint64_t)slice->byte_offset + slice->byte_length;
  }
  const uint8_t* scaling = (const uint8_t*)picture->scaling_matrices;
  for (uint32_t i = 0; i < sizeof(*picture->scaling_matrices); ++i) {
    if (scaling[i] == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_h264_validate_encode_configuration(
    const amdf_media_configuration_t* configuration,
    const amdf_media_h264_configuration_t* h264,
    const amdf_media_encode_configuration_t* encode,
    const amdf_media_h264_encode_configuration_t* h264_encode) {
  amdf_status_t status = amdf_structure_validate_header(
      h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION, sizeof(*h264));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_media_encode_validate_configuration(configuration, encode);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      h264_encode, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION,
      sizeof(*h264_encode));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (h264->next != encode || encode->next != h264_encode ||
      h264->profile != AMDF_MEDIA_H264_PROFILE_MAIN ||
      !amdf_media_h264_level_idc_valid(h264->level_idc) ||
      h264->level_idc > 52 ||
      configuration->format != AMDF_MEDIA_PIXEL_FORMAT_NV12 ||
      h264_encode->picture_order_count_type != 2 ||
      h264_encode->log2_max_frame_number_minus4 > 12 ||
      h264_encode->log2_max_picture_order_count_lsb_minus4 != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_h264_validate_encode_picture(
    const amdf_media_configuration_info_t* requirements,
    const amdf_media_encode_configuration_t* configuration,
    const amdf_media_encode_picture_info_t* picture,
    const amdf_media_h264_encode_configuration_t* h264_encode,
    const amdf_media_h264_encode_picture_info_t* h264_picture) {
  amdf_status_t status =
      amdf_media_encode_validate_picture(requirements, configuration, picture);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  status = amdf_structure_validate_input(
      h264_picture, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO,
      sizeof(*h264_picture));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (picture->next != h264_picture || h264_picture->used_as_reference > 1 ||
      (h264_picture->reference_count != 0 &&
       h264_picture->references == NULL)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t maximum_frame_number =
      1u << (h264_encode->log2_max_frame_number_minus4 + 4);
  if (h264_picture->frame_number >= maximum_frame_number) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (h264_picture->frame_type == AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR) {
    if (h264_picture->frame_number != 0 ||
        h264_picture->picture_order_count != 0 ||
        h264_picture->reference_count != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  } else if (h264_picture->frame_type == AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P) {
    if (h264_picture->reference_count != 1) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  } else {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t index = 0; index < h264_picture->reference_count; ++index) {
    const amdf_media_h264_encode_reference_t* reference =
        &h264_picture->references[index];
    if (reference->binding.frame == NULL ||
        reference->binding.slot_index >=
            requirements->configuration.reference_image_count ||
        reference->binding.slot_index == picture->reconstruction_slot_index ||
        (reference->frame_type != AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR &&
         reference->frame_type != AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P) ||
        reference->frame_number >= maximum_frame_number ||
        reference->temporal_layer_index >=
            configuration->rate_control.temporal_layer_count) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  return AMDF_STATUS_OK;
}
