// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/encode.h"

#include "libamdf/src/structure.h"

amdf_status_t amdf_media_encode_validate_configuration(
    const amdf_media_configuration_t* configuration,
    const amdf_media_encode_configuration_t* encode) {
  amdf_status_t status = amdf_structure_validate_header(
      encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION, sizeof(*encode));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (configuration->direction != AMDF_MEDIA_DIRECTION_ENCODE ||
      (configuration->codec != AMDF_MEDIA_CODEC_H264 &&
       configuration->codec != AMDF_MEDIA_CODEC_HEVC &&
       configuration->codec != AMDF_MEDIA_CODEC_AV1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (configuration->reference_image_count < 2 ||
      encode->intra_picture_period == 0 || encode->inter_picture_period != 1 ||
      encode->rate_control.temporal_layer_count == 0 ||
      encode->rate_control.temporal_layers == NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (encode->rate_control.temporal_layer_count >
      AMDF_MEDIA_ENCODE_TEMPORAL_LAYER_STORAGE_CAPACITY) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const bool av1 = configuration->codec == AMDF_MEDIA_CODEC_AV1;
  const uint32_t minimum_quantization_parameter = av1 ? 1 : 0;
  const uint32_t maximum_quantization_parameter = av1 ? 255 : 51;
  if (encode->rate_control.mode != AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP &&
      encode->rate_control.mode != AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR &&
      encode->rate_control.mode != AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_VBR) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  for (uint32_t index = 0; index < encode->rate_control.temporal_layer_count;
       ++index) {
    const amdf_media_encode_rate_control_layer_t* layer =
        &encode->rate_control.temporal_layers[index];
    if (layer->frame_rate_numerator == 0 ||
        layer->frame_rate_denominator == 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    if (encode->rate_control.mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP) {
      const amdf_media_encode_cqp_t* cqp = &layer->control.cqp;
      if (cqp->intra_quantization_parameter < minimum_quantization_parameter ||
          cqp->intra_quantization_parameter > maximum_quantization_parameter ||
          cqp->predicted_quantization_parameter <
              minimum_quantization_parameter ||
          cqp->predicted_quantization_parameter >
              maximum_quantization_parameter) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      continue;
    }
    const amdf_media_encode_bit_rate_t* bit_rate = &layer->control.bit_rate;
    if (bit_rate->minimum_quantization_parameter <
            minimum_quantization_parameter ||
        bit_rate->minimum_quantization_parameter >
            bit_rate->maximum_quantization_parameter ||
        bit_rate->maximum_quantization_parameter >
            maximum_quantization_parameter ||
        bit_rate->target_bit_rate == 0 ||
        bit_rate->virtual_buffer_capacity == 0 ||
        bit_rate->initial_virtual_buffer_fullness >
            bit_rate->virtual_buffer_capacity ||
        (encode->rate_control.mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR
             ? bit_rate->peak_bit_rate != bit_rate->target_bit_rate
             : bit_rate->peak_bit_rate < bit_rate->target_bit_rate)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t amdf_media_encode_validate_picture(
    const amdf_media_configuration_info_t* requirements,
    const amdf_media_encode_configuration_t* configuration,
    const amdf_media_encode_picture_info_t* picture) {
  const amdf_status_t status = amdf_structure_validate_header(
      picture, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO, sizeof(*picture));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (picture->input == NULL || picture->reconstruction == NULL ||
      picture->compressed.memory == NULL ||
      picture->reconstruction_slot_index >=
          requirements->configuration.reference_image_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (picture->temporal_layer_index >=
      configuration->rate_control.temporal_layer_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (picture->compressed.byte_offset %
              requirements->compressed_byte_offset_alignment !=
          0 ||
      picture->compressed.byte_length %
              requirements->compressed_byte_length_alignment !=
          0 ||
      picture->compressed.byte_length <
          requirements->minimum_compressed_byte_length) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  return AMDF_STATUS_OK;
}
