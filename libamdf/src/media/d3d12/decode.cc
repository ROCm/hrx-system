// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/decode.h"

#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {
namespace {

template <typename T>
amdf_status_t PrepareReferences(const Session& session, uint32_t current_slot,
                                uint32_t count, const T* references,
                                DecodePicture* result) {
  const uint32_t capacity = session.info.configuration.reference_image_count;
  if (current_slot >= capacity) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  uint32_t used = 1u << current_slot;
  for (uint32_t index = 0; index < count; ++index) {
    const auto& binding = references[index].binding;
    if (binding.slot_index >= capacity) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    if ((used & (1u << binding.slot_index)) != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    used |= 1u << binding.slot_index;
    result->references[index] = binding;
  }
  result->reference_count = count;
  return AMDF_STATUS_OK;
}

}  // namespace

amdf_status_t PrepareDecode(const Session& session, const void* input,
                            uint64_t compressed_byte_length,
                            uint32_t current_reference_slot,
                            uint32_t maximum_reference_count,
                            uint32_t maximum_slice_count, void* slices,
                            DecodePicture* result) {
  auto& arguments = result->arguments;
  switch (session.info.configuration.codec) {
    case AMDF_MEDIA_CODEC_H264: {
      amdf_status_t status = amdf_structure_validate_input(
          input, AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO,
          sizeof(amdf_media_h264_decode_picture_info_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& picture =
          *static_cast<const amdf_media_h264_decode_picture_info_t*>(input);
      if (picture.reference_count > maximum_reference_count ||
          picture.slice_count > maximum_slice_count) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      status = ValidateH264(session, picture, compressed_byte_length);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      status = PrepareReferences(session, current_reference_slot,
                                 picture.reference_count, picture.references,
                                 result);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      LowerH264(session, current_reference_slot, picture,
                &result->parameters.h264, &result->matrices.h264,
                static_cast<DXVA_Slice_H264_Short*>(slices));
      arguments.NumFrameArguments = 3;
      arguments.FrameArguments[0] = {
          D3D12_VIDEO_DECODE_ARGUMENT_TYPE_PICTURE_PARAMETERS,
          sizeof(result->parameters.h264), &result->parameters.h264};
      arguments.FrameArguments[1] = {
          D3D12_VIDEO_DECODE_ARGUMENT_TYPE_INVERSE_QUANTIZATION_MATRIX,
          sizeof(result->matrices.h264), &result->matrices.h264};
      arguments.FrameArguments[2] = {
          D3D12_VIDEO_DECODE_ARGUMENT_TYPE_SLICE_CONTROL,
          static_cast<UINT>(picture.slice_count *
                            sizeof(DXVA_Slice_H264_Short)),
          slices};
      break;
    }
    case AMDF_MEDIA_CODEC_HEVC: {
      amdf_status_t status = amdf_structure_validate_input(
          input, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO,
          sizeof(amdf_media_hevc_decode_picture_info_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& picture =
          *static_cast<const amdf_media_hevc_decode_picture_info_t*>(input);
      if (picture.reference_count > maximum_reference_count ||
          picture.slice_count > maximum_slice_count) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      status = ValidateHevc(session, picture, compressed_byte_length);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      status = PrepareReferences(session, current_reference_slot,
                                 picture.reference_count, picture.references,
                                 result);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      LowerHevc(session, current_reference_slot, picture,
                &result->parameters.hevc, &result->matrices.hevc,
                static_cast<DXVA_Slice_HEVC_Short*>(slices));
      arguments.NumFrameArguments = 1;
      arguments.FrameArguments[0] = {
          D3D12_VIDEO_DECODE_ARGUMENT_TYPE_PICTURE_PARAMETERS,
          sizeof(result->parameters.hevc), &result->parameters.hevc};
      if (picture.sequence.scaling_list_enabled) {
        arguments.FrameArguments[arguments.NumFrameArguments++] = {
            D3D12_VIDEO_DECODE_ARGUMENT_TYPE_INVERSE_QUANTIZATION_MATRIX,
            sizeof(result->matrices.hevc), &result->matrices.hevc};
      }
      arguments.FrameArguments[arguments.NumFrameArguments++] = {
          D3D12_VIDEO_DECODE_ARGUMENT_TYPE_SLICE_CONTROL,
          static_cast<UINT>(picture.slice_count *
                            sizeof(DXVA_Slice_HEVC_Short)),
          slices};
      break;
    }
    default:
      amdf_abort();
  }
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
