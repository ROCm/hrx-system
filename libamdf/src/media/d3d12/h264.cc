// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/h264.h"

namespace amdf::media::d3d12 {

amdf_status_t ValidateH264(const Session& session,
                           const amdf_media_h264_decode_picture_info_t& picture,
                           uint64_t compressed_byte_length) {
  const auto& sequence = picture.sequence;
  const auto& parameters = picture.picture;
  if (picture.scaling_matrices == nullptr || picture.slice_count == 0 ||
      picture.slices == nullptr || picture.reference_count > 16 ||
      (picture.reference_count != 0 && picture.references == nullptr) ||
      sequence.maximum_reference_frame_count > 16 ||
      sequence.maximum_reference_frame_count + 1 >
          session.info.configuration.reference_image_count ||
      picture.reference_count > sequence.maximum_reference_frame_count ||
      sequence.log2_max_frame_num_minus4 > 12 ||
      sequence.picture_order_count_type > 2 ||
      sequence.log2_max_picture_order_count_lsb_minus4 > 12 ||
      (sequence.picture_order_count_type != 0 &&
       sequence.log2_max_picture_order_count_lsb_minus4 != 0) ||
      (sequence.picture_order_count_type != 1 &&
       sequence.delta_picture_order_always_zero != 0) ||
      sequence.delta_picture_order_always_zero > 1 ||
      sequence.direct_8x8_inference > 1 || picture.is_reference > 1 ||
      picture.is_intra > 1 || parameters.entropy_coding_mode > 1 ||
      parameters.bottom_field_picture_order_present > 1 ||
      parameters.default_reference_l0_count_minus1 > 31 ||
      parameters.default_reference_l1_count_minus1 > 31 ||
      parameters.weighted_prediction > 1 ||
      parameters.weighted_biprediction > 2 ||
      parameters.initial_qp_minus26 < -26 ||
      parameters.initial_qp_minus26 > 25 ||
      parameters.initial_qs_minus26 < -26 ||
      parameters.initial_qs_minus26 > 25 ||
      parameters.chroma_qp_index_offset < -12 ||
      parameters.chroma_qp_index_offset > 12 ||
      parameters.second_chroma_qp_index_offset < -12 ||
      parameters.second_chroma_qp_index_offset > 12 ||
      parameters.deblocking_filter_control_present > 1 ||
      parameters.constrained_intra_prediction > 1 ||
      parameters.redundant_picture_count_present > 1 ||
      parameters.transform_8x8_mode > 1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if ((session.h264.profile != AMDF_MEDIA_H264_PROFILE_HIGH &&
       parameters.transform_8x8_mode != 0) ||
      (session.h264.profile == AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE &&
       (parameters.entropy_coding_mode != 0 ||
        parameters.weighted_prediction != 0 ||
        parameters.weighted_biprediction != 0))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t maximum_frame_number =
      1u << (sequence.log2_max_frame_num_minus4 + 4);
  if (picture.frame_number >= maximum_frame_number) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& reference = picture.references[index];
    if (reference.binding.frame == nullptr || reference.long_term > 1 ||
        reference.frame_number > UINT16_MAX ||
        (reference.long_term == 0 &&
         reference.frame_number >= maximum_frame_number)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  uint64_t end = 0;
  for (uint32_t index = 0; index < picture.slice_count; ++index) {
    const auto& slice = picture.slices[index];
    if (slice.byte_length < 4 || slice.byte_offset < end ||
        slice.byte_offset > compressed_byte_length ||
        slice.byte_length > compressed_byte_length - slice.byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    end = uint64_t(slice.byte_offset) + slice.byte_length;
  }
  for (const auto& matrix : picture.scaling_matrices->scaling_4x4) {
    for (uint8_t value : matrix) {
      if (value == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  for (const auto& matrix : picture.scaling_matrices->scaling_8x8) {
    for (uint8_t value : matrix) {
      if (value == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  return AMDF_STATUS_OK;
}

void LowerH264(const Session& session, uint32_t current_reference_slot,
               const amdf_media_h264_decode_picture_info_t& picture,
               DXVA_PicParams_H264* native, DXVA_Qmatrix_H264* matrices,
               DXVA_Slice_H264_Short* slices) {
  const auto& sequence = picture.sequence;
  const auto& parameters = picture.picture;
  *native = {};
  native->wFrameWidthInMbsMinus1 =
      static_cast<USHORT>(session.info.configuration.coded_width / 16 - 1);
  native->wFrameHeightInMbsMinus1 =
      static_cast<USHORT>(session.info.configuration.coded_height / 16 - 1);
  native->CurrPic.bPicEntry = static_cast<UCHAR>(current_reference_slot);
  native->num_ref_frames =
      static_cast<UCHAR>(sequence.maximum_reference_frame_count);
  native->chroma_format_idc = 1;
  native->RefPicFlag = picture.is_reference;
  native->constrained_intra_pred_flag = parameters.constrained_intra_prediction;
  native->MbsConsecutiveFlag = 1;
  native->frame_mbs_only_flag = 1;
  native->MinLumaBipredSize8x8Flag = session.h264.level_idc >= 31;
  native->IntraPicFlag = picture.is_intra;
  // DXVA's current off-host bitstream contract uses these fixed values even
  // though their field names suggest the original partial-bitstream protocol.
  native->Reserved16Bits = 3;
  native->ContinuationFlag = 1;
  native->StatusReportFeedbackNumber = 1;
  native->CurrFieldOrderCnt[0] = picture.top_field_order_count;
  native->CurrFieldOrderCnt[1] = picture.bottom_field_order_count;
  native->pic_init_qp_minus26 =
      static_cast<CHAR>(parameters.initial_qp_minus26);
  native->pic_init_qs_minus26 =
      static_cast<CHAR>(parameters.initial_qs_minus26);
  native->chroma_qp_index_offset =
      static_cast<CHAR>(parameters.chroma_qp_index_offset);
  native->second_chroma_qp_index_offset =
      static_cast<CHAR>(parameters.second_chroma_qp_index_offset);
  native->frame_num = static_cast<USHORT>(picture.frame_number);
  native->log2_max_frame_num_minus4 =
      static_cast<UCHAR>(sequence.log2_max_frame_num_minus4);
  native->pic_order_cnt_type =
      static_cast<UCHAR>(sequence.picture_order_count_type);
  native->log2_max_pic_order_cnt_lsb_minus4 =
      static_cast<UCHAR>(sequence.log2_max_picture_order_count_lsb_minus4);
  native->delta_pic_order_always_zero_flag =
      static_cast<UCHAR>(sequence.delta_picture_order_always_zero);
  native->direct_8x8_inference_flag =
      static_cast<UCHAR>(sequence.direct_8x8_inference);
  native->entropy_coding_mode_flag =
      static_cast<UCHAR>(parameters.entropy_coding_mode);
  native->pic_order_present_flag =
      static_cast<UCHAR>(parameters.bottom_field_picture_order_present);
  native->num_ref_idx_l0_active_minus1 =
      static_cast<UCHAR>(parameters.default_reference_l0_count_minus1);
  native->num_ref_idx_l1_active_minus1 =
      static_cast<UCHAR>(parameters.default_reference_l1_count_minus1);
  native->weighted_pred_flag =
      static_cast<UCHAR>(parameters.weighted_prediction);
  native->weighted_bipred_idc =
      static_cast<UCHAR>(parameters.weighted_biprediction);
  native->deblocking_filter_control_present_flag =
      static_cast<UCHAR>(parameters.deblocking_filter_control_present);
  native->redundant_pic_cnt_present_flag =
      static_cast<UCHAR>(parameters.redundant_picture_count_present);
  native->transform_8x8_mode_flag =
      static_cast<UCHAR>(parameters.transform_8x8_mode);
  for (auto& reference : native->RefFrameList) {
    reference.bPicEntry = 0xff;
  }
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& reference = picture.references[index];
    native->RefFrameList[index].bPicEntry = static_cast<UCHAR>(
        reference.binding.slot_index | (reference.long_term << 7));
    native->FrameNumList[index] = static_cast<USHORT>(reference.frame_number);
    native->FieldOrderCntList[index][0] = reference.top_field_order_count;
    native->FieldOrderCntList[index][1] = reference.bottom_field_order_count;
    native->UsedForReferenceFlags |= 3u << (index * 2);
  }
  static constexpr uint8_t kScan4x4[16] = {0, 1,  4,  8,  5, 2,  3,  6,
                                           9, 12, 13, 10, 7, 11, 14, 15};
  static constexpr uint8_t kScan8x8[64] = {
      0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
      12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
      35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
      58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};
  for (uint32_t matrix = 0; matrix < 6; ++matrix) {
    for (uint32_t index = 0; index < 16; ++index) {
      matrices->bScalingLists4x4[matrix][index] =
          picture.scaling_matrices->scaling_4x4[matrix][kScan4x4[index]];
    }
  }
  for (uint32_t matrix = 0; matrix < 2; ++matrix) {
    for (uint32_t index = 0; index < 64; ++index) {
      matrices->bScalingLists8x8[matrix][index] =
          picture.scaling_matrices->scaling_8x8[matrix][kScan8x8[index]];
    }
  }
  for (uint32_t index = 0; index < picture.slice_count; ++index) {
    slices[index] = {picture.slices[index].byte_offset,
                     picture.slices[index].byte_length, 0};
  }
}

}  // namespace amdf::media::d3d12
