// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/hevc.h"

#include <cstring>

namespace amdf::media::d3d12 {
namespace {

bool ValidReferenceSet(const amdf_media_hevc_decode_picture_info_t& picture,
                       const amdf_media_hevc_reference_set_t& set,
                       uint32_t kind, uint32_t* used) {
  if (set.count > 8) {
    return false;
  }
  int32_t previous = picture.picture_order_count;
  for (uint32_t index = 0; index < set.count; ++index) {
    const uint32_t reference_index = set.indices[index];
    if (reference_index >= picture.reference_count ||
        (*used & (1u << reference_index)) != 0) {
      return false;
    }
    const auto& reference = picture.references[reference_index];
    if ((reference.long_term != 0) != (kind == 2) ||
        (kind == 0 && reference.picture_order_count >= previous) ||
        (kind == 1 && reference.picture_order_count <= previous)) {
      return false;
    }
    previous = reference.picture_order_count;
    *used |= 1u << reference_index;
  }
  return true;
}

template <size_t N>
void DiagonalScalingList(const uint8_t (&source)[N], uint8_t (&target)[N]) {
  // HEVC scans every diagonal in the same direction. H264 alternates them.
  constexpr uint32_t width = N == 16 ? 4 : 8;
  uint32_t position = 0;
  for (uint32_t diagonal = 0; diagonal < 2 * width - 1; ++diagonal) {
    for (uint32_t x = 0; x < width; ++x) {
      if (diagonal >= x && diagonal - x < width) {
        target[position++] = source[(diagonal - x) * width + x];
      }
    }
  }
}

}  // namespace

amdf_status_t ValidateHevc(const Session& session,
                           const amdf_media_hevc_decode_picture_info_t& picture,
                           uint64_t compressed_byte_length) {
  const auto& sequence = picture.sequence;
  const auto& parameters = picture.picture;
  const uint32_t flags =
      sequence.scaling_list_enabled |
      sequence.asymmetric_motion_partitions_enabled |
      sequence.sample_adaptive_offset_enabled | sequence.pcm_enabled |
      sequence.pcm_loop_filter_disabled |
      sequence.long_term_references_present |
      sequence.temporal_motion_vector_prediction_enabled |
      sequence.strong_intra_smoothing_enabled |
      parameters.dependent_slice_segments_enabled |
      parameters.output_flag_present | parameters.sign_data_hiding_enabled |
      parameters.cabac_initialization_present |
      parameters.constrained_intra_prediction |
      parameters.transform_skip_enabled |
      parameters.coding_unit_qp_delta_enabled |
      parameters.slice_chroma_qp_offsets_present |
      parameters.weighted_prediction | parameters.weighted_biprediction |
      parameters.transform_quantization_bypass_enabled |
      parameters.tiles_enabled |
      parameters.entropy_coding_synchronization_enabled |
      parameters.uniform_tile_spacing |
      parameters.loop_filter_across_tiles_enabled |
      parameters.loop_filter_across_slices_enabled |
      parameters.deblocking_filter_override_enabled |
      parameters.deblocking_filter_disabled |
      parameters.reference_lists_modification_present |
      parameters.slice_segment_header_extension_present |
      picture.is_random_access | picture.is_idr | picture.is_intra;
  if (flags > 1 || picture.slice_count == 0 || picture.slices == nullptr ||
      picture.reference_count > 15 ||
      (picture.reference_count != 0 && picture.references == nullptr) ||
      sequence.maximum_decoded_picture_buffer_count_minus1 > 15 ||
      sequence.maximum_decoded_picture_buffer_count_minus1 + 1 >
          session.info.configuration.reference_image_count ||
      picture.reference_count >
          sequence.maximum_decoded_picture_buffer_count_minus1 ||
      sequence.log2_max_picture_order_count_lsb_minus4 > 12 ||
      sequence.log2_min_luma_coding_block_size_minus3 > 3 ||
      sequence.log2_diff_max_min_luma_coding_block_size > 3 ||
      sequence.log2_min_luma_transform_block_size_minus2 > 3 ||
      sequence.log2_diff_max_min_luma_transform_block_size > 3 ||
      sequence.short_term_reference_set_count > 64 ||
      sequence.long_term_reference_count > 32 ||
      (!sequence.long_term_references_present &&
       sequence.long_term_reference_count != 0) ||
      parameters.default_reference_l0_count_minus1 > 14 ||
      parameters.default_reference_l1_count_minus1 > 14 ||
      parameters.initial_qp_minus26 <
          -26 - 6 * int32_t(session.hevc.sample_bit_depth - 8) ||
      parameters.initial_qp_minus26 > 25 ||
      parameters.extra_slice_header_bit_count > 7 ||
      parameters.cb_qp_offset < -12 || parameters.cb_qp_offset > 12 ||
      parameters.cr_qp_offset < -12 || parameters.cr_qp_offset > 12 ||
      parameters.beta_offset_div2 < -6 || parameters.beta_offset_div2 > 6 ||
      parameters.tc_offset_div2 < -6 || parameters.tc_offset_div2 > 6 ||
      parameters.tile_column_count_minus1 > 19 ||
      parameters.tile_row_count_minus1 > 21 ||
      picture.predicted_reference_set_delta_count > 15 ||
      picture.short_term_reference_set_bit_count > UINT16_MAX ||
      (picture.is_random_access && !picture.is_intra) ||
      (picture.is_idr &&
       (!picture.is_random_access || picture.picture_order_count != 0)) ||
      (sequence.scaling_list_enabled && picture.scaling_matrices == nullptr)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t minimum_block_log2 =
      sequence.log2_min_luma_coding_block_size_minus3 + 3;
  const uint32_t maximum_block_log2 =
      minimum_block_log2 + sequence.log2_diff_max_min_luma_coding_block_size;
  const uint32_t minimum_transform_log2 =
      sequence.log2_min_luma_transform_block_size_minus2 + 2;
  const uint32_t maximum_transform_log2 =
      minimum_transform_log2 +
      sequence.log2_diff_max_min_luma_transform_block_size;
  if (maximum_block_log2 < 4 || maximum_block_log2 > 6 ||
      minimum_transform_log2 >= minimum_block_log2 ||
      maximum_transform_log2 > 5 ||
      maximum_transform_log2 > maximum_block_log2 ||
      sequence.maximum_transform_hierarchy_depth_inter >
          maximum_block_log2 - minimum_transform_log2 ||
      sequence.maximum_transform_hierarchy_depth_intra >
          maximum_block_log2 - minimum_transform_log2 ||
      ((session.info.configuration.coded_width |
        session.info.configuration.coded_height) &
       ((1u << minimum_block_log2) - 1)) != 0 ||
      parameters.coding_unit_qp_delta_depth >
          sequence.log2_diff_max_min_luma_coding_block_size ||
      (!parameters.coding_unit_qp_delta_enabled &&
       parameters.coding_unit_qp_delta_depth != 0) ||
      parameters.log2_parallel_merge_level_minus2 > maximum_block_log2 - 2) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (sequence.pcm_enabled &&
      (sequence.pcm_luma_bit_depth_minus1 >= session.hevc.sample_bit_depth ||
       sequence.pcm_chroma_bit_depth_minus1 >= session.hevc.sample_bit_depth ||
       sequence.log2_min_pcm_luma_coding_block_size_minus3 > 2 ||
       sequence.log2_diff_max_min_pcm_luma_coding_block_size > 2 ||
       sequence.log2_min_pcm_luma_coding_block_size_minus3 + 3 <
           minimum_block_log2 ||
       sequence.log2_min_pcm_luma_coding_block_size_minus3 + 3 +
               sequence.log2_diff_max_min_pcm_luma_coding_block_size >
           maximum_block_log2)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const uint32_t block_size = 1u << maximum_block_log2;
  const uint32_t columns =
      (session.info.configuration.coded_width + block_size - 1) / block_size;
  const uint32_t rows =
      (session.info.configuration.coded_height + block_size - 1) / block_size;
  if (parameters.tile_column_count_minus1 >= columns ||
      parameters.tile_row_count_minus1 >= rows ||
      (!parameters.tiles_enabled &&
       (parameters.tile_column_count_minus1 != 0 ||
        parameters.tile_row_count_minus1 != 0 ||
        parameters.uniform_tile_spacing != 0 ||
        parameters.loop_filter_across_tiles_enabled != 0))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (parameters.tiles_enabled && !parameters.uniform_tile_spacing) {
    uint64_t width = 0, height = 0;
    for (uint32_t index = 0; index < parameters.tile_column_count_minus1;
         ++index) {
      width += uint64_t(parameters.tile_column_width_minus1[index]) + 1;
    }
    for (uint32_t index = 0; index < parameters.tile_row_count_minus1;
         ++index) {
      height += uint64_t(parameters.tile_row_height_minus1[index]) + 1;
    }
    if (width >= columns || height >= rows) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
  }
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& reference = picture.references[index];
    if (reference.binding.frame == nullptr || reference.long_term > 1 ||
        (reference.long_term && !sequence.long_term_references_present)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  uint32_t used = 0;
  if (!ValidReferenceSet(picture, picture.short_term_before, 0, &used) ||
      !ValidReferenceSet(picture, picture.short_term_after, 1, &used) ||
      !ValidReferenceSet(picture, picture.long_term, 2, &used) ||
      (!picture.is_intra && used == 0) || (picture.is_idr && used != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  uint64_t end = 0;
  for (uint32_t index = 0; index < picture.slice_count; ++index) {
    const auto& slice = picture.slices[index];
    if (slice.byte_length < 6 || slice.byte_offset < end ||
        slice.byte_offset > compressed_byte_length ||
        slice.byte_length > compressed_byte_length - slice.byte_offset) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    end = uint64_t(slice.byte_offset) + slice.byte_length;
  }
  if (sequence.scaling_list_enabled) {
    for (uint8_t value : picture.scaling_matrices->dc_16x16) {
      if (value == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
    for (uint8_t value : picture.scaling_matrices->dc_32x32) {
      if (value == 0) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  return AMDF_STATUS_OK;
}

void LowerHevc(const Session& session, uint32_t current_reference_slot,
               const amdf_media_hevc_decode_picture_info_t& picture,
               DXVA_PicParams_HEVC* native, DXVA_Qmatrix_HEVC* matrices,
               DXVA_Slice_HEVC_Short* slices) {
  const auto& sequence = picture.sequence;
  const auto& parameters = picture.picture;
  *native = {};
  native->PicWidthInMinCbsY = static_cast<USHORT>(
      session.info.configuration.coded_width >>
      (sequence.log2_min_luma_coding_block_size_minus3 + 3));
  native->PicHeightInMinCbsY = static_cast<USHORT>(
      session.info.configuration.coded_height >>
      (sequence.log2_min_luma_coding_block_size_minus3 + 3));
  native->chroma_format_idc = 1;
  native->bit_depth_luma_minus8 = session.hevc.sample_bit_depth - 8;
  native->bit_depth_chroma_minus8 = session.hevc.sample_bit_depth - 8;
  native->log2_max_pic_order_cnt_lsb_minus4 =
      sequence.log2_max_picture_order_count_lsb_minus4;
  native->CurrPic.bPicEntry = static_cast<UCHAR>(current_reference_slot);
  native->sps_max_dec_pic_buffering_minus1 =
      static_cast<UCHAR>(sequence.maximum_decoded_picture_buffer_count_minus1);
  native->log2_min_luma_coding_block_size_minus3 =
      static_cast<UCHAR>(sequence.log2_min_luma_coding_block_size_minus3);
  native->log2_diff_max_min_luma_coding_block_size =
      static_cast<UCHAR>(sequence.log2_diff_max_min_luma_coding_block_size);
  native->log2_min_transform_block_size_minus2 =
      static_cast<UCHAR>(sequence.log2_min_luma_transform_block_size_minus2);
  native->log2_diff_max_min_transform_block_size =
      static_cast<UCHAR>(sequence.log2_diff_max_min_luma_transform_block_size);
  native->max_transform_hierarchy_depth_inter =
      static_cast<UCHAR>(sequence.maximum_transform_hierarchy_depth_inter);
  native->max_transform_hierarchy_depth_intra =
      static_cast<UCHAR>(sequence.maximum_transform_hierarchy_depth_intra);
  native->num_short_term_ref_pic_sets =
      static_cast<UCHAR>(sequence.short_term_reference_set_count);
  native->num_long_term_ref_pics_sps =
      static_cast<UCHAR>(sequence.long_term_reference_count);
  native->num_ref_idx_l0_default_active_minus1 =
      static_cast<UCHAR>(parameters.default_reference_l0_count_minus1);
  native->num_ref_idx_l1_default_active_minus1 =
      static_cast<UCHAR>(parameters.default_reference_l1_count_minus1);
  native->init_qp_minus26 = static_cast<CHAR>(parameters.initial_qp_minus26);
  native->ucNumDeltaPocsOfRefRpsIdx =
      static_cast<UCHAR>(picture.predicted_reference_set_delta_count);
  native->wNumBitsForShortTermRPSInSlice =
      static_cast<USHORT>(picture.short_term_reference_set_bit_count);
  native->scaling_list_enabled_flag = sequence.scaling_list_enabled;
  native->amp_enabled_flag = sequence.asymmetric_motion_partitions_enabled;
  native->sample_adaptive_offset_enabled_flag =
      sequence.sample_adaptive_offset_enabled;
  native->pcm_enabled_flag = sequence.pcm_enabled;
  if (sequence.pcm_enabled) {
    native->pcm_sample_bit_depth_luma_minus1 =
        sequence.pcm_luma_bit_depth_minus1;
    native->pcm_sample_bit_depth_chroma_minus1 =
        sequence.pcm_chroma_bit_depth_minus1;
    native->log2_min_pcm_luma_coding_block_size_minus3 =
        sequence.log2_min_pcm_luma_coding_block_size_minus3;
    native->log2_diff_max_min_pcm_luma_coding_block_size =
        sequence.log2_diff_max_min_pcm_luma_coding_block_size;
    native->pcm_loop_filter_disabled_flag = sequence.pcm_loop_filter_disabled;
  }
  native->long_term_ref_pics_present_flag =
      sequence.long_term_references_present;
  native->sps_temporal_mvp_enabled_flag =
      sequence.temporal_motion_vector_prediction_enabled;
  native->strong_intra_smoothing_enabled_flag =
      sequence.strong_intra_smoothing_enabled;
  native->dependent_slice_segments_enabled_flag =
      parameters.dependent_slice_segments_enabled;
  native->output_flag_present_flag = parameters.output_flag_present;
  native->num_extra_slice_header_bits = parameters.extra_slice_header_bit_count;
  native->sign_data_hiding_enabled_flag = parameters.sign_data_hiding_enabled;
  native->cabac_init_present_flag = parameters.cabac_initialization_present;
  native->constrained_intra_pred_flag = parameters.constrained_intra_prediction;
  native->transform_skip_enabled_flag = parameters.transform_skip_enabled;
  native->cu_qp_delta_enabled_flag = parameters.coding_unit_qp_delta_enabled;
  native->pps_slice_chroma_qp_offsets_present_flag =
      parameters.slice_chroma_qp_offsets_present;
  native->weighted_pred_flag = parameters.weighted_prediction;
  native->weighted_bipred_flag = parameters.weighted_biprediction;
  native->transquant_bypass_enabled_flag =
      parameters.transform_quantization_bypass_enabled;
  native->tiles_enabled_flag = parameters.tiles_enabled;
  native->entropy_coding_sync_enabled_flag =
      parameters.entropy_coding_synchronization_enabled;
  native->uniform_spacing_flag = parameters.uniform_tile_spacing;
  native->loop_filter_across_tiles_enabled_flag =
      parameters.loop_filter_across_tiles_enabled;
  native->pps_loop_filter_across_slices_enabled_flag =
      parameters.loop_filter_across_slices_enabled;
  native->deblocking_filter_override_enabled_flag =
      parameters.deblocking_filter_override_enabled;
  native->pps_deblocking_filter_disabled_flag =
      parameters.deblocking_filter_disabled;
  native->lists_modification_present_flag =
      parameters.reference_lists_modification_present;
  native->slice_segment_header_extension_present_flag =
      parameters.slice_segment_header_extension_present;
  native->IrapPicFlag = picture.is_random_access;
  native->IdrPicFlag = picture.is_idr;
  native->IntraPicFlag = picture.is_intra;
  native->pps_cb_qp_offset = static_cast<CHAR>(parameters.cb_qp_offset);
  native->pps_cr_qp_offset = static_cast<CHAR>(parameters.cr_qp_offset);
  native->pps_beta_offset_div2 = static_cast<CHAR>(parameters.beta_offset_div2);
  native->pps_tc_offset_div2 = static_cast<CHAR>(parameters.tc_offset_div2);
  native->num_tile_columns_minus1 =
      static_cast<UCHAR>(parameters.tile_column_count_minus1);
  native->num_tile_rows_minus1 =
      static_cast<UCHAR>(parameters.tile_row_count_minus1);
  if (parameters.tiles_enabled && !parameters.uniform_tile_spacing) {
    for (uint32_t index = 0; index < parameters.tile_column_count_minus1;
         ++index) {
      native->column_width_minus1[index] =
          static_cast<USHORT>(parameters.tile_column_width_minus1[index]);
    }
    for (uint32_t index = 0; index < parameters.tile_row_count_minus1;
         ++index) {
      native->row_height_minus1[index] =
          static_cast<USHORT>(parameters.tile_row_height_minus1[index]);
    }
  }
  native->diff_cu_qp_delta_depth =
      static_cast<UCHAR>(parameters.coding_unit_qp_delta_depth);
  native->log2_parallel_merge_level_minus2 =
      static_cast<UCHAR>(parameters.log2_parallel_merge_level_minus2);
  native->CurrPicOrderCntVal = picture.picture_order_count;
  std::memset(native->RefPicList, 0xff, sizeof(native->RefPicList));
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& reference = picture.references[index];
    native->RefPicList[index].bPicEntry = static_cast<UCHAR>(
        reference.binding.slot_index | (reference.long_term << 7));
    native->PicOrderCntValList[index] = reference.picture_order_count;
  }
  const amdf_media_hevc_reference_set_t* sets[] = {&picture.short_term_before,
                                                   &picture.short_term_after,
                                                   &picture.long_term};
  uint8_t* native_sets[] = {native->RefPicSetStCurrBefore,
                            native->RefPicSetStCurrAfter,
                            native->RefPicSetLtCurr};
  for (uint32_t set = 0; set < 3; ++set) {
    std::memset(native_sets[set], 0xff, 8);
    std::memcpy(native_sets[set], sets[set]->indices, sets[set]->count);
  }
  native->StatusReportFeedbackNumber = 1;
  if (sequence.scaling_list_enabled) {
    const auto& source = *picture.scaling_matrices;
    for (uint32_t index = 0; index < 6; ++index) {
      DiagonalScalingList(source.scaling_4x4[index],
                          matrices->ucScalingLists0[index]);
      DiagonalScalingList(source.scaling_8x8[index],
                          matrices->ucScalingLists1[index]);
      DiagonalScalingList(source.scaling_16x16[index],
                          matrices->ucScalingLists2[index]);
      matrices->ucScalingListDCCoefSizeID2[index] = source.dc_16x16[index];
    }
    for (uint32_t index = 0; index < 2; ++index) {
      DiagonalScalingList(source.scaling_32x32[index],
                          matrices->ucScalingLists3[index]);
      matrices->ucScalingListDCCoefSizeID3[index] = source.dc_32x32[index];
    }
  }
  for (uint32_t index = 0; index < picture.slice_count; ++index) {
    slices[index] = {picture.slices[index].byte_offset,
                     picture.slices[index].byte_length, 0};
  }
}

}  // namespace amdf::media::d3d12
