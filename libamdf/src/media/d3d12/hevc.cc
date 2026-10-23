// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/hevc.h"

#include <cstring>

#include "libamdf/src/media/hevc.h"

namespace amdf::media::d3d12 {

amdf_status_t ValidateHevc(const Session& session,
                           const amdf_media_hevc_decode_picture_info_t& picture,
                           uint64_t compressed_byte_length) {
  return amdf_media_hevc_validate(&session.info.configuration, &session.hevc,
                                  &picture, compressed_byte_length);
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
      amdf_media_hevc_scan_scaling_list(source.scaling_4x4[index], 4,
                                        matrices->ucScalingLists0[index]);
      amdf_media_hevc_scan_scaling_list(source.scaling_8x8[index], 8,
                                        matrices->ucScalingLists1[index]);
      amdf_media_hevc_scan_scaling_list(source.scaling_16x16[index], 8,
                                        matrices->ucScalingLists2[index]);
      matrices->ucScalingListDCCoefSizeID2[index] = source.dc_16x16[index];
    }
    for (uint32_t index = 0; index < 2; ++index) {
      amdf_media_hevc_scan_scaling_list(source.scaling_32x32[index], 8,
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
