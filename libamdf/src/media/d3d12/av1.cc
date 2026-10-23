// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/av1.h"

#include <cstring>

namespace amdf::media::d3d12 {

void LowerAv1(const Session& session, uint32_t current_reference_slot,
              const amdf_media_av1_decode_picture_info_t& picture,
              DXVA_PicParams_AV1* native, DXVA_Tile_AV1* tiles) {
  *native = {};
  const auto& sequence = picture.sequence;
  native->width = session.info.configuration.coded_width;
  native->height = session.info.configuration.coded_height;
  native->max_width = native->width;
  native->max_height = native->height;
  native->CurrPicTextureIndex = static_cast<UCHAR>(current_reference_slot);
  native->seq_profile = static_cast<UCHAR>(session.av1.profile);
  native->bitdepth = static_cast<UCHAR>(session.av1.sample_bit_depth);
  native->superres_denom = static_cast<UCHAR>(picture.superres_denominator);
  native->tiles.cols = static_cast<UCHAR>(picture.tile_column_count);
  native->tiles.rows = static_cast<UCHAR>(picture.tile_row_count);
  native->tiles.context_update_id =
      static_cast<USHORT>(picture.context_update_tile);
  for (uint32_t i = 0; i < picture.tile_column_count; ++i) {
    native->tiles.widths[i] =
        picture.tile_column_starts[i + 1] - picture.tile_column_starts[i];
  }
  for (uint32_t i = 0; i < picture.tile_row_count; ++i) {
    native->tiles.heights[i] =
        picture.tile_row_starts[i + 1] - picture.tile_row_starts[i];
  }
  native->coding.use_128x128_superblock = sequence.use_128x128_superblock;
  native->coding.intra_edge_filter = sequence.intra_edge_filter;
  native->coding.interintra_compound = sequence.interintra_compound;
  native->coding.masked_compound = sequence.masked_compound;
  native->coding.dual_filter = sequence.dual_filter;
  native->coding.jnt_comp = sequence.joint_compound;
  native->coding.cdef = sequence.cdef;
  native->coding.restoration = sequence.restoration;
  native->coding.film_grain = sequence.film_grain;
  native->coding.filter_intra = sequence.filter_intra;
  native->coding.enable_ref_frame_mvs = sequence.reference_frame_motion_vectors;
  native->coding.warped_motion = picture.warped_motion;
  native->coding.screen_content_tools = picture.screen_content_tools;
  native->coding.integer_mv = picture.integer_motion_vectors;
  native->coding.intrabc = picture.intra_block_copy;
  native->coding.high_precision_mv = picture.high_precision_motion_vectors;
  native->coding.switchable_motion_mode = picture.switchable_motion_mode;
  native->coding.disable_frame_end_update_cdf =
      picture.disable_frame_end_cdf_update;
  native->coding.disable_cdf_update = picture.disable_cdf_update;
  native->coding.reference_mode = picture.reference_select;
  native->coding.skip_mode = picture.skip_mode;
  native->coding.reduced_tx_set = picture.reduced_transform_set;
  native->coding.tx_mode = picture.transform_mode;
  native->coding.use_ref_frame_mvs = picture.use_reference_motion_vectors;
  native->coding.superres = picture.superres_denominator != 8;
  // The operation always produces an explicitly owned reconstructed target.
  native->coding.reference_frame_update = 1;
  native->format.frame_type = picture.frame_type;
  native->format.show_frame = picture.show_frame;
  native->format.showable_frame = picture.showable_frame;
  native->format.subsampling_x = 1;
  native->format.subsampling_y = 1;
  native->primary_ref_frame = static_cast<UCHAR>(picture.primary_reference);
  native->order_hint = static_cast<UCHAR>(picture.order_hint);
  native->order_hint_bits = static_cast<UCHAR>(sequence.order_hint_bit_count);
  native->interp_filter = static_cast<UCHAR>(picture.interpolation_filter);
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t index = picture.reference_map[i];
    native->RefFrameMapTextureIndex[i] =
        index == 255 ? 255
                     : static_cast<UCHAR>(picture.references[index].slot_index);
  }
  for (uint32_t i = 0; i < 7; ++i) {
    auto& reference = native->frame_refs[i];
    // AV1's active indices address the logical map, not physical surfaces.
    reference.Index = picture.active_reference_indices[i];
    if (reference.Index != 255) {
      reference.width = native->width;
      reference.height = native->height;
      reference.wmtype = picture.global_motion[i].type;
      reference.wminvalid = picture.global_motion[i].invalid;
      std::memcpy(reference.wmmat, picture.global_motion[i].parameters,
                  sizeof(reference.wmmat));
    }
  }
  const auto& filter = picture.loop_filter;
  std::memcpy(native->loop_filter.filter_level, filter.levels, 2);
  native->loop_filter.filter_level_u = filter.levels[2];
  native->loop_filter.filter_level_v = filter.levels[3];
  native->loop_filter.sharpness_level = static_cast<UCHAR>(filter.sharpness);
  native->loop_filter.mode_ref_delta_enabled = filter.deltas_enabled;
  native->loop_filter.mode_ref_delta_update = filter.deltas_updated;
  native->loop_filter.delta_lf_present = filter.delta_present;
  native->loop_filter.delta_lf_multi = filter.delta_multi;
  native->loop_filter.delta_lf_res =
      static_cast<UCHAR>(filter.delta_resolution);
  std::memcpy(native->loop_filter.ref_deltas, filter.reference_deltas, 8);
  std::memcpy(native->loop_filter.mode_deltas, filter.mode_deltas, 2);
  for (uint32_t i = 0; i < 3; ++i) {
    native->loop_filter.frame_restoration_type[i] = filter.restoration_types[i];
    native->loop_filter.log2_restoration_unit_size[i] =
        filter.log2_restoration_unit_sizes[i];
  }
  const auto& quantization = picture.quantization;
  native->quantization.delta_q_present = quantization.delta_present;
  native->quantization.delta_q_res = quantization.delta_resolution;
  native->quantization.base_qindex =
      static_cast<UCHAR>(quantization.base_index);
  native->quantization.y_dc_delta_q =
      static_cast<CHAR>(quantization.y_dc_delta);
  native->quantization.u_dc_delta_q = quantization.chroma_dc_deltas[0];
  native->quantization.v_dc_delta_q = quantization.chroma_dc_deltas[1];
  native->quantization.u_ac_delta_q = quantization.chroma_ac_deltas[0];
  native->quantization.v_ac_delta_q = quantization.chroma_ac_deltas[1];
  native->quantization.qm_y = quantization.matrix_levels[0];
  native->quantization.qm_u = quantization.matrix_levels[1];
  native->quantization.qm_v = quantization.matrix_levels[2];
  native->cdef.damping = picture.cdef.damping_minus3;
  native->cdef.bits = picture.cdef.bits;
  for (uint32_t i = 0; i < 8; ++i) {
    native->cdef.y_strengths[i].primary = picture.cdef.luma_primary[i];
    native->cdef.y_strengths[i].secondary = picture.cdef.luma_secondary[i];
    native->cdef.uv_strengths[i].primary = picture.cdef.chroma_primary[i];
    native->cdef.uv_strengths[i].secondary = picture.cdef.chroma_secondary[i];
  }
  const auto& segments = picture.segmentation;
  native->segmentation.enabled = segments.enabled;
  if (segments.enabled) {
    native->segmentation.update_map = segments.update_map;
    native->segmentation.update_data = segments.update_data;
    native->segmentation.temporal_update = segments.temporal_update;
    for (uint32_t i = 0; i < 8; ++i) {
      native->segmentation.feature_mask[i].mask = segments.feature_masks[i];
    }
    std::memcpy(native->segmentation.feature_data, segments.feature_values,
                sizeof(segments.feature_values));
  }
  const auto& grain = picture.film_grain;
  auto& output = native->film_grain;
  output.apply_grain = grain.apply;
  if (grain.apply) {
    output.grain_seed = static_cast<USHORT>(grain.seed);
    output.scaling_shift_minus8 = grain.scaling_shift_minus8;
    output.chroma_scaling_from_luma = grain.chroma_scaling_from_luma;
    output.ar_coeff_lag = grain.coefficient_lag;
    output.ar_coeff_shift_minus6 = grain.coefficient_shift_minus6;
    output.grain_scale_shift = grain.scale_shift;
    output.overlap_flag = grain.overlap;
    output.clip_to_restricted_range = grain.clip_to_restricted_range;
    output.num_y_points = static_cast<UCHAR>(grain.luma_point_count);
    output.num_cb_points = static_cast<UCHAR>(grain.u_point_count);
    output.num_cr_points = static_cast<UCHAR>(grain.v_point_count);
    std::memcpy(output.scaling_points_y, grain.luma_points,
                sizeof(grain.luma_points));
    std::memcpy(output.scaling_points_cb, grain.u_points,
                sizeof(grain.u_points));
    std::memcpy(output.scaling_points_cr, grain.v_points,
                sizeof(grain.v_points));
    // DXVA retains the bitstream's unsigned coefficient bias.
    for (uint32_t i = 0; i < 24; ++i) {
      output.ar_coeffs_y[i] =
          static_cast<UCHAR>(grain.luma_coefficients[i] + 128);
    }
    for (uint32_t i = 0; i < 25; ++i) {
      output.ar_coeffs_cb[i] =
          static_cast<UCHAR>(grain.u_coefficients[i] + 128);
      output.ar_coeffs_cr[i] =
          static_cast<UCHAR>(grain.v_coefficients[i] + 128);
    }
    output.cb_mult = grain.chroma_multipliers[0];
    output.cr_mult = grain.chroma_multipliers[1];
    output.cb_luma_mult = grain.chroma_luma_multipliers[0];
    output.cr_luma_mult = grain.chroma_luma_multipliers[1];
    output.cb_offset = static_cast<SHORT>(grain.chroma_offsets[0]);
    output.cr_offset = static_cast<SHORT>(grain.chroma_offsets[1]);
  }
  for (uint32_t i = 0; i < picture.tile_column_count * picture.tile_row_count;
       ++i) {
    tiles[i] = {picture.tiles[i].byte_offset,
                picture.tiles[i].byte_length,
                static_cast<USHORT>(i / picture.tile_column_count),
                static_cast<USHORT>(i % picture.tile_column_count),
                0,
                255,
                0};
  }
  native->StatusReportFeedbackNumber = 1;
}

}  // namespace amdf::media::d3d12
