// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/vp9.h"

#include <cstring>

namespace amdf::media::d3d12 {

void LowerVp9(const Session& session, uint32_t current_reference_slot,
              const amdf_media_vp9_decode_picture_info_t& picture,
              DXVA_PicParams_VP9* native) {
  *native = {};
  native->CurrPic.bPicEntry = static_cast<UCHAR>(current_reference_slot);
  native->profile = static_cast<UCHAR>(session.vp9.profile);
  native->frame_type = picture.frame_type;
  native->show_frame = picture.show_frame;
  native->error_resilient_mode = picture.error_resilient;
  native->subsampling_x = 1;
  native->subsampling_y = 1;
  native->refresh_frame_context = picture.refresh_frame_context;
  native->frame_parallel_decoding_mode = picture.parallel_decoding;
  native->intra_only = picture.intra_only;
  native->frame_context_idx = picture.frame_context_index;
  native->reset_frame_context = picture.reset_frame_context;
  native->allow_high_precision_mv = picture.high_precision_motion_vectors;
  native->width = session.info.configuration.coded_width;
  native->height = session.info.configuration.coded_height;
  native->BitDepthMinus8Luma =
      static_cast<UCHAR>(session.vp9.sample_bit_depth - 8);
  native->BitDepthMinus8Chroma = native->BitDepthMinus8Luma;
  native->interp_filter = static_cast<UCHAR>(picture.interpolation_filter);
  for (uint32_t i = 0; i < 8; ++i) {
    const uint32_t index = picture.reference_map[i];
    native->ref_frame_map[i].bPicEntry =
        index == 255 ? 255
                     : static_cast<UCHAR>(picture.references[index].slot_index);
    if (index != 255) {
      native->ref_frame_coded_width[i] = native->width;
      native->ref_frame_coded_height[i] = native->height;
    }
  }
  for (uint32_t i = 0; i < 3; ++i) {
    const uint32_t index = picture.active_reference_indices[i];
    native->frame_refs[i].bPicEntry =
        index == 255 ? 255 : native->ref_frame_map[index].bPicEntry;
    native->ref_frame_sign_bias[i + 1] = picture.reference_sign_bias[i];
  }
  native->filter_level = static_cast<UCHAR>(picture.loop_filter.level);
  native->sharpness_level = static_cast<UCHAR>(picture.loop_filter.sharpness);
  native->mode_ref_delta_enabled = picture.loop_filter.deltas_enabled;
  native->mode_ref_delta_update = picture.loop_filter.deltas_updated;
  native->use_prev_in_find_mv_refs = picture.use_previous_frame_motion_vectors;
  std::memcpy(native->ref_deltas, picture.loop_filter.reference_deltas, 4);
  std::memcpy(native->mode_deltas, picture.loop_filter.mode_deltas, 2);
  native->base_qindex = static_cast<SHORT>(picture.base_quantizer_index);
  native->y_dc_delta_q = static_cast<CHAR>(picture.y_dc_quantizer_delta);
  native->uv_dc_delta_q = static_cast<CHAR>(picture.uv_dc_quantizer_delta);
  native->uv_ac_delta_q = static_cast<CHAR>(picture.uv_ac_quantizer_delta);
  auto& segments = native->stVP9Segments;
  segments.enabled = picture.segmentation.enabled;
  // Inactive native masks are zero: forwarding retained disabled features
  // changes decoded samples on this backend.
  if (picture.segmentation.enabled) {
    segments.update_map = picture.segmentation.update_map;
    segments.temporal_update = picture.segmentation.temporal_update;
    segments.abs_delta = picture.segmentation.absolute_values;
    std::memcpy(segments.tree_probs, picture.segmentation.tree_probabilities,
                7);
    std::memcpy(segments.pred_probs,
                picture.segmentation.prediction_probabilities, 3);
    std::memcpy(segments.feature_mask, picture.segmentation.feature_masks, 8);
    std::memcpy(segments.feature_data, picture.segmentation.feature_values,
                sizeof(segments.feature_data));
  }
  native->log2_tile_cols = static_cast<UCHAR>(picture.log2_tile_columns);
  native->log2_tile_rows = static_cast<UCHAR>(picture.log2_tile_rows);
  native->uncompressed_header_size_byte_aligned =
      static_cast<USHORT>(picture.uncompressed_header_byte_length);
  native->first_partition_size =
      static_cast<USHORT>(picture.compressed_header_byte_length);
  native->StatusReportFeedbackNumber = 1;
}

}  // namespace amdf::media::d3d12
