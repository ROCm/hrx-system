// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/encode.h"

#include "libamdf/src/media/av1.h"
#include "libamdf/src/media/h264.h"
#include "libamdf/src/media/hevc.h"
#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {
namespace {

constexpr UINT kAv1InvalidDpbResourceIndex = 0xffu;

void PrepareSequence(const Session& session, uint32_t temporal_layer_index,
                     EncodePicture* picture) {
  const auto& native = session.native_encode;
  picture->arguments.SequenceControlDesc.RateControl =
      native.temporal_layer_rate_controls[temporal_layer_index].descriptor;
  picture->arguments.SequenceControlDesc.PictureTargetResolution =
      native.resolution;
  picture->arguments.SequenceControlDesc.SelectedLayoutMode =
      native.subregion_mode;
  if (native.codec == D3D12_VIDEO_ENCODER_CODEC_AV1) {
    picture->arguments.SequenceControlDesc.FrameSubregionsLayoutData = {
        sizeof(native.tiles)};
    picture->arguments.SequenceControlDesc.FrameSubregionsLayoutData
        .pTilesPartition_AV1 = &native.tiles;
  }
  picture->arguments.SequenceControlDesc.CodecGopSequence = native.gop;
}

amdf_status_t PrepareH264(const Session& session,
                          const amdf_media_encode_picture_info_t& encode,
                          EncodePicture* picture) {
  const auto* source =
      static_cast<const amdf_media_h264_encode_picture_info_t*>(encode.next);
  amdf_status_t status = amdf_media_h264_validate_encode_picture(
      &session.requirements, &session.encode, &encode, &session.h264_encode,
      source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  picture->h264.FrameType =
      source->frame_type == AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR
          ? D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_IDR_FRAME
          : D3D12_VIDEO_ENCODER_FRAME_TYPE_H264_P_FRAME;
  picture->h264.PictureOrderCountNumber = source->picture_order_count;
  // D3D12 names this field by its decode-order role, but defines its syntax
  // value as frame_num. libamdf keeps the provider task identity separate.
  picture->h264.FrameDecodingOrderNumber = source->frame_number;
  picture->h264.TemporalLayerIndex = encode.temporal_layer_index;
  picture->h264.List0ReferenceFramesCount = source->reference_count;
  picture->h264.pList0ReferenceFrames =
      source->reference_count != 0 ? picture->list0 : nullptr;
  picture->h264.ReferenceFramesReconPictureDescriptorsCount =
      source->reference_count;
  picture->h264.pReferenceFramesReconPictureDescriptors =
      source->reference_count != 0 ? picture->h264_references : nullptr;
  for (uint32_t index = 0; index < source->reference_count; ++index) {
    const auto& reference = source->references[index];
    picture->list0[index] = index;
    picture->references[index] = reference.binding;
    picture->h264_references[index] = {
        .ReconstructedPictureResourceIndex = index,
        .PictureOrderCountNumber =
            static_cast<UINT>(reference.picture_order_count),
        .FrameDecodingOrderNumber = reference.frame_number,
        .TemporalLayerIndex = reference.temporal_layer_index,
    };
  }
  picture->reference_count = source->reference_count;
  picture->used_as_reference = source->used_as_reference != 0;
  picture->arguments.PictureControlDesc.PictureControlCodecData = {
      sizeof(picture->h264)};
  picture->arguments.PictureControlDesc.PictureControlCodecData.pH264PicData =
      &picture->h264;
  return AMDF_STATUS_OK;
}

amdf_status_t PrepareHevc(const Session& session,
                          const amdf_media_encode_picture_info_t& encode,
                          EncodePicture* picture) {
  const auto* source =
      static_cast<const amdf_media_hevc_encode_picture_info_t*>(encode.next);
  amdf_status_t status = amdf_media_hevc_validate_encode_picture(
      &session.requirements, &session.encode, &encode, source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  picture->hevc.FrameType =
      source->frame_type == AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR
          ? D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_IDR_FRAME
          : D3D12_VIDEO_ENCODER_FRAME_TYPE_HEVC_P_FRAME;
  picture->hevc.PictureOrderCountNumber = source->picture_order_count;
  picture->hevc.TemporalLayerIndex = encode.temporal_layer_index;
  picture->hevc.List0ReferenceFramesCount = source->reference_count;
  picture->hevc.pList0ReferenceFrames =
      source->reference_count != 0 ? picture->list0 : nullptr;
  picture->hevc.ReferenceFramesReconPictureDescriptorsCount =
      source->reference_count;
  picture->hevc.pReferenceFramesReconPictureDescriptors =
      source->reference_count != 0 ? picture->hevc_references : nullptr;
  for (uint32_t index = 0; index < source->reference_count; ++index) {
    const auto& reference = source->references[index];
    picture->list0[index] = index;
    picture->references[index] = reference.binding;
    picture->hevc_references[index] = {
        .ReconstructedPictureResourceIndex = index,
        .IsRefUsedByCurrentPic = TRUE,
        .IsLongTermReference = FALSE,
        .PictureOrderCountNumber =
            static_cast<UINT>(reference.picture_order_count),
        .TemporalLayerIndex = reference.temporal_layer_index,
    };
  }
  picture->reference_count = source->reference_count;
  picture->used_as_reference = source->used_as_reference != 0;
  picture->arguments.PictureControlDesc.PictureControlCodecData = {
      sizeof(picture->hevc)};
  picture->arguments.PictureControlDesc.PictureControlCodecData.pHEVCPicData =
      &picture->hevc;
  return AMDF_STATUS_OK;
}

void CopyLoopFilter(
    const amdf_media_av1_loop_filter_t& source,
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_CONFIG* filter,
    D3D12_VIDEO_ENCODER_CODEC_AV1_LOOP_FILTER_DELTA_CONFIG* delta) {
  filter->LoopFilterLevel[0] = source.levels[0];
  filter->LoopFilterLevel[1] = source.levels[1];
  filter->LoopFilterLevelU = source.levels[2];
  filter->LoopFilterLevelV = source.levels[3];
  filter->LoopFilterSharpnessLevel = source.sharpness;
  filter->LoopFilterDeltaEnabled = source.deltas_enabled;
  filter->UpdateRefDelta = source.deltas_updated;
  filter->UpdateModeDelta = source.deltas_updated;
  for (uint32_t index = 0; index < 8; ++index) {
    filter->RefDeltas[index] = source.reference_deltas[index];
  }
  for (uint32_t index = 0; index < 2; ++index) {
    filter->ModeDeltas[index] = source.mode_deltas[index];
  }
  delta->DeltaLFPresent = source.delta_present;
  delta->DeltaLFMulti = source.delta_multi;
  delta->DeltaLFRes = source.delta_resolution;
}

void CopyQuantization(
    const amdf_media_av1_quantization_t& source,
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_CONFIG* quantization,
    D3D12_VIDEO_ENCODER_CODEC_AV1_QUANTIZATION_DELTA_CONFIG* delta) {
  quantization->BaseQIndex = source.base_index;
  quantization->YDCDeltaQ = source.y_dc_delta;
  quantization->UDCDeltaQ = source.chroma_dc_deltas[0];
  quantization->UACDeltaQ = source.chroma_ac_deltas[0];
  quantization->VDCDeltaQ = source.chroma_dc_deltas[1];
  quantization->VACDeltaQ = source.chroma_ac_deltas[1];
  quantization->UsingQMatrix = source.matrix_levels[0] != 255;
  quantization->QMY = source.matrix_levels[0];
  quantization->QMU = source.matrix_levels[1];
  quantization->QMV = source.matrix_levels[2];
  delta->DeltaQPresent = source.delta_present;
  delta->DeltaQRes = source.delta_resolution;
}

void CopyCdef(const amdf_media_av1_cdef_t& source,
              D3D12_VIDEO_ENCODER_AV1_CDEF_CONFIG* cdef) {
  cdef->CdefBits = source.bits;
  cdef->CdefDampingMinus3 = source.damping_minus3;
  for (uint32_t index = 0; index < 8; ++index) {
    cdef->CdefYPriStrength[index] = source.luma_primary[index];
    cdef->CdefYSecStrength[index] = source.luma_secondary[index];
    cdef->CdefUVPriStrength[index] = source.chroma_primary[index];
    cdef->CdefUVSecStrength[index] = source.chroma_secondary[index];
  }
}

amdf_status_t PrepareAv1(const Session& session,
                         const amdf_media_encode_picture_info_t& encode,
                         EncodePicture* picture) {
  const auto* source =
      static_cast<const amdf_media_av1_encode_picture_info_t*>(encode.next);
  amdf_status_t status = amdf_media_av1_validate_encode_picture(
      &session.requirements, &session.encode, &encode, &session.av1_encode,
      source);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  picture->av1.FrameType =
      static_cast<D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE>(source->frame_type);
  picture->av1.InterpolationFilter =
      static_cast<D3D12_VIDEO_ENCODER_AV1_INTERPOLATION_FILTERS>(
          source->interpolation_filter);
  picture->av1.TxMode =
      static_cast<D3D12_VIDEO_ENCODER_AV1_TX_MODE>(source->transform_mode);
  picture->av1.SuperResDenominator = 8;
  picture->av1.OrderHint = source->order_hint;
  picture->av1.PictureIndex = source->picture_index;
  const bool scalable = session.encode.rate_control.temporal_layer_count > 1;
  picture->av1.TemporalLayerIndexPlus1 =
      scalable ? encode.temporal_layer_index + 1 : 0;
  picture->av1.PrimaryRefFrame = source->primary_reference;
  picture->av1.RefreshFrameFlags = source->refresh_frame_flags;
  CopyLoopFilter(source->loop_filter, &picture->av1.LoopFilter,
                 &picture->av1.LoopFilterDelta);
  CopyQuantization(source->quantization, &picture->av1.Quantization,
                   &picture->av1.QuantizationDelta);
  CopyCdef(source->cdef, &picture->av1.CDEF);
  for (auto& reference : picture->av1.ReferenceFramesReconPictureDescriptors) {
    reference.ReconstructedPictureResourceIndex = kAv1InvalidDpbResourceIndex;
  }
  for (uint32_t index = 0; index < source->reference_count; ++index) {
    const auto& reference = source->references[index];
    picture->references[index] = reference.binding;
    auto& native = picture->av1.ReferenceFramesReconPictureDescriptors[index];
    native.ReconstructedPictureResourceIndex = index;
    native.FrameType =
        static_cast<D3D12_VIDEO_ENCODER_AV1_FRAME_TYPE>(reference.frame_type);
    native.OrderHint = reference.order_hint;
    native.PictureIndex = reference.picture_index;
    native.TemporalLayerIndexPlus1 =
        scalable ? reference.temporal_layer_index + 1 : 0;
  }
  for (uint32_t index = 0; index < 7; ++index) {
    picture->av1.ReferenceIndices[index] =
        source->reference_indices[index] == UINT8_MAX
            ? 0u
            : source->reference_indices[index];
  }
  picture->reference_count = source->reference_count;
  picture->used_as_reference = source->refresh_frame_flags != 0;
  picture->arguments.PictureControlDesc.PictureControlCodecData = {
      sizeof(picture->av1)};
  picture->arguments.PictureControlDesc.PictureControlCodecData.pAV1PicData =
      &picture->av1;
  return AMDF_STATUS_OK;
}

bool NativeAv1ResultValid(
    const D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES& source) {
  if (source.CompoundPredictionType >
          D3D12_VIDEO_ENCODER_AV1_COMP_PREDICTION_TYPE_COMPOUND_REFERENCE ||
      source.LoopFilter.LoopFilterLevel[0] > 63 ||
      source.LoopFilter.LoopFilterLevel[1] > 63 ||
      source.LoopFilter.LoopFilterLevelU > 63 ||
      source.LoopFilter.LoopFilterLevelV > 63 ||
      source.LoopFilter.LoopFilterSharpnessLevel > 7 ||
      source.LoopFilter.LoopFilterDeltaEnabled > 1 ||
      source.LoopFilter.UpdateRefDelta > 1 ||
      source.LoopFilter.UpdateModeDelta > 1 ||
      source.LoopFilterDelta.DeltaLFPresent > 1 ||
      source.LoopFilterDelta.DeltaLFMulti > 1 ||
      source.LoopFilterDelta.DeltaLFRes > 3 ||
      source.Quantization.BaseQIndex > 255 ||
      source.Quantization.YDCDeltaQ < -64 ||
      source.Quantization.YDCDeltaQ > 63 ||
      source.Quantization.UDCDeltaQ < -64 ||
      source.Quantization.UDCDeltaQ > 63 ||
      source.Quantization.UACDeltaQ < -64 ||
      source.Quantization.UACDeltaQ > 63 ||
      source.Quantization.VDCDeltaQ < -64 ||
      source.Quantization.VDCDeltaQ > 63 ||
      source.Quantization.VACDeltaQ < -64 ||
      source.Quantization.VACDeltaQ > 63 ||
      source.Quantization.UsingQMatrix > 1 ||
      (source.Quantization.UsingQMatrix &&
       (source.Quantization.QMY > 15 || source.Quantization.QMU > 15 ||
        source.Quantization.QMV > 15)) ||
      source.QuantizationDelta.DeltaQPresent > 1 ||
      source.QuantizationDelta.DeltaQRes > 3 || source.CDEF.CdefBits > 3 ||
      source.CDEF.CdefDampingMinus3 > 3 ||
      source.SegmentationConfig.UpdateMap > 1 ||
      source.SegmentationConfig.TemporalUpdate > 1 ||
      source.SegmentationConfig.UpdateData > 1 ||
      source.SegmentationConfig.NumSegments > 8 || source.PrimaryRefFrame > 7) {
    return false;
  }
  const uint32_t strength_count = 1u << source.CDEF.CdefBits;
  for (uint32_t index = 0; index < 8; ++index) {
    if (source.LoopFilter.RefDeltas[index] < -64 ||
        source.LoopFilter.RefDeltas[index] > 63 ||
        (index < strength_count &&
         (source.CDEF.CdefYPriStrength[index] > 15 ||
          source.CDEF.CdefYSecStrength[index] > 3 ||
          source.CDEF.CdefUVPriStrength[index] > 15 ||
          source.CDEF.CdefUVSecStrength[index] > 3))) {
      return false;
    }
    if (index >= source.SegmentationConfig.NumSegments) {
      continue;
    }
    const auto& segment = source.SegmentationConfig.SegmentsData[index];
    constexpr uint64_t kKnownFeatureMask = UINT64_C(0x1FE);
    if ((segment.EnabledFeatures & ~kKnownFeatureMask) != 0) {
      return false;
    }
    for (uint32_t feature = 0; feature < 8; ++feature) {
      if ((segment.EnabledFeatures & (UINT64_C(1) << (feature + 1))) == 0) {
        continue;
      }
      const int64_t value = segment.FeatureValue[feature];
      const bool valid = feature == 0   ? value >= -255 && value <= 255
                         : feature <= 4 ? value >= -63 && value <= 63
                         : feature == 5 ? value >= 0 && value <= 7
                                        : value == 0;
      if (!valid) {
        return false;
      }
    }
  }
  for (uint32_t index = 0; index < 2; ++index) {
    if (source.LoopFilter.ModeDeltas[index] < -64 ||
        source.LoopFilter.ModeDeltas[index] > 63) {
      return false;
    }
  }
  for (uint32_t index = 0; index < 7; ++index) {
    if (source.ReferenceIndices[index] > 7) {
      return false;
    }
  }
  return true;
}

void CopyAv1Result(const D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES& source,
                   amdf_media_av1_encode_result_info_t* result) {
  result->compound_prediction_type = source.CompoundPredictionType;
  auto& filter = result->loop_filter;
  filter.levels[0] = static_cast<uint8_t>(source.LoopFilter.LoopFilterLevel[0]);
  filter.levels[1] = static_cast<uint8_t>(source.LoopFilter.LoopFilterLevel[1]);
  filter.levels[2] = static_cast<uint8_t>(source.LoopFilter.LoopFilterLevelU);
  filter.levels[3] = static_cast<uint8_t>(source.LoopFilter.LoopFilterLevelV);
  filter.sharpness =
      static_cast<uint32_t>(source.LoopFilter.LoopFilterSharpnessLevel);
  filter.deltas_enabled =
      static_cast<uint32_t>(source.LoopFilter.LoopFilterDeltaEnabled);
  filter.deltas_updated = static_cast<uint32_t>(
      source.LoopFilter.UpdateRefDelta || source.LoopFilter.UpdateModeDelta);
  filter.delta_present =
      static_cast<uint32_t>(source.LoopFilterDelta.DeltaLFPresent);
  filter.delta_multi =
      static_cast<uint32_t>(source.LoopFilterDelta.DeltaLFMulti);
  filter.delta_resolution =
      static_cast<uint32_t>(source.LoopFilterDelta.DeltaLFRes);
  for (uint32_t index = 0; index < 8; ++index) {
    filter.reference_deltas[index] =
        static_cast<int8_t>(source.LoopFilter.RefDeltas[index]);
  }
  for (uint32_t index = 0; index < 2; ++index) {
    filter.mode_deltas[index] =
        static_cast<int8_t>(source.LoopFilter.ModeDeltas[index]);
  }
  auto& quantization = result->quantization;
  quantization.base_index =
      static_cast<uint32_t>(source.Quantization.BaseQIndex);
  quantization.y_dc_delta = static_cast<int32_t>(source.Quantization.YDCDeltaQ);
  quantization.chroma_dc_deltas[0] =
      static_cast<int8_t>(source.Quantization.UDCDeltaQ);
  quantization.chroma_ac_deltas[0] =
      static_cast<int8_t>(source.Quantization.UACDeltaQ);
  quantization.chroma_dc_deltas[1] =
      static_cast<int8_t>(source.Quantization.VDCDeltaQ);
  quantization.chroma_ac_deltas[1] =
      static_cast<int8_t>(source.Quantization.VACDeltaQ);
  quantization.matrix_levels[0] =
      source.Quantization.UsingQMatrix
          ? static_cast<uint8_t>(source.Quantization.QMY)
          : UINT8_MAX;
  quantization.matrix_levels[1] =
      source.Quantization.UsingQMatrix
          ? static_cast<uint8_t>(source.Quantization.QMU)
          : UINT8_MAX;
  quantization.matrix_levels[2] =
      source.Quantization.UsingQMatrix
          ? static_cast<uint8_t>(source.Quantization.QMV)
          : UINT8_MAX;
  quantization.delta_present =
      static_cast<uint32_t>(source.QuantizationDelta.DeltaQPresent);
  quantization.delta_resolution =
      static_cast<uint32_t>(source.QuantizationDelta.DeltaQRes);
  auto& cdef = result->cdef;
  cdef.bits = static_cast<uint32_t>(source.CDEF.CdefBits);
  cdef.damping_minus3 = static_cast<uint32_t>(source.CDEF.CdefDampingMinus3);
  for (uint32_t index = 0; index < 8; ++index) {
    if (index < (1u << cdef.bits)) {
      cdef.luma_primary[index] =
          static_cast<uint8_t>(source.CDEF.CdefYPriStrength[index]);
      cdef.luma_secondary[index] =
          static_cast<uint8_t>(source.CDEF.CdefYSecStrength[index]);
      cdef.chroma_primary[index] =
          static_cast<uint8_t>(source.CDEF.CdefUVPriStrength[index]);
      cdef.chroma_secondary[index] =
          static_cast<uint8_t>(source.CDEF.CdefUVSecStrength[index]);
    }
  }
  auto& segments = result->segmentation;
  segments.enabled = source.SegmentationConfig.NumSegments != 0;
  segments.update_map =
      static_cast<uint32_t>(source.SegmentationConfig.UpdateMap);
  segments.temporal_update =
      static_cast<uint32_t>(source.SegmentationConfig.TemporalUpdate);
  segments.update_data =
      static_cast<uint32_t>(source.SegmentationConfig.UpdateData);
  for (uint32_t segment = 0; segment < source.SegmentationConfig.NumSegments;
       ++segment) {
    const auto& native = source.SegmentationConfig.SegmentsData[segment];
    segments.feature_masks[segment] =
        static_cast<uint8_t>(native.EnabledFeatures >> 1);
    for (uint32_t feature = 0; feature < 8; ++feature) {
      if ((segments.feature_masks[segment] & (1u << feature)) != 0) {
        segments.feature_values[segment][feature] =
            static_cast<int16_t>(native.FeatureValue[feature]);
      }
    }
  }
  result->primary_reference = static_cast<uint32_t>(source.PrimaryRefFrame);
  for (uint32_t index = 0; index < 7; ++index) {
    result->reference_indices[index] =
        static_cast<uint8_t>(source.ReferenceIndices[index]);
  }
}

amdf_status_t ValidateEncodeResultOutput(
    const Session& session, amdf_media_encode_result_info_t* out_info,
    amdf_media_av1_encode_result_info_t** out_av1_result) {
  amdf_status_t status = amdf_structure_validate_header(
      out_info, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
      sizeof(*out_info));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (out_info->region_capacity != 0 && out_info->regions == nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  *out_av1_result = nullptr;
  if (session.info.configuration.codec == AMDF_MEDIA_CODEC_AV1) {
    if (out_info->next != nullptr) {
      auto* av1_result =
          static_cast<amdf_media_av1_encode_result_info_t*>(out_info->next);
      status = amdf_structure_validate_output(
          av1_result, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO,
          sizeof(*av1_result));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      *out_av1_result = av1_result;
    }
  } else if (out_info->next != nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

void ResetEncodeResult(amdf_media_encode_result_info_t* out_info,
                       amdf_media_av1_encode_result_info_t* av1_result) {
  const uint32_t region_capacity = out_info->region_capacity;
  amdf_media_encoded_region_t* regions = out_info->regions;
  void* next = out_info->next;
  const uint32_t structure_size = out_info->structure_size;
  if (av1_result != nullptr) {
    const uint32_t av1_structure_size = av1_result->structure_size;
    *av1_result = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO,
        .structure_size = av1_structure_size,
    };
  }
  *out_info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
      .structure_size = structure_size,
      .next = next,
      .region_capacity = region_capacity,
      .regions = regions,
  };
}

}  // namespace

amdf_status_t PrepareEncode(const Session& session,
                            const amdf_media_encode_picture_info_t& encode,
                            uint32_t maximum_reference_count,
                            EncodePicture* out_picture) {
  EncodePicture picture = {};
  amdf_status_t status = AMDF_STATUS_OK;
  switch (session.info.configuration.codec) {
    case AMDF_MEDIA_CODEC_H264:
      status = PrepareH264(session, encode, &picture);
      break;
    case AMDF_MEDIA_CODEC_HEVC:
      status = PrepareHevc(session, encode, &picture);
      break;
    case AMDF_MEDIA_CODEC_AV1:
      status = PrepareAv1(session, encode, &picture);
      break;
    default:
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
      break;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  PrepareSequence(session, encode.temporal_layer_index, &picture);
  if (picture.reference_count > maximum_reference_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (picture.used_as_reference) {
    picture.arguments.PictureControlDesc.Flags =
        D3D12_VIDEO_ENCODER_PICTURE_CONTROL_FLAG_USED_AS_REFERENCE_PICTURE;
  }
  *out_picture = picture;
  // Repair self-references after the value copy.
  out_picture->arguments.SequenceControlDesc.RateControl =
      session.native_encode
          .temporal_layer_rate_controls[encode.temporal_layer_index]
          .descriptor;
  out_picture->arguments.SequenceControlDesc.CodecGopSequence =
      session.native_encode.gop;
  if (session.native_encode.codec == D3D12_VIDEO_ENCODER_CODEC_AV1) {
    out_picture->arguments.SequenceControlDesc.FrameSubregionsLayoutData
        .pTilesPartition_AV1 = &session.native_encode.tiles;
  }
  if (session.info.configuration.codec == AMDF_MEDIA_CODEC_H264) {
    out_picture->h264.pList0ReferenceFrames =
        out_picture->reference_count != 0 ? out_picture->list0 : nullptr;
    out_picture->h264.pReferenceFramesReconPictureDescriptors =
        out_picture->reference_count != 0 ? out_picture->h264_references
                                          : nullptr;
    out_picture->arguments.PictureControlDesc.PictureControlCodecData
        .pH264PicData = &out_picture->h264;
  } else if (session.info.configuration.codec == AMDF_MEDIA_CODEC_HEVC) {
    out_picture->hevc.pList0ReferenceFrames =
        out_picture->reference_count != 0 ? out_picture->list0 : nullptr;
    out_picture->hevc.pReferenceFramesReconPictureDescriptors =
        out_picture->reference_count != 0 ? out_picture->hevc_references
                                          : nullptr;
    out_picture->arguments.PictureControlDesc.PictureControlCodecData
        .pHEVCPicData = &out_picture->hevc;
  } else {
    out_picture->arguments.PictureControlDesc.PictureControlCodecData
        .pAV1PicData = &out_picture->av1;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t ParseEncodeResult(const Session& session, const uint8_t* feedback,
                                uint64_t feedback_byte_length,
                                uint64_t compressed_byte_length,
                                amdf_media_encode_result_info_t* out_info) {
  amdf_media_av1_encode_result_info_t* av1_result = nullptr;
  amdf_status_t status =
      ValidateEncodeResultOutput(session, out_info, &av1_result);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (feedback_byte_length < sizeof(D3D12_VIDEO_ENCODER_OUTPUT_METADATA)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const auto* metadata =
      reinterpret_cast<const D3D12_VIDEO_ENCODER_OUTPUT_METADATA*>(feedback);
  if (metadata->EncodeErrorFlags !=
      D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_NO_ERROR) {
    ResetEncodeResult(out_info, av1_result);
    return AMDF_STATUS_OK;
  }
  if (metadata->WrittenSubregionsCount != 1 ||
      metadata->EncodedBitstreamWrittenBytesCount == 0 ||
      metadata->EncodedBitstreamWrittenBytesCount > compressed_byte_length ||
      feedback_byte_length <
          sizeof(*metadata) +
              sizeof(D3D12_VIDEO_ENCODER_FRAME_SUBREGION_METADATA)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  const auto* region =
      reinterpret_cast<const D3D12_VIDEO_ENCODER_FRAME_SUBREGION_METADATA*>(
          feedback + sizeof(*metadata));
  if (region->bStartOffset >= region->bSize ||
      region->bSize > metadata->EncodedBitstreamWrittenBytesCount) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  }
  if (session.info.configuration.codec == AMDF_MEDIA_CODEC_AV1) {
    const uint64_t av1_offset = sizeof(*metadata) + sizeof(*region);
    const uint64_t required =
        av1_offset +
        sizeof(
            D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES) +
        sizeof(D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES);
    if (feedback_byte_length < required) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    const auto* tiles = reinterpret_cast<
        const D3D12_VIDEO_ENCODER_AV1_PICTURE_CONTROL_SUBREGIONS_LAYOUT_DATA_TILES*>(
        feedback + av1_offset);
    if (tiles->RowCount != 1 || tiles->ColCount != 1) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
    const auto* post =
        reinterpret_cast<const D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES*>(
            feedback + av1_offset + sizeof(*tiles));
    if (!NativeAv1ResultValid(*post)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
    }
  }
  amdf_media_av1_encode_result_info_t av1_value = {};
  if (av1_result != nullptr) {
    av1_value.type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO;
    av1_value.structure_size = av1_result->structure_size;
    const uint64_t av1_offset = sizeof(*metadata) + sizeof(*region) +
                                sizeof(session.native_encode.tiles);
    const auto* post =
        reinterpret_cast<const D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES*>(
            feedback + av1_offset);
    CopyAv1Result(*post, &av1_value);
  }
  const amdf_media_encoded_region_t region_value = {
      .kind = static_cast<amdf_media_encoded_region_kind_t>(
          session.info.configuration.codec == AMDF_MEDIA_CODEC_H264
              ? AMDF_MEDIA_ENCODED_REGION_H264_ANNEX_B_SLICE
          : session.info.configuration.codec == AMDF_MEDIA_CODEC_HEVC
              ? AMDF_MEDIA_ENCODED_REGION_HEVC_ANNEX_B_SLICE
              : AMDF_MEDIA_ENCODED_REGION_AV1_TILE_GROUP_PAYLOAD),
      .byte_offset = region->bStartOffset,
      .byte_length = region->bSize - region->bStartOffset,
  };
  const uint32_t region_capacity = out_info->region_capacity;
  amdf_media_encoded_region_t* regions = out_info->regions;
  ResetEncodeResult(out_info, av1_result);
  out_info->written_byte_length = metadata->EncodedBitstreamWrittenBytesCount;
  out_info->region_count = 1;
  if (av1_result != nullptr) {
    *av1_result = av1_value;
  }
  if (region_capacity == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  }
  regions[0] = region_value;
  return AMDF_STATUS_OK;
}

amdf_status_t ClearEncodeResult(const Session& session,
                                amdf_media_encode_result_info_t* out_info) {
  amdf_media_av1_encode_result_info_t* av1_result = nullptr;
  const amdf_status_t status =
      ValidateEncodeResultOutput(session, out_info, &av1_result);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  ResetEncodeResult(out_info, av1_result);
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
