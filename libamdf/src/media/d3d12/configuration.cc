// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/configuration.h"

#include <cstdint>

#include "amdf/media/vp9.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/media/av1.h"
#include "libamdf/src/media/d3d12/device.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/media/h264.h"
#include "libamdf/src/media/hevc.h"
#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {

D3D12_RESOURCE_DESC ImageDescription(
    const amdf_media_image_description_t& description) {
  D3D12_RESOURCE_DESC native = {
      .Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
  };
  native.Width = description.width;
  native.Height = description.height;
  native.DepthOrArraySize = static_cast<UINT16>(description.array_layer_count);
  native.MipLevels = 1;
  native.Format = description.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
                      ? DXGI_FORMAT_NV12
                      : DXGI_FORMAT_P010;
  native.SampleDesc.Count = 1;
  if (description.roles == AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE) {
    native.Flags = D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY |
                   D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
  } else if (description.roles == AMDF_MEDIA_IMAGE_ROLE_ENCODE_INPUT) {
    native.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
  } else if (description.roles == AMDF_MEDIA_IMAGE_ROLE_ENCODE_RECONSTRUCTION) {
    native.Flags = D3D12_RESOURCE_FLAG_VIDEO_ENCODE_REFERENCE_ONLY |
                   D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE;
  }
  return native;
}

amdf_status_t ImageRequirements(
    Device* device, const amdf_media_image_description_t& description,
    amdf_media_image_requirements_t* out_requirements) {
  if (description.width == 0 || description.height == 0 ||
      description.array_layer_count == 0 || description.roles == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  const bool supported_roles =
      description.roles == AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT ||
      description.roles == AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE ||
      description.roles == (AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT |
                            AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE) ||
      description.roles == AMDF_MEDIA_IMAGE_ROLE_ENCODE_INPUT ||
      description.roles == AMDF_MEDIA_IMAGE_ROLE_ENCODE_RECONSTRUCTION;
  if ((description.format != AMDF_MEDIA_PIXEL_FORMAT_NV12 &&
       description.format != AMDF_MEDIA_PIXEL_FORMAT_P010) ||
      description.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      description.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      description.array_layer_count >
          D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION ||
      ((description.width | description.height) & 1) != 0 || !supported_roles) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const D3D12_RESOURCE_DESC native = ImageDescription(description);
  const D3D12_RESOURCE_ALLOCATION_INFO allocation =
      device->native->GetResourceAllocationInfo(0, 1, &native);
  if (allocation.SizeInBytes == UINT64_MAX || allocation.SizeInBytes == 0 ||
      allocation.Alignment == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_requirements = {description, allocation.SizeInBytes,
                       allocation.Alignment};
  return AMDF_STATUS_OK;
}

static amdf_status_t QueryDecodeConfiguration(
    Device* device, const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    D3D12_VIDEO_DECODE_CONFIGURATION* out_native_decode) {
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
      .configuration = configuration,
  };
  D3D12_FEATURE_DATA_VIDEO_DECODE_SUPPORT support = {};
  uint32_t dimension_alignment = 0;
  uint32_t maximum_reference_count = 0;
  amdf_status_t status = AMDF_STATUS_OK;
  switch (configuration.codec) {
    case AMDF_MEDIA_CODEC_H264: {
      status = amdf_structure_validate_input(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
          sizeof(amdf_media_h264_configuration_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& h264 = *static_cast<const amdf_media_h264_configuration_t*>(
          codec_configuration);
      if (h264.profile < AMDF_MEDIA_H264_PROFILE_CONSTRAINED_BASELINE ||
          h264.profile > AMDF_MEDIA_H264_PROFILE_HIGH || h264.level_idc > 255) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
      } else if (configuration.format != AMDF_MEDIA_PIXEL_FORMAT_NV12) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_FORMAT;
      }
      support.Configuration.DecodeProfile = {
          0x1b81be68,
          0xa0c7,
          0x11d3,
          {0xb9, 0x84, 0x00, 0xc0, 0x4f, 0x2e, 0x73, 0xc5}};
      dimension_alignment = 16;
      maximum_reference_count = 17;
      break;
    }
    case AMDF_MEDIA_CODEC_HEVC: {
      status = amdf_structure_validate_input(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
          sizeof(amdf_media_hevc_configuration_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& hevc = *static_cast<const amdf_media_hevc_configuration_t*>(
          codec_configuration);
      if ((hevc.profile != AMDF_MEDIA_HEVC_PROFILE_MAIN &&
           hevc.profile != AMDF_MEDIA_HEVC_PROFILE_MAIN10) ||
          (hevc.sample_bit_depth != 8 && hevc.sample_bit_depth != 10) ||
          (hevc.profile == AMDF_MEDIA_HEVC_PROFILE_MAIN &&
           hevc.sample_bit_depth != 8)) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
      } else if (configuration.format != (hevc.sample_bit_depth == 8
                                              ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                              : AMDF_MEDIA_PIXEL_FORMAT_P010)) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_FORMAT;
      }
      static constexpr GUID profiles[] = {
          {0x5b11d51b,
           0x2f4c,
           0x4452,
           {0xbc, 0xc3, 0x09, 0xf2, 0xa1, 0x16, 0x0c, 0xc0}},
          {0x107af0e0,
           0xef1a,
           0x4d19,
           {0xab, 0xa8, 0x67, 0xa1, 0x63, 0x07, 0x3d, 0x13}},
      };
      // Main10 permits eight-bit pictures with the same coding tools as Main.
      // Select the native decoding mode by the actual sample representation;
      // the bitstream's profile declaration need not name the narrowest mode.
      support.Configuration.DecodeProfile =
          profiles[hevc.sample_bit_depth == 10];
      dimension_alignment = 8;
      maximum_reference_count = 16;
      break;
    }
    case AMDF_MEDIA_CODEC_VP9: {
      status = amdf_structure_validate_input(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_VP9_CONFIGURATION,
          sizeof(amdf_media_vp9_configuration_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& vp9 = *static_cast<const amdf_media_vp9_configuration_t*>(
          codec_configuration);
      if (!((vp9.profile == AMDF_MEDIA_VP9_PROFILE_0 &&
             vp9.sample_bit_depth == 8) ||
            (vp9.profile == AMDF_MEDIA_VP9_PROFILE_2 &&
             vp9.sample_bit_depth == 10))) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
      } else if (configuration.format != (vp9.sample_bit_depth == 8
                                              ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                              : AMDF_MEDIA_PIXEL_FORMAT_P010)) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_FORMAT;
      }
      static constexpr GUID profiles[] = {
          {0x463707f8,
           0xa1d0,
           0x4585,
           {0x87, 0x6d, 0x83, 0xaa, 0x6d, 0x60, 0xb8, 0x9e}},
          {0xa4c749ef,
           0x6ecf,
           0x48aa,
           {0x84, 0x48, 0x50, 0xa7, 0xa1, 0x16, 0x5f, 0xf7}},
      };
      support.Configuration.DecodeProfile =
          profiles[vp9.sample_bit_depth == 10];
      dimension_alignment = 2;
      maximum_reference_count = 9;
      break;
    }
    case AMDF_MEDIA_CODEC_AV1: {
      status = amdf_structure_validate_input(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
          sizeof(amdf_media_av1_configuration_t));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      const auto& av1 = *static_cast<const amdf_media_av1_configuration_t*>(
          codec_configuration);
      if (av1.profile != 0 ||
          (av1.sample_bit_depth != 8 && av1.sample_bit_depth != 10)) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
      } else if (configuration.format != (av1.sample_bit_depth == 8
                                              ? AMDF_MEDIA_PIXEL_FORMAT_NV12
                                              : AMDF_MEDIA_PIXEL_FORMAT_P010)) {
        info.reason = AMDF_MEDIA_CONFIGURATION_REASON_FORMAT;
      }
      support.Configuration.DecodeProfile = {
          0xb8be4ccb,
          0xcf53,
          0x46ba,
          {0x8d, 0x59, 0xd6, 0xb8, 0xa6, 0xda, 0x5d, 0x2a}};
      dimension_alignment = 2;
      maximum_reference_count = 9;
      break;
    }
    default:
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
      break;
  }
  if (info.reason != AMDF_MEDIA_CONFIGURATION_REASON_NONE) {
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  if (configuration.coded_width == 0 || configuration.coded_height == 0 ||
      configuration.coded_width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      configuration.coded_height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      ((configuration.coded_width | configuration.coded_height) &
       (dimension_alignment - 1)) != 0) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS;
  } else if (configuration.reference_image_count == 0 ||
             configuration.reference_image_count > maximum_reference_count) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE;
  }
  if (info.reason != AMDF_MEDIA_CONFIGURATION_REASON_NONE) {
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  support.Width = configuration.coded_width;
  support.Height = configuration.coded_height;
  support.DecodeFormat = configuration.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
                             ? DXGI_FORMAT_NV12
                             : DXGI_FORMAT_P010;
  // Unknown rates ask for the complete syntax envelope, not an invented rate
  // target. The session reports the native state charge for this envelope.
  status = Hresult(device->video->CheckFeatureSupport(
      D3D12_FEATURE_VIDEO_DECODE_SUPPORT, &support, sizeof(support)));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if ((support.SupportFlags & D3D12_VIDEO_DECODE_SUPPORT_FLAG_SUPPORTED) == 0) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION;
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  const bool separate_reference =
      (support.ConfigurationFlags &
       D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_REFERENCE_ONLY_ALLOCATIONS_REQUIRED) !=
      0;
  if (separate_reference) {
    D3D12_FEATURE_DATA_VIDEO_DECODE_CONVERSION_SUPPORT conversion = {
        .Configuration = support.Configuration,
        .DecodeSample = {configuration.coded_width,
                         configuration.coded_height,
                         {support.DecodeFormat,
                          DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601}},
    };
    conversion.OutputFormat = conversion.DecodeSample.Format;
    status = Hresult(device->video->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_DECODE_CONVERSION_SUPPORT, &conversion,
        sizeof(conversion)));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if ((conversion.SupportFlags &
         D3D12_VIDEO_DECODE_CONVERSION_SUPPORT_FLAG_SUPPORTED) == 0) {
      info.reason = AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE;
      *out_info = info;
      return AMDF_STATUS_OK;
    }
  }
  uint32_t storage_height = configuration.coded_height;
  if ((support.ConfigurationFlags &
       D3D12_VIDEO_DECODE_CONFIGURATION_FLAG_HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED) !=
      0) {
    storage_height = (storage_height + 31) & ~31u;
  }
  amdf_media_image_description_t description = {
      configuration.format, configuration.coded_width, storage_height, 1,
      AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT};
  // VP9/AV1 reference sampling needs complete sixteen-row reconstruction
  // blocks, including the chroma border. Native admission does not report this
  // through HEIGHT_ALIGNMENT_MULTIPLE_32_REQUIRED. Keep coded and output
  // extents exact when a separate reference allocation carries this padding.
  const uint32_t reference_height =
      (configuration.codec == AMDF_MEDIA_CODEC_VP9 ||
       configuration.codec == AMDF_MEDIA_CODEC_AV1)
          ? (storage_height + 15) & ~15u
          : storage_height;
  if (!separate_reference) {
    description.roles |= AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE;
    description.height = reference_height;
  }
  status = ImageRequirements(device, description, &info.output);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (separate_reference) {
    description.roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE;
    description.height = reference_height;
    status = ImageRequirements(device, description, &info.reference);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  if (!device->existing_heaps) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION;
    *out_info = info;
    return AMDF_STATUS_OK;
  }
  info.support = AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED;
  info.image_heap_memory_scope =
      amdf_endpoint_local_memory_scope(device->base.base.endpoint);
  info.image_heap_memory_profile_ordinal = 0;
  info.compressed_memory_scope = info.image_heap_memory_scope;
  info.compressed_memory_profile_ordinal = 0;
  info.feedback_memory_scope =
      &device->base.base.provider_instance->system_memory_scope;
  info.feedback_memory_profile_ordinal = 0;
  info.compressed_byte_offset_alignment = 128;
  info.compressed_byte_length_alignment = 128;
  info.feedback_byte_length = sizeof(D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS);
  info.feedback_byte_offset_alignment = 8;
  *out_info = info;
  if (out_native_decode != nullptr) {
    *out_native_decode = support.Configuration;
  }
  return AMDF_STATUS_OK;
}

namespace {

bool H264Level(uint32_t level_idc, D3D12_VIDEO_ENCODER_LEVELS_H264* out_level) {
  static constexpr uint8_t kLevelIndicators[] = {
      10, 9,  11, 12, 13, 20, 21, 22, 30, 31,
      32, 40, 41, 42, 50, 51, 52, 60, 61, 62,
  };
  for (uint32_t index = 0; index < sizeof(kLevelIndicators); ++index) {
    if (kLevelIndicators[index] == level_idc) {
      *out_level = static_cast<D3D12_VIDEO_ENCODER_LEVELS_H264>(index);
      return true;
    }
  }
  return false;
}

bool HevcLevel(uint32_t level_idc, D3D12_VIDEO_ENCODER_LEVELS_HEVC* out_level) {
  static constexpr uint8_t kLevelIndicators[] = {
      30, 60, 63, 90, 93, 120, 123, 150, 153, 156, 180, 183, 186,
  };
  for (uint32_t index = 0; index < sizeof(kLevelIndicators); ++index) {
    if (kLevelIndicators[index] == level_idc) {
      *out_level = static_cast<D3D12_VIDEO_ENCODER_LEVELS_HEVC>(index);
      return true;
    }
  }
  return false;
}

void AnswerUnsupported(amdf_media_configuration_info_t* info,
                       amdf_media_configuration_reason_t reason,
                       amdf_media_configuration_info_t* out_info) {
  info->reason = reason;
  *out_info = *info;
}

amdf_status_t QueryEncodeConfiguration(
    Device* device, const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    EncodeConfiguration* out_native_encode) {
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
      .configuration = configuration,
  };
  const amdf_media_encode_configuration_t* encode = nullptr;
  const amdf_media_h264_configuration_t* h264 = nullptr;
  const amdf_media_h264_encode_configuration_t* h264_encode = nullptr;
  const amdf_media_hevc_configuration_t* hevc = nullptr;
  const amdf_media_hevc_encode_configuration_t* hevc_encode = nullptr;
  const amdf_media_av1_configuration_t* av1 = nullptr;
  const amdf_media_av1_encode_configuration_t* av1_encode = nullptr;
  amdf_media_pixel_format_t expected_format = AMDF_MEDIA_PIXEL_FORMAT_UNKNOWN;
  amdf_status_t status = AMDF_STATUS_OK;
  switch (configuration.codec) {
    case AMDF_MEDIA_CODEC_H264:
      status = amdf_structure_validate_header(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
          sizeof(*h264));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      h264 = static_cast<const amdf_media_h264_configuration_t*>(
          codec_configuration);
      encode =
          static_cast<const amdf_media_encode_configuration_t*>(h264->next);
      status = amdf_structure_validate_header(
          encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION,
          sizeof(*encode));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      h264_encode = static_cast<const amdf_media_h264_encode_configuration_t*>(
          encode->next);
      status = amdf_structure_validate_input(
          h264_encode, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION,
          sizeof(*h264_encode));
      expected_format = AMDF_MEDIA_PIXEL_FORMAT_NV12;
      break;
    case AMDF_MEDIA_CODEC_HEVC:
      status = amdf_structure_validate_header(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION,
          sizeof(*hevc));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      hevc = static_cast<const amdf_media_hevc_configuration_t*>(
          codec_configuration);
      encode =
          static_cast<const amdf_media_encode_configuration_t*>(hevc->next);
      status = amdf_structure_validate_header(
          encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION,
          sizeof(*encode));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      hevc_encode = static_cast<const amdf_media_hevc_encode_configuration_t*>(
          encode->next);
      status = amdf_structure_validate_input(
          hevc_encode, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_CONFIGURATION,
          sizeof(*hevc_encode));
      if (amdf_status_is_ok(status)) {
        expected_format = hevc->sample_bit_depth == 10
                              ? AMDF_MEDIA_PIXEL_FORMAT_P010
                              : AMDF_MEDIA_PIXEL_FORMAT_NV12;
      }
      break;
    case AMDF_MEDIA_CODEC_AV1:
      status = amdf_structure_validate_header(
          codec_configuration, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION,
          sizeof(*av1));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      av1 = static_cast<const amdf_media_av1_configuration_t*>(
          codec_configuration);
      encode = static_cast<const amdf_media_encode_configuration_t*>(av1->next);
      status = amdf_structure_validate_header(
          encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION,
          sizeof(*encode));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      av1_encode = static_cast<const amdf_media_av1_encode_configuration_t*>(
          encode->next);
      status = amdf_structure_validate_input(
          av1_encode, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION,
          sizeof(*av1_encode));
      if (amdf_status_is_ok(status)) {
        expected_format = av1->sample_bit_depth == 10
                              ? AMDF_MEDIA_PIXEL_FORMAT_P010
                              : AMDF_MEDIA_PIXEL_FORMAT_NV12;
      }
      break;
    default:
      AnswerUnsupported(&info, AMDF_MEDIA_CONFIGURATION_REASON_CODEC, out_info);
      return AMDF_STATUS_OK;
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (configuration.format != expected_format) {
    AnswerUnsupported(&info, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT, out_info);
    return AMDF_STATUS_OK;
  }
  if (configuration.reference_image_count < 2 ||
      configuration.reference_image_count >
          D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION) {
    AnswerUnsupported(&info, AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE,
                      out_info);
    return AMDF_STATUS_OK;
  }
  if (configuration.coded_width == 0 || configuration.coded_height == 0 ||
      configuration.coded_width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      configuration.coded_height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      ((configuration.coded_width | configuration.coded_height) & 1) != 0) {
    AnswerUnsupported(&info, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS,
                      out_info);
    return AMDF_STATUS_OK;
  }
  if (h264 != nullptr) {
    status = amdf_media_h264_validate_encode_configuration(&configuration, h264,
                                                           encode, h264_encode);
  } else if (hevc != nullptr) {
    status = amdf_media_hevc_validate_encode_configuration(&configuration, hevc,
                                                           encode, hevc_encode);
  } else {
    status = amdf_media_av1_validate_encode_configuration(&configuration, av1,
                                                          encode, av1_encode);
  }
  if (!amdf_status_is_ok(status)) {
    AnswerUnsupported(&info, AMDF_MEDIA_CONFIGURATION_REASON_CODEC, out_info);
    return AMDF_STATUS_OK;
  }

  EncodeConfiguration local_native = {};
  EncodeConfiguration* native =
      out_native_encode != nullptr ? out_native_encode : &local_native;
  *native = {};
  D3D12_FEATURE_DATA_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT codec_support =
      {};
  if (h264 != nullptr) {
    native->codec = D3D12_VIDEO_ENCODER_CODEC_H264;
    native->h264_profile = D3D12_VIDEO_ENCODER_PROFILE_H264_MAIN;
    native->profile = {sizeof(native->h264_profile)};
    native->profile.pH264Profile = &native->h264_profile;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_H264 limits = {};
    codec_support.Codec = native->codec;
    codec_support.Profile = native->profile;
    codec_support.CodecSupportLimits = {sizeof(limits)};
    codec_support.CodecSupportLimits.pH264Support = &limits;
    status = Hresult(device->encoder->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT, &codec_support,
        sizeof(codec_support)));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (!codec_support.IsSupported ||
        !H264Level(h264->level_idc, &native->h264_level)) {
      AnswerUnsupported(&info,
                        AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION,
                        out_info);
      return AMDF_STATUS_OK;
    }
    native->codec_configuration = {sizeof(native->h264)};
    native->codec_configuration.pH264Config = &native->h264;
    native->h264_gop = {
        encode->intra_picture_period,
        encode->inter_picture_period,
        static_cast<UCHAR>(h264_encode->picture_order_count_type),
        static_cast<UCHAR>(h264_encode->log2_max_frame_number_minus4),
        static_cast<UCHAR>(
            h264_encode->log2_max_picture_order_count_lsb_minus4),
    };
    native->gop = {sizeof(native->h264_gop)};
    native->gop.pH264GroupOfPictures = &native->h264_gop;
    native->level = {sizeof(native->h264_level)};
    native->level.pH264LevelSetting = &native->h264_level;
  } else if (hevc != nullptr) {
    native->codec = D3D12_VIDEO_ENCODER_CODEC_HEVC;
    native->hevc_profile = hevc->sample_bit_depth == 10
                               ? D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN10
                               : D3D12_VIDEO_ENCODER_PROFILE_HEVC_MAIN;
    native->profile = {sizeof(native->hevc_profile)};
    native->profile.pHEVCProfile = &native->hevc_profile;
    D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC limits = {};
    limits.MinLumaCodingUnitSize =
        static_cast<D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE>(
            hevc_encode->minimum_luma_coding_unit_log2 - 3);
    limits.MaxLumaCodingUnitSize =
        static_cast<D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_CUSIZE>(
            hevc_encode->maximum_luma_coding_unit_log2 - 3);
    limits.MinLumaTransformUnitSize =
        static_cast<D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE>(
            hevc_encode->minimum_luma_transform_unit_log2 - 2);
    limits.MaxLumaTransformUnitSize =
        static_cast<D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_HEVC_TUSIZE>(
            hevc_encode->maximum_luma_transform_unit_log2 - 2);
    limits.max_transform_hierarchy_depth_inter = static_cast<UCHAR>(
        hevc_encode->maximum_transform_hierarchy_depth_inter);
    limits.max_transform_hierarchy_depth_intra = static_cast<UCHAR>(
        hevc_encode->maximum_transform_hierarchy_depth_intra);
    codec_support.Codec = native->codec;
    codec_support.Profile = native->profile;
    codec_support.CodecSupportLimits = {sizeof(limits)};
    codec_support.CodecSupportLimits.pHEVCSupport = &limits;
    status = Hresult(device->encoder->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT, &codec_support,
        sizeof(codec_support)));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    constexpr uint32_t kUnmodeledRequiredFlags =
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_ASYMETRIC_MOTION_PARTITION_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_ROTATION_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_TRANSFORM_SKIP_CONTEXT_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_IMPLICIT_RDPCM_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXPLICIT_RDPCM_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_EXTENDED_PRECISION_PROCESSING_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_INTRA_SMOOTHING_DISABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_HIGH_PRECISION_OFFSETS_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_PERSISTENT_RICE_ADAPTATION_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CABAC_BYPASS_ALIGNMENT_ENABLED_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CROSS_COMPONENT_PREDICTION_ENABLED_FLAG_REQUIRED |
        D3D12_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT_HEVC_FLAG_CHROMA_QP_OFFSET_LIST_ENABLED_FLAG_REQUIRED;
    if (!codec_support.IsSupported ||
        (static_cast<uint32_t>(limits.SupportFlags) &
         kUnmodeledRequiredFlags) != 0 ||
        !HevcLevel(hevc_encode->level_idc, &native->hevc_level.Level)) {
      AnswerUnsupported(&info,
                        AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION,
                        out_info);
      return AMDF_STATUS_OK;
    }
    native->hevc_level.Tier = hevc_encode->tier == AMDF_MEDIA_HEVC_TIER_HIGH
                                  ? D3D12_VIDEO_ENCODER_TIER_HEVC_HIGH
                                  : D3D12_VIDEO_ENCODER_TIER_HEVC_MAIN;
    native->hevc.MinLumaCodingUnitSize = limits.MinLumaCodingUnitSize;
    native->hevc.MaxLumaCodingUnitSize = limits.MaxLumaCodingUnitSize;
    native->hevc.MinLumaTransformUnitSize = limits.MinLumaTransformUnitSize;
    native->hevc.MaxLumaTransformUnitSize = limits.MaxLumaTransformUnitSize;
    native->hevc.max_transform_hierarchy_depth_inter =
        limits.max_transform_hierarchy_depth_inter;
    native->hevc.max_transform_hierarchy_depth_intra =
        limits.max_transform_hierarchy_depth_intra;
    native->codec_configuration = {sizeof(native->hevc)};
    native->codec_configuration.pHEVCConfig = &native->hevc;
    native->hevc_gop = {
        encode->intra_picture_period,
        encode->inter_picture_period,
        static_cast<UCHAR>(
            hevc_encode->log2_max_picture_order_count_lsb_minus4),
    };
    native->gop = {sizeof(native->hevc_gop)};
    native->gop.pHEVCGroupOfPictures = &native->hevc_gop;
    native->level = {sizeof(native->hevc_level)};
    native->level.pHEVCLevelSetting = &native->hevc_level;
  } else {
    native->codec = D3D12_VIDEO_ENCODER_CODEC_AV1;
    native->av1_profile = D3D12_VIDEO_ENCODER_AV1_PROFILE_MAIN;
    native->profile = {sizeof(native->av1_profile)};
    native->profile.pAV1Profile = &native->av1_profile;
    D3D12_VIDEO_ENCODER_AV1_CODEC_CONFIGURATION_SUPPORT limits = {};
    codec_support.Codec = native->codec;
    codec_support.Profile = native->profile;
    codec_support.CodecSupportLimits = {sizeof(limits)};
    codec_support.CodecSupportLimits.pAV1Support = &limits;
    status = Hresult(device->encoder->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_ENCODER_CODEC_CONFIGURATION_SUPPORT, &codec_support,
        sizeof(codec_support)));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    uint32_t requested_features = 0;
    if ((av1_encode->feature_flags &
         AMDF_MEDIA_AV1_ENCODE_FEATURE_ORDER_HINT) != 0) {
      requested_features |=
          D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_ORDER_HINT_TOOLS;
    }
    if ((av1_encode->feature_flags & AMDF_MEDIA_AV1_ENCODE_FEATURE_CDEF) != 0) {
      requested_features |= D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAG_CDEF_FILTERING;
    }
    if (!codec_support.IsSupported || av1_encode->level > 23 ||
        (static_cast<uint32_t>(limits.RequiredFeatureFlags) &
         ~requested_features) != 0 ||
        (requested_features &
         ~static_cast<uint32_t>(limits.SupportedFeatureFlags)) != 0) {
      AnswerUnsupported(&info,
                        AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION,
                        out_info);
      return AMDF_STATUS_OK;
    }
    native->av1.FeatureFlags =
        static_cast<D3D12_VIDEO_ENCODER_AV1_FEATURE_FLAGS>(requested_features);
    native->av1.OrderHintBitsMinus1 =
        av1_encode->order_hint_bit_count == 0
            ? 0
            : av1_encode->order_hint_bit_count - 1;
    native->codec_configuration = {sizeof(native->av1)};
    native->codec_configuration.pAV1Config = &native->av1;
    native->av1_gop = {encode->intra_picture_period,
                       encode->inter_picture_period};
    native->gop = {sizeof(native->av1_gop)};
    native->gop.pAV1SequenceStructure = &native->av1_gop;
    native->av1_level.Level =
        static_cast<D3D12_VIDEO_ENCODER_AV1_LEVELS>(av1_encode->level);
    native->av1_level.Tier = av1_encode->tier == AMDF_MEDIA_AV1_TIER_HIGH
                                 ? D3D12_VIDEO_ENCODER_AV1_TIER_HIGH
                                 : D3D12_VIDEO_ENCODER_AV1_TIER_MAIN;
    native->level = {sizeof(native->av1_level)};
    native->level.pAV1LevelSetting = &native->av1_level;
  }

  native->rate_control.TargetFrameRate = {
      encode->rate_control.frame_rate_numerator,
      encode->rate_control.frame_rate_denominator,
  };
  if (encode->rate_control.mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP) {
    const auto& source = encode->rate_control.control.cqp;
    native->cqp = {source.intra_quantization_parameter,
                   source.predicted_quantization_parameter,
                   source.predicted_quantization_parameter};
    native->rate_control.Mode = D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CQP;
    native->rate_control.ConfigParams = {sizeof(native->cqp)};
    native->rate_control.ConfigParams.pConfiguration_CQP = &native->cqp;
  } else {
    const auto& source = encode->rate_control.control.bit_rate;
    native->rate_control.Flags =
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_QP_RANGE |
        D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_VBV_SIZES;
    if (source.maximum_frame_bit_count != 0) {
      native->rate_control.Flags |=
          D3D12_VIDEO_ENCODER_RATE_CONTROL_FLAG_ENABLE_MAX_FRAME_SIZE;
    }
    if (encode->rate_control.mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR) {
      native->cbr = {0,
                     source.minimum_quantization_parameter,
                     source.maximum_quantization_parameter,
                     source.maximum_frame_bit_count,
                     source.target_bit_rate,
                     source.virtual_buffer_capacity,
                     source.initial_virtual_buffer_fullness};
      native->rate_control.Mode = D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_CBR;
      native->rate_control.ConfigParams = {sizeof(native->cbr)};
      native->rate_control.ConfigParams.pConfiguration_CBR = &native->cbr;
    } else {
      native->vbr = {0,
                     source.minimum_quantization_parameter,
                     source.maximum_quantization_parameter,
                     source.maximum_frame_bit_count,
                     source.target_bit_rate,
                     source.peak_bit_rate,
                     source.virtual_buffer_capacity,
                     source.initial_virtual_buffer_fullness};
      native->rate_control.Mode = D3D12_VIDEO_ENCODER_RATE_CONTROL_MODE_VBR;
      native->rate_control.ConfigParams = {sizeof(native->vbr)};
      native->rate_control.ConfigParams.pConfiguration_VBR = &native->vbr;
    }
  }
  native->resolution = {configuration.coded_width, configuration.coded_height};
  native->subregion_mode =
      D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_FULL_FRAME;
  D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOLUTION_SUPPORT_LIMITS limits = {};
  D3D12_FEATURE_DATA_VIDEO_ENCODER_SUPPORT1 support = {
      .Codec = native->codec,
      .InputFormat = configuration.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
                         ? DXGI_FORMAT_NV12
                         : DXGI_FORMAT_P010,
      .CodecConfiguration = native->codec_configuration,
      .CodecGopSequence = native->gop,
      .RateControl = native->rate_control,
      .IntraRefresh = D3D12_VIDEO_ENCODER_INTRA_REFRESH_MODE_NONE,
      .SubregionFrameEncoding =
          D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_FULL_FRAME,
      .ResolutionsListCount = 1,
      .pResolutionList = &native->resolution,
      .MaxReferenceFramesInDPB = 1,
      .SuggestedProfile = native->profile,
      .SuggestedLevel = native->level,
      .pResolutionDependentSupport = &limits,
  };
  if (native->codec == D3D12_VIDEO_ENCODER_CODEC_AV1) {
    native->tiles.RowCount = 1;
    native->tiles.ColCount = 1;
    native->tiles.RowHeights[0] = (configuration.coded_height + 63) / 64;
    native->tiles.ColWidths[0] = (configuration.coded_width + 63) / 64;
    support.SubregionFrameEncoding =
        D3D12_VIDEO_ENCODER_FRAME_SUBREGION_LAYOUT_MODE_UNIFORM_GRID_PARTITION;
    native->subregion_mode = support.SubregionFrameEncoding;
    support.SubregionFrameEncodingData = {sizeof(native->tiles)};
    support.SubregionFrameEncodingData.pTilesPartition_AV1 = &native->tiles;
  }
  status = Hresult(device->encoder->CheckFeatureSupport(
      D3D12_FEATURE_VIDEO_ENCODER_SUPPORT1, &support, sizeof(support)));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (support.ValidationFlags != D3D12_VIDEO_ENCODER_VALIDATION_FLAG_NONE ||
      (support.SupportFlags &
       D3D12_VIDEO_ENCODER_SUPPORT_FLAG_GENERAL_SUPPORT_OK) == 0) {
    const auto reason =
        support.ValidationFlags ==
                D3D12_VIDEO_ENCODER_VALIDATION_FLAG_RESOLUTION_NOT_SUPPORTED_IN_LIST
            ? AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS
            : AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION;
    AnswerUnsupported(&info, reason, out_info);
    return AMDF_STATUS_OK;
  }
  D3D12_FEATURE_DATA_VIDEO_ENCODER_RESOURCE_REQUIREMENTS resources = {
      .Codec = native->codec,
      .Profile = native->profile,
      .InputFormat = support.InputFormat,
      .PictureTargetResolution = native->resolution,
  };
  status = Hresult(device->encoder->CheckFeatureSupport(
      D3D12_FEATURE_VIDEO_ENCODER_RESOURCE_REQUIREMENTS, &resources,
      sizeof(resources)));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!resources.IsSupported ||
      resources.CompressedBitstreamBufferAccessAlignment == 0 ||
      resources.EncoderMetadataBufferAccessAlignment == 0 ||
      resources.MaxEncoderOutputMetadataBufferSize == 0) {
    AnswerUnsupported(
        &info, AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION, out_info);
    return AMDF_STATUS_OK;
  }

  amdf_media_image_description_t description = {
      configuration.format,
      configuration.coded_width,
      configuration.coded_height,
      1,
      AMDF_MEDIA_IMAGE_ROLE_ENCODE_INPUT,
  };
  status = ImageRequirements(device, description, &info.input);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  description.array_layer_count = configuration.reference_image_count;
  description.roles = AMDF_MEDIA_IMAGE_ROLE_ENCODE_RECONSTRUCTION;
  status = ImageRequirements(device, description, &info.reference);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!device->existing_heaps) {
    AnswerUnsupported(
        &info, AMDF_MEDIA_CONFIGURATION_REASON_NATIVE_CONFIGURATION, out_info);
    return AMDF_STATUS_OK;
  }
  const uint64_t storage_width =
      (uint64_t(configuration.coded_width) + 255) & ~UINT64_C(255);
  const uint64_t storage_height =
      (uint64_t(configuration.coded_height) + 63) & ~UINT64_C(63);
  const uint64_t sample_bytes =
      configuration.format == AMDF_MEDIA_PIXEL_FORMAT_P010 ? 2 : 1;
  info.support = AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED;
  info.image_heap_memory_scope =
      amdf_endpoint_local_memory_scope(device->base.base.endpoint);
  info.image_heap_memory_profile_ordinal = 0;
  info.compressed_memory_scope = info.image_heap_memory_scope;
  info.compressed_memory_profile_ordinal = 0;
  info.feedback_memory_scope =
      &device->base.base.provider_instance->system_memory_scope;
  info.feedback_memory_profile_ordinal = 0;
  info.compressed_byte_offset_alignment =
      resources.CompressedBitstreamBufferAccessAlignment;
  info.compressed_byte_length_alignment =
      resources.CompressedBitstreamBufferAccessAlignment;
  info.minimum_compressed_byte_length =
      storage_width * storage_height * 3 / 2 * sample_bytes;
  info.feedback_byte_offset_alignment =
      resources.EncoderMetadataBufferAccessAlignment;
  info.feedback_byte_length =
      sizeof(D3D12_VIDEO_ENCODER_OUTPUT_METADATA) +
      sizeof(D3D12_VIDEO_ENCODER_FRAME_SUBREGION_METADATA);
  if (native->codec == D3D12_VIDEO_ENCODER_CODEC_AV1) {
    info.feedback_byte_length +=
        sizeof(native->tiles) +
        sizeof(D3D12_VIDEO_ENCODER_AV1_POST_ENCODE_VALUES);
  }
  native->metadata_byte_length = resources.MaxEncoderOutputMetadataBufferSize;
  native->metadata_alignment = resources.EncoderMetadataBufferAccessAlignment;
  *out_info = info;
  return AMDF_STATUS_OK;
}

}  // namespace

amdf_status_t QueryConfiguration(
    Device* device, uint32_t family_ordinal,
    const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    D3D12_VIDEO_DECODE_CONFIGURATION* out_native_decode,
    EncodeConfiguration* out_native_encode) {
  if (family_ordinal == kDecodeFamilyOrdinal &&
      configuration.direction == AMDF_MEDIA_DIRECTION_DECODE) {
    return QueryDecodeConfiguration(device, configuration, codec_configuration,
                                    out_info, out_native_decode);
  }
  if (family_ordinal == kEncodeFamilyOrdinal && device->encoder != nullptr &&
      configuration.direction == AMDF_MEDIA_DIRECTION_ENCODE) {
    return QueryEncodeConfiguration(device, configuration, codec_configuration,
                                    out_info, out_native_encode);
  }
  *out_info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
      .reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC,
      .configuration = configuration,
  };
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
