// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/media/hevc.h"
#include "libamdf/src/endpoint.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/media/d3d12/device.h"
#include "libamdf/src/media/d3d12/object.h"
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
  if ((description.format != AMDF_MEDIA_PIXEL_FORMAT_NV12 &&
       description.format != AMDF_MEDIA_PIXEL_FORMAT_P010) ||
      description.width > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      description.height > D3D12_REQ_TEXTURE2D_U_OR_V_DIMENSION ||
      description.array_layer_count >
          D3D12_REQ_TEXTURE2D_ARRAY_AXIS_DIMENSION ||
      ((description.width | description.height) & 1) != 0 ||
      (description.roles & ~(AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT |
                             AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE)) != 0) {
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

amdf_status_t QueryConfiguration(
    Device* device, const amdf_media_configuration_t& configuration,
    const void* codec_configuration, amdf_media_configuration_info_t* out_info,
    D3D12_VIDEO_DECODE_CONFIGURATION* out_native_configuration) {
  amdf_media_configuration_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = out_info->structure_size,
      .configuration = configuration,
  };
  if (configuration.direction != AMDF_MEDIA_DIRECTION_DECODE) {
    info.reason = AMDF_MEDIA_CONFIGURATION_REASON_CODEC;
    *out_info = info;
    return AMDF_STATUS_OK;
  }
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
  if (!separate_reference) {
    description.roles |= AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE;
  }
  status = ImageRequirements(device, description, &info.output);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (separate_reference) {
    description.roles = AMDF_MEDIA_IMAGE_ROLE_DECODE_REFERENCE;
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
  if (out_native_configuration != nullptr) {
    *out_native_configuration = support.Configuration;
  }
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
