// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/session.h"

#include <cstring>

#include "libamdf/src/media/d3d12/configuration.h"
#include "libamdf/src/media/d3d12/object.h"

namespace amdf::media::d3d12 {

amdf_status_t AMDF_CALL
SessionCreate(amdf_kernel_queue_t* queue,
              const amdf_media_session_create_info_t* create_info,
              amdf_media_session_t** out_session) {
  const auto direction = create_info->configuration.direction;
  if (direction != AMDF_MEDIA_DIRECTION_DECODE &&
      direction != AMDF_MEDIA_DIRECTION_ENCODE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  auto* device = reinterpret_cast<Device*>(queue->device);
  Session* session = nullptr;
  amdf_status_t status = Allocate(queue->host_allocator, &session);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  session->base.queue = queue;
  session->requirements.structure_size = sizeof(session->requirements);
  session->info.type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO;
  session->info.structure_size = sizeof(session->info);
  session->info.queue = queue;
  session->info.configuration = create_info->configuration;
  amdf_child_tracker_initialize(&session->commands);

  D3D12_VIDEO_DECODER_DESC decoder = {};
  status = QueryConfiguration(
      device, queue->info.queue_family_ordinal, create_info->configuration,
      create_info->next, &session->requirements,
      direction == AMDF_MEDIA_DIRECTION_DECODE ? &decoder.Configuration
                                               : nullptr,
      direction == AMDF_MEDIA_DIRECTION_ENCODE ? &session->native_encode
                                               : nullptr);
  if (!amdf_status_is_ok(status) ||
      session->requirements.support !=
          AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
    Delete(queue->host_allocator, session);
    return amdf_status_is_ok(status)
               ? amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)
               : status;
  }

  if (create_info->configuration.codec == AMDF_MEDIA_CODEC_H264) {
    session->h264 =
        *static_cast<const amdf_media_h264_configuration_t*>(create_info->next);
  } else if (create_info->configuration.codec == AMDF_MEDIA_CODEC_HEVC) {
    session->hevc =
        *static_cast<const amdf_media_hevc_configuration_t*>(create_info->next);
  } else if (create_info->configuration.codec == AMDF_MEDIA_CODEC_AV1) {
    session->av1 =
        *static_cast<const amdf_media_av1_configuration_t*>(create_info->next);
  } else {
    session->vp9 =
        *static_cast<const amdf_media_vp9_configuration_t*>(create_info->next);
  }

  if (direction == AMDF_MEDIA_DIRECTION_DECODE) {
    if (create_info->configuration.codec == AMDF_MEDIA_CODEC_H264) {
      session->h264.next = nullptr;
    } else if (create_info->configuration.codec == AMDF_MEDIA_CODEC_HEVC) {
      session->hevc.next = nullptr;
    } else if (create_info->configuration.codec == AMDF_MEDIA_CODEC_AV1) {
      session->av1.next = nullptr;
    } else {
      session->vp9.next = nullptr;
    }
    D3D12_FEATURE_DATA_VIDEO_DECODER_HEAP_SIZE heap = {};
    heap.VideoDecoderHeapDesc.Configuration = decoder.Configuration;
    heap.VideoDecoderHeapDesc.DecodeWidth =
        create_info->configuration.coded_width;
    heap.VideoDecoderHeapDesc.DecodeHeight =
        create_info->configuration.coded_height;
    heap.VideoDecoderHeapDesc.Format =
        create_info->configuration.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
            ? DXGI_FORMAT_NV12
            : DXGI_FORMAT_P010;
    heap.VideoDecoderHeapDesc.MaxDecodePictureBufferCount =
        create_info->configuration.reference_image_count;
    status = Hresult(device->video->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_DECODER_HEAP_SIZE, &heap, sizeof(heap)));
    if (amdf_status_is_ok(status)) {
      session->info.system_state_byte_length = heap.MemoryPoolL0Size;
      session->info.local_state_byte_length = heap.MemoryPoolL1Size;
      status = Hresult(device->video->CreateVideoDecoder(
          &decoder, IID_PPV_ARGS(&session->decoder)));
    }
    if (amdf_status_is_ok(status)) {
      status = Hresult(device->video->CreateVideoDecoderHeap(
          &heap.VideoDecoderHeapDesc, IID_PPV_ARGS(&session->decoder_heap)));
    }
  } else {
    const auto* encode = static_cast<const amdf_media_encode_configuration_t*>(
        static_cast<const amdf_input_structure_t*>(create_info->next)->next);
    session->encode = *encode;
    std::memcpy(session->rate_control_layers,
                encode->rate_control.temporal_layers,
                encode->rate_control.temporal_layer_count *
                    sizeof(session->rate_control_layers[0]));
    session->encode.rate_control.temporal_layers = session->rate_control_layers;
    if (create_info->configuration.codec == AMDF_MEDIA_CODEC_H264) {
      session->h264_encode =
          *static_cast<const amdf_media_h264_encode_configuration_t*>(
              encode->next);
      session->h264.next = &session->encode;
      session->encode.next = &session->h264_encode;
      session->h264_encode.next = nullptr;
    } else if (create_info->configuration.codec == AMDF_MEDIA_CODEC_HEVC) {
      session->hevc_encode =
          *static_cast<const amdf_media_hevc_encode_configuration_t*>(
              encode->next);
      session->hevc.next = &session->encode;
      session->encode.next = &session->hevc_encode;
      session->hevc_encode.next = nullptr;
    } else {
      session->av1_encode =
          *static_cast<const amdf_media_av1_encode_configuration_t*>(
              encode->next);
      session->av1.next = &session->encode;
      session->encode.next = &session->av1_encode;
      session->av1_encode.next = nullptr;
    }
    const auto& native = session->native_encode;
    D3D12_VIDEO_ENCODER_DESC encoder = {
        .EncodeCodec = native.codec,
        .EncodeProfile = native.profile,
        .InputFormat =
            create_info->configuration.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
                ? DXGI_FORMAT_NV12
                : DXGI_FORMAT_P010,
        .CodecConfiguration = native.codec_configuration,
    };
    D3D12_FEATURE_DATA_VIDEO_ENCODER_HEAP_SIZE heap = {};
    heap.HeapDesc.EncodeCodec = native.codec;
    heap.HeapDesc.EncodeProfile = native.profile;
    heap.HeapDesc.EncodeLevel = native.level;
    heap.HeapDesc.ResolutionsListCount = 1;
    heap.HeapDesc.pResolutionList = &native.resolution;
    status = Hresult(device->encoder->CheckFeatureSupport(
        D3D12_FEATURE_VIDEO_ENCODER_HEAP_SIZE, &heap, sizeof(heap)));
    if (amdf_status_is_ok(status) && !heap.IsSupported) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (amdf_status_is_ok(status)) {
      session->info.system_state_byte_length = heap.MemoryPoolL0Size;
      session->info.local_state_byte_length = heap.MemoryPoolL1Size;
      status = Hresult(device->encoder->CreateVideoEncoder(
          &encoder, IID_PPV_ARGS(&session->encoder)));
    }
    if (amdf_status_is_ok(status)) {
      status = Hresult(device->encoder->CreateVideoEncoderHeap(
          &heap.HeapDesc, IID_PPV_ARGS(&session->encoder_heap)));
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&queue->children);
  }
  if (!amdf_status_is_ok(status)) {
    Delete(queue->host_allocator, session);
    return status;
  }
  *out_session = &session->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL SessionQueryInfo(amdf_media_session_t* base,
                                         amdf_media_session_info_t* out_info) {
  const auto* session = reinterpret_cast<const Session*>(base);
  const uint32_t structure_size = out_info->structure_size;
  *out_info = session->info;
  out_info->structure_size = structure_size;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL SessionDestroy(amdf_media_session_t* base) {
  auto* session = reinterpret_cast<Session*>(base);
  if (amdf_child_tracker_count(&session->commands) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_kernel_queue_t* queue = base->queue;
  Delete(queue->host_allocator, session);
  amdf_child_tracker_unregister(&queue->children);
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
