// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/session.h"

#include "libamdf/src/media/d3d12/object.h"

namespace amdf::media::d3d12 {

amdf_status_t AMDF_CALL
SessionCreate(amdf_kernel_queue_t* queue,
              const amdf_media_session_create_info_t* create_info,
              amdf_media_session_t** out_session) {
  auto* device = reinterpret_cast<Device*>(queue->device);
  amdf_media_configuration_info_t requirements = {
      .structure_size = sizeof(requirements),
  };
  D3D12_VIDEO_DECODER_DESC decoder = {};
  amdf_status_t status =
      QueryConfiguration(device, create_info->configuration, create_info->next,
                         &requirements, &decoder.Configuration);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (requirements.support != AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Session* session = nullptr;
  status = Allocate(queue->host_allocator, &session);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  session->base.queue = queue;
  session->requirements = requirements;
  if (create_info->configuration.codec == AMDF_MEDIA_CODEC_H264) {
    session->h264 =
        *static_cast<const amdf_media_h264_configuration_t*>(create_info->next);
  } else {
    session->hevc =
        *static_cast<const amdf_media_hevc_configuration_t*>(create_info->next);
  }
  session->info.type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO;
  session->info.structure_size = sizeof(session->info);
  session->info.queue = queue;
  session->info.configuration = create_info->configuration;
  session->info.system_state_byte_length = heap.MemoryPoolL0Size;
  session->info.local_state_byte_length = heap.MemoryPoolL1Size;
  amdf_child_tracker_initialize(&session->commands);
  status = Hresult(device->video->CreateVideoDecoder(
      &decoder, IID_PPV_ARGS(&session->decoder)));
  if (amdf_status_is_ok(status)) {
    status = Hresult(device->video->CreateVideoDecoderHeap(
        &heap.VideoDecoderHeapDesc, IID_PPV_ARGS(&session->heap)));
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
