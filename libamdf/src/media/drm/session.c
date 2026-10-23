// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/session.h"

#include <string.h>
#include <sys/random.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/linux/file.h"

amdf_status_t AMDF_CALL amdf_media_drm_session_create(
    amdf_kernel_queue_t* queue,
    const amdf_media_session_create_info_t* create_info,
    amdf_media_session_t** out_session) {
  amdf_media_configuration_info_t requirements = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(requirements),
  };
  amdf_status_t status = amdf_media_drm_configuration(
      (amdf_media_drm_device_t*)queue->device, queue->info.queue_family_ordinal,
      &create_info->configuration, create_info->next, &requirements);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (requirements.support != AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_session_t* session = NULL;
  status =
      amdf_calloc(queue->host_allocator, sizeof(*session),
                  amdf_alignof(amdf_media_drm_session_t), (void**)&session);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  session->base.queue = queue;
  session->requirements = requirements;
  amdf_child_tracker_initialize(&session->commands);
  if (requirements.requires_begin_end) {
    session->h264 = *(const amdf_media_h264_configuration_t*)create_info->next;
    session->h264.next = NULL;
    const uint32_t macroblock_width =
        create_info->configuration.coded_width / 16;
    const uint32_t macroblock_height =
        ((create_info->configuration.coded_height / 16) + 1) & ~1u;
    session->context_byte_length =
        create_info->configuration.reference_image_count *
        ((macroblock_width * macroblock_height * 192 + 255) & ~255u);
    amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)queue->device;
    const uint64_t length =
        (131072 + session->context_byte_length + device->page_size - 1) &
        ~(device->page_size - 1);
    status = amdf_media_drm_buffer_prepare(device, -1, length, true,
                                           &session->storage);
    if (amdf_status_is_ok(status)) {
      memset(session->storage->host_pointer, 0, length);
      // Firmware uses an opaque 32-bit stream cookie alongside explicit context
      // storage. A cold entropy request avoids process-global identity state.
      const ssize_t received =
          getrandom(&session->stream_handle, sizeof(session->stream_handle), 0);
      if (received < 0) {
        status = amdf_linux_error(errno);
      } else if (received != sizeof(session->stream_handle)) {
        status = amdf_linux_error(EIO);
      }
    }
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&queue->children);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup =
        amdf_media_drm_buffer_release(&session->storage);
    amdf_media_drm_buffer_abandon(&session->storage);
    amdf_free(queue->host_allocator, session);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_session = &session->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_session_query_info(
    amdf_media_session_t* base, amdf_media_session_info_t* out_info) {
  const amdf_media_drm_session_t* session =
      (const amdf_media_drm_session_t*)base;
  *out_info = (amdf_media_session_info_t){
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_INFO,
      .structure_size = out_info->structure_size,
      .queue = base->queue,
      .configuration = session->requirements.configuration,
      .system_state_byte_length =
          session->storage != NULL ? session->storage->byte_length : 0,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_media_drm_session_destroy(amdf_media_session_t* base) {
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)base;
  if (amdf_child_tracker_count(&session->commands) != 0 ||
      session->operation == AMDF_MEDIA_COMMAND_OPERATION_BEGIN) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_kernel_queue_t* queue = base->queue;
  const amdf_status_t status = amdf_media_drm_buffer_release(&session->storage);
  amdf_media_drm_buffer_abandon(&session->storage);
  amdf_free(queue->host_allocator, session);
  amdf_child_tracker_unregister(&queue->children);
  return status;
}
