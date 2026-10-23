// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/session.h"

#include "libamdf/src/allocator.h"

amdf_status_t AMDF_CALL amdf_media_drm_session_create(
    amdf_kernel_queue_t* queue,
    const amdf_media_session_create_info_t* create_info,
    amdf_media_session_t** out_session) {
  amdf_media_configuration_info_t requirements = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(requirements),
  };
  amdf_status_t status = amdf_media_drm_configuration(
      (amdf_media_drm_device_t*)queue->device, &create_info->configuration,
      create_info->next, &requirements);
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
  status = amdf_child_tracker_register(&queue->children);
  if (!amdf_status_is_ok(status)) {
    amdf_free(queue->host_allocator, session);
    return status;
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
  };
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_media_drm_session_destroy(amdf_media_session_t* base) {
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)base;
  if (amdf_child_tracker_count(&session->commands) != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_kernel_queue_t* queue = base->queue;
  amdf_free(queue->host_allocator, session);
  amdf_child_tracker_unregister(&queue->children);
  return AMDF_STATUS_OK;
}
