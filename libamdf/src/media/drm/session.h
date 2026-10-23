// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_SESSION_H_
#define AMDF_SRC_MEDIA_DRM_SESSION_H_

#include "libamdf/src/media/drm/device.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_session_t {
  // Borrowed queue and provider identity.
  amdf_media_session_t base;
  // Exact admitted configuration and resource geometry.
  amdf_media_configuration_info_t requirements;
  // Recordings borrowing this codec state, independent of submission status.
  amdf_child_tracker_t commands;
} amdf_media_drm_session_t;

amdf_status_t AMDF_CALL amdf_media_drm_session_create(
    amdf_kernel_queue_t* queue,
    const amdf_media_session_create_info_t* create_info,
    amdf_media_session_t** out_session);
amdf_status_t AMDF_CALL amdf_media_drm_session_query_info(
    amdf_media_session_t* session, amdf_media_session_info_t* out_info);
amdf_status_t AMDF_CALL
amdf_media_drm_session_destroy(amdf_media_session_t* session);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_SESSION_H_
