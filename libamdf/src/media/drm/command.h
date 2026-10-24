// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_SRC_MEDIA_DRM_COMMAND_H_
#define AMDF_SRC_MEDIA_DRM_COMMAND_H_

#include "libamdf/src/media/drm/image.h"
#include "libamdf/src/media/drm/queue.h"
#include "libamdf/src/media/drm/session.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amdf_media_drm_command_t {
  // Exact borrowed session and provider identity.
  amdf_media_command_t base;
  // Native command storage provisioned once, writable only after retirement.
  amdf_media_drm_buffer_t* storage;
  // Caller-owned feedback backing borrowed through command destruction.
  amdf_media_drm_buffer_t* feedback_buffer;
  // Native address of the exact feedback record.
  uint64_t feedback_address;
  // Direct host view of the exact feedback record.
  uint32_t* feedback_pointer;
  // Compressed-output capacity retained for feedback-bound validation.
  uint32_t compressed_byte_length;
  // Complete immutable native IB byte extent after preparation.
  uint32_t byte_length;
  // Provisioned bound for application-authored identity activations.
  uint32_t maximum_activation_count;
  // Parsed slice bound admitted by the provisioned recording.
  uint32_t maximum_slice_count;
  // Reference operand bound provisioned in the trailing residency storage.
  uint32_t maximum_reference_count;
  // Native BO residency list; repeated operands share one resident entry.
  struct drm_amdgpu_bo_list_entry* buffers;
  // Initialized residency prefix consumed by the publication ioctl.
  uint32_t buffer_count;
  // Explicit operation selected during successful preparation.
  amdf_media_command_operation_t operation;
  // Last accepted point borrowing the recording and all its operands.
  uint64_t submission;
} amdf_media_drm_command_t;

amdf_status_t AMDF_CALL amdf_media_drm_command_create(
    amdf_media_session_t* session,
    const amdf_media_command_create_info_t* create_info,
    amdf_media_command_t** out_command);
amdf_status_t AMDF_CALL amdf_media_drm_command_prepare(
    amdf_media_command_t* command,
    const amdf_media_command_prepare_info_t* prepare_info);
amdf_status_t AMDF_CALL amdf_media_drm_command_query_result(
    amdf_media_command_t* command, amdf_media_command_result_info_t* out_info);
amdf_status_t AMDF_CALL
amdf_media_drm_command_destroy(amdf_media_command_t* command);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // AMDF_SRC_MEDIA_DRM_COMMAND_H_
