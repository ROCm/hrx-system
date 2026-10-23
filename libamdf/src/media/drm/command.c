// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/command.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/media/drm/jpeg.h"
#include "libamdf/src/structure.h"

amdf_status_t AMDF_CALL amdf_media_drm_command_create(
    amdf_kernel_queue_t* queue,
    const amdf_media_command_create_info_t* create_info,
    amdf_media_command_t** out_command) {
  if (create_info->feedback.memory != NULL ||
      create_info->feedback.access_ordinal != 0 ||
      create_info->feedback.byte_offset != 0 ||
      create_info->feedback.byte_length != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_command_t* command = NULL;
  amdf_status_t status =
      amdf_calloc(queue->host_allocator, sizeof(*command),
                  amdf_alignof(amdf_media_drm_command_t), (void**)&command);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  command->base.queue = queue;
  command->maximum_activation_count = create_info->maximum_activation_count;
  amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)queue->device;
  status = amdf_media_drm_buffer_prepare(device, -1, device->page_size, true,
                                         &command->storage);
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&queue->children);
  }
  if (!amdf_status_is_ok(status)) {
    const amdf_status_t cleanup =
        amdf_media_drm_buffer_release(&command->storage);
    amdf_media_drm_buffer_abandon(&command->storage);
    amdf_free(queue->host_allocator, command);
    return amdf_status_is_ok(cleanup) ? status : cleanup;
  }
  *out_command = &command->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_command_prepare(
    amdf_media_session_t* session_base, amdf_media_command_t* command_base,
    const amdf_media_command_prepare_info_t* prepare_info) {
  amdf_media_drm_command_t* command = (amdf_media_drm_command_t*)command_base;
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)session_base;
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)command_base->queue;
  amdf_media_drm_device_t* device =
      (amdf_media_drm_device_t*)queue->base.device;
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_status_t status =
      amdf_atomic_uint64_load_acquire(&queue->terminal_status);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (prepare_info->activation_count > command->maximum_activation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_structure_validate_input(
      prepare_info->decode, AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
      sizeof(*prepare_info->decode));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_media_decode_picture_info_t* decode = prepare_info->decode;
  if (decode->reference_output != NULL || decode->reference_slot_index != 0 ||
      decode->compressed.byte_offset % 128 != 0 ||
      decode->compressed.byte_length % 128 != 0 ||
      decode->compressed.byte_length > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_drm_buffer_t* compressed;
  uint64_t address;
  status = amdf_media_drm_memory_range(
      device, decode->compressed.memory, decode->compressed.access_ordinal,
      decode->compressed.byte_offset, decode->compressed.byte_length,
      &compressed, &address);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (address % 128 != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_frame_t output;
  status = amdf_media_drm_frame_prepare(
      device, decode->output, &session->requirements.output.description,
      &output);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (compressed == output.buffers[0] || compressed == output.buffers[1]) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  for (uint32_t i = 0;
       amdf_status_is_ok(status) && i < prepare_info->activation_count; ++i) {
    const amdf_media_image_activation_t* activation =
        &prepare_info->activations[i];
    if (activation->after == NULL ||
        activation->after->device != &device->base ||
        (activation->before != NULL &&
         activation->before->device != &device->base)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  if (amdf_status_is_ok(status) && command->session != session) {
    status = amdf_child_tracker_register(&session->commands);
    if (amdf_status_is_ok(status)) {
      if (command->session != NULL) {
        amdf_child_tracker_unregister(&command->session->commands);
      }
      command->session = session;
    }
  }
  if (amdf_status_is_ok(status)) {
    command->submission = 0;
    command->byte_length = amdf_media_drm_jpeg_emit(
        command->storage->host_pointer, address,
        (uint32_t)decode->compressed.byte_length, &output);
    command->buffers[0].bo_handle = command->storage->handle;
    command->buffers[1].bo_handle = compressed->handle;
    command->buffers[2].bo_handle = output.buffers[0]->handle;
    command->buffers[3].bo_handle = output.buffers[1]->handle;
  }
  return status;
}

amdf_status_t AMDF_CALL amdf_media_drm_command_query_result(
    amdf_media_command_t* base, amdf_media_command_result_info_t* out_info) {
  const amdf_media_drm_command_t* command =
      (const amdf_media_drm_command_t*)base;
  const amdf_media_drm_queue_t* queue =
      (const amdf_media_drm_queue_t*)base->queue;
  if (command->submission == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  *out_info = (amdf_media_command_result_info_t){
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
      .structure_size = out_info->structure_size,
      .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
      .submission = command->submission,
      // PACKETJ has no codec-feedback writeback in this admitted protocol.
      .codec_result = AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE,
  };
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
amdf_media_drm_command_destroy(amdf_media_command_t* base) {
  amdf_media_drm_command_t* command = (amdf_media_drm_command_t*)base;
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base->queue;
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_media_drm_buffer_release(&command->storage);
  amdf_media_drm_buffer_abandon(&command->storage);
  if (command->session != NULL) {
    amdf_child_tracker_unregister(&command->session->commands);
  }
  amdf_child_tracker_unregister(&queue->base.children);
  amdf_free(queue->base.host_allocator, command);
  return status;
}
