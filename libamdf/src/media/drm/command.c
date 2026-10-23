// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/command.h"

#include "libamdf/src/allocator.h"
#include "libamdf/src/media/drm/jpeg.h"
#include "libamdf/src/media/drm/vcn.h"
#include "libamdf/src/media/h264.h"
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
  if (create_info->maximum_reference_count > 16) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const bool jpeg =
      ((const amdf_media_drm_queue_t*)queue)->ip_type == AMDGPU_HW_IP_VCN_JPEG;
  const uint32_t capacity =
      jpeg ? 4 : 7 + 2 * create_info->maximum_reference_count;
  amdf_status_t status =
      amdf_calloc(queue->host_allocator,
                  sizeof(*command) + capacity * sizeof(*command->buffers),
                  amdf_alignof(amdf_media_drm_command_t), (void**)&command);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  command->base.queue = queue;
  command->maximum_activation_count = create_info->maximum_activation_count;
  command->maximum_slice_count = create_info->maximum_slice_count;
  command->maximum_reference_count = create_info->maximum_reference_count;
  command->buffers = (struct drm_amdgpu_bo_list_entry*)(command + 1);
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

static bool amdf_media_drm_frames_overlap(
    const amdf_media_drm_frame_t* first, const amdf_media_drm_frame_t* second) {
  for (uint32_t i = 0; i < 2; ++i) {
    for (uint32_t j = 0; j < 2; ++j) {
      if (first->addresses[i] <
              second->addresses[j] +
                  (uint64_t)second->row_pitches[j] * second->row_counts[j] &&
          second->addresses[j] <
              first->addresses[i] +
                  (uint64_t)first->row_pitches[i] * first->row_counts[i]) {
        return true;
      }
    }
  }
  return false;
}

static amdf_status_t amdf_media_drm_video_frame(
    const amdf_media_drm_frame_t* frame, const amdf_media_drm_frame_t* layout,
    const amdf_media_drm_buffer_t* compressed) {
  if (frame->buffers[0] == compressed || frame->buffers[1] == compressed ||
      frame->addresses[1] - frame->addresses[0] +
              (uint64_t)frame->row_pitches[1] * frame->row_counts[1] >
          UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (layout != NULL &&
      (frame->buffers[0] != frame->buffers[1] ||
       frame->addresses[1] !=
           frame->addresses[0] +
               (uint64_t)frame->row_pitches[0] * frame->row_counts[0] ||
       frame->row_pitches[0] != layout->row_pitches[0] ||
       frame->row_pitches[1] != layout->row_pitches[1])) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  return AMDF_STATUS_OK;
}

static void amdf_media_drm_command_add_buffer(
    amdf_media_drm_command_t* command, const amdf_media_drm_buffer_t* buffer) {
  command->buffers[command->buffer_count++].bo_handle = buffer->handle;
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
  const bool jpeg =
      session->requirements.configuration.codec == AMDF_MEDIA_CODEC_JPEG;
  const bool picture_operation =
      prepare_info->operation == AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  if (!picture_operation &&
      (jpeg || (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_BEGIN &&
                prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_END))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (!picture_operation &&
      (prepare_info->decode != NULL || prepare_info->activation_count != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (prepare_info->activation_count > command->maximum_activation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const amdf_media_decode_picture_info_t* decode = prepare_info->decode;
  amdf_media_drm_buffer_t* compressed = NULL;
  uint64_t address = 0;
  amdf_media_drm_frame_t output = {0}, current = {0}, references[16];
  const amdf_media_h264_decode_picture_info_t* h264 = NULL;
  if (picture_operation) {
    status = amdf_structure_validate_header(
        decode, AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO, sizeof(*decode));
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (decode->compressed.byte_offset %
                session->requirements.compressed_byte_offset_alignment !=
            0 ||
        decode->compressed.byte_length % 128 != 0 ||
        decode->compressed.byte_length > UINT32_MAX) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    status = amdf_media_drm_memory_range(
        device, decode->compressed.memory, decode->compressed.access_ordinal,
        decode->compressed.byte_offset, decode->compressed.byte_length,
        &compressed, &address);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (address % session->requirements.compressed_byte_offset_alignment != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    status = amdf_media_drm_frame_prepare(
        device, decode->output, &session->requirements.output.description,
        &output);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (jpeg) {
      if (decode->next != NULL || decode->reference_output != NULL ||
          decode->reference_slot_index != 0 ||
          output.buffers[0] == compressed || output.buffers[1] == compressed) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    } else {
      status = amdf_media_drm_video_frame(&output, NULL, compressed);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      h264 = decode->next;
      status = amdf_structure_validate_input(
          h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO,
          sizeof(*h264));
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      if (h264->slice_count > command->maximum_slice_count ||
          h264->reference_count > command->maximum_reference_count ||
          decode->reference_slot_index >=
              session->requirements.configuration.reference_image_count) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      status = amdf_media_h264_validate(&session->requirements.configuration,
                                        &session->h264, h264,
                                        decode->compressed.byte_length);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      status = amdf_media_drm_frame_prepare(
          device, decode->reference_output,
          &session->requirements.reference.description, &current);
      if (amdf_status_is_ok(status)) {
        status = amdf_media_drm_video_frame(&current, &current, compressed);
      }
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      if (amdf_media_drm_frames_overlap(&current, &output)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
      uint32_t used_slots = 1u << decode->reference_slot_index;
      for (uint32_t i = 0;
           amdf_status_is_ok(status) && i < h264->reference_count; ++i) {
        const amdf_media_reference_t* binding = &h264->references[i].binding;
        if (binding->slot_index >=
                session->requirements.configuration.reference_image_count ||
            (used_slots & (1u << binding->slot_index)) != 0) {
          status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
          continue;
        }
        used_slots |= 1u << binding->slot_index;
        status = amdf_media_drm_frame_prepare(
            device, binding->frame,
            &session->requirements.reference.description, &references[i]);
        if (amdf_status_is_ok(status)) {
          status =
              amdf_media_drm_video_frame(&references[i], &current, compressed);
        }
        if (amdf_status_is_ok(status) &&
            (amdf_media_drm_frames_overlap(&references[i], &current) ||
             amdf_media_drm_frames_overlap(&references[i], &output))) {
          status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
        }
        for (uint32_t prior = 0; amdf_status_is_ok(status) && prior < i;
             ++prior) {
          if (amdf_media_drm_frames_overlap(&references[i],
                                            &references[prior])) {
            status = amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
          }
        }
      }
    }
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
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  command->submission = 0;
  command->operation = prepare_info->operation;
  command->buffer_count = 0;
  amdf_media_drm_command_add_buffer(command, command->storage);
  if (!jpeg) {
    amdf_media_drm_command_add_buffer(command, session->storage);
  }
  if (!picture_operation) {
    command->byte_length = amdf_media_drm_vcn_control(command->storage, session,
                                                      prepare_info->operation);
  } else {
    amdf_media_drm_command_add_buffer(command, compressed);
    for (uint32_t i = 0; i < 2; ++i) {
      amdf_media_drm_command_add_buffer(command, output.buffers[i]);
    }
    if (jpeg) {
      command->byte_length = amdf_media_drm_jpeg_emit(
          command->storage->host_pointer, address,
          (uint32_t)decode->compressed.byte_length, &output);
    } else {
      for (uint32_t i = 0; i < 2; ++i) {
        amdf_media_drm_command_add_buffer(command, current.buffers[i]);
      }
      for (uint32_t i = 0; i < h264->reference_count; ++i) {
        for (uint32_t plane = 0; plane < 2; ++plane) {
          amdf_media_drm_command_add_buffer(command,
                                            references[i].buffers[plane]);
        }
      }
      command->byte_length = amdf_media_drm_vcn_h264(
          command->storage, session, address,
          (uint32_t)decode->compressed.byte_length, &output, &current,
          references, decode->reference_slot_index, h264);
    }
  }
  return AMDF_STATUS_OK;
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
      .operation = command->operation,
      .submission = command->submission,
      // Queue retirement does not certify codec success. This provider exposes
      // no interpreted codec-feedback protocol for these native operations.
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
