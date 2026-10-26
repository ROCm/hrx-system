// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/drm/command.h"

#include <string.h>

#include "libamdf/src/allocator.h"
#include "libamdf/src/media/av1.h"
#include "libamdf/src/media/drm/jpeg.h"
#include "libamdf/src/media/drm/vcn.h"
#include "libamdf/src/media/drm/vcn_encode.h"
#include "libamdf/src/media/h264.h"
#include "libamdf/src/media/hevc.h"
#include "libamdf/src/media/vp9.h"
#include "libamdf/src/structure.h"

amdf_status_t AMDF_CALL amdf_media_drm_command_create(
    amdf_media_session_t* session_base,
    const amdf_media_command_create_info_t* create_info,
    amdf_media_command_t** out_command) {
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)session_base;
  amdf_kernel_queue_t* queue = session_base->queue;
  amdf_media_drm_device_t* device = (amdf_media_drm_device_t*)queue->device;
  const amdf_media_codec_t codec = session->requirements.configuration.codec;
  const bool encode = session->requirements.configuration.direction ==
                      AMDF_MEDIA_DIRECTION_ENCODE;
  if (create_info->maximum_reference_count > (encode ? 8u : 16u)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const bool jpeg =
      ((const amdf_media_drm_queue_t*)queue)->ip_type == AMDGPU_HW_IP_VCN_JPEG;
  if (jpeg ? codec != AMDF_MEDIA_CODEC_JPEG
           : codec != AMDF_MEDIA_CODEC_H264 && codec != AMDF_MEDIA_CODEC_HEVC &&
                 codec != AMDF_MEDIA_CODEC_VP9 &&
                 codec != AMDF_MEDIA_CODEC_AV1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  amdf_media_drm_buffer_t* feedback_buffer = NULL;
  uint64_t feedback_address = 0;
  if (encode) {
    if (create_info->feedback.byte_length !=
            session->requirements.feedback_byte_length ||
        create_info->feedback.byte_offset %
                session->requirements.feedback_byte_offset_alignment !=
            0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
    }
    amdf_status_t status = amdf_media_drm_memory_range(
        device, create_info->feedback.memory,
        create_info->feedback.access_ordinal, create_info->feedback.byte_offset,
        create_info->feedback.byte_length, &feedback_buffer, &feedback_address);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (feedback_address %
                session->requirements.feedback_byte_offset_alignment !=
            0 ||
        feedback_buffer->host_pointer == NULL) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
  } else if (create_info->feedback.memory != NULL ||
             create_info->feedback.access_ordinal != 0 ||
             create_info->feedback.byte_offset != 0 ||
             create_info->feedback.byte_length != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }

  const uint32_t capacity = encode ? 7
                            : jpeg
                                ? 4
                                : 7 + 2 * create_info->maximum_reference_count;
  amdf_media_drm_command_t* command = NULL;
  amdf_status_t status =
      amdf_calloc(queue->host_allocator,
                  sizeof(*command) + capacity * sizeof(*command->buffers),
                  amdf_alignof(amdf_media_drm_command_t), (void**)&command);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  command->base.session = session_base;
  command->feedback_buffer = feedback_buffer;
  command->feedback_address = feedback_address;
  if (feedback_buffer != NULL) {
    command->feedback_pointer =
        (uint32_t*)((uint8_t*)feedback_buffer->host_pointer + feedback_address -
                    feedback_buffer->address);
  }
  command->maximum_activation_count = create_info->maximum_activation_count;
  command->maximum_slice_count = create_info->maximum_slice_count;
  command->maximum_reference_count = create_info->maximum_reference_count;
  command->buffers = (struct drm_amdgpu_bo_list_entry*)(command + 1);
  const amdf_media_drm_engine_t* engine =
      &device->engines[queue->info.queue_family_ordinal];
  uint64_t storage_length = device->page_size;
  if (!encode && codec == AMDF_MEDIA_CODEC_AV1) {
    storage_length =
        (AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_OFFSET +
         AMDF_MEDIA_DRM_VCN_AV1_AUXILIARY_BYTES + device->page_size - 1) &
        ~(device->page_size - 1);
  } else if (encode &&
             engine->command_protocol == AMDF_MEDIA_DRM_COMMAND_PROTOCOL_VCN5) {
    storage_length = 2 * device->page_size;
  }
  status = amdf_media_drm_buffer_prepare(device, -1, storage_length, true,
                                         &command->storage);
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&queue->children);
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&session->commands);
    if (!amdf_status_is_ok(status)) {
      amdf_child_tracker_unregister(&queue->children);
    }
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
  for (uint32_t i = 0; i < command->buffer_count; ++i) {
    if (command->buffers[i].bo_handle == buffer->handle) {
      return;
    }
  }
  command->buffers[command->buffer_count++].bo_handle = buffer->handle;
}

static bool amdf_media_drm_image_requirements_equal(
    const amdf_media_image_requirements_t* first,
    const amdf_media_image_requirements_t* second) {
  return first->description.format == second->description.format &&
         first->description.width == second->description.width &&
         first->description.height == second->description.height &&
         first->description.array_layer_count ==
             second->description.array_layer_count &&
         first->description.roles == second->description.roles &&
         first->byte_length == second->byte_length &&
         first->byte_offset_alignment == second->byte_offset_alignment;
}

static bool amdf_media_drm_ranges_overlap(uint64_t first_address,
                                          uint64_t first_length,
                                          uint64_t second_address,
                                          uint64_t second_length) {
  return first_address < second_address + second_length &&
         second_address < first_address + first_length;
}

static bool amdf_media_drm_range_overlaps_frame(
    const amdf_media_drm_buffer_t* buffer, uint64_t address, uint64_t length,
    const amdf_media_drm_frame_t* frame) {
  for (uint32_t plane = 0; plane < 2; ++plane) {
    if (frame->buffers[plane] == buffer &&
        amdf_media_drm_ranges_overlap(
            address, length, frame->addresses[plane],
            (uint64_t)frame->row_pitches[plane] * frame->row_counts[plane])) {
      return true;
    }
  }
  return false;
}

static bool amdf_media_drm_bytes_are_zero(const void* data,
                                          uint32_t byte_length) {
  const uint8_t* bytes = data;
  for (uint32_t i = 0; i < byte_length; ++i) {
    if (bytes[i] != 0) {
      return false;
    }
  }
  return true;
}

static bool amdf_media_drm_av1_encode_picture_supported(
    const amdf_media_av1_encode_picture_info_t* picture) {
  const amdf_media_av1_loop_filter_t* filter = &picture->loop_filter;
  const amdf_media_av1_quantization_t* quantization = &picture->quantization;
  const amdf_media_av1_cdef_t* cdef = &picture->cdef;
  const uint32_t transform_mode = picture->frame_type == 0 ? 1 : 2;
  return picture->refresh_frame_flags == 255 &&
         picture->interpolation_filter == 4 &&
         picture->transform_mode == transform_mode && filter->sharpness == 0 &&
         filter->deltas_enabled == 0 && filter->deltas_updated == 0 &&
         filter->delta_present == 0 && filter->delta_resolution == 0 &&
         filter->delta_multi == 0 &&
         amdf_media_drm_bytes_are_zero(filter->levels,
                                       sizeof(filter->levels)) &&
         amdf_media_drm_bytes_are_zero(filter->reference_deltas,
                                       sizeof(filter->reference_deltas)) &&
         amdf_media_drm_bytes_are_zero(filter->mode_deltas,
                                       sizeof(filter->mode_deltas)) &&
         amdf_media_drm_bytes_are_zero(filter->restoration_types,
                                       sizeof(filter->restoration_types)) &&
         amdf_media_drm_bytes_are_zero(
             filter->log2_restoration_unit_sizes,
             sizeof(filter->log2_restoration_unit_sizes)) &&
         quantization->base_index == 20 && quantization->y_dc_delta == 0 &&
         quantization->delta_present == 0 &&
         quantization->delta_resolution == 0 &&
         amdf_media_drm_bytes_are_zero(
             quantization->chroma_dc_deltas,
             sizeof(quantization->chroma_dc_deltas)) &&
         amdf_media_drm_bytes_are_zero(
             quantization->chroma_ac_deltas,
             sizeof(quantization->chroma_ac_deltas)) &&
         quantization->matrix_levels[0] == 255 &&
         quantization->matrix_levels[1] == 255 &&
         quantization->matrix_levels[2] == 255 && cdef->damping_minus3 == 0 &&
         cdef->bits == 0 &&
         amdf_media_drm_bytes_are_zero(cdef->luma_primary,
                                       sizeof(cdef->luma_primary)) &&
         amdf_media_drm_bytes_are_zero(cdef->luma_secondary,
                                       sizeof(cdef->luma_secondary)) &&
         amdf_media_drm_bytes_are_zero(cdef->chroma_primary,
                                       sizeof(cdef->chroma_primary)) &&
         amdf_media_drm_bytes_are_zero(cdef->chroma_secondary,
                                       sizeof(cdef->chroma_secondary));
}

static const amdf_media_reference_t* amdf_media_drm_encode_reference(
    amdf_media_codec_t codec, const amdf_media_h264_encode_picture_info_t* h264,
    const amdf_media_hevc_encode_picture_info_t* hevc,
    const amdf_media_av1_encode_picture_info_t* av1, uint32_t index) {
  return codec == AMDF_MEDIA_CODEC_H264   ? &h264->references[index].binding
         : codec == AMDF_MEDIA_CODEC_HEVC ? &hevc->references[index].binding
                                          : &av1->references[index].binding;
}

static amdf_status_t amdf_media_drm_command_prepare_encode(
    amdf_media_drm_command_t* command, amdf_media_drm_session_t* session,
    amdf_media_drm_device_t* device,
    const amdf_media_command_prepare_info_t* prepare_info) {
  if (prepare_info->decode != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
    if (prepare_info->encode != NULL || prepare_info->activation_count != 0) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    command->submission = 0;
    command->operation = prepare_info->operation;
    command->compressed_byte_length = 0;
    command->buffer_count = 0;
    amdf_media_drm_command_add_buffer(command, command->storage);
    amdf_media_drm_command_add_buffer(command, session->storage);
    command->byte_length = amdf_media_drm_vcn_encode_control(
        command->storage, session, prepare_info->operation);
    return AMDF_STATUS_OK;
  }

  const amdf_media_encode_picture_info_t* encode = prepare_info->encode;
  const amdf_media_h264_encode_picture_info_t* h264 = NULL;
  const amdf_media_hevc_encode_picture_info_t* hevc = NULL;
  const amdf_media_av1_encode_picture_info_t* av1 = NULL;
  uint32_t reference_count = 0;
  const amdf_media_codec_t codec = session->requirements.configuration.codec;
  amdf_status_t status = AMDF_STATUS_OK;
  if (codec == AMDF_MEDIA_CODEC_H264) {
    h264 = encode != NULL ? encode->next : NULL;
    status = amdf_media_h264_validate_encode_picture(
        &session->requirements, &session->encode, encode,
        &session->encode_codec.h264, h264);
    if (amdf_status_is_ok(status)) {
      reference_count = h264->reference_count;
    }
  } else if (codec == AMDF_MEDIA_CODEC_HEVC) {
    hevc = encode != NULL ? encode->next : NULL;
    status = amdf_media_hevc_validate_encode_picture(
        &session->requirements, &session->encode, encode, hevc);
    if (amdf_status_is_ok(status)) {
      reference_count = hevc->reference_count;
    }
  } else {
    av1 = encode != NULL ? encode->next : NULL;
    status = amdf_media_av1_validate_encode_picture(
        &session->requirements, &session->encode, encode,
        &session->encode_codec.av1, av1);
    if (amdf_status_is_ok(status) &&
        !amdf_media_drm_av1_encode_picture_supported(av1)) {
      status = amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    if (amdf_status_is_ok(status)) {
      reference_count = av1->reference_count;
    }
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (reference_count > command->maximum_reference_count ||
      encode->compressed.byte_length > UINT32_MAX) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }

  amdf_media_drm_buffer_t* compressed = NULL;
  uint64_t compressed_address = 0;
  status = amdf_media_drm_memory_range(
      device, encode->compressed.memory, encode->compressed.access_ordinal,
      encode->compressed.byte_offset, encode->compressed.byte_length,
      &compressed, &compressed_address);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (compressed_address %
          session->requirements.compressed_byte_offset_alignment !=
      0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (compressed == command->feedback_buffer &&
      amdf_media_drm_ranges_overlap(
          compressed_address, encode->compressed.byte_length,
          command->feedback_address,
          session->requirements.feedback_byte_length)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  amdf_media_drm_frame_t input = {0};
  status = amdf_media_drm_frame_prepare(
      device, encode->input, &session->requirements.input.description, &input);
  if (amdf_status_is_ok(status)) {
    status = amdf_media_drm_video_frame(&input, NULL, compressed);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  if (encode->reconstruction->storage_kind !=
          AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE ||
      encode->reconstruction->storage.image.array_layer !=
          encode->reconstruction_slot_index) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_drm_frame_t reconstruction = {0};
  status = amdf_media_drm_frame_prepare(
      device, encode->reconstruction,
      &session->requirements.reference.description, &reconstruction);
  if (amdf_status_is_ok(status)) {
    status = amdf_media_drm_video_frame(&reconstruction, &reconstruction,
                                        compressed);
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const amdf_media_drm_image_t* reconstruction_image =
      (const amdf_media_drm_image_t*)
          encode->reconstruction->storage.image.image;
  if (!amdf_media_drm_image_requirements_equal(
          &reconstruction_image->requirements,
          &session->requirements.reference) ||
      amdf_media_drm_frames_overlap(&input, &reconstruction)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  amdf_media_drm_buffer_t* reconstruction_buffer = NULL;
  uint64_t reconstruction_address = 0;
  status = amdf_media_drm_memory_range(
      device, reconstruction_image->memory,
      reconstruction_image->access_ordinal, reconstruction_image->byte_offset,
      reconstruction_image->requirements.byte_length, &reconstruction_buffer,
      &reconstruction_address);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (reconstruction_buffer != reconstruction.buffers[0] ||
      amdf_media_drm_range_overlaps_frame(
          command->feedback_buffer, command->feedback_address,
          session->requirements.feedback_byte_length, &input) ||
      (command->feedback_buffer == reconstruction_buffer &&
       amdf_media_drm_ranges_overlap(
           command->feedback_address,
           session->requirements.feedback_byte_length, reconstruction_address,
           reconstruction_image->requirements.byte_length))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }

  for (uint32_t i = 0; i < reference_count; ++i) {
    const amdf_media_reference_t* binding =
        amdf_media_drm_encode_reference(codec, h264, hevc, av1, i);
    if (binding->frame->storage_kind !=
            AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE ||
        binding->frame->storage.image.image != &reconstruction_image->base ||
        binding->frame->storage.image.array_layer != binding->slot_index) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    amdf_media_drm_frame_t reference = {0};
    status = amdf_media_drm_frame_prepare(
        device, binding->frame, &session->requirements.reference.description,
        &reference);
    if (amdf_status_is_ok(status)) {
      status =
          amdf_media_drm_video_frame(&reference, &reconstruction, compressed);
    }
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  for (uint32_t i = 0; i < prepare_info->activation_count; ++i) {
    const amdf_media_image_activation_t* activation =
        &prepare_info->activations[i];
    if (activation->after == NULL ||
        activation->after->device != &device->base ||
        (activation->before != NULL &&
         activation->before->device != &device->base) ||
        activation->before == activation->after ||
        ((const amdf_media_drm_image_t*)activation->after)->memory ==
            encode->compressed.memory) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }

  command->submission = 0;
  command->operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  command->compressed_byte_length = (uint32_t)encode->compressed.byte_length;
  command->buffer_count = 0;
  memset(command->feedback_pointer, 0,
         session->requirements.feedback_byte_length);
  amdf_media_drm_command_add_buffer(command, command->storage);
  amdf_media_drm_command_add_buffer(command, session->storage);
  amdf_media_drm_command_add_buffer(command, command->feedback_buffer);
  amdf_media_drm_command_add_buffer(command, compressed);
  for (uint32_t plane = 0; plane < 2; ++plane) {
    amdf_media_drm_command_add_buffer(command, input.buffers[plane]);
  }
  amdf_media_drm_command_add_buffer(command, reconstruction_buffer);

  const amdf_media_drm_vcn_encode_resources_t resources = {
      .input = input,
      .reconstruction_address = reconstruction_address,
      .reconstruction_layer_byte_length =
          (uint32_t)reconstruction_image->layer_byte_length,
      .reconstruction_luma_byte_length =
          (uint32_t)(reconstruction_image->row_pitch *
                     reconstruction_image->allocated_height),
      .reconstruction_row_pitch = (uint32_t)reconstruction_image->row_pitch,
      .reconstruction_swizzle_mode = reconstruction_image->swizzle_mode,
      .reconstruction_slot_count =
          session->requirements.configuration.reference_image_count,
      .compressed_address = compressed_address,
      .compressed_byte_length = command->compressed_byte_length,
      .feedback_address = command->feedback_address,
  };
  command->byte_length =
      h264 != NULL ? amdf_media_drm_vcn_encode_h264(
                         command->storage, session, &resources,
                         encode->reconstruction_slot_index,
                         encode->temporal_layer_index, h264)
      : hevc != NULL
          ? amdf_media_drm_vcn_encode_hevc(command->storage, session,
                                           &resources,
                                           encode->reconstruction_slot_index,
                                           encode->temporal_layer_index, hevc)
          : amdf_media_drm_vcn_encode_av1(command->storage, session, &resources,
                                          encode->reconstruction_slot_index,
                                          encode->temporal_layer_index, av1);
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_command_prepare(
    amdf_media_command_t* command_base,
    const amdf_media_command_prepare_info_t* prepare_info) {
  amdf_media_session_t* session_base = command_base->session;
  amdf_media_drm_command_t* command = (amdf_media_drm_command_t*)command_base;
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)session_base;
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)session_base->queue;
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
  const bool encode_session = session->requirements.configuration.direction ==
                              AMDF_MEDIA_DIRECTION_ENCODE;
  const bool picture_operation =
      prepare_info->operation == AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  if (!picture_operation &&
      (jpeg || (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_BEGIN &&
                prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_END))) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (!picture_operation &&
      (prepare_info->decode != NULL || prepare_info->encode != NULL ||
       prepare_info->activation_count != 0)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (prepare_info->activation_count > command->maximum_activation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (encode_session) {
    return amdf_media_drm_command_prepare_encode(command, session, device,
                                                 prepare_info);
  }
  if (picture_operation && prepare_info->encode != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const amdf_media_decode_picture_info_t* decode = prepare_info->decode;
  amdf_media_drm_buffer_t* compressed = NULL;
  uint64_t address = 0;
  amdf_media_drm_frame_t output = {0}, current = {0}, references[16];
  const amdf_media_h264_decode_picture_info_t* h264 = NULL;
  const amdf_media_hevc_decode_picture_info_t* hevc = NULL;
  const amdf_media_vp9_decode_picture_info_t* vp9 = NULL;
  const amdf_media_av1_decode_picture_info_t* av1 = NULL;
  uint32_t reference_count = 0, slice_count = 0;
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
      if (session->requirements.configuration.codec == AMDF_MEDIA_CODEC_H264) {
        h264 = decode->next;
        status = amdf_structure_validate_input(
            h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO,
            sizeof(*h264));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        reference_count = h264->reference_count;
        slice_count = h264->slice_count;
      } else if (session->requirements.configuration.codec ==
                 AMDF_MEDIA_CODEC_AV1) {
        av1 = decode->next;
        status = amdf_structure_validate_input(
            av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_DECODE_PICTURE_INFO,
            sizeof(*av1));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        // Validation owns tile multiplication/range safety before sizing.
        status = amdf_media_av1_validate(&session->requirements.configuration,
                                         av1, decode->compressed.byte_length);
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        reference_count = av1->reference_count;
        slice_count = av1->tile_column_count * av1->tile_row_count;
      } else if (session->requirements.configuration.codec ==
                 AMDF_MEDIA_CODEC_VP9) {
        vp9 = decode->next;
        status = amdf_structure_validate_input(
            vp9, AMDF_STRUCTURE_TYPE_MEDIA_VP9_DECODE_PICTURE_INFO,
            sizeof(*vp9));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        reference_count = vp9->reference_count;
        slice_count = 1;
      } else {
        hevc = decode->next;
        status = amdf_structure_validate_input(
            hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_DECODE_PICTURE_INFO,
            sizeof(*hevc));
        if (!amdf_status_is_ok(status)) {
          return status;
        }
        reference_count = hevc->reference_count;
        slice_count = hevc->slice_count;
      }
      if (slice_count > command->maximum_slice_count ||
          reference_count > command->maximum_reference_count ||
          decode->reference_slot_index >=
              session->requirements.configuration.reference_image_count) {
        return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
      }
      status =
          av1 != NULL ? AMDF_STATUS_OK
          : h264 != NULL
              ? amdf_media_h264_validate(&session->requirements.configuration,
                                         &session->codec.h264, h264,
                                         decode->compressed.byte_length)
          : vp9 != NULL
              ? amdf_media_vp9_validate(&session->requirements.configuration,
                                        vp9, decode->compressed.byte_length)
              : amdf_media_hevc_validate(&session->requirements.configuration,
                                         &session->codec.hevc, hevc,
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
      for (uint32_t i = 0; amdf_status_is_ok(status) && i < reference_count;
           ++i) {
        const amdf_media_reference_t* binding =
            h264 != NULL  ? &h264->references[i].binding
            : av1 != NULL ? &av1->references[i]
            : vp9 != NULL ? &vp9->references[i]
                          : &hevc->references[i].binding;
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
      const amdf_media_drm_engine_t* engine =
          &device->engines[queue->base.info.queue_family_ordinal];
      command->byte_length = amdf_media_drm_jpeg_emit(
          engine->command_protocol, command->storage->host_pointer, address,
          (uint32_t)decode->compressed.byte_length, &output);
    } else {
      for (uint32_t i = 0; i < 2; ++i) {
        amdf_media_drm_command_add_buffer(command, current.buffers[i]);
      }
      for (uint32_t i = 0; i < reference_count; ++i) {
        for (uint32_t plane = 0; plane < 2; ++plane) {
          amdf_media_drm_command_add_buffer(command,
                                            references[i].buffers[plane]);
        }
      }
      command->byte_length =
          av1 != NULL
              ? amdf_media_drm_vcn_av1(command->storage, session, address,
                                       (uint32_t)decode->compressed.byte_length,
                                       &output, &current, references,
                                       decode->reference_slot_index, av1)
          : h264 != NULL
              ? amdf_media_drm_vcn_h264(
                    command->storage, session, address,
                    (uint32_t)decode->compressed.byte_length, &output, &current,
                    references, decode->reference_slot_index, h264)
          : vp9 != NULL
              ? amdf_media_drm_vcn_vp9(command->storage, session, address,
                                       (uint32_t)decode->compressed.byte_length,
                                       &output, &current, references,
                                       decode->reference_slot_index, vp9)
              : amdf_media_drm_vcn_hevc(
                    command->storage, session, address,
                    (uint32_t)decode->compressed.byte_length, &output, &current,
                    references, decode->reference_slot_index, hevc);
    }
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL amdf_media_drm_command_query_result(
    amdf_media_command_t* base, amdf_media_command_result_info_t* out_info) {
  const amdf_media_drm_command_t* command =
      (const amdf_media_drm_command_t*)base;
  const amdf_media_drm_session_t* session =
      (const amdf_media_drm_session_t*)base->session;
  const amdf_media_drm_queue_t* queue =
      (const amdf_media_drm_queue_t*)base->session->queue;
  if (command->submission == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_status_t status = AMDF_STATUS_OK;
  amdf_media_codec_result_t codec_result = AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE;
  if (session->requirements.configuration.direction ==
      AMDF_MEDIA_DIRECTION_ENCODE) {
    amdf_media_encoded_region_t ignored_region;
    amdf_media_encode_result_info_t ignored_result = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
        .structure_size = sizeof(ignored_result),
        .region_capacity = 1,
        .regions = &ignored_region,
    };
    amdf_media_encode_result_info_t* encode =
        out_info->next != NULL ? out_info->next : &ignored_result;
    if (command->operation == AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
      status = amdf_media_drm_vcn_encode_result(
          session, command->feedback_pointer, command->compressed_byte_length,
          encode);
      codec_result = command->feedback_pointer[1] != 0
                         ? AMDF_MEDIA_CODEC_RESULT_SUCCEEDED
                         : AMDF_MEDIA_CODEC_RESULT_FAILED;
    } else if (out_info->next != NULL) {
      status = amdf_media_drm_vcn_encode_result_clear(session, encode);
    }
    if (!amdf_status_is_ok(status) &&
        amdf_status_code(status) != AMDF_STATUS_CODE_BUFFER_TOO_SMALL) {
      return status;
    }
  } else if (out_info->next != NULL) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  out_info->operation = command->operation;
  out_info->submission = command->submission;
  out_info->codec_result = codec_result;
  return status;
}

amdf_status_t AMDF_CALL
amdf_media_drm_command_destroy(amdf_media_command_t* base) {
  amdf_media_drm_command_t* command = (amdf_media_drm_command_t*)base;
  amdf_media_drm_session_t* session = (amdf_media_drm_session_t*)base->session;
  amdf_media_drm_queue_t* queue = (amdf_media_drm_queue_t*)base->session->queue;
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  const amdf_status_t status = amdf_media_drm_buffer_release(&command->storage);
  amdf_media_drm_buffer_abandon(&command->storage);
  amdf_child_tracker_unregister(&session->commands);
  amdf_child_tracker_unregister(&queue->base.children);
  amdf_free(queue->base.host_allocator, command);
  return status;
}
