// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/command.h"

#include "libamdf/src/media/d3d12/decode.h"
#include "libamdf/src/media/d3d12/image.h"
#include "libamdf/src/media/d3d12/memory.h"
#include "libamdf/src/media/d3d12/object.h"
#include "libamdf/src/media/d3d12/queue.h"
#include "libamdf/src/structure.h"

namespace amdf::media::d3d12 {
namespace {

amdf_status_t ValidateBuffer(Device* device,
                             const amdf_media_buffer_range_t& range,
                             MemoryProfile profile, uint64_t offset_alignment,
                             uint64_t length_alignment, uint64_t minimum_length,
                             Memory** out_memory) {
  if (range.memory == nullptr ||
      range.access_ordinal >= range.memory->info.access_count ||
      range.memory->accesses[range.access_ordinal].device !=
          &device->base.base) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (range.byte_offset > range.memory->info.byte_length ||
      range.byte_length > range.memory->info.byte_length - range.byte_offset ||
      range.byte_length < minimum_length ||
      range.byte_offset % offset_alignment != 0 ||
      range.byte_length % length_alignment != 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  Memory* memory = NativeMemory(range.memory, range.access_ordinal);
  if (memory->profile != profile) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  *out_memory = memory;
  return AMDF_STATUS_OK;
}

amdf_status_t ValidateImage(Device* device, const amdf_media_frame_t* frame,
                            const amdf_media_image_requirements_t& requirements,
                            amdf_memory_t* compressed_memory,
                            Image** out_image) {
  amdf_status_t status =
      ValidateFrame(device, frame, requirements.description.roles, out_image);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (frame->format != requirements.description.format ||
      frame->width != requirements.description.width ||
      frame->height != requirements.description.height ||
      (*out_image)->memory == compressed_memory) {
    // The compressed buffer is a full-heap view. Its activation cannot coexist
    // with an image in that same heap, even when the logical payloads differ.
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  return AMDF_STATUS_OK;
}

bool SameSubresource(const amdf_media_frame_t* lhs,
                     const amdf_media_frame_t* rhs) {
  return lhs->storage.image.image == rhs->storage.image.image &&
         lhs->storage.image.array_layer == rhs->storage.image.array_layer;
}

D3D12_RESOURCE_BARRIER Transition(ID3D12Resource* resource, UINT subresource,
                                  D3D12_RESOURCE_STATES state) {
  D3D12_RESOURCE_BARRIER barrier = {
      .Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
  };
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = subresource;
  barrier.Transition.StateBefore = D3D12_RESOURCE_STATE_COMMON;
  barrier.Transition.StateAfter = state;
  return barrier;
}

void AppendImageTransitions(Image* image, uint32_t layer,
                            D3D12_RESOURCE_STATES state,
                            D3D12_RESOURCE_BARRIER* barriers, uint32_t* count) {
  // One mip, two planes. Other layers may be live references in this recording.
  const uint32_t layers = image->requirements.description.array_layer_count;
  barriers[(*count)++] = Transition(image->resource.Get(), layer, state);
  barriers[(*count)++] =
      Transition(image->resource.Get(), layer + layers, state);
}

}  // namespace

amdf_status_t AMDF_CALL
CommandCreate(amdf_kernel_queue_t* queue,
              const amdf_media_command_create_info_t* create_info,
              amdf_media_command_t** out_command) {
  if (create_info->codec != AMDF_MEDIA_CODEC_H264 &&
      create_info->codec != AMDF_MEDIA_CODEC_HEVC &&
      create_info->codec != AMDF_MEDIA_CODEC_VP9 &&
      create_info->codec != AMDF_MEDIA_CODEC_AV1) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const size_t slice_byte_length = DecodeSliceByteLength(create_info->codec);
  auto* device = reinterpret_cast<Device*>(queue->device);
  Memory* feedback = nullptr;
  amdf_status_t status = ValidateBuffer(
      device, create_info->feedback, kFeedbackBuffer, 8, 1,
      sizeof(D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS), &feedback);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->maximum_reference_count > 16 ||
      create_info->maximum_slice_count > UINT32_MAX / slice_byte_length ||
      create_info->maximum_activation_count > UINT32_MAX - 38) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t barrier_count =
      create_info->maximum_activation_count + 2 +
      2 * (uint64_t(create_info->maximum_reference_count) + 2);
  const uint64_t trailing_length =
      barrier_count * sizeof(D3D12_RESOURCE_BARRIER) +
      uint64_t(create_info->maximum_slice_count) * slice_byte_length;
  if (trailing_length > SIZE_MAX - sizeof(Command)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  void* storage = nullptr;
  status = amdf_calloc_with_trailing(queue->host_allocator, sizeof(Command),
                                     static_cast<size_t>(trailing_length),
                                     alignof(Command), &storage);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  auto* command = new (storage) Command{};
  command->base.queue = queue;
  command->base.codec = create_info->codec;
  command->feedback = create_info->feedback;
  command->feedback_resource = feedback->resource.Get();
  command->feedback_pointer =
      reinterpret_cast<const D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS*>(
          static_cast<const uint8_t*>(feedback->host_pointer) +
          create_info->feedback.byte_offset);
  command->maximum_slice_count = create_info->maximum_slice_count;
  command->maximum_reference_count = create_info->maximum_reference_count;
  command->maximum_activation_count = create_info->maximum_activation_count;
  command->barriers = reinterpret_cast<D3D12_RESOURCE_BARRIER*>(command + 1);
  command->slices = command->barriers + barrier_count;
  status = Hresult(device->native->CreateCommandAllocator(
      D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE, IID_PPV_ARGS(&command->allocator)));
  if (amdf_status_is_ok(status)) {
    status = Hresult(device->native->CreateCommandList(
        0, D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE, command->allocator.Get(),
        nullptr, IID_PPV_ARGS(&command->native)));
  }
  if (amdf_status_is_ok(status)) {
    status = Hresult(command->native->Close());
  }
  if (amdf_status_is_ok(status)) {
    const D3D12_QUERY_HEAP_DESC description = {
        D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS, 1, 0};
    status = Hresult(device->native->CreateQueryHeap(
        &description, IID_PPV_ARGS(&command->query)));
  }
  if (amdf_status_is_ok(status)) {
    status = amdf_child_tracker_register(&queue->children);
  }
  if (!amdf_status_is_ok(status)) {
    Delete(queue->host_allocator, command);
    return status;
  }
  *out_command = &command->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CommandPrepare(
    amdf_media_session_t* session_base, amdf_media_command_t* command_base,
    const amdf_media_command_prepare_info_t* prepare_info) {
  auto* session = reinterpret_cast<Session*>(session_base);
  auto* command = reinterpret_cast<Command*>(command_base);
  auto* queue = reinterpret_cast<Queue*>(command_base->queue);
  auto* device = reinterpret_cast<Device*>(queue->base.device);
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_status_t status = QueueTerminalStatus(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (command_base->codec != session->info.configuration.codec) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  if (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (prepare_info->activation_count > command->maximum_activation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  status = amdf_structure_validate_header(
      prepare_info->decode, AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
      sizeof(amdf_media_decode_picture_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const auto& decode = *prepare_info->decode;
  DecodePicture picture = {};
  status = PrepareDecode(
      *session, decode.next, decode.compressed.byte_length,
      decode.reference_slot_index, command->maximum_reference_count,
      command->maximum_slice_count, command->slices, &picture);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Memory* compressed = nullptr;
  status = ValidateBuffer(device, decode.compressed, kSharedHeap, 128, 128, 128,
                          &compressed);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Image* output_image = nullptr;
  status = ValidateImage(device, decode.output, session->requirements.output,
                         decode.compressed.memory, &output_image);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const bool separate_reference =
      session->requirements.reference.byte_length != 0;
  Image* current_image = output_image;
  const amdf_media_frame_t* current = decode.output;
  if (separate_reference) {
    status = ValidateImage(device, decode.reference_output,
                           session->requirements.reference,
                           decode.compressed.memory, &current_image);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    current = decode.reference_output;
    if (SameSubresource(current, decode.output)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  } else if (decode.reference_output != nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  Image* reference_images[16];
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto* reference = picture.references[index].frame;
    status = ValidateImage(device, reference,
                           separate_reference ? session->requirements.reference
                                              : session->requirements.output,
                           decode.compressed.memory, &reference_images[index]);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (SameSubresource(reference, current) ||
        SameSubresource(reference, decode.output)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (SameSubresource(reference, picture.references[prior].frame)) {
        return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
      }
    }
  }
  for (uint32_t index = 0; index < prepare_info->activation_count; ++index) {
    const auto& activation = prepare_info->activations[index];
    if (activation.after == nullptr ||
        activation.after->device != &device->base ||
        (activation.before != nullptr &&
         activation.before->device != &device->base) ||
        activation.before == activation.after ||
        reinterpret_cast<Image*>(activation.after)->memory ==
            decode.compressed.memory) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }
  // Preparation is a transition of this live owner. Retain the old native
  // session until reset has stopped borrowing it; a failed reset remains safe
  // to destroy and cannot publish either the old or an incomplete recording.
  status = amdf_child_tracker_register(&session->commands);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  command->prepared = false;
  command->executed = false;
  command->submission = 0;
  status = Hresult(command->allocator->Reset());
  if (amdf_status_is_ok(status)) {
    status = Hresult(command->native->Reset(command->allocator.Get()));
  }
  if (!amdf_status_is_ok(status)) {
    amdf_child_tracker_unregister(&session->commands);
    return status;
  }
  if (command->session != nullptr) {
    amdf_child_tracker_unregister(&command->session->commands);
  }
  command->session = session;

  ID3D12Resource* references[17] = {};
  UINT subresources[17] = {};
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& binding = picture.references[index];
    references[binding.slot_index] = reference_images[index]->resource.Get();
    subresources[binding.slot_index] = binding.frame->storage.image.array_layer;
  }
  references[decode.reference_slot_index] = current_image->resource.Get();
  subresources[decode.reference_slot_index] =
      current->storage.image.array_layer;
  auto& arguments = picture.arguments;
  arguments.ReferenceFrames = {
      session->info.configuration.reference_image_count, references,
      subresources, nullptr};
  arguments.CompressedBitstream = {compressed->resource.Get(),
                                   decode.compressed.byte_offset,
                                   decode.compressed.byte_length};
  arguments.pHeap = session->heap.Get();
  D3D12_VIDEO_DECODE_OUTPUT_STREAM_ARGUMENTS output = {
      .pOutputTexture2D = output_image->resource.Get(),
      .OutputSubresource = decode.output->storage.image.array_layer,
  };
  if (separate_reference) {
    output.ConversionArguments.Enable = TRUE;
    output.ConversionArguments.pReferenceTexture2D =
        current_image->resource.Get();
    output.ConversionArguments.ReferenceSubresource =
        current->storage.image.array_layer;
    output.ConversionArguments.DecodeColorSpace =
        DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601;
    output.ConversionArguments.OutputColorSpace =
        DXGI_COLOR_SPACE_YCBCR_FULL_G22_LEFT_P601;
  }
  uint32_t barrier_count = 0;
  for (uint32_t index = 0; index < prepare_info->activation_count; ++index) {
    const auto& activation = prepare_info->activations[index];
    auto& barrier = command->barriers[barrier_count++];
    barrier = {};
    barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
    barrier.Aliasing.pResourceBefore =
        activation.before != nullptr
            ? reinterpret_cast<Image*>(activation.before)->resource.Get()
            : nullptr;
    barrier.Aliasing.pResourceAfter =
        reinterpret_cast<Image*>(activation.after)->resource.Get();
  }
  auto& input_activation = command->barriers[barrier_count++];
  input_activation = {};
  input_activation.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
  input_activation.Aliasing.pResourceAfter = compressed->resource.Get();
  command->native->ResourceBarrier(barrier_count, command->barriers);
  barrier_count = 0;
  command->barriers[barrier_count++] = Transition(
      compressed->resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_STATE_VIDEO_DECODE_READ);
  AppendImageTransitions(output_image, decode.output->storage.image.array_layer,
                         D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE,
                         command->barriers, &barrier_count);
  if (separate_reference) {
    AppendImageTransitions(current_image, current->storage.image.array_layer,
                           D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE,
                           command->barriers, &barrier_count);
  }
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    AppendImageTransitions(
        reference_images[index],
        picture.references[index].frame->storage.image.array_layer,
        D3D12_RESOURCE_STATE_VIDEO_DECODE_READ, command->barriers,
        &barrier_count);
  }
  command->native->ResourceBarrier(barrier_count, command->barriers);
  command->native->DecodeFrame(session->decoder.Get(), &output, &arguments);
  command->native->EndQuery(command->query.Get(),
                            D3D12_QUERY_TYPE_VIDEO_DECODE_STATISTICS, 0);
  command->native->ResolveQueryData(
      command->query.Get(), D3D12_QUERY_TYPE_VIDEO_DECODE_STATISTICS, 0, 1,
      command->feedback_resource, command->feedback.byte_offset);
  for (uint32_t index = 0; index < barrier_count; ++index) {
    auto& transition = command->barriers[index].Transition;
    transition.StateBefore = transition.StateAfter;
    transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  }
  command->native->ResourceBarrier(barrier_count, command->barriers);
  status = Hresult(command->native->Close());
  command->prepared = amdf_status_is_ok(status);
  return status;
}

amdf_status_t AMDF_CALL CommandQueryResult(
    amdf_media_command_t* base, amdf_media_command_result_info_t* out_info) {
  const auto* command = reinterpret_cast<const Command*>(base);
  const auto* queue = reinterpret_cast<const Queue*>(base->queue);
  if (command->submission == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  out_info->operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  out_info->submission = command->submission;
  out_info->codec_result = AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE;
  if (command->executed) {
    out_info->codec_result =
        command->feedback_pointer->Status == D3D12_VIDEO_DECODE_STATUS_OK
            ? AMDF_MEDIA_CODEC_RESULT_SUCCEEDED
            : AMDF_MEDIA_CODEC_RESULT_FAILED;
  }
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL CommandDestroy(amdf_media_command_t* base) {
  auto* command = reinterpret_cast<Command*>(base);
  auto* queue = reinterpret_cast<Queue*>(base->queue);
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  Session* session = command->session;
  Delete(queue->base.host_allocator, command);
  if (session != nullptr) {
    amdf_child_tracker_unregister(&session->commands);
  }
  amdf_child_tracker_unregister(&queue->base.children);
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
