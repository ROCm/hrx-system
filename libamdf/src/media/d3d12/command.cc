// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/media/d3d12/command.h"

#include "libamdf/src/media/d3d12/decode.h"
#include "libamdf/src/media/d3d12/encode.h"
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

amdf_status_t CreateDefaultBuffer(Device* device, uint64_t byte_length,
                                  ID3D12Resource** out_resource) {
  D3D12_HEAP_PROPERTIES heap = {
      .Type = D3D12_HEAP_TYPE_DEFAULT,
      .CreationNodeMask = 1,
      .VisibleNodeMask = 1,
  };
  D3D12_RESOURCE_DESC description = {
      .Dimension = D3D12_RESOURCE_DIMENSION_BUFFER,
      .Width = byte_length,
      .Height = 1,
      .DepthOrArraySize = 1,
      .MipLevels = 1,
      .SampleDesc = {.Count = 1},
      .Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
  };
  return Hresult(device->native->CreateCommittedResource(
      &heap, D3D12_HEAP_FLAG_NONE, &description, D3D12_RESOURCE_STATE_COMMON,
      nullptr, IID_PPV_ARGS(out_resource)));
}

bool SameImageRequirements(const amdf_media_image_requirements_t& lhs,
                           const amdf_media_image_requirements_t& rhs) {
  return lhs.description.format == rhs.description.format &&
         lhs.description.width == rhs.description.width &&
         lhs.description.height == rhs.description.height &&
         lhs.description.array_layer_count ==
             rhs.description.array_layer_count &&
         lhs.description.roles == rhs.description.roles &&
         lhs.byte_length == rhs.byte_length &&
         lhs.byte_offset_alignment == rhs.byte_offset_alignment;
}

amdf_status_t PrepareEncodeCommand(
    Session* session, Command* command, Device* device,
    const amdf_media_command_prepare_info_t* prepare_info) {
  amdf_status_t status = amdf_structure_validate_header(
      prepare_info->encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO,
      sizeof(amdf_media_encode_picture_info_t));
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  const auto& encode = *prepare_info->encode;
  EncodePicture picture = {};
  status = PrepareEncode(*session, encode, command->maximum_reference_count,
                         &picture);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Memory* compressed = nullptr;
  status = ValidateBuffer(
      device, encode.compressed, kSharedHeap,
      session->requirements.compressed_byte_offset_alignment,
      session->requirements.compressed_byte_length_alignment,
      session->requirements.minimum_compressed_byte_length, &compressed);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Image* input_image = nullptr;
  status = ValidateImage(device, encode.input, session->requirements.input,
                         encode.compressed.memory, &input_image);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  Image* reconstruction_image = nullptr;
  status = ValidateImage(device, encode.reconstruction,
                         session->requirements.reference,
                         encode.compressed.memory, &reconstruction_image);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (!SameImageRequirements(input_image->requirements,
                             session->requirements.input) ||
      !SameImageRequirements(reconstruction_image->requirements,
                             session->requirements.reference) ||
      SameSubresource(encode.input, encode.reconstruction)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
  }
  Image* reference_images[8] = {};
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    const auto& reference = picture.references[index];
    status =
        ValidateImage(device, reference.frame, session->requirements.reference,
                      encode.compressed.memory, &reference_images[index]);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
    if (reference_images[index] != reconstruction_image ||
        reference.slot_index == encode.reconstruction_slot_index ||
        SameSubresource(reference.frame, encode.reconstruction)) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
    for (uint32_t prior = 0; prior < index; ++prior) {
      if (reference.slot_index == picture.references[prior].slot_index ||
          SameSubresource(reference.frame, picture.references[prior].frame)) {
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
            encode.compressed.memory) {
      return amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT);
    }
  }

  command->prepared = false;
  command->executed = false;
  command->submission = 0;
  command->operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  command->compressed = encode.compressed;
  status = Hresult(command->allocator->Reset());
  if (amdf_status_is_ok(status)) {
    status = Hresult(command->encode->Reset(command->allocator.Get()));
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }

  ID3D12Resource* reference_resources[8] = {};
  UINT reference_subresources[8] = {};
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    reference_resources[index] = reference_images[index]->resource.Get();
    reference_subresources[index] =
        picture.references[index].frame->storage.image.array_layer;
  }
  picture.arguments.PictureControlDesc.ReferenceFrames = {
      picture.reference_count,
      picture.reference_count != 0 ? reference_resources : nullptr,
      picture.reference_count != 0 ? reference_subresources : nullptr};
  picture.arguments.pInputFrame = input_image->resource.Get();
  picture.arguments.InputFrameSubresource =
      encode.input->storage.image.array_layer;
  D3D12_VIDEO_ENCODER_ENCODEFRAME_OUTPUT_ARGUMENTS output = {
      .Bitstream = {compressed->resource.Get(), encode.compressed.byte_offset},
      .ReconstructedPicture =
          {
              reconstruction_image->resource.Get(),
              encode.reconstruction->storage.image.array_layer,
          },
      .EncoderOutputMetadata = {command->raw_metadata.Get(), 0},
  };

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
  auto& output_activation = command->barriers[barrier_count++];
  output_activation = {};
  output_activation.Type = D3D12_RESOURCE_BARRIER_TYPE_ALIASING;
  output_activation.Aliasing.pResourceAfter = compressed->resource.Get();
  command->encode->ResourceBarrier(barrier_count, command->barriers);

  barrier_count = 0;
  AppendImageTransitions(input_image, encode.input->storage.image.array_layer,
                         D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ,
                         command->barriers, &barrier_count);
  AppendImageTransitions(reconstruction_image,
                         encode.reconstruction->storage.image.array_layer,
                         D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE,
                         command->barriers, &barrier_count);
  for (uint32_t index = 0; index < picture.reference_count; ++index) {
    AppendImageTransitions(
        reference_images[index],
        picture.references[index].frame->storage.image.array_layer,
        D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ, command->barriers,
        &barrier_count);
  }
  const uint32_t raw_metadata_barrier = barrier_count;
  command->barriers[barrier_count++] = Transition(
      command->raw_metadata.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE);
  command->barriers[barrier_count++] = Transition(
      command->feedback_resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE);
  command->barriers[barrier_count++] = Transition(
      compressed->resource.Get(), D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
      D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE);
  command->encode->ResourceBarrier(barrier_count, command->barriers);
  command->encode->EncodeFrame(session->encoder.Get(),
                               session->encoder_heap.Get(), &picture.arguments,
                               &output);
  auto raw_to_read = Transition(command->raw_metadata.Get(),
                                D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ);
  raw_to_read.Transition.StateBefore = D3D12_RESOURCE_STATE_VIDEO_ENCODE_WRITE;
  command->encode->ResourceBarrier(1, &raw_to_read);
  const auto& native = session->native_encode;
  D3D12_VIDEO_ENCODER_RESOLVE_METADATA_INPUT_ARGUMENTS resolve = {
      .EncoderCodec = native.codec,
      .EncoderProfile = native.profile,
      .EncoderInputFormat =
          session->info.configuration.format == AMDF_MEDIA_PIXEL_FORMAT_NV12
              ? DXGI_FORMAT_NV12
              : DXGI_FORMAT_P010,
      .EncodedPictureEffectiveResolution = native.resolution,
      .HWLayoutMetadata = {command->raw_metadata.Get(), 0},
  };
  D3D12_VIDEO_ENCODER_RESOLVE_METADATA_OUTPUT_ARGUMENTS resolved = {
      .ResolvedLayoutMetadata = {command->feedback_resource,
                                 command->feedback.byte_offset},
  };
  command->encode->ResolveEncoderOutputMetadata(&resolve, &resolved);
  command->barriers[raw_metadata_barrier].Transition.StateAfter =
      D3D12_RESOURCE_STATE_VIDEO_ENCODE_READ;
  for (uint32_t index = 0; index < barrier_count; ++index) {
    auto& transition = command->barriers[index].Transition;
    transition.StateBefore = transition.StateAfter;
    transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  }
  command->encode->ResourceBarrier(barrier_count, command->barriers);
  status = Hresult(command->encode->Close());
  command->prepared = amdf_status_is_ok(status);
  return status;
}

}  // namespace

amdf_status_t AMDF_CALL
CommandCreate(amdf_media_session_t* session_base,
              const amdf_media_command_create_info_t* create_info,
              amdf_media_command_t** out_command) {
  auto* session = reinterpret_cast<Session*>(session_base);
  amdf_kernel_queue_t* queue = session_base->queue;
  const auto direction = session->info.configuration.direction;
  const amdf_media_codec_t codec = session->info.configuration.codec;
  const bool decode = direction == AMDF_MEDIA_DIRECTION_DECODE;
  if ((decode && codec != AMDF_MEDIA_CODEC_H264 &&
       codec != AMDF_MEDIA_CODEC_HEVC && codec != AMDF_MEDIA_CODEC_VP9 &&
       codec != AMDF_MEDIA_CODEC_AV1) ||
      (!decode && codec != AMDF_MEDIA_CODEC_H264 &&
       codec != AMDF_MEDIA_CODEC_HEVC && codec != AMDF_MEDIA_CODEC_AV1)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  const size_t slice_byte_length = decode ? DecodeSliceByteLength(codec) : 0;
  auto* device = reinterpret_cast<Device*>(queue->device);
  Memory* feedback = nullptr;
  amdf_status_t status =
      ValidateBuffer(device, create_info->feedback, kFeedbackBuffer,
                     session->requirements.feedback_byte_offset_alignment, 1,
                     session->requirements.feedback_byte_length, &feedback);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (create_info->maximum_reference_count > (decode ? 16u : 8u) ||
      (decode &&
       create_info->maximum_slice_count > UINT32_MAX / slice_byte_length) ||
      create_info->maximum_activation_count > UINT32_MAX - 38) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  const uint64_t barrier_count =
      create_info->maximum_activation_count + 8 +
      2 * uint64_t(create_info->maximum_reference_count);
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
  command->base.session = session_base;
  command->feedback = create_info->feedback;
  command->feedback_resource = feedback->resource.Get();
  command->feedback_pointer =
      static_cast<const uint8_t*>(feedback->host_pointer) +
      create_info->feedback.byte_offset;
  command->maximum_slice_count = create_info->maximum_slice_count;
  command->maximum_reference_count = create_info->maximum_reference_count;
  command->maximum_activation_count = create_info->maximum_activation_count;
  command->barriers = reinterpret_cast<D3D12_RESOURCE_BARRIER*>(command + 1);
  command->slices = command->barriers + barrier_count;
  const auto command_type = decode ? D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE
                                   : D3D12_COMMAND_LIST_TYPE_VIDEO_ENCODE;
  status = Hresult(device->native->CreateCommandAllocator(
      command_type, IID_PPV_ARGS(&command->allocator)));
  if (amdf_status_is_ok(status)) {
    if (decode) {
      status = Hresult(device->native->CreateCommandList(
          0, command_type, command->allocator.Get(), nullptr,
          IID_PPV_ARGS(&command->decode)));
      command->executable = command->decode.Get();
    } else {
      status = Hresult(device->native->CreateCommandList(
          0, command_type, command->allocator.Get(), nullptr,
          IID_PPV_ARGS(&command->encode)));
      command->executable = command->encode.Get();
    }
  }
  if (amdf_status_is_ok(status)) {
    status =
        Hresult(decode ? command->decode->Close() : command->encode->Close());
  }
  if (amdf_status_is_ok(status) && decode) {
    const D3D12_QUERY_HEAP_DESC description = {
        D3D12_QUERY_HEAP_TYPE_VIDEO_DECODE_STATISTICS, 1, 0};
    status = Hresult(device->native->CreateQueryHeap(
        &description, IID_PPV_ARGS(&command->query)));
  }
  if (amdf_status_is_ok(status) && !decode) {
    status =
        CreateDefaultBuffer(device, session->native_encode.metadata_byte_length,
                            &command->raw_metadata);
  }
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
    Delete(queue->host_allocator, command);
    return status;
  }
  *out_command = &command->base;
  return AMDF_STATUS_OK;
}

amdf_status_t AMDF_CALL
CommandPrepare(amdf_media_command_t* command_base,
               const amdf_media_command_prepare_info_t* prepare_info) {
  auto* session_base = command_base->session;
  auto* session = reinterpret_cast<Session*>(session_base);
  auto* command = reinterpret_cast<Command*>(command_base);
  auto* queue = reinterpret_cast<Queue*>(session_base->queue);
  auto* device = reinterpret_cast<Device*>(queue->base.device);
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  amdf_status_t status = QueueTerminalStatus(queue);
  if (!amdf_status_is_ok(status)) {
    return status;
  }
  if (prepare_info->operation != AMDF_MEDIA_COMMAND_OPERATION_PICTURE) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
  }
  if (prepare_info->activation_count > command->maximum_activation_count) {
    return amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE);
  }
  if (session->info.configuration.direction == AMDF_MEDIA_DIRECTION_ENCODE) {
    if (prepare_info->decode != nullptr) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    return PrepareEncodeCommand(session, command, device, prepare_info);
  }
  if (prepare_info->encode != nullptr) {
    return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
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
  // Preparation is a transition of this live session child. A failed reset
  // remains safe to destroy and cannot publish an incomplete recording.
  command->prepared = false;
  command->executed = false;
  command->submission = 0;
  command->operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE;
  status = Hresult(command->allocator->Reset());
  if (amdf_status_is_ok(status)) {
    status = Hresult(command->decode->Reset(command->allocator.Get()));
  }
  if (!amdf_status_is_ok(status)) {
    return status;
  }

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
  arguments.pHeap = session->decoder_heap.Get();
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
  command->decode->ResourceBarrier(barrier_count, command->barriers);
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
  command->decode->ResourceBarrier(barrier_count, command->barriers);
  command->decode->DecodeFrame(session->decoder.Get(), &output, &arguments);
  command->decode->EndQuery(command->query.Get(),
                            D3D12_QUERY_TYPE_VIDEO_DECODE_STATISTICS, 0);
  command->decode->ResolveQueryData(
      command->query.Get(), D3D12_QUERY_TYPE_VIDEO_DECODE_STATISTICS, 0, 1,
      command->feedback_resource, command->feedback.byte_offset);
  for (uint32_t index = 0; index < barrier_count; ++index) {
    auto& transition = command->barriers[index].Transition;
    transition.StateBefore = transition.StateAfter;
    transition.StateAfter = D3D12_RESOURCE_STATE_COMMON;
  }
  command->decode->ResourceBarrier(barrier_count, command->barriers);
  status = Hresult(command->decode->Close());
  command->prepared = amdf_status_is_ok(status);
  return status;
}

amdf_status_t AMDF_CALL CommandQueryResult(
    amdf_media_command_t* base, amdf_media_command_result_info_t* out_info) {
  const auto* command = reinterpret_cast<const Command*>(base);
  const auto* session = reinterpret_cast<const Session*>(base->session);
  const auto* queue = reinterpret_cast<const Queue*>(base->session->queue);
  if (command->submission == 0) {
    return amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION);
  }
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  if (session->info.configuration.direction == AMDF_MEDIA_DIRECTION_DECODE) {
    if (out_info->next != nullptr) {
      return amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    }
    out_info->operation = command->operation;
    out_info->submission = command->submission;
    out_info->codec_result = AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE;
    if (command->executed) {
      const auto* feedback =
          reinterpret_cast<const D3D12_QUERY_DATA_VIDEO_DECODE_STATISTICS*>(
              command->feedback_pointer);
      out_info->codec_result = feedback->Status == D3D12_VIDEO_DECODE_STATUS_OK
                                   ? AMDF_MEDIA_CODEC_RESULT_SUCCEEDED
                                   : AMDF_MEDIA_CODEC_RESULT_FAILED;
    }
    return AMDF_STATUS_OK;
  }
  amdf_status_t status = AMDF_STATUS_OK;
  if (command->executed && out_info->next != nullptr) {
    status = ParseEncodeResult(
        *session, command->feedback_pointer, command->feedback.byte_length,
        command->compressed.byte_length,
        static_cast<amdf_media_encode_result_info_t*>(out_info->next));
    if (!amdf_status_is_ok(status) &&
        amdf_status_code(status) != AMDF_STATUS_CODE_BUFFER_TOO_SMALL) {
      return status;
    }
  } else if (out_info->next != nullptr) {
    auto* encode =
        static_cast<amdf_media_encode_result_info_t*>(out_info->next);
    status = ClearEncodeResult(*session, encode);
    if (!amdf_status_is_ok(status)) {
      return status;
    }
  }
  out_info->operation = command->operation;
  out_info->submission = command->submission;
  out_info->codec_result = AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE;
  if (command->executed) {
    const auto* metadata =
        reinterpret_cast<const D3D12_VIDEO_ENCODER_OUTPUT_METADATA*>(
            command->feedback_pointer);
    out_info->codec_result =
        metadata->EncodeErrorFlags ==
                D3D12_VIDEO_ENCODER_ENCODE_ERROR_FLAG_NO_ERROR
            ? AMDF_MEDIA_CODEC_RESULT_SUCCEEDED
            : AMDF_MEDIA_CODEC_RESULT_FAILED;
  }
  return status;
}

amdf_status_t AMDF_CALL CommandDestroy(amdf_media_command_t* base) {
  auto* command = reinterpret_cast<Command*>(base);
  auto* session = reinterpret_cast<Session*>(base->session);
  auto* queue = reinterpret_cast<Queue*>(base->session->queue);
  if (command->submission > amdf_atomic_uint64_load_acquire(&queue->retired)) {
    return amdf_make_api_status(AMDF_STATUS_CODE_BUSY);
  }
  Delete(queue->base.host_allocator, command);
  amdf_child_tracker_unregister(&session->commands);
  amdf_child_tracker_unregister(&queue->base.children);
  return AMDF_STATUS_OK;
}

}  // namespace amdf::media::d3d12
