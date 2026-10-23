// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/interop/gpu/media/decode_fixture.h"

#include <dxgi1_4.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include "amdf/media/gfx11.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/interop/gpu/media/consume.h"

using Microsoft::WRL::ComPtr;
namespace {
amdf_media_frame_t Frame(amdf_memory_t* memory, amdf_media_image_t* image,
                         uint64_t offset,
                         const amdf_media_image_description_t& description) {
  amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = description.format,
  };
  frame.width = description.width;
  frame.height = description.height;
  frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE;
  frame.storage.image.memory = memory;
  frame.storage.image.byte_offset = offset;
  frame.storage.image.image = image;
  return frame;
}

void CheckPixels(const uint32_t* pixels, uint32_t generation, uint32_t width,
                 uint32_t height, bool p010, uint64_t expected_hash) {
  const uint32_t pixel_count = width * height;
  uint64_t hash = 0xcbf29ce484222325;
  for (uint32_t pixel = 0; pixel < pixel_count; ++pixel) {
    ASSERT_EQ(pixels[pixel * 2 + 1] >> 16, generation) << "pixel=" << pixel;
    const uint32_t values[] = {pixels[pixel * 2] & 65535u,
                               pixels[pixel * 2] >> 16,
                               pixels[pixel * 2 + 1] & 65535u};
    for (uint32_t value : values) {
      if (p010) {
        ASSERT_EQ(value & 63u, 0u) << "P010 low storage bits";
      } else {
        ASSERT_LE(value, 255u);
      }
      hash = (hash ^ (value & 255u)) * 0x100000001b3;
      if (p010) {
        hash = (hash ^ (value >> 8)) * 0x100000001b3;
      }
    }
  }
  ASSERT_EQ(hash, expected_hash) << "generation=" << generation;
  for (uint32_t pixel = 0; pixel < 4096; ++pixel) {
    const uint32_t x = generation * 26 % (width - 127) + (pixel % 64) * 2;
    const uint32_t y = generation * 18 % (height - 127) + (pixel / 64) * 2;
    const uint32_t first = pixels[(y * width + x) * 2];
    const uint32_t second = pixels[(y * width + x) * 2 + 1];
    const double divisor = p010 ? 256.0 : 1.0;
    const double luma = (first & 65535u) / divisor;
    const double u = (first >> 16) / divisor - 128.0;
    const double v = (second & 65535u) / divisor - 128.0;
    const double expected[] = {
        luma + 1.402 * v, luma - 0.344136 * u - 0.714136 * v, luma + 1.772 * u};
    const uint32_t result = pixels[pixel_count * 2 + pixel];
    ASSERT_EQ(result >> 24, 255u);
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const int rounded = static_cast<int>(
          std::nearbyint(std::fmin(255.0, std::fmax(0.0, expected[channel]))));
      ASSERT_LE(std::abs(int((result >> (channel * 8)) & 255u) - rounded), 1)
          << "generation=" << generation << " pixel=" << pixel;
    }
  }
}

}  // namespace

MediaDecodeInteropTest::MediaDecodeInteropTest()
    : GpuCommandTest({
          .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
          .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
          .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
          .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                              AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
          .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
          .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      }) {}

void MediaDecodeInteropTest::SetUp() {
  ComPtr<ID3D12Debug> debug;
  if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
    GTEST_SKIP() << "D3D12 media validation requires Windows Graphics Tools";
  }
  debug->EnableDebugLayer();
  GpuCommandTest::SetUp();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const void* extension = nullptr;
  const auto query_status = api_->query_extension(
      AMDF_EXTENSION_MEDIA, AMDF_MEDIA_EXTENSION_VERSION_1,
      AMDF_MEDIA_EXTENSION_VERSION_LATEST, &extension);
  if (query_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "media extension is not enabled";
  }
  ASSERT_EQ(query_status, AMDF_STATUS_OK);
  media_ = static_cast<const amdf_media_api_t*>(extension);
  const amdf_media_device_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  ASSERT_EQ(media_->device_create(endpoint_, &create, &media_device_),
            AMDF_STATUS_OK);
  // D3D12 returns the same device for this adapter. Keep an independent
  // validation view until after every media object has been released.
  amdf_endpoint_info_t endpoint = {
      .type = AMDF_STRUCTURE_TYPE_ENDPOINT_INFO,
      .structure_size = sizeof(endpoint),
  };
  ASSERT_EQ(api_->endpoint_query_info(endpoint_, &endpoint), AMDF_STATUS_OK);
  ASSERT_EQ(endpoint.native_identity.type,
            AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER);
  const auto& identity = endpoint.native_identity.value.windows_adapter;
  const LUID luid = {static_cast<DWORD>(identity.luid),
                     static_cast<LONG>(identity.luid >> 32)};
  ComPtr<IDXGIFactory4> factory;
  ComPtr<IDXGIAdapter1> adapter;
  ASSERT_TRUE(SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))));
  ASSERT_TRUE(
      SUCCEEDED(factory->EnumAdapterByLuid(luid, IID_PPV_ARGS(&adapter))));
  ComPtr<ID3D12Device> native;
  ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0,
                                          IID_PPV_ARGS(&native))));
  ASSERT_TRUE(SUCCEEDED(native.As(&validation_)));
  const amdf_media_kernel_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .required_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
      .maximum_pending_submission_count = 32,
  };
  ASSERT_EQ(
      media_->kernel_queue_create(media_device_, &queue_create, &decode_queue_),
      AMDF_STATUS_OK);
  const amdf_gpu_kernel_queue_create_info_t gpu_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(gpu_create),
      .queue_family_ordinal = family_.ordinal,
      .maximum_pending_submission_count = 32,
      .required_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
  };
  ASSERT_EQ(gpu_api_->kernel_queue_create(device_, &gpu_create, &gpu_queue_),
            AMDF_STATUS_OK);
  const amdf_kernel_queue_dependency_create_info_t dependency_create = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_CREATE_INFO,
      .structure_size = sizeof(dependency_create),
  };
  ASSERT_EQ(api_->kernel_queue_dependency_create(
                gpu_queue_, decode_queue_, &dependency_create, &gpu_to_decode_),
            AMDF_STATUS_OK);
  ASSERT_EQ(api_->kernel_queue_dependency_create(
                decode_queue_, gpu_queue_, &dependency_create, &decode_to_gpu_),
            AMDF_STATUS_OK);
}

void MediaDecodeInteropTest::CreateSharedMemory(uint64_t byte_length,
                                                amdf_memory_t** out_memory,
                                                uint64_t* out_address) {
  const amdf_memory_device_access_t accesses[] = {
      {media_device_,
       {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}},
      {device_,
       {
           .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
           .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
       }},
  };
  amdf_memory_profile_t profile = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
      .structure_size = sizeof(profile),
  };
  amdf_memory_access_capabilities_t capabilities[2] = {};
  for (auto& capability : capabilities) {
    capability.type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES;
    capability.structure_size = sizeof(capability);
  }
  ASSERT_EQ(api_->memory_scope_query_device_profile(
                configuration_.image_heap_memory_scope,
                configuration_.image_heap_memory_profile_ordinal, 2, accesses,
                &profile, capabilities),
            AMDF_STATUS_OK);
  ASSERT_EQ(profile.availability, AMDF_STATUS_OK);
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal =
          configuration_.image_heap_memory_profile_ordinal,
      .access_count = 2,
      .required_flags = AMDF_MEMORY_FLAG_SHAREABLE,
      .byte_length = byte_length,
      .minimum_alignment = 65536,
      .accesses = accesses,
  };
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(api_->memory_create(configuration_.image_heap_memory_scope, &create,
                                &memory, nullptr),
            AMDF_STATUS_OK);
  memories_.push_back(memory);
  *out_memory = memory;
  amdf_memory_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->memory_query_info(memory, &info), AMDF_STATUS_OK);
  ASSERT_EQ(info.access_count, 2u);
  EXPECT_EQ(info.byte_length, byte_length);
  const uint64_t granularity =
      profile.allocation.native_byte_length_granularity;
  ASSERT_NE(granularity, 0u);
  EXPECT_EQ(info.native_allocation_byte_length,
            (byte_length + profile.allocation.native_byte_length_prefix +
             granularity - 1) &
                ~(granularity - 1));
  EXPECT_EQ(info.native_allocation_granularity, granularity);
  ASSERT_EQ(api_->memory_query_address(memory, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       out_address),
            AMDF_STATUS_OK);
}

void MediaDecodeInteropTest::CreateRecordings(uint32_t count) {
  const amdf_memory_device_access_t access = {
      media_device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = configuration_.feedback_memory_profile_ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = 65536,
      .accesses = &access,
  };
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(api_->memory_create(configuration_.feedback_memory_scope, &create,
                                &memory, nullptr),
            AMDF_STATUS_OK);
  memories_.push_back(memory);
  const amdf_memory_map_info_t map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(map),
      .byte_length = 65536,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api_->memory_map(memory, &map, &feedback_mapping_), AMDF_STATUS_OK);
  amdf_host_mapping_info_t mapping = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(mapping),
  };
  ASSERT_EQ(api_->host_mapping_query_info(feedback_mapping_, &mapping),
            AMDF_STATUS_OK);
  feedback_ = static_cast<uint8_t*>(mapping.pointer);
  std::memset(feedback_, 0xa5, 65536);
  for (uint32_t index = 0; index < count; ++index) {
    const amdf_media_command_create_info_t command_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
        .structure_size = sizeof(command_create),
        .maximum_slice_count = 1,
        .maximum_reference_count =
            configuration_.configuration.reference_image_count - 1,
        .maximum_activation_count = 2,
        .feedback = {memory, 0,
                     64 + index * configuration_.feedback_byte_length,
                     configuration_.feedback_byte_length},
    };
    amdf_media_command_t* command = nullptr;
    ASSERT_EQ(media_->command_create(decode_queue_, &command_create, &command),
              AMDF_STATUS_OK);
    commands_.push_back(command);
  }
}

void MediaDecodeInteropTest::SubmitGpu(
    const amdf_gpu_kernel_command_t& command,
    const amdf_kernel_queue_dependency_wait_t* wait) {
  const amdf_gpu_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command_count = 1,
      .wait_count = wait != nullptr ? 1u : 0u,
      .commands = &command,
      .waits = wait,
  };
  uint64_t point = 0;
  const auto status =
      gpu_api_->kernel_queue_submit(gpu_queue_, &submit, &point);
  if (point != 0) {
    gpu_point_ = point;
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
}

void MediaDecodeInteropTest::RunPictures(
    std::span<const DecodePictureInput> fixture, uint32_t frame_count,
    uint32_t recording_count) {
  const auto* kernel =
      kernels::media_consume::kKernels.Find(gpu_endpoint_info_);
  if (kernel == nullptr) {
    GTEST_SKIP() << "the raw-layout witness targets gfx1150";
  }
  const auto* profile = Pm4CommandProfile::Find(gpu_endpoint_info_);
  ASSERT_NE(profile, nullptr);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 64u);
  ASSERT_FALSE(fixture.empty());
  const uint32_t width = configuration_.configuration.coded_width;
  const uint32_t height = configuration_.configuration.coded_height;
  const uint32_t pixel_count = width * height;
  const uint32_t reference_count =
      configuration_.configuration.reference_image_count;
  ASSERT_LE(reference_count, reference_frames_.size());
  ASSERT_GE(width, 128u);
  ASSERT_GE(height, 128u);
  const bool p010 =
      configuration_.configuration.format == AMDF_MEDIA_PIXEL_FORMAT_P010;
  ASSERT_NO_FATAL_FAILURE(CreateRecordings(recording_count));
  const uint64_t output_heap_length =
      131072 + 2 * configuration_.output.byte_length;
  const uint64_t reference_heap_length =
      131072 + reference_count * configuration_.reference.byte_length;
  ASSERT_NE(configuration_.reference.byte_length, 0u);
  amdf_memory_t* output_heap = nullptr;
  amdf_memory_t* reference_heap = nullptr;
  amdf_memory_t* compressed = nullptr;
  uint64_t output_address = 0, reference_address = 0, compressed_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      CreateSharedMemory(output_heap_length, &output_heap, &output_address));
  ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
      reference_heap_length, &reference_heap, &reference_address));
  ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(recording_count * 65536,
                                             &compressed, &compressed_address));
  std::array<amdf_media_frame_t, 2> outputs;
  auto& references = reference_frames_;
  std::array<amdf_media_gfx11_image_layout_t, 2> layouts = {};
  for (uint32_t index = 0; index < outputs.size(); ++index) {
    amdf_media_image_t* image = nullptr;
    const uint64_t offset = 65536 + index * configuration_.output.byte_length;
    ASSERT_NO_FATAL_FAILURE(CreateImage(
        output_heap, offset, configuration_.output.description, &image));
    outputs[index] =
        Frame(output_heap, image, offset, configuration_.output.description);
    auto& layout = layouts[index];
    layout.type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT;
    layout.structure_size = sizeof(layout);
    amdf_media_image_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
        .structure_size = sizeof(info),
        .next = &layout,
    };
    const amdf_media_image_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
        .structure_size = sizeof(query),
    };
    ASSERT_EQ(media_->image_query_info(image, 1, &query, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(layout.plane_count, 2u);
    for (const auto& plane : layout.planes) {
      ASSERT_EQ(plane.swizzle_mode, 26u);
      ASSERT_EQ(plane.pipe_bank_xor, 0u);
    }
  }
  for (uint32_t index = 0; index < reference_count; ++index) {
    amdf_media_image_t* image = nullptr;
    const uint64_t offset =
        65536 + index * configuration_.reference.byte_length;
    ASSERT_NO_FATAL_FAILURE(CreateImage(
        reference_heap, offset, configuration_.reference.description, &image));
    references[index] = Frame(reference_heap, image, offset,
                              configuration_.reference.description);
  }
  const uint64_t result_bytes = (uint64_t(pixel_count) * 2 + 4096) * 4;
  const uint64_t result_stride = result_bytes + 128;
  GpuMemory* code = nullptr;
  GpuMemory* controls = nullptr;
  GpuMemory* source = nullptr;
  GpuMemory* results = nullptr;
  GpuMemory* heap_copy = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      kernel->executable.byte_length, &code, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      (recording_count + 3) * 4096, &controls, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       recording_count * 65536, &source,
                                       AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                   recording_count * result_stride, &results,
                   AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, output_heap_length,
      &heap_copy, AMDF_MEMORY_FLAG_HOST_COHERENT));
  std::memcpy(code->host.pointer, kernel->executable.words,
              kernel->executable.byte_length);
  const Pm4ComputeProgram program = {
      code->device_address + kernel->entry_byte_offset,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {64, 1, 1}};
  auto* control_bytes = static_cast<uint8_t*>(controls->host.pointer);
  std::memset(control_bytes, 0, controls->info.byte_length);
  std::vector<amdf_gpu_kernel_command_t> consumers(recording_count);
  std::vector<std::vector<uint8_t>> immutable(recording_count);
  for (uint32_t index = 0; index < recording_count; ++index) {
    Pm4CommandWriter writer(
        reinterpret_cast<uint32_t*>(control_bytes + index * 4096 + 256),
        *profile);
    writer.SystemBarrier();
    writer.BindCompute(program, controls->device_address + index * 4096,
                       family_.format_features);
    writer.Dispatch(program, pixel_count, 1, 1);
    writer.SystemBarrier();
    writer.PadToEightWords();
    consumers[index] = {
        .memory = controls->memory,
        .byte_offset = index * 4096 + 256,
        .byte_length = writer.word_count() * sizeof(uint32_t),
    };
    immutable[index].assign(control_bytes + consumers[index].byte_offset,
                            control_bytes + consumers[index].byte_offset +
                                consumers[index].byte_length);
  }
  auto make_command = [&](uint32_t slot, auto emit) {
    Pm4CommandWriter writer(
        reinterpret_cast<uint32_t*>(control_bytes + slot * 4096), *profile);
    writer.SystemBarrier();
    emit(writer);
    writer.WaitDma();
    writer.SystemBarrier();
    writer.PadToEightWords();
    return amdf_gpu_kernel_command_t{
        .memory = controls->memory,
        .byte_offset = slot * 4096,
        .byte_length = writer.word_count() * sizeof(uint32_t),
    };
  };
  const auto initialize =
      make_command(recording_count, [&](Pm4CommandWriter& writer) {
        writer.DmaFill32(output_address, 0xa5a5a5a5,
                         static_cast<uint32_t>(output_heap_length));
        writer.DmaFill32(reference_address, 0xa5a5a5a5,
                         static_cast<uint32_t>(reference_heap_length));
      });
  const auto upload =
      make_command(recording_count + 1, [&](Pm4CommandWriter& writer) {
        writer.DmaCopy(source->device_address, compressed_address,
                       recording_count * 65536);
      });
  const auto capture =
      make_command(recording_count + 2, [&](Pm4CommandWriter& writer) {
        writer.DmaCopy(output_address, heap_copy->device_address,
                       static_cast<uint32_t>(output_heap_length));
      });
  ASSERT_NO_FATAL_FAILURE(code->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(controls->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(initialize));
  bool reference_active[17] = {};
  uint64_t consumed[2] = {};
  for (uint32_t batch = 0; batch < frame_count; batch += recording_count) {
    const uint32_t count = std::min(recording_count, frame_count - batch);
    std::memset(source->host.pointer, 0, source->info.byte_length);
    std::memset(results->host.pointer, 0xa5, results->info.byte_length);
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t generation = batch + index, slot = generation % 2;
      auto& entry = fixture[generation % fixture.size()];
      std::memcpy(static_cast<uint8_t*>(source->host.pointer) + index * 65536,
                  entry.bytes, entry.byte_length);
      const auto& layout = layouts[slot];
      const uint64_t pointers[] = {
          output_address + layout.planes[0].byte_offset,
          results->device_address + index * result_stride + 64};
      const uint32_t constants[] = {
          generation * 26 % (width - 127),
          generation * 18 % (height - 127),
          generation,
          static_cast<uint32_t>(layout.planes[1].byte_offset -
                                layout.planes[0].byte_offset),
          0,
          0,
          width,
          height,
          p010 ? 1u : 0u};
      std::memcpy(control_bytes + index * 4096, pointers, sizeof(pointers));
      std::memcpy(control_bytes + index * 4096 + sizeof(pointers), constants,
                  sizeof(constants));
      amdf_media_image_activation_t activations[2];
      uint32_t activation_count = 0;
      if (generation < 2) {
        activations[activation_count++] = {nullptr,
                                           outputs[slot].storage.image.image};
      }
      if (!reference_active[entry.reference_slot]) {
        activations[activation_count++] = {
            nullptr, references[entry.reference_slot].storage.image.image};
        reference_active[entry.reference_slot] = true;
      }
      const amdf_media_decode_picture_info_t decode = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
          .structure_size = sizeof(decode),
          .next = entry.parameters,
          .compressed = {compressed, 0, index * 65536, entry.byte_length},
          .output = &outputs[slot],
          .reference_output = &references[entry.reference_slot],
          .reference_slot_index = entry.reference_slot,
      };
      const amdf_media_command_prepare_info_t prepare = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
          .structure_size = sizeof(prepare),
          .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
          .activation_count = activation_count,
          .activations = activations,
          .decode = &decode,
      };
      ASSERT_EQ(media_->command_prepare(session_, commands_[index], &prepare),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(source->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_NO_FATAL_FAILURE(controls->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_NO_FATAL_FAILURE(results->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    ASSERT_NO_FATAL_FAILURE(SubmitGpu(upload));
    const uint64_t uploaded = gpu_point_;
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t slot = (batch + index) % 2;
      const amdf_kernel_queue_dependency_wait_t ready = {
          gpu_to_decode_, std::max(uploaded, consumed[slot])};
      const amdf_media_kernel_queue_submission_info_t submit = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
          .structure_size = sizeof(submit),
          .command = commands_[index],
          .wait_count = 1,
          .waits = &ready,
      };
      uint64_t point = 0;
      const auto status =
          media_->kernel_queue_submit(decode_queue_, &submit, &point);
      if (point != 0) {
        decode_point_ = point;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      const amdf_kernel_queue_dependency_wait_t decoded = {decode_to_gpu_,
                                                           point};
      ASSERT_NO_FATAL_FAILURE(SubmitGpu(consumers[index], &decoded));
      consumed[slot] = gpu_point_;
    }
    ASSERT_NO_FATAL_FAILURE(SubmitGpu(capture));
    ASSERT_EQ(api_->kernel_queue_wait(gpu_queue_, gpu_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t progress = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(progress),
    };
    ASSERT_EQ(api_->kernel_queue_refresh_status(decode_queue_, &progress),
              AMDF_STATUS_OK);
    ASSERT_EQ(progress.terminal_status, AMDF_STATUS_OK);
    ASSERT_EQ(progress.retired_submission, decode_point_);
    ASSERT_NO_FATAL_FAILURE(results->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    ASSERT_NO_FATAL_FAILURE(heap_copy->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    for (uint32_t index = 0; index < count; ++index) {
      amdf_media_command_result_info_t result = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
          .structure_size = sizeof(result),
      };
      ASSERT_EQ(media_->command_query_result(commands_[index], &result),
                AMDF_STATUS_OK);
      ASSERT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_SUCCEEDED);
      const auto* bytes = static_cast<const uint8_t*>(results->host.pointer) +
                          index * result_stride;
      ASSERT_NO_FATAL_FAILURE(CheckPixels(
          reinterpret_cast<const uint32_t*>(bytes + 64), batch + index, width,
          height, p010, fixture[(batch + index) % fixture.size()].pixel_hash));
      for (uint32_t guard = 0; guard < 64; ++guard) {
        ASSERT_EQ(bytes[guard], 0xa5u);
        ASSERT_EQ(bytes[64 + result_bytes + guard], 0xa5u);
      }
      ASSERT_EQ(std::memcmp(control_bytes + consumers[index].byte_offset,
                            immutable[index].data(), immutable[index].size()),
                0);
    }
    const auto* bytes = static_cast<const uint8_t*>(heap_copy->host.pointer);
    for (uint32_t guard = 0; guard < 65536; ++guard) {
      ASSERT_EQ(bytes[guard], 0xa5u);
      ASSERT_EQ(bytes[output_heap_length - 65536 + guard], 0xa5u);
    }
    for (uint32_t guard = 0; guard < 65536; ++guard) {
      if (guard < 64 ||
          guard >= 64 + recording_count * configuration_.feedback_byte_length) {
        ASSERT_EQ(feedback_[guard], 0xa5u);
      }
    }
  }
  RecordProperty("pictures", frame_count);
  RecordProperty("recordings", recording_count);
  RecordProperty("batch_observations",
                 (frame_count + recording_count - 1) / recording_count);
  RecordProperty("media_kernel_sha256", kernel->executable.sha256);
}

void MediaDecodeInteropTest::CreateImage(
    amdf_memory_t* memory, uint64_t offset,
    const amdf_media_image_description_t& description,
    amdf_media_image_t** out_image) {
  const uint32_t gpu_access = 1;
  const bool output =
      (description.roles & AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT) != 0;
  const amdf_media_image_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory = memory,
      .description = description,
      .byte_offset = offset,
      .layout_access_count = output ? 1u : 0u,
      .layout_access_ordinals = output ? &gpu_access : nullptr,
  };
  amdf_media_image_t* image = nullptr;
  ASSERT_EQ(media_->image_create(media_device_, &create, &image),
            AMDF_STATUS_OK);
  images_.push_back(image);
  *out_image = image;
}

void MediaDecodeInteropTest::TearDown() {
  if (decode_point_ != 0) {
    ASSERT_EQ(api_->kernel_queue_wait(decode_queue_, decode_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  if (gpu_point_ != 0) {
    ASSERT_EQ(api_->kernel_queue_wait(gpu_queue_, gpu_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  for (auto* command : commands_) {
    ASSERT_EQ(media_->command_destroy(command), AMDF_STATUS_OK);
  }
  if (session_ != nullptr) {
    ASSERT_EQ(media_->session_destroy(session_), AMDF_STATUS_OK);
  }
  for (auto* image : images_) {
    media_->image_destroy(image);
  }
  if (feedback_mapping_ != nullptr) {
    ASSERT_EQ(api_->host_mapping_destroy(feedback_mapping_), AMDF_STATUS_OK);
  }
  for (auto* memory : memories_) {
    ASSERT_EQ(api_->memory_destroy(memory, nullptr), AMDF_STATUS_OK);
  }
  for (auto* dependency : {gpu_to_decode_, decode_to_gpu_}) {
    if (dependency != nullptr) {
      ASSERT_EQ(api_->kernel_queue_dependency_destroy(dependency),
                AMDF_STATUS_OK);
    }
  }
  for (auto* queue : {decode_queue_, gpu_queue_}) {
    if (queue != nullptr) {
      ASSERT_EQ(api_->kernel_queue_destroy(queue), AMDF_STATUS_OK);
    }
  }
  if (media_device_ != nullptr) {
    ASSERT_EQ(api_->device_destroy(media_device_), AMDF_STATUS_OK);
  }
  if (validation_ != nullptr) {
    for (UINT64 index = 0; index < validation_->GetNumStoredMessages();
         ++index) {
      SIZE_T size = 0;
      ASSERT_TRUE(SUCCEEDED(validation_->GetMessage(index, nullptr, &size)));
      std::vector<uint8_t> storage(size);
      auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
      ASSERT_TRUE(SUCCEEDED(validation_->GetMessage(index, message, &size)));
      EXPECT_GT(message->Severity, D3D12_MESSAGE_SEVERITY_WARNING)
          << message->ID << ": " << message->pDescription;
    }
    validation_.Reset();
  }
  GpuCommandTest::TearDown();
}

void MediaDecodeInteropTest::CreateSession(
    const amdf_media_configuration_t& configuration,
    const void* codec_configuration) {
  const amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = codec_configuration,
      .configuration = configuration,
  };
  configuration_.type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO;
  configuration_.structure_size = sizeof(configuration_);
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query,
                                               &configuration_),
            AMDF_STATUS_OK);
  ASSERT_EQ(configuration_.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = codec_configuration,
      .configuration = configuration_.configuration,
  };
  ASSERT_EQ(media_->session_create(decode_queue_, &session_create, &session_),
            AMDF_STATUS_OK);
}
