// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "amdf/media/h264.h"

#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <utility>
#include <vector>

#include "amdf/media.h"
#include "amdf/media/gfx11.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "libamdf/cts/interop/gpu/media/consume.h"
#include "libamdf/cts/interop/gpu/media/reference_main.h"

namespace {

using Microsoft::WRL::ComPtr;

// FNV-1a over each raster pixel's Y/U/V bytes from independent FFmpeg decode,
// verified against the encoder reconstruction. Generation tags are checked
// separately so stale output cannot satisfy a repeated stream's pixel oracle.
constexpr uint64_t kPixelHashes[12] = {
    0x4f93364832b3f2d7, 0xdea0cbf06a7e24cd, 0x7e2ee8aa35fe6ce0,
    0xb160a8118523b73f, 0x5945a85843a5a234, 0xde9a0b9c42fd47bc,
    0x8193763875ecb1e9, 0x42798e3de4cf9fbc, 0x62edb9ad1a92162e,
    0x3330931192bf43b3, 0x17c4311a4e05c771, 0x1c443a7a7c988909};

struct FixturePicture {
  // Complete padded Annex B slice bytes retained by the embedded fixture.
  const uint8_t* bytes;
  // Native compressed extent, including alignment padding.
  uint32_t byte_length;
  // Application-selected reconstructed output slot.
  uint32_t reference_slot;
  // Complete public parsed picture metadata, with borrowed arrays below.
  amdf_media_h264_decode_picture_info_t picture;
  // Explicit physical reference-slot choices, independent of native indices.
  uint32_t reference_slots[3];
  // Parsed reference markings and POCs; frames are bound by the caller.
  amdf_media_h264_reference_t references[3];
  // One complete slice, already packed for the native byte framing.
  amdf_media_h264_slice_t slice;
};

void LoadFixture(std::array<FixturePicture, 12>* pictures) {
  const auto& file = amdf_cts_media_reference_main_create()[0];
  const auto* cursor = reinterpret_cast<const uint8_t*>(file.data);
  ASSERT_GE(file.size, 68u);
  ASSERT_EQ(std::memcmp(cursor, "H264REFS", 8), 0);
  cursor += 8;
  auto word = [&]() {
    uint32_t value;
    std::memcpy(&value, cursor, 4);
    cursor += 4;
    return value;
  };
  ASSERT_EQ(word(), pictures->size());
  ASSERT_EQ(word(), 256u);
  ASSERT_EQ(word(), 256u);
  const uint32_t frame_bits = word(), direct8 = word(), deblock = word();
  const uint32_t constrained = word(), level = word();
  const int32_t qp = static_cast<int32_t>(word());
  const int32_t qs = static_cast<int32_t>(word());
  const int32_t chroma = static_cast<int32_t>(word());
  const uint32_t poc_bits = word(), maximum_references = word();
  const uint32_t l0 = word(), l1 = word();
  ASSERT_EQ(level, 13u);
  for (uint32_t index = 0; index < pictures->size(); ++index) {
    ASSERT_LE(cursor + 72,
              reinterpret_cast<const uint8_t*>(file.data) + file.size);
    auto& entry = (*pictures)[index];
    ASSERT_EQ(word(), index);
    entry.byte_length = word();
    auto& picture = entry.picture;
    picture.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_DECODE_PICTURE_INFO;
    picture.structure_size = sizeof(picture);
    picture.frame_number = word();
    picture.top_field_order_count = static_cast<int32_t>(word());
    picture.bottom_field_order_count = picture.top_field_order_count;
    entry.reference_slot = word();
    const uint32_t display_index = word();
    ASSERT_EQ(display_index * 2, picture.top_field_order_count);
    picture.is_reference = word();
    picture.is_intra = word();
    picture.reference_count = word();
    ASSERT_LE(picture.reference_count, 3u);
    for (uint32_t reference = 0; reference < 3; ++reference) {
      entry.reference_slots[reference] = word();
      entry.references[reference].frame_number = word();
      entry.references[reference].top_field_order_count =
          static_cast<int32_t>(word());
      entry.references[reference].bottom_field_order_count =
          entry.references[reference].top_field_order_count;
    }
    picture.sequence.maximum_reference_frame_count = maximum_references;
    picture.sequence.log2_max_frame_num_minus4 = frame_bits;
    picture.sequence.log2_max_picture_order_count_lsb_minus4 = poc_bits;
    picture.sequence.direct_8x8_inference = direct8;
    picture.picture.default_reference_l0_count_minus1 = l0;
    picture.picture.default_reference_l1_count_minus1 = l1;
    picture.picture.initial_qp_minus26 = qp;
    picture.picture.initial_qs_minus26 = qs;
    picture.picture.chroma_qp_index_offset = chroma;
    picture.picture.second_chroma_qp_index_offset = chroma;
    picture.picture.deblocking_filter_control_present = deblock;
    picture.picture.constrained_intra_prediction = constrained;
    entry.slice = {0, entry.byte_length};
    picture.slice_count = 1;
    picture.slices = &entry.slice;
    picture.references = entry.references;
    ASSERT_LE(entry.byte_length, 65536u);
    ASSERT_LE(cursor + entry.byte_length,
              reinterpret_cast<const uint8_t*>(file.data) + file.size);
    entry.bytes = cursor;
    cursor += entry.byte_length;
  }
  ASSERT_EQ(cursor, reinterpret_cast<const uint8_t*>(file.data) + file.size);
}

amdf_media_frame_t Frame(amdf_memory_t* memory, amdf_media_image_t* image,
                         uint64_t offset) {
  amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
  };
  frame.width = frame.height = 256;
  frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE;
  frame.storage.image.memory = memory;
  frame.storage.image.byte_offset = offset;
  frame.storage.image.image = image;
  return frame;
}

void CheckPixels(const uint32_t* pixels, uint32_t generation) {
  uint64_t hash = 0xcbf29ce484222325;
  for (uint32_t pixel = 0; pixel < 65536; ++pixel) {
    ASSERT_EQ(pixels[pixel] >> 24, generation) << "pixel=" << pixel;
    for (uint32_t channel = 0; channel < 3; ++channel) {
      hash = (hash ^ ((pixels[pixel] >> (channel * 8)) & 255u)) * 0x100000001b3;
    }
  }
  ASSERT_EQ(hash, kPixelHashes[generation % 12]);
  for (uint32_t pixel = 0; pixel < 4096; ++pixel) {
    const uint32_t x = generation * 26 % 128 + (pixel % 64) * 2;
    const uint32_t y = generation * 18 % 128 + (pixel / 64) * 2;
    const uint32_t source = pixels[y * 256 + x];
    const double luma = source & 255u;
    const double u = double((source >> 8) & 255u) - 128.0;
    const double v = double((source >> 16) & 255u) - 128.0;
    const double expected[] = {
        luma + 1.402 * v, luma - 0.344136 * u - 0.714136 * v, luma + 1.772 * u};
    const uint32_t result = pixels[65536 + pixel];
    ASSERT_EQ(result >> 24, 255u);
    for (uint32_t channel = 0; channel < 3; ++channel) {
      const int rounded = static_cast<int>(
          std::nearbyint(std::fmin(255.0, std::fmax(0.0, expected[channel]))));
      ASSERT_LE(std::abs(int((result >> (channel * 8)) & 255u) - rounded), 1)
          << "generation=" << generation << " pixel=" << pixel;
    }
  }
}

class MediaH264InteropTest : public GpuCommandTest {
 protected:
  MediaH264InteropTest()
      : GpuCommandTest({
            .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
            .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
            .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
            .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
            .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
            .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
        }) {}

  void SetUp() override {
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
    ASSERT_TRUE(SUCCEEDED(D3D12CreateDevice(
        adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&native))));
    ASSERT_TRUE(SUCCEEDED(native.As(&validation_)));
    h264_.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION;
    h264_.structure_size = sizeof(h264_);
    h264_.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
    h264_.level_idc = 13;
    const amdf_media_configuration_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
        .structure_size = sizeof(query),
        .next = &h264_,
        .configuration = {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                          AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 4},
    };
    configuration_.type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO;
    configuration_.structure_size = sizeof(configuration_);
    ASSERT_EQ(media_->device_query_configuration(media_device_, &query,
                                                 &configuration_),
              AMDF_STATUS_OK);
    ASSERT_EQ(configuration_.support,
              AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
    const amdf_media_kernel_queue_create_info_t queue_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_create),
        .required_capabilities =
            AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
        .maximum_pending_submission_count = 32,
    };
    ASSERT_EQ(media_->kernel_queue_create(media_device_, &queue_create,
                                          &decode_queue_),
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
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            gpu_queue_, decode_queue_, &dependency_create, &gpu_to_decode_),
        AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            decode_queue_, gpu_queue_, &dependency_create, &decode_to_gpu_),
        AMDF_STATUS_OK);
    const amdf_media_session_create_info_t session_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
        .structure_size = sizeof(session_create),
        .next = &h264_,
        .configuration = configuration_.configuration,
    };
    ASSERT_EQ(media_->session_create(decode_queue_, &session_create, &session_),
              AMDF_STATUS_OK);
  }

  void CreateSharedMemory(uint64_t byte_length, amdf_memory_t** out_memory,
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
    ASSERT_EQ(api_->memory_create(configuration_.image_heap_memory_scope,
                                  &create, &memory, nullptr),
              AMDF_STATUS_OK);
    memories_.push_back(memory);
    *out_memory = memory;
    amdf_memory_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->memory_query_info(memory, &info), AMDF_STATUS_OK);
    ASSERT_EQ(info.access_count, 2u);
    ASSERT_EQ(api_->memory_query_address(memory, 1, AMDF_MEMORY_ADDRESS_GPU,
                                         out_address),
              AMDF_STATUS_OK);
  }

  void CreateRecordings(uint32_t count) {
    const amdf_memory_device_access_t access = {
        media_device_,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal =
            configuration_.feedback_memory_profile_ordinal,
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
    ASSERT_EQ(api_->memory_map(memory, &map, &feedback_mapping_),
              AMDF_STATUS_OK);
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
          .maximum_reference_count = 3,
          .maximum_activation_count = 2,
          .feedback = {memory, 0,
                       64 + index * configuration_.feedback_byte_length,
                       configuration_.feedback_byte_length},
      };
      amdf_media_command_t* command = nullptr;
      ASSERT_EQ(
          media_->command_create(decode_queue_, &command_create, &command),
          AMDF_STATUS_OK);
      commands_.push_back(command);
    }
  }

  void SubmitGpu(const amdf_gpu_kernel_command_t& command,
                 const amdf_kernel_queue_dependency_wait_t* wait = nullptr) {
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

  void RunPictures(uint32_t frame_count, uint32_t recording_count) {
    const auto* kernel =
        kernels::media_consume::kKernels.Find(gpu_endpoint_info_);
    if (kernel == nullptr) {
      GTEST_SKIP() << "the raw-layout witness targets gfx1150";
    }
    const auto* profile = Pm4CommandProfile::Find(gpu_endpoint_info_);
    ASSERT_NE(profile, nullptr);
    ASSERT_EQ(kernel->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel->workgroup_size(), 64u);
    std::array<FixturePicture, 12> fixture = {};
    ASSERT_NO_FATAL_FAILURE(LoadFixture(&fixture));
    amdf_media_h264_scaling_matrices_t matrices;
    std::memset(&matrices, 16, sizeof(matrices));
    ASSERT_NO_FATAL_FAILURE(CreateRecordings(recording_count));
    const uint64_t output_heap_length =
        131072 + 2 * configuration_.output.byte_length;
    const uint64_t reference_heap_length =
        131072 + 4 * configuration_.reference.byte_length;
    ASSERT_NE(configuration_.reference.byte_length, 0u);
    amdf_memory_t* output_heap = nullptr;
    amdf_memory_t* reference_heap = nullptr;
    amdf_memory_t* compressed = nullptr;
    uint64_t output_address = 0, reference_address = 0, compressed_address = 0;
    ASSERT_NO_FATAL_FAILURE(
        CreateSharedMemory(output_heap_length, &output_heap, &output_address));
    ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
        reference_heap_length, &reference_heap, &reference_address));
    ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
        recording_count * 65536, &compressed, &compressed_address));
    std::array<amdf_media_frame_t, 2> outputs;
    std::array<amdf_media_frame_t, 4> references;
    std::array<amdf_media_gfx11_image_layout_t, 2> layouts = {};
    for (uint32_t index = 0; index < outputs.size(); ++index) {
      amdf_media_image_t* image = nullptr;
      const uint64_t offset = 65536 + index * configuration_.output.byte_length;
      ASSERT_NO_FATAL_FAILURE(CreateImage(
          output_heap, offset, configuration_.output.description, &image));
      outputs[index] = Frame(output_heap, image, offset);
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
    for (uint32_t index = 0; index < references.size(); ++index) {
      amdf_media_image_t* image = nullptr;
      const uint64_t offset =
          65536 + index * configuration_.reference.byte_length;
      ASSERT_NO_FATAL_FAILURE(CreateImage(reference_heap, offset,
                                          configuration_.reference.description,
                                          &image));
      references[index] = Frame(reference_heap, image, offset);
    }
    for (auto& entry : fixture) {
      entry.picture.scaling_matrices = &matrices;
      for (uint32_t index = 0; index < entry.picture.reference_count; ++index) {
        ASSERT_LT(entry.reference_slots[index], references.size());
        entry.references[index].frame =
            &references[entry.reference_slots[index]];
      }
    }
    constexpr uint64_t kResultBytes = (65536 + 4096) * 4;
    constexpr uint64_t kResultStride = kResultBytes + 128;
    GpuMemory* code = nullptr;
    GpuMemory* controls = nullptr;
    GpuMemory* source = nullptr;
    GpuMemory* results = nullptr;
    GpuMemory* heap_copy = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
        kernel->executable.byte_length, &code, AMDF_MEMORY_FLAG_HOST_COHERENT));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                     (recording_count + 3) * 4096, &controls,
                     AMDF_MEMORY_FLAG_HOST_COHERENT));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                         recording_count * 65536, &source,
                                         AMDF_MEMORY_FLAG_HOST_COHERENT));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     recording_count * kResultStride, &results,
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
      writer.Dispatch(program, 65536, 1, 1);
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
    bool reference_active[4] = {};
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
            results->device_address + index * kResultStride + 64};
        const uint32_t constants[] = {
            generation * 26 % 128,
            generation * 18 % 128,
            generation,
            static_cast<uint32_t>(layout.planes[1].byte_offset -
                                  layout.planes[0].byte_offset),
            0,
            0};
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
            .next = &entry.picture,
            .compressed = {compressed, 0, index * 65536, entry.byte_length},
            .output = &outputs[slot],
            .reference_output = &references[entry.reference_slot],
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
                            index * kResultStride;
        ASSERT_NO_FATAL_FAILURE(CheckPixels(
            reinterpret_cast<const uint32_t*>(bytes + 64), batch + index));
        for (uint32_t guard = 0; guard < 64; ++guard) {
          ASSERT_EQ(bytes[guard], 0xa5u);
          ASSERT_EQ(bytes[64 + kResultBytes + guard], 0xa5u);
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
            guard >=
                64 + recording_count * configuration_.feedback_byte_length) {
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

  void CreateImage(amdf_memory_t* memory, uint64_t offset,
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

  void TearDown() override {
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

  // Optional media table negotiated independently of compute.
  const amdf_media_api_t* media_ = nullptr;
  // Case-owned media activation on the compute endpoint.
  amdf_device_t* media_device_ = nullptr;
  // Exact native admission and storage requirements.
  amdf_media_configuration_info_t configuration_ = {};
  // Parsed stream envelope supplied independently of the provider.
  amdf_media_h264_configuration_t h264_ = {};
  // Case-owned native media ordering domain.
  amdf_kernel_queue_t* decode_queue_ = nullptr;
  // Separate native GPU KMQ consuming the same shared bytes.
  amdf_kernel_queue_t* gpu_queue_ = nullptr;
  // Cold producer bindings in both directions.
  amdf_kernel_queue_dependency_t* gpu_to_decode_ = nullptr;
  // Cold decode-to-GPU binding, retained through every accepted consumer.
  amdf_kernel_queue_dependency_t* decode_to_gpu_ = nullptr;
  // Explicit codec state without an image pool.
  amdf_media_session_t* session_ = nullptr;
  // Last accepted media prefix, retained through failure and teardown.
  uint64_t decode_point_ = 0;
  // Last accepted GPU prefix, retained through failure and teardown.
  uint64_t gpu_point_ = 0;
  // Application-owned backing destroyed after its identities and last uses.
  std::vector<amdf_memory_t*> memories_;
  // Explicit placed identities, destroyed before backing.
  std::vector<amdf_media_image_t*> images_;
  // Fixed recording owners, destroyed before their session.
  std::vector<amdf_media_command_t*> commands_;
  // Explicit host view of caller-owned feedback.
  amdf_host_mapping_t* feedback_mapping_ = nullptr;
  // Direct host pointer borrowed from the explicit feedback mapping.
  uint8_t* feedback_ = nullptr;
  // Native debug view kept through complete media teardown.
  ComPtr<ID3D12InfoQueue> validation_;
};

TEST_F(MediaH264InteropTest, PlacesImagesInCoreMemoryAndBindsNativeQueues) {
  amdf_memory_t* memory = nullptr;
  uint64_t address = 0;
  ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
      2 * 65536 + 2 * configuration_.output.byte_length, &memory, &address));
  amdf_media_image_t* image = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateImage(memory, 65536, configuration_.output.description, &image));
  amdf_media_gfx11_image_layout_t layout = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT,
      .structure_size = sizeof(layout),
  };
  amdf_media_image_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
      .structure_size = sizeof(info),
      .next = &layout,
  };
  const amdf_media_image_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
      .structure_size = sizeof(query),
  };
  ASSERT_EQ(media_->image_query_info(image, 1, &query, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.memory, memory);
  EXPECT_EQ(info.byte_offset, 65536u);
  EXPECT_EQ(layout.plane_count, 2u);
  amdf_kernel_queue_dependency_info_t dependency = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_INFO,
      .structure_size = sizeof(dependency),
  };
  ASSERT_EQ(
      api_->kernel_queue_dependency_query_info(decode_to_gpu_, &dependency),
      AMDF_STATUS_OK);
  EXPECT_EQ(dependency.producer_signal_operation_count, 0u);
  EXPECT_EQ(dependency.consumer_wait_operation_count, 1u);
  EXPECT_EQ(api_->device_destroy(media_device_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
}

TEST_F(MediaH264InteropTest, DecodesReorderedReferencesAndConsumesOnGpu) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(12, 12));
}

TEST_F(MediaH264InteropTest, AdmissionPreservesTheRequestedCodecEnvelope) {
  amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = &h264_,
      .configuration = configuration_.configuration,
  };
  auto info = configuration_;
  query.configuration.coded_height = 258;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS);
  EXPECT_EQ(info.configuration.coded_height, 258u);
  EXPECT_EQ(info.output.byte_length, 0u);

  query.configuration = configuration_.configuration;
  query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_FORMAT);
  EXPECT_EQ(info.configuration.format, AMDF_MEDIA_PIXEL_FORMAT_P010);

  query.configuration = configuration_.configuration;
  query.configuration.direction = AMDF_MEDIA_DIRECTION_ENCODE;
  ASSERT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
  EXPECT_EQ(info.reason, AMDF_MEDIA_CONFIGURATION_REASON_CODEC);
  EXPECT_EQ(info.configuration.direction, AMDF_MEDIA_DIRECTION_ENCODE);

  // A malformed schema fails the call instead of answering admission. The
  // caller's preceding result remains intact when validation fails.
  const auto unchanged = info;
  query.configuration = configuration_.configuration;
  query.next = nullptr;
  EXPECT_EQ(media_->device_query_configuration(media_device_, &query, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  EXPECT_EQ(std::memcmp(&info, &unchanged, sizeof(info)), 0);
}

TEST_F(MediaH264InteropTest, PlansArrayPlacementBeforeAllocatingItsBacking) {
  for (auto format :
       {AMDF_MEDIA_PIXEL_FORMAT_NV12, AMDF_MEDIA_PIXEL_FORMAT_P010}) {
    const amdf_media_image_description_t description = {
        static_cast<amdf_media_pixel_format_t>(format), 256, 256, 2,
        AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT};
    amdf_media_image_requirements_t requirements = {};
    ASSERT_EQ(media_->device_query_image_requirements(
                  media_device_, &description, &requirements),
              AMDF_STATUS_OK);
    ASSERT_NE(requirements.byte_length, 0u);
    const uint64_t offset = requirements.byte_offset_alignment;
    amdf_memory_t* memory = nullptr;
    uint64_t address = 0;
    ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
        offset + requirements.byte_length, &memory, &address));
    amdf_media_image_t* image = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateImage(memory, offset, description, &image));
    for (uint32_t layer = 0; layer < 2; ++layer) {
      amdf_media_gfx11_image_layout_t layout = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT,
          .structure_size = sizeof(layout),
      };
      amdf_media_image_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
          .structure_size = sizeof(info),
          .next = &layout,
      };
      const amdf_media_image_query_t query = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
          .structure_size = sizeof(query),
          .array_layer = layer,
      };
      ASSERT_EQ(media_->image_query_info(image, 1, &query, &info),
                AMDF_STATUS_OK);
      EXPECT_EQ(info.requirements.byte_length, requirements.byte_length);
      EXPECT_EQ(info.byte_offset, offset);
      EXPECT_EQ(layout.array_layer, layer);
      EXPECT_EQ(layout.plane_count, 2u);
    }
    // The first consumer layout is realized before the duplicate is rejected.
    // Failure must release that native identity without publishing an owner.
    const uint32_t duplicate_accesses[] = {1, 1};
    const amdf_media_image_create_info_t invalid = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
        .structure_size = sizeof(invalid),
        .memory = memory,
        .description = description,
        .byte_offset = offset,
        .layout_access_count = 2,
        .layout_access_ordinals = duplicate_accesses,
    };
    amdf_media_image_t* unchanged = image;
    EXPECT_EQ(media_->image_create(media_device_, &invalid, &unchanged),
              amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
    EXPECT_EQ(unchanged, image);
  }
}

TEST_F(MediaH264InteropTest, ReusesFourRecordingsAcrossThirtyTwoPictures) {
  ASSERT_NO_FATAL_FAILURE(RunPictures(32, 4));
}

}  // namespace
