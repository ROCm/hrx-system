// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/interop/gpu/media/encode_fixture.h"

#if defined(_WIN32)
#include <dxgi1_4.h>
#endif  // _WIN32

#include <algorithm>
#include <array>
#include <cstring>

#include "amdf/media/av1.h"
#include "amdf/media/gfx11.h"
#include "amdf/media/h264.h"
#include "amdf/media/hevc.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/interop/gpu/media/produce.h"

#if defined(_WIN32)
using Microsoft::WRL::ComPtr;
#endif  // _WIN32

namespace {

template <typename T>
void Initialize(T* value, amdf_structure_type_t type) {
  value->type = type;
  value->structure_size = sizeof(*value);
}

uint64_t AlignUp(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

amdf_media_frame_t Frame(amdf_memory_t* memory, amdf_media_image_t* image,
                         uint64_t offset,
                         const amdf_media_image_description_t& description,
                         uint32_t array_layer = 0) {
  amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = description.format,
      .width = description.width,
      .height = description.height,
      .storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE,
  };
  frame.storage.image.memory = memory;
  frame.storage.image.byte_offset = offset;
  frame.storage.image.image = image;
  frame.storage.image.array_layer = array_layer;
  return frame;
}

struct ConfigurationChain {
  // Direction-common GOP and rate-control configuration.
  amdf_media_encode_configuration_t encode = {};
  // H264 profile and level at the head of an H264 chain.
  amdf_media_h264_configuration_t h264 = {};
  // H264 ordering geometry following the common configuration.
  amdf_media_h264_encode_configuration_t h264_encode = {};
  // HEVC profile and precision at the head of an HEVC chain.
  amdf_media_hevc_configuration_t hevc = {};
  // HEVC coding geometry following the common configuration.
  amdf_media_hevc_encode_configuration_t hevc_encode = {};
  // AV1 profile and precision at the head of an AV1 chain.
  amdf_media_av1_configuration_t av1 = {};
  // AV1 sequence tools following the common configuration.
  amdf_media_av1_encode_configuration_t av1_encode = {};

  const void* InitializeFor(
      const EncodeExecutionCase& test_case,
      amdf_media_encode_rate_control_mode_t rate_control_mode) {
    Initialize(&encode, AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_CONFIGURATION);
    encode.intra_picture_period = 12;
    encode.inter_picture_period = 1;
    encode.rate_control.mode = rate_control_mode;
    encode.rate_control.frame_rate_numerator = 30;
    encode.rate_control.frame_rate_denominator = 1;
    if (rate_control_mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CQP) {
      encode.rate_control.control.cqp = {20, 20};
    } else {
      encode.rate_control.control.bit_rate = {
          .minimum_quantization_parameter =
              test_case.codec == AMDF_MEDIA_CODEC_AV1 ? 8u : 1u,
          .maximum_quantization_parameter =
              test_case.codec == AMDF_MEDIA_CODEC_AV1 ? 255u : 51u,
          .target_bit_rate = 1200000,
          .peak_bit_rate =
              rate_control_mode == AMDF_MEDIA_ENCODE_RATE_CONTROL_MODE_CBR
                  ? 1200000u
                  : 2400000u,
          .virtual_buffer_capacity = 1200000,
          .initial_virtual_buffer_fullness = 600000,
      };
    }
    if (test_case.codec == AMDF_MEDIA_CODEC_H264) {
      Initialize(&h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION);
      Initialize(&h264_encode,
                 AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_CONFIGURATION);
      h264.next = &encode;
      h264.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
      h264.level_idc = 30;
      encode.next = &h264_encode;
      h264_encode.picture_order_count_type = 2;
      return &h264;
    }
    if (test_case.codec == AMDF_MEDIA_CODEC_HEVC) {
      Initialize(&hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_CONFIGURATION);
      Initialize(&hevc_encode,
                 AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_CONFIGURATION);
      hevc.next = &encode;
      hevc.profile = test_case.sample_bit_depth == 10
                         ? AMDF_MEDIA_HEVC_PROFILE_MAIN10
                         : AMDF_MEDIA_HEVC_PROFILE_MAIN;
      hevc.sample_bit_depth = test_case.sample_bit_depth;
      encode.next = &hevc_encode;
      hevc_encode.level_idc = 90;
      hevc_encode.tier = AMDF_MEDIA_HEVC_TIER_MAIN;
      hevc_encode.minimum_luma_coding_unit_log2 = 3;
      hevc_encode.maximum_luma_coding_unit_log2 = 6;
      hevc_encode.minimum_luma_transform_unit_log2 = 2;
      hevc_encode.maximum_luma_transform_unit_log2 = 5;
      hevc_encode.maximum_transform_hierarchy_depth_inter = 4;
      hevc_encode.maximum_transform_hierarchy_depth_intra = 4;
      return &hevc;
    }
    Initialize(&av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_CONFIGURATION);
    Initialize(&av1_encode, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_CONFIGURATION);
    av1.next = &encode;
    av1.profile = 0;
    av1.sample_bit_depth = test_case.sample_bit_depth;
    encode.next = &av1_encode;
    av1_encode.level = 0;
    av1_encode.tier = AMDF_MEDIA_AV1_TIER_MAIN;
    av1_encode.feature_flags = AMDF_MEDIA_AV1_ENCODE_FEATURE_ORDER_HINT |
                               AMDF_MEDIA_AV1_ENCODE_FEATURE_CDEF;
    av1_encode.order_hint_bit_count = 4;
    return &av1;
  }
};

struct PictureControls {
  // H264 reference storage used by h264 when encoding a P picture.
  amdf_media_h264_encode_reference_t h264_reference = {};
  // Complete H264 picture controls.
  amdf_media_h264_encode_picture_info_t h264 = {};
  // HEVC reference storage used by hevc when encoding a P picture.
  amdf_media_hevc_encode_reference_t hevc_reference = {};
  // Complete HEVC picture controls.
  amdf_media_hevc_encode_picture_info_t hevc = {};
  // AV1 reference storage used by av1 when encoding an inter picture.
  amdf_media_av1_encode_reference_t av1_reference = {};
  // Complete AV1 picture controls.
  amdf_media_av1_encode_picture_info_t av1 = {};

  const void* InitializeFor(const EncodeExecutionCase& test_case,
                            uint32_t generation,
                            const amdf_media_frame_t* reference,
                            uint32_t reference_slot) {
    const bool key = generation == 0;
    if (test_case.codec == AMDF_MEDIA_CODEC_H264) {
      Initialize(&h264, AMDF_STRUCTURE_TYPE_MEDIA_H264_ENCODE_PICTURE_INFO);
      h264.frame_type = key ? AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR
                            : AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
      h264.frame_number = generation;
      h264.picture_order_count = static_cast<int32_t>(generation * 2);
      h264.frame_decoding_order_number = generation;
      h264.used_as_reference = 1;
      if (!key) {
        h264_reference.binding = {reference, reference_slot};
        h264_reference.frame_type = generation == 1
                                        ? AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_IDR
                                        : AMDF_MEDIA_H264_ENCODE_FRAME_TYPE_P;
        h264_reference.frame_number = generation - 1;
        h264_reference.picture_order_count =
            static_cast<int32_t>((generation - 1) * 2);
        h264_reference.frame_decoding_order_number = generation - 1;
        h264.reference_count = 1;
        h264.references = &h264_reference;
      }
      return &h264;
    }
    if (test_case.codec == AMDF_MEDIA_CODEC_HEVC) {
      Initialize(&hevc, AMDF_STRUCTURE_TYPE_MEDIA_HEVC_ENCODE_PICTURE_INFO);
      hevc.frame_type = key ? AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_IDR
                            : AMDF_MEDIA_HEVC_ENCODE_FRAME_TYPE_P;
      hevc.picture_order_count = static_cast<int32_t>(generation);
      hevc.used_as_reference = 1;
      if (!key) {
        hevc_reference.binding = {reference, reference_slot};
        hevc_reference.picture_order_count =
            static_cast<int32_t>(generation - 1);
        hevc.reference_count = 1;
        hevc.references = &hevc_reference;
      }
      return &hevc;
    }
    Initialize(&av1, AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_PICTURE_INFO);
    av1.frame_type = key ? 0u : 1u;
    av1.order_hint = generation;
    av1.picture_index = generation;
    av1.primary_reference = key ? 7u : 0u;
    av1.refresh_frame_flags = 255;
    av1.interpolation_filter = 4;
    av1.transform_mode = key ? 1u : 2u;
    av1.quantization.base_index = 20;
    std::fill(std::begin(av1.quantization.matrix_levels),
              std::end(av1.quantization.matrix_levels), UINT8_MAX);
    std::fill(std::begin(av1.reference_indices),
              std::end(av1.reference_indices), key ? UINT8_MAX : uint8_t{0});
    if (!key) {
      av1_reference.binding = {reference, reference_slot};
      av1_reference.frame_type = generation == 1 ? 0u : 1u;
      av1_reference.order_hint = generation - 1;
      av1_reference.picture_index = generation - 1;
      av1.reference_count = 1;
      av1.references = &av1_reference;
    }
    return &av1;
  }
};

}  // namespace

MediaEncodeInteropTest::MediaEncodeInteropTest()
    : GpuCommandTest({
          .command_type = AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
          .roles = AMDF_QUEUE_ROLE_TRANSFER | AMDF_QUEUE_ROLE_CACHE_CONTROL,
          .format_features = AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR,
          .cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                              AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM,
          .cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
          .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                               AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
      }) {}

void MediaEncodeInteropTest::SetUp() {
#if defined(_WIN32)
  ComPtr<ID3D12Debug> debug;
  if (FAILED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
    GTEST_SKIP() << "D3D12 media validation requires Windows Graphics Tools";
  }
  debug->EnableDebugLayer();
#endif  // _WIN32
  GpuCommandTest::SetUp();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const void* extension = nullptr;
  const auto query_status = api_->query_extension(
      AMDF_EXTENSION_MEDIA, AMDF_MEDIA_EXTENSION_VERSION_1,
      AMDF_MEDIA_EXTENSION_VERSION_LATEST, &extension);
  if (query_status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "media encode extension is not enabled";
  }
  ASSERT_EQ(query_status, AMDF_STATUS_OK);
  media_ = static_cast<const amdf_media_api_t*>(extension);
  const amdf_media_device_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  ASSERT_EQ(media_->device_create(endpoint_, &create, &media_device_),
            AMDF_STATUS_OK);

#if defined(_WIN32)
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
#endif  // _WIN32

  amdf_device_info_t device_info = {
      .type = AMDF_STRUCTURE_TYPE_DEVICE_INFO,
      .structure_size = sizeof(device_info),
  };
  ASSERT_EQ(api_->device_query_info(media_device_, &device_info),
            AMDF_STATUS_OK);
  encode_family_ = device_info.queue_family_count;
  decode_family_ = device_info.queue_family_count;
  for (uint32_t ordinal = 0; ordinal < device_info.queue_family_count;
       ++ordinal) {
    uint32_t count = 0;
    ASSERT_EQ(media_->device_enumerate_codecs(media_device_, ordinal, 0,
                                              nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_media_codec_info_t> codecs(count);
    ASSERT_EQ(media_->device_enumerate_codecs(media_device_, ordinal, count,
                                              codecs.data(), &count),
              AMDF_STATUS_OK);
    const auto supports = [&](amdf_media_codec_t codec,
                              amdf_media_direction_t direction) {
      return std::any_of(codecs.begin(), codecs.end(), [&](const auto& entry) {
        return entry.codec == codec && entry.direction == direction;
      });
    };
    if (encode_family_ == device_info.queue_family_count &&
        (supports(AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_ENCODE) ||
         supports(AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_ENCODE) ||
         supports(AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_ENCODE))) {
      encode_family_ = ordinal;
      for (const auto& codec : codecs) {
        if (codec.direction == AMDF_MEDIA_DIRECTION_ENCODE) {
          encode_codecs_.push_back(codec.codec);
        }
      }
    }
    if (decode_family_ == device_info.queue_family_count &&
        supports(AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE) &&
        supports(AMDF_MEDIA_CODEC_HEVC, AMDF_MEDIA_DIRECTION_DECODE) &&
        supports(AMDF_MEDIA_CODEC_AV1, AMDF_MEDIA_DIRECTION_DECODE)) {
      decode_family_ = ordinal;
    }
  }
  if (encode_family_ == device_info.queue_family_count) {
    GTEST_SKIP() << "no native encode family";
  }
  if (decode_family_ == device_info.queue_family_count) {
    GTEST_SKIP() << "no complete native decode family";
  }
  const amdf_media_kernel_queue_create_info_t encode_queue_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(encode_queue_create),
      .queue_family_ordinal = encode_family_,
      .required_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
      .maximum_pending_submission_count = 32,
  };
  ASSERT_EQ(media_->kernel_queue_create(media_device_, &encode_queue_create,
                                        &encode_queue_),
            AMDF_STATUS_OK);
  const amdf_media_kernel_queue_create_info_t decode_queue_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(decode_queue_create),
      .queue_family_ordinal = decode_family_,
      .required_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
      .maximum_pending_submission_count = 4,
  };
  ASSERT_EQ(media_->kernel_queue_create(media_device_, &decode_queue_create,
                                        &decode_queue_),
            AMDF_STATUS_OK);
#if defined(_WIN32)
  const amdf_gpu_kernel_queue_create_info_t gpu_queue_create = {
      .type = AMDF_STRUCTURE_TYPE_GPU_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(gpu_queue_create),
      .queue_family_ordinal = family_.ordinal,
      .maximum_pending_submission_count = 32,
      .required_capabilities =
          AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
  };
  ASSERT_EQ(
      gpu_api_->kernel_queue_create(device_, &gpu_queue_create, &gpu_queue_),
      AMDF_STATUS_OK);
#else
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&gpu_command_queue_, 65536));
  ASSERT_NE(gpu_command_queue_, nullptr);
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096,
                   &gpu_completion_));
  std::memset(gpu_completion_->host.pointer, 0,
              gpu_completion_->info.byte_length);
#endif  // _WIN32
  const amdf_kernel_queue_dependency_create_info_t dependency_create = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_CREATE_INFO,
      .structure_size = sizeof(dependency_create),
  };
  if (gpu_queue_ != nullptr) {
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            gpu_queue_, encode_queue_, &dependency_create, &gpu_to_encode_),
        AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            encode_queue_, gpu_queue_, &dependency_create, &encode_to_gpu_),
        AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            gpu_queue_, decode_queue_, &dependency_create, &gpu_to_decode_),
        AMDF_STATUS_OK);
  }
  ASSERT_EQ(
      api_->kernel_queue_dependency_create(
          encode_queue_, decode_queue_, &dependency_create, &encode_to_decode_),
      AMDF_STATUS_OK);
  if (gpu_queue_ != nullptr) {
    ASSERT_EQ(
        api_->kernel_queue_dependency_create(
            decode_queue_, gpu_queue_, &dependency_create, &decode_to_gpu_),
        AMDF_STATUS_OK);
  }
}

void MediaEncodeInteropTest::CreateSharedMemory(amdf_memory_scope_t* scope,
                                                uint32_t profile_ordinal,
                                                uint64_t byte_length,
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
                scope, profile_ordinal, 2, accesses, &profile, capabilities),
            AMDF_STATUS_OK);
  ASSERT_EQ(profile.availability, AMDF_STATUS_OK);
  ASSERT_NE(profile.roles & AMDF_MEMORY_PROFILE_ROLE_CREATE, 0u);
  ASSERT_NE(profile.allocation.byte_length_granularity, 0u);
  for (uint32_t index = 0; index < std::size(accesses); ++index) {
    ASSERT_EQ(accesses[index].requirements.access &
                  ~capabilities[index].supported_access,
              0u);
    ASSERT_EQ(accesses[index].requirements.flags &
                  ~capabilities[index].supported_flags,
              0u);
  }
  const uint64_t allocation_length =
      AlignUp(byte_length, profile.allocation.byte_length_granularity);
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal = profile_ordinal,
      .access_count = 2,
      .required_flags = profile.supported_flags & AMDF_MEMORY_FLAG_SHAREABLE
                            ? AMDF_MEMORY_FLAG_SHAREABLE
                            : 0u,
      .byte_length = allocation_length,
      .accesses = accesses,
  };
  amdf_memory_t* memory = nullptr;
  ASSERT_EQ(api_->memory_create(scope, &create, &memory, nullptr),
            AMDF_STATUS_OK);
  memories_.push_back(memory);
  *out_memory = memory;
  ASSERT_EQ(api_->memory_query_address(memory, 1, AMDF_MEMORY_ADDRESS_GPU,
                                       out_address),
            AMDF_STATUS_OK);
}

void MediaEncodeInteropTest::SubmitGpu(
    const amdf_gpu_kernel_command_t& command,
    const amdf_kernel_queue_dependency_wait_t* wait) {
  if (gpu_queue_ == nullptr) {
    ASSERT_EQ(wait, nullptr);
    uint64_t command_address = 0;
    ASSERT_EQ(api_->memory_query_address(
                  command.memory, 0, AMDF_MEMORY_ADDRESS_GPU, &command_address),
              AMDF_STATUS_OK);
    ASSERT_EQ(command.byte_offset % sizeof(uint32_t), 0u);
    ASSERT_EQ(command.byte_length % sizeof(uint32_t), 0u);
    const auto* profile = Pm4CommandProfile::Find(gpu_endpoint_info_);
    ASSERT_NE(profile, nullptr);
    Pm4CommandWriter writer(
        gpu_command_queue_->words().data() + gpu_command_word_count_, *profile);
    writer.CallIndirectBuffer(command_address + command.byte_offset,
                              static_cast<uint32_t>(command.byte_length / 4));
    ++gpu_point_;
    writer.ReleaseSystem32(gpu_completion_->device_address,
                           static_cast<uint32_t>(gpu_point_));
    writer.PadToEightWords();
    gpu_command_word_count_ += writer.word_count();
    ASSERT_NO_FATAL_FAILURE(
        gpu_command_queue_->Publish(api_, gpu_api_, gpu_command_word_count_));
    ASSERT_TRUE(GpuWaitEqual<uint32_t>(
        reinterpret_cast<uintptr_t>(gpu_completion_->host.pointer),
        static_cast<uint32_t>(gpu_point_)));
    ASSERT_NO_FATAL_FAILURE(gpu_command_queue_->WaitRetired(api_));
    return;
  }
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

void MediaEncodeInteropTest::SubmitMediaControl(
    amdf_kernel_queue_t* queue, amdf_media_command_t* command,
    amdf_media_command_operation_t operation, uint64_t* out_point) {
  const amdf_media_command_prepare_info_t prepare = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      .structure_size = sizeof(prepare),
      .operation = operation,
  };
  ASSERT_EQ(media_->command_prepare(command, &prepare), AMDF_STATUS_OK);
  const amdf_media_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command = command,
  };
  uint64_t point = 0;
  ASSERT_EQ(media_->kernel_queue_submit(queue, &submit, &point),
            AMDF_STATUS_OK);
  ASSERT_NE(point, 0u);
  *out_point = point;
  ASSERT_EQ(api_->kernel_queue_wait(queue, point, AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
  amdf_media_command_result_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
      .structure_size = sizeof(result),
  };
  ASSERT_EQ(media_->command_query_result(command, &result), AMDF_STATUS_OK);
  ASSERT_EQ(result.operation, operation);
  ASSERT_EQ(result.submission, point);
  ASSERT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE);
}

void MediaEncodeInteropTest::QueryKernelImageLayout(
    amdf_media_image_t* image, uint32_t array_layer,
    MediaKernelImageLayout* out_layout) {
  *out_layout = {};
  const amdf_media_image_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
      .structure_size = sizeof(query),
      .array_layer = array_layer,
  };
  amdf_media_image_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
      .structure_size = sizeof(info),
  };
#if defined(_WIN32)
  amdf_media_gfx11_image_layout_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_GFX11_IMAGE_LAYOUT,
      .structure_size = sizeof(native),
  };
  info.next = &native;
  ASSERT_EQ(media_->image_query_info(image, 1, &query, &info), AMDF_STATUS_OK);
  ASSERT_EQ(native.plane_count, 2u);
  for (uint32_t plane = 0; plane < native.plane_count; ++plane) {
    ASSERT_EQ(native.planes[plane].swizzle_mode, 26u);
    out_layout->byte_offsets[plane] = native.planes[plane].byte_offset;
    out_layout->pipe_bank_xors[plane] = native.planes[plane].pipe_bank_xor;
  }
#else
  amdf_media_linear_image_layout_t native = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT,
      .structure_size = sizeof(native),
  };
  info.next = &native;
  ASSERT_EQ(media_->image_query_info(image, 1, &query, &info), AMDF_STATUS_OK);
  ASSERT_EQ(native.plane_count, 2u);
  ASSERT_EQ(native.array_layer, array_layer);
  out_layout->linear = 1;
  for (uint32_t plane = 0; plane < native.plane_count; ++plane) {
    ASSERT_LE(native.planes[plane].row_pitch, UINT32_MAX);
    out_layout->byte_offsets[plane] = native.planes[plane].byte_offset;
    out_layout->row_pitches[plane] =
        static_cast<uint32_t>(native.planes[plane].row_pitch);
  }
#endif  // _WIN32
}

void MediaEncodeInteropTest::RunPictures(
    const EncodeExecutionCase& test_case,
    amdf_media_encode_rate_control_mode_t rate_control_mode, uint32_t width,
    uint32_t height, uint32_t frame_count, uint32_t recording_count) {
  if (std::find(encode_codecs_.begin(), encode_codecs_.end(),
                test_case.codec) == encode_codecs_.end()) {
    GTEST_SKIP() << "encode codec is not advertised";
  }
  ASSERT_GT(frame_count, 0u);
  ASSERT_GT(recording_count, 0u);
  ASSERT_LE(recording_count, frame_count);
  ASSERT_LE(frame_count, 12u);
  const auto* kernel =
      kernels::media_produce::kKernels.Find(gpu_endpoint_info_);
  if (kernel == nullptr) {
    GTEST_SKIP() << "no media image kernel for this GPU";
  }
  const auto* profile = Pm4CommandProfile::Find(gpu_endpoint_info_);
  ASSERT_NE(profile, nullptr);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 64u);

  ConfigurationChain configuration_chain;
  const void* codec_configuration =
      configuration_chain.InitializeFor(test_case, rate_control_mode);
  const amdf_media_configuration_query_t configuration_query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(configuration_query),
      .next = codec_configuration,
      .queue_family_ordinal = encode_family_,
      .configuration =
          {
              .codec = test_case.codec,
              .direction = AMDF_MEDIA_DIRECTION_ENCODE,
              .format = test_case.format,
              .coded_width = width,
              .coded_height = height,
              .reference_image_count = 2,
          },
  };
  amdf_media_configuration_info_t configuration = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
      .structure_size = sizeof(configuration),
  };
  ASSERT_EQ(media_->device_query_configuration(
                media_device_, &configuration_query, &configuration),
            AMDF_STATUS_OK);
  ASSERT_EQ(configuration.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED);
  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = codec_configuration,
      .configuration = configuration.configuration,
  };
  ASSERT_EQ(
      media_->session_create(encode_queue_, &session_create, &encode_session_),
      AMDF_STATUS_OK);

  const uint64_t feedback_start =
      AlignUp(64, configuration.feedback_byte_offset_alignment);
  const uint64_t feedback_stride =
      AlignUp(configuration.feedback_byte_length,
              configuration.feedback_byte_offset_alignment);
  const uint64_t feedback_length =
      feedback_start + recording_count * feedback_stride + 64;
  const amdf_memory_device_access_t feedback_access = {
      media_device_,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
  const amdf_memory_create_info_t feedback_create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(feedback_create),
      .memory_profile_ordinal = configuration.feedback_memory_profile_ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = feedback_length,
      .minimum_alignment = configuration.feedback_byte_offset_alignment,
      .accesses = &feedback_access,
  };
  amdf_memory_t* feedback_memory = nullptr;
  ASSERT_EQ(api_->memory_create(configuration.feedback_memory_scope,
                                &feedback_create, &feedback_memory, nullptr),
            AMDF_STATUS_OK);
  memories_.push_back(feedback_memory);
  const amdf_memory_map_info_t feedback_map = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_MAP_INFO,
      .structure_size = sizeof(feedback_map),
      .byte_length = feedback_length,
      .flags = AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE,
  };
  ASSERT_EQ(api_->memory_map(feedback_memory, &feedback_map,
                             &encode_feedback_mapping_),
            AMDF_STATUS_OK);
  amdf_host_mapping_info_t feedback_mapping = {
      .type = AMDF_STRUCTURE_TYPE_HOST_MAPPING_INFO,
      .structure_size = sizeof(feedback_mapping),
  };
  ASSERT_EQ(api_->host_mapping_query_info(encode_feedback_mapping_,
                                          &feedback_mapping),
            AMDF_STATUS_OK);
  encode_feedback_ = static_cast<uint8_t*>(feedback_mapping.pointer);
  std::memset(encode_feedback_, 0xa5, feedback_length);
  for (uint32_t index = 0; index < recording_count; ++index) {
    const amdf_media_command_create_info_t command_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
        .structure_size = sizeof(command_create),
        .maximum_slice_count = 1,
        .maximum_reference_count = 1,
        .maximum_activation_count = 2,
        .feedback = {feedback_memory, 0,
                     feedback_start + index * feedback_stride,
                     configuration.feedback_byte_length},
    };
    amdf_media_command_t* command = nullptr;
    ASSERT_EQ(
        media_->command_create(encode_session_, &command_create, &command),
        AMDF_STATUS_OK);
    commands_.push_back(command);
  }
  if (configuration.requires_begin_end != 0) {
    ASSERT_NO_FATAL_FAILURE(
        SubmitMediaControl(encode_queue_, commands_[0],
                           AMDF_MEDIA_COMMAND_OPERATION_BEGIN, &encode_point_));
    encode_session_active_ = true;
  }

  const uint64_t input_start =
      AlignUp(65536, configuration.input.byte_offset_alignment);
  const uint64_t input_stride =
      AlignUp(configuration.input.byte_length,
              configuration.input.byte_offset_alignment);
  const uint64_t input_heap_length = input_start + 2 * input_stride + 65536;
  const uint64_t reconstruction_start =
      AlignUp(65536, configuration.reference.byte_offset_alignment);
  const uint64_t reconstruction_heap_length =
      reconstruction_start + configuration.reference.byte_length + 65536;
  amdf_memory_t* input_memory = nullptr;
  amdf_memory_t* reconstruction_memory = nullptr;
  uint64_t input_address = 0;
  uint64_t reconstruction_address = 0;
  ASSERT_NO_FATAL_FAILURE(
      CreateSharedMemory(configuration.image_heap_memory_scope,
                         configuration.image_heap_memory_profile_ordinal,
                         input_heap_length, &input_memory, &input_address));
  ASSERT_NO_FATAL_FAILURE(
      CreateSharedMemory(configuration.image_heap_memory_scope,
                         configuration.image_heap_memory_profile_ordinal,
                         reconstruction_heap_length, &reconstruction_memory,
                         &reconstruction_address));

  std::array<amdf_media_frame_t, 2> input_frames = {};
  std::array<MediaKernelImageLayout, 2> input_layouts = {};
  std::array<amdf_media_image_t*, 2> input_images = {};
  const uint32_t gpu_access_ordinal = 1;
  for (uint32_t index = 0; index < input_frames.size(); ++index) {
    const uint64_t offset = input_start + index * input_stride;
    const amdf_media_image_create_info_t image_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
        .structure_size = sizeof(image_create),
        .memory = input_memory,
        .requirements = configuration.input,
        .byte_offset = offset,
        .layout_access_count = 1,
        .layout_access_ordinals = &gpu_access_ordinal,
    };
    ASSERT_EQ(media_->image_create(media_device_, &image_create,
                                   &input_images[index]),
              AMDF_STATUS_OK);
    images_.push_back(input_images[index]);
    input_frames[index] = Frame(input_memory, input_images[index], offset,
                                configuration.input.description);
    ASSERT_NO_FATAL_FAILURE(
        QueryKernelImageLayout(input_images[index], 0, &input_layouts[index]));
  }
  amdf_media_image_t* reconstruction_image = nullptr;
  const amdf_media_image_create_info_t reconstruction_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      .structure_size = sizeof(reconstruction_create),
      .memory = reconstruction_memory,
      .requirements = configuration.reference,
      .byte_offset = reconstruction_start,
  };
  ASSERT_EQ(media_->image_create(media_device_, &reconstruction_create,
                                 &reconstruction_image),
            AMDF_STATUS_OK);
  images_.push_back(reconstruction_image);
  std::array<amdf_media_frame_t, 2> reconstructions = {
      Frame(reconstruction_memory, reconstruction_image, reconstruction_start,
            configuration.reference.description, 0),
      Frame(reconstruction_memory, reconstruction_image, reconstruction_start,
            configuration.reference.description, 1),
  };

  const uint64_t compressed_start =
      AlignUp(65536, configuration.compressed_byte_offset_alignment);
  const uint64_t compressed_stride =
      AlignUp(configuration.minimum_compressed_byte_length,
              configuration.compressed_byte_length_alignment);
  const uint64_t compressed_length =
      compressed_start + recording_count * compressed_stride + 65536;
  amdf_memory_t* compressed_memory = nullptr;
  uint64_t compressed_address = 0;
  ASSERT_NO_FATAL_FAILURE(CreateSharedMemory(
      configuration.compressed_memory_scope,
      configuration.compressed_memory_profile_ordinal, compressed_length,
      &compressed_memory, &compressed_address));

  GpuMemory* code = nullptr;
  GpuMemory* controls = nullptr;
  GpuMemory* captured = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      kernel->executable.byte_length, &code, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
      (recording_count + 2) * 4096, &controls, AMDF_MEMORY_FLAG_HOST_COHERENT));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, compressed_length,
      &captured, AMDF_MEMORY_FLAG_HOST_COHERENT));
  std::memcpy(code->host.pointer, kernel->executable.words,
              kernel->executable.byte_length);
  auto* control_bytes = static_cast<uint8_t*>(controls->host.pointer);
  std::memset(control_bytes, 0, controls->info.byte_length);
  std::memset(captured->host.pointer, 0xa5, captured->info.byte_length);
  const Pm4ComputeProgram program = {
      code->device_address + kernel->entry_byte_offset,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {64, 1, 1},
  };
  std::vector<amdf_gpu_kernel_command_t> producers(recording_count);
  std::vector<std::vector<uint8_t>> immutable_commands(recording_count);
  for (uint32_t index = 0; index < recording_count; ++index) {
    Pm4CommandWriter writer(
        reinterpret_cast<uint32_t*>(control_bytes + index * 4096 + 256),
        *profile);
    writer.SystemBarrier();
    writer.BindCompute(program, controls->device_address + index * 4096,
                       family_.format_features);
    writer.Dispatch(program, width * height, 1, 1);
    writer.SystemBarrier();
    writer.PadToEightWords();
    producers[index] = {
        .memory = controls->memory,
        .byte_offset = index * 4096 + 256,
        .byte_length = writer.word_count() * sizeof(uint32_t),
    };
    immutable_commands[index].assign(
        control_bytes + producers[index].byte_offset,
        control_bytes + producers[index].byte_offset +
            producers[index].byte_length);
  }
  auto make_transfer = [&](uint32_t slot, auto emit) {
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
  ASSERT_LE(input_heap_length, UINT32_MAX);
  ASSERT_LE(reconstruction_heap_length, UINT32_MAX);
  ASSERT_LE(compressed_length, UINT32_MAX);
  const auto initialize =
      make_transfer(recording_count, [&](Pm4CommandWriter& writer) {
        writer.DmaFill32(input_address, 0xa5a5a5a5,
                         static_cast<uint32_t>(input_heap_length));
        writer.DmaFill32(reconstruction_address, 0xa5a5a5a5,
                         static_cast<uint32_t>(reconstruction_heap_length));
        writer.DmaFill32(compressed_address, 0xa5a5a5a5,
                         static_cast<uint32_t>(compressed_length));
      });
  const auto capture =
      make_transfer(recording_count + 1, [&](Pm4CommandWriter& writer) {
        writer.DmaCopy(compressed_address, captured->device_address,
                       static_cast<uint32_t>(compressed_length));
      });
  ASSERT_NO_FATAL_FAILURE(code->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(controls->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(captured->TransferHostCache(
      api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
  ASSERT_NO_FATAL_FAILURE(SubmitGpu(initialize));

  bool input_active[2] = {};
  bool reconstruction_active = false;
  uint64_t input_last_use[2] = {};
  for (uint32_t batch = 0; batch < frame_count; batch += recording_count) {
    const uint32_t count = std::min(recording_count, frame_count - batch);
    std::vector<PictureControls> picture_controls(count);
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t generation = batch + index;
      const uint32_t input_slot = generation % 2;
      const uint32_t reconstruction_slot = generation % 2;
      const uint32_t reference_slot = 1 - reconstruction_slot;
      const auto& layout = input_layouts[input_slot];
      const uint64_t pointer = input_address + layout.byte_offsets[0];
      const uint32_t arguments[] = {
          generation,
          static_cast<uint32_t>(layout.byte_offsets[1] -
                                layout.byte_offsets[0]),
          layout.pipe_bank_xors[0],
          layout.pipe_bank_xors[1],
          width,
          height,
          test_case.sample_bit_depth == 10 ? 1u : 0u,
          layout.linear,
          layout.row_pitches[0],
          layout.row_pitches[1],
      };
      std::memcpy(control_bytes + index * 4096, &pointer, sizeof(pointer));
      std::memcpy(control_bytes + index * 4096 + sizeof(pointer), arguments,
                  sizeof(arguments));
      amdf_media_image_activation_t activations[2] = {};
      uint32_t activation_count = 0;
      if (!input_active[input_slot]) {
        activations[activation_count++].after = input_images[input_slot];
        input_active[input_slot] = true;
      }
      if (!reconstruction_active) {
        activations[activation_count++].after = reconstruction_image;
        reconstruction_active = true;
      }
      const void* codec_picture_controls =
          picture_controls[index].InitializeFor(
              test_case, generation,
              generation == 0 ? nullptr : &reconstructions[reference_slot],
              reference_slot);
      const amdf_media_encode_picture_info_t encode = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_PICTURE_INFO,
          .structure_size = sizeof(encode),
          .next = codec_picture_controls,
          .input = &input_frames[input_slot],
          .reconstruction = &reconstructions[reconstruction_slot],
          .reconstruction_slot_index = reconstruction_slot,
          .compressed = {compressed_memory, 0,
                         compressed_start + index * compressed_stride,
                         compressed_stride},
      };
      const amdf_media_command_prepare_info_t prepare = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
          .structure_size = sizeof(prepare),
          .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
          .activation_count = activation_count,
          .activations = activations,
          .encode = &encode,
      };
      ASSERT_EQ(media_->command_prepare(commands_[index], &prepare),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(controls->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_FLUSH));
    uint64_t last_encode = 0;
    for (uint32_t index = 0; index < count; ++index) {
      const uint32_t generation = batch + index;
      const uint32_t input_slot = generation % 2;
      amdf_kernel_queue_dependency_wait_t reusable = {
          encode_to_gpu_, input_last_use[input_slot]};
      if (gpu_queue_ == nullptr && input_last_use[input_slot] != 0) {
        ASSERT_EQ(
            api_->kernel_queue_wait(encode_queue_, input_last_use[input_slot],
                                    AMDF_TIMEOUT_INFINITE, 0),
            AMDF_STATUS_OK);
      }
      ASSERT_NO_FATAL_FAILURE(SubmitGpu(
          producers[index],
          gpu_queue_ != nullptr && input_last_use[input_slot] != 0 ? &reusable
                                                                   : nullptr));
      const amdf_kernel_queue_dependency_wait_t produced = {gpu_to_encode_,
                                                            gpu_point_};
      const amdf_media_kernel_queue_submission_info_t submit = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
          .structure_size = sizeof(submit),
          .command = commands_[index],
          .wait_count = gpu_queue_ != nullptr ? 1u : 0u,
          .waits = gpu_queue_ != nullptr ? &produced : nullptr,
      };
      uint64_t point = 0;
      const auto status =
          media_->kernel_queue_submit(encode_queue_, &submit, &point);
      if (point != 0) {
        encode_point_ = point;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      ASSERT_NE(point, 0u);
      last_encode = point;
      input_last_use[input_slot] = point;
    }
    const amdf_kernel_queue_dependency_wait_t encoded = {encode_to_gpu_,
                                                         last_encode};
    if (gpu_queue_ == nullptr) {
      ASSERT_EQ(api_->kernel_queue_wait(encode_queue_, last_encode,
                                        AMDF_TIMEOUT_INFINITE, 0),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(
        SubmitGpu(capture, gpu_queue_ != nullptr ? &encoded : nullptr));
    if (gpu_queue_ != nullptr) {
      ASSERT_EQ(api_->kernel_queue_wait(gpu_queue_, gpu_point_,
                                        AMDF_TIMEOUT_INFINITE, 0),
                AMDF_STATUS_OK);
    }
    amdf_kernel_queue_status_t progress = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(progress),
    };
    ASSERT_EQ(api_->kernel_queue_refresh_status(encode_queue_, &progress),
              AMDF_STATUS_OK);
    ASSERT_EQ(progress.terminal_status, AMDF_STATUS_OK);
    ASSERT_EQ(progress.retired_submission, encode_point_);
    ASSERT_NO_FATAL_FAILURE(captured->TransferHostCache(
        api_, family_.ordinal, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    const auto* captured_bytes =
        static_cast<const uint8_t*>(captured->host.pointer);
    std::vector<EncodedPictureRegion> decoded_pictures;
    for (uint32_t index = 0; index < count; ++index) {
      amdf_media_encoded_region_t region = {};
      amdf_media_av1_encode_result_info_t av1_result = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_AV1_ENCODE_RESULT_INFO,
          .structure_size = sizeof(av1_result),
      };
      amdf_media_encode_result_info_t encode_result = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_ENCODE_RESULT_INFO,
          .structure_size = sizeof(encode_result),
          .next =
              test_case.codec == AMDF_MEDIA_CODEC_AV1 ? &av1_result : nullptr,
          .region_capacity = index == 0 ? 0u : 1u,
          .regions = index == 0 ? nullptr : &region,
      };
      amdf_media_command_result_info_t result = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
          .structure_size = sizeof(result),
          .next = &encode_result,
      };
      auto status = media_->command_query_result(commands_[index], &result);
      if (index == 0) {
        ASSERT_EQ(status,
                  amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
        ASSERT_EQ(encode_result.region_count, 1u);
        encode_result.region_capacity = 1;
        encode_result.regions = &region;
        status = media_->command_query_result(commands_[index], &result);
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      ASSERT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_SUCCEEDED);
      ASSERT_EQ(result.operation, AMDF_MEDIA_COMMAND_OPERATION_PICTURE);
      ASSERT_EQ(encode_result.region_count, 1u);
      ASSERT_GT(encode_result.written_byte_length, 0u);
      ASSERT_LE(encode_result.written_byte_length, compressed_stride);
      ASSERT_GT(region.byte_length, 0u);
      ASSERT_LE(region.byte_offset + region.byte_length,
                encode_result.written_byte_length);
      if (batch == 0) {
        const uint64_t output_start =
            compressed_start + index * compressed_stride;
        const auto* unwritten_begin =
            captured_bytes + output_start + encode_result.written_byte_length;
        const auto* unwritten_end =
            captured_bytes + output_start + compressed_stride;
        const auto* native_write =
            std::find_if(unwritten_begin, unwritten_end,
                         [](uint8_t value) { return value != 0xa5; });
        ASSERT_EQ(native_write, unwritten_end)
            << "native write after reported extent at byte "
            << native_write - (captured_bytes + output_start);
      }
      const auto expected_kind =
          test_case.codec == AMDF_MEDIA_CODEC_H264
              ? AMDF_MEDIA_ENCODED_REGION_H264_ANNEX_B_SLICE
          : test_case.codec == AMDF_MEDIA_CODEC_HEVC
              ? AMDF_MEDIA_ENCODED_REGION_HEVC_ANNEX_B_SLICE
          : configuration.requires_begin_end != 0
              ? AMDF_MEDIA_ENCODED_REGION_AV1_FRAME_OBU
              : AMDF_MEDIA_ENCODED_REGION_AV1_TILE_GROUP_PAYLOAD;
      ASSERT_EQ(region.kind, expected_kind);
      const uint64_t output_offset =
          compressed_start + index * compressed_stride + region.byte_offset;
      const auto* output = captured_bytes + output_offset;
      ASSERT_FALSE(std::all_of(output, output + region.byte_length,
                               [](uint8_t value) { return value == 0xa5; }));
      if (test_case.codec != AMDF_MEDIA_CODEC_AV1) {
        ASSERT_GE(region.byte_length, 5u);
        EXPECT_EQ(output[0], 0u);
        EXPECT_EQ(output[1], 0u);
        EXPECT_EQ(output[2], 0u);
        EXPECT_EQ(output[3], 1u);
      }
      if (batch == 0) {
        decoded_pictures.push_back({
            .byte_offset = output_offset,
            .byte_length = region.byte_length,
            .byte_capacity = compressed_stride - region.byte_offset,
            .kind = region.kind,
            .bytes = output,
            .av1_result = av1_result,
        });
      }
    }
    if (batch == 0) {
      ASSERT_NO_FATAL_FAILURE(DecodePictures(test_case, width, height,
                                             compressed_memory,
                                             decoded_pictures, last_encode));
    }
    for (uint64_t index = 0; index < compressed_start; ++index) {
      ASSERT_EQ(captured_bytes[index], 0xa5u);
    }
    for (uint64_t index =
             compressed_start + recording_count * compressed_stride;
         index < compressed_length; ++index) {
      ASSERT_EQ(captured_bytes[index], 0xa5u);
    }
    for (uint32_t index = 0; index < recording_count; ++index) {
      ASSERT_EQ(std::memcmp(control_bytes + producers[index].byte_offset,
                            immutable_commands[index].data(),
                            immutable_commands[index].size()),
                0);
    }
  }
  if (configuration.requires_begin_end != 0) {
    ASSERT_NO_FATAL_FAILURE(SubmitMediaControl(encode_queue_, commands_[0],
                                               AMDF_MEDIA_COMMAND_OPERATION_END,
                                               &encode_point_));
    encode_session_active_ = false;
  }
  for (uint64_t index = 0; index < feedback_length; ++index) {
    bool result_byte = false;
    for (uint32_t recording = 0; recording < recording_count; ++recording) {
      const uint64_t begin = feedback_start + recording * feedback_stride;
      result_byte |=
          index >= begin && index < begin + configuration.feedback_byte_length;
    }
    if (!result_byte) {
      ASSERT_EQ(encode_feedback_[index], 0xa5u);
    }
  }
  RecordProperty("codec", test_case.codec);
  RecordProperty("sample_bit_depth", test_case.sample_bit_depth);
  RecordProperty("rate_control_mode", rate_control_mode);
  RecordProperty("pictures", frame_count);
  RecordProperty("recordings", recording_count);
  RecordProperty("media_kernel_sha256", kernel->executable.sha256);
}

void MediaEncodeInteropTest::TearDown() {
  if (encode_point_ != 0) {
    ASSERT_EQ(api_->kernel_queue_wait(encode_queue_, encode_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  if (gpu_queue_ != nullptr && gpu_point_ != 0) {
    ASSERT_EQ(api_->kernel_queue_wait(gpu_queue_, gpu_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  if (decode_point_ != 0) {
    ASSERT_EQ(api_->kernel_queue_wait(decode_queue_, decode_point_,
                                      AMDF_TIMEOUT_INFINITE, 0),
              AMDF_STATUS_OK);
  }
  if (decode_session_active_) {
    ASSERT_FALSE(decode_commands_.empty());
    ASSERT_NO_FATAL_FAILURE(
        SubmitMediaControl(decode_queue_, decode_commands_[0],
                           AMDF_MEDIA_COMMAND_OPERATION_END, &decode_point_));
    decode_session_active_ = false;
  }
  if (encode_session_active_) {
    ASSERT_FALSE(commands_.empty());
    ASSERT_NO_FATAL_FAILURE(SubmitMediaControl(encode_queue_, commands_[0],
                                               AMDF_MEDIA_COMMAND_OPERATION_END,
                                               &encode_point_));
    encode_session_active_ = false;
  }
  for (auto* command : commands_) {
    ASSERT_EQ(media_->command_destroy(command), AMDF_STATUS_OK);
  }
  for (auto* command : decode_commands_) {
    ASSERT_EQ(media_->command_destroy(command), AMDF_STATUS_OK);
  }
  if (encode_session_ != nullptr) {
    ASSERT_EQ(media_->session_destroy(encode_session_), AMDF_STATUS_OK);
  }
  if (decode_session_ != nullptr) {
    ASSERT_EQ(media_->session_destroy(decode_session_), AMDF_STATUS_OK);
  }
  for (auto* image : images_) {
    media_->image_destroy(image);
  }
  if (encode_feedback_mapping_ != nullptr) {
    ASSERT_EQ(api_->host_mapping_destroy(encode_feedback_mapping_),
              AMDF_STATUS_OK);
  }
  if (decode_feedback_mapping_ != nullptr) {
    ASSERT_EQ(api_->host_mapping_destroy(decode_feedback_mapping_),
              AMDF_STATUS_OK);
  }
  for (auto* memory : memories_) {
    ASSERT_EQ(api_->memory_destroy(memory, nullptr), AMDF_STATUS_OK);
  }
  for (auto* dependency : {gpu_to_encode_, encode_to_gpu_, gpu_to_decode_,
                           encode_to_decode_, decode_to_gpu_}) {
    if (dependency != nullptr) {
      ASSERT_EQ(api_->kernel_queue_dependency_destroy(dependency),
                AMDF_STATUS_OK);
    }
  }
  for (auto* queue : {encode_queue_, decode_queue_}) {
    if (queue != nullptr) {
      ASSERT_EQ(api_->kernel_queue_destroy(queue), AMDF_STATUS_OK);
    }
  }
#if defined(_WIN32)
  if (gpu_queue_ != nullptr) {
    ASSERT_EQ(api_->kernel_queue_destroy(gpu_queue_), AMDF_STATUS_OK);
    gpu_queue_ = nullptr;
  }
#endif  // _WIN32
  if (media_device_ != nullptr) {
    ASSERT_EQ(api_->device_destroy(media_device_), AMDF_STATUS_OK);
  }
#if defined(_WIN32)
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
#endif  // _WIN32
  GpuCommandTest::TearDown();
}
