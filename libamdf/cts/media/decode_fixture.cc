// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/media/decode_fixture.h"

#include <algorithm>
#include <cstring>

#include "util/device_cache.h"
#include "util/provider.h"

namespace amdf::cts::media {

void MediaDecodeTest::SetUpMedia(
    const amdf_media_configuration_t& configuration,
    const void* codec_configuration, bool linear_frames) {
  linear_frames_ = linear_frames;
  ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                          AMDF_ABI_VERSION_LATEST, &api_),
            AMDF_STATUS_OK);
  const void* extension = nullptr;
  auto status = api_->query_extension(
      AMDF_EXTENSION_MEDIA, AMDF_MEDIA_EXTENSION_VERSION_1,
      AMDF_MEDIA_EXTENSION_VERSION_LATEST, &extension);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "media extension is not enabled";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  media_ = static_cast<const amdf_media_api_t*>(extension);
  amdf_instance_t* instance = nullptr;
  ASSERT_EQ(GetCtsDeviceCache().GetInstance(&instance), AMDF_STATUS_OK);
  uint32_t count = 0;
  ASSERT_EQ(api_->endpoint_enumerate(instance, 0, nullptr, &count),
            AMDF_STATUS_OK);
  std::vector<amdf_endpoint_summary_t> summaries(count);
  ASSERT_EQ(api_->endpoint_enumerate(instance, count, summaries.data(), &count),
            AMDF_STATUS_OK);
  amdf_endpoint_t* endpoint = nullptr;
  for (const auto& summary : summaries) {
    if (summary.engine_kind == AMDF_ENGINE_KIND_GPU &&
        GetCtsDeviceCache().IsGpuEndpointSelected(summary.id)) {
      ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint),
                AMDF_STATUS_OK);
      break;
    }
  }
  if (endpoint == nullptr) {
    GTEST_SKIP() << "no selected GPU endpoint";
  }
  const amdf_media_device_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
      .structure_size = sizeof(create),
  };
  status = media_->device_create(endpoint, &create, &device_);
  if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
    GTEST_SKIP() << "native media provider is unavailable";
  }
  ASSERT_EQ(status, AMDF_STATUS_OK);
  amdf_device_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_DEVICE_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->device_query_info(device_, &info), AMDF_STATUS_OK);
  uint32_t family = info.queue_family_count;
  for (uint32_t i = 0; i < info.queue_family_count; ++i) {
    ASSERT_EQ(media_->device_enumerate_codecs(device_, i, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_media_codec_info_t> codecs(count);
    ASSERT_EQ(media_->device_enumerate_codecs(device_, i, count, codecs.data(),
                                              &count),
              AMDF_STATUS_OK);
    if (std::any_of(codecs.begin(), codecs.end(), [&](const auto& codec) {
          return codec.codec == configuration.codec &&
                 codec.direction == AMDF_MEDIA_DIRECTION_DECODE;
        })) {
      family = i;
      break;
    }
  }
  if (family == info.queue_family_count) {
    GTEST_SKIP() << "no native decode family for codec=" << configuration.codec;
  }
  family_ = family;
  const amdf_media_configuration_query_t query = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
      .structure_size = sizeof(query),
      .next = codec_configuration,
      .queue_family_ordinal = family,
      .configuration = configuration,
  };
  configuration_.type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO;
  configuration_.structure_size = sizeof(configuration_);
  ASSERT_EQ(
      media_->device_query_configuration(device_, &query, &configuration_),
      AMDF_STATUS_OK);
  if (configuration_.support != AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
    GTEST_SKIP() << "exact decode configuration is unsupported";
  }
  ASSERT_NE(configuration_.requires_begin_end, 0u);
  const amdf_media_kernel_queue_create_info_t queue_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
      .structure_size = sizeof(queue_create),
      .queue_family_ordinal = family,
      .maximum_pending_submission_count = 6,
  };
  ASSERT_EQ(media_->kernel_queue_create(device_, &queue_create, &queue_),
            AMDF_STATUS_OK);
  const amdf_media_session_create_info_t session_create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
      .structure_size = sizeof(session_create),
      .next = codec_configuration,
      .configuration = query.configuration,
  };
  ASSERT_EQ(media_->session_create(queue_, &session_create, &session_),
            AMDF_STATUS_OK);
  references_.resize(configuration.reference_image_count);
  reference_layouts_.resize(configuration.reference_image_count);
}

void MediaDecodeTest::CreateCommand(amdf_media_command_t** out) {
  const amdf_media_command_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
      .structure_size = sizeof(create),
      .codec = configuration_.configuration.codec,
      .maximum_slice_count =
          configuration_.configuration.codec == AMDF_MEDIA_CODEC_AV1 ? 256u
                                                                     : 1u,
      .maximum_reference_count =
          configuration_.configuration.reference_image_count - 1,
  };
  ASSERT_EQ(media_->command_create(queue_, &create, out), AMDF_STATUS_OK);
}

void MediaDecodeTest::CreateMemory(uint64_t length, CtsMappedMemory* memory) {
  const amdf_memory_device_access_t access = {
      device_, {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
  const amdf_memory_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory_profile_ordinal =
          configuration_.image_heap_memory_profile_ordinal,
      .access_count = 1,
      .required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE,
      .byte_length = length,
      .accesses = &access,
  };
  ASSERT_NO_FATAL_FAILURE(
      memory->Create(api_, configuration_.image_heap_memory_scope, create));
  std::fill(memory->bytes().begin(), memory->bytes().end(), 0xa5);
}

void MediaDecodeTest::CreateFrames(
    const amdf_media_image_requirements_t& requirements,
    CtsMappedMemory* memory, amdf_media_image_t** image,
    std::span<amdf_media_linear_image_layout_t> layouts,
    std::span<amdf_media_frame_t> frames) {
  auto description = requirements.description;
  description.array_layer_count = static_cast<uint32_t>(frames.size());
  amdf_media_image_requirements_t array;
  ASSERT_EQ(
      media_->device_query_image_requirements(device_, &description, &array),
      AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(CreateMemory(512 + array.byte_length, memory));
  const amdf_media_image_create_info_t create = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
      .structure_size = sizeof(create),
      .memory = memory->memory,
      .description = description,
      .byte_offset = 256,
  };
  ASSERT_EQ(media_->image_create(device_, &create, image), AMDF_STATUS_OK);
  for (uint32_t i = 0; i < frames.size(); ++i) {
    auto& layout = layouts[i];
    layout.type = AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT;
    layout.structure_size = sizeof(layout);
    const amdf_media_image_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
        .structure_size = sizeof(query),
        .array_layer = i,
    };
    amdf_media_image_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
        .structure_size = sizeof(info),
        .next = &layout,
    };
    ASSERT_EQ(media_->image_query_info(*image, 0, &query, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(layout.plane_count, 2u);
    auto& frame = frames[i];
    frame.type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME;
    frame.structure_size = sizeof(frame);
    frame.format = description.format;
    frame.width = description.width;
    frame.height = description.height;
    if (linear_frames_) {
      frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES;
      frame.storage.linear = {2, layout.planes};
    } else {
      frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE;
      frame.storage.image = {memory->memory, 256, *image, i};
    }
  }
}

void MediaDecodeTest::CreatePictures() {
  ASSERT_NO_FATAL_FAILURE(CreateMemory(4 * 65536 + 256, &input_));
  ASSERT_NO_FATAL_FAILURE(CreateFrames(configuration_.output, &output_memory_,
                                       &output_image_, output_layouts_,
                                       outputs_));
  ASSERT_NO_FATAL_FAILURE(CreateFrames(configuration_.reference,
                                       &reference_memory_, &reference_image_,
                                       reference_layouts_, references_));
  for (auto& command : commands_) {
    ASSERT_NO_FATAL_FAILURE(CreateCommand(&command));
  }
  ASSERT_NO_FATAL_FAILURE(CreateCommand(&control_));
  ASSERT_NO_FATAL_FAILURE(
      Cache(reference_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
}

void MediaDecodeTest::Cache(CtsMappedMemory& memory,
                            amdf_host_cache_operation_t operation) {
  ASSERT_EQ(api_->host_mapping_cache_control(memory.mapping, operation, 0,
                                             memory.info.byte_length),
            AMDF_STATUS_OK);
}

amdf_status_t MediaDecodeTest::PreparePicture(const DecodeInput& entry,
                                              uint32_t slot) {
  const uint64_t offset = 256 + uint64_t(slot) * 65536;
  std::memcpy(input_.bytes().data() + offset, entry.bytes, entry.byte_length);
  const amdf_media_decode_picture_info_t decode = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
      .structure_size = sizeof(decode),
      .next = entry.picture,
      .compressed = {input_.memory, 0, offset, entry.byte_length},
      .output = &outputs_[slot],
      .reference_output = &references_[entry.reference_slot],
      .reference_slot_index = entry.reference_slot,
  };
  const amdf_media_command_prepare_info_t prepare = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      .structure_size = sizeof(prepare),
      .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
      .decode = &decode,
  };
  return media_->command_prepare(session_, commands_[slot], &prepare);
}

void MediaDecodeTest::PrepareControl(amdf_media_command_operation_t operation) {
  const amdf_media_command_prepare_info_t prepare = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      .structure_size = sizeof(prepare),
      .operation = operation,
  };
  ASSERT_EQ(media_->command_prepare(session_, control_, &prepare),
            AMDF_STATUS_OK);
}

amdf_status_t MediaDecodeTest::Submit(amdf_media_command_t* command) {
  const amdf_media_kernel_queue_submission_info_t submit = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
      .structure_size = sizeof(submit),
      .command = command,
  };
  uint64_t accepted = 0;
  const auto status = media_->kernel_queue_submit(queue_, &submit, &accepted);
  if (accepted != 0) {
    last_submission_ = accepted;
  }
  EXPECT_EQ(accepted != 0, status == AMDF_STATUS_OK);
  return status;
}

void MediaDecodeTest::Join() {
  ASSERT_EQ(api_->kernel_queue_wait(queue_, last_submission_, UINT64_MAX, 0),
            AMDF_STATUS_OK);
  amdf_kernel_queue_status_t status = {
      .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
      .structure_size = sizeof(status),
  };
  ASSERT_EQ(api_->kernel_queue_query_status(queue_, &status), AMDF_STATUS_OK);
  ASSERT_EQ(status.retired_submission, last_submission_);
  ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
}

void MediaDecodeTest::Begin() {
  ASSERT_NO_FATAL_FAILURE(PrepareControl(AMDF_MEDIA_COMMAND_OPERATION_BEGIN));
  ASSERT_EQ(Submit(control_), AMDF_STATUS_OK);
  active_ = true;
}

void MediaDecodeTest::End() {
  ASSERT_NO_FATAL_FAILURE(PrepareControl(AMDF_MEDIA_COMMAND_OPERATION_END));
  ASSERT_EQ(Submit(control_), AMDF_STATUS_OK);
  active_ = false;
  ASSERT_NO_FATAL_FAILURE(Join());
  amdf_media_command_result_info_t result = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
      .structure_size = sizeof(result),
  };
  ASSERT_EQ(media_->command_query_result(control_, &result), AMDF_STATUS_OK);
  EXPECT_EQ(result.operation, AMDF_MEDIA_COMMAND_OPERATION_END);
  EXPECT_EQ(result.submission, last_submission_);
  EXPECT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE);
}

void MediaDecodeTest::CheckGuards(
    CtsMappedMemory& memory,
    std::span<const amdf_media_linear_image_layout_t> layouts) {
  ASSERT_NO_FATAL_FAILURE(Cache(memory, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
  std::vector<bool> writable(memory.info.byte_length, false);
  for (const auto& layout : layouts) {
    for (const auto& plane : layout.planes) {
      if (plane.byte_length == 0) {
        continue;
      }
      ASSERT_LE(plane.byte_offset + plane.byte_length, writable.size());
      std::fill(writable.begin() + plane.byte_offset,
                writable.begin() + plane.byte_offset + plane.byte_length, true);
    }
  }
  for (size_t i = 0; i < writable.size(); ++i) {
    if (!writable[i]) {
      ASSERT_EQ(memory.bytes()[i], 0xa5) << "guard=" << i;
    }
  }
}

void MediaDecodeTest::CheckPixels(std::span<const DecodeInput> fixture,
                                  uint32_t first) {
  ASSERT_NO_FATAL_FAILURE(CheckGuards(output_memory_, output_layouts_));
  ASSERT_NO_FATAL_FAILURE(CheckGuards(reference_memory_, reference_layouts_));
  const auto& shape = configuration_.configuration;
  const uint32_t sample_bytes =
      shape.format == AMDF_MEDIA_PIXEL_FORMAT_P010 ? 2 : 1;
  for (uint32_t slot = 0; slot < 4; ++slot) {
    const auto& y = output_layouts_[slot].planes[0];
    const auto& uv = output_layouts_[slot].planes[1];
    uint64_t hash = UINT64_C(14695981039346656037);
    for (uint32_t row = 0; row < shape.coded_height; ++row) {
      for (uint32_t column = 0; column < shape.coded_width; ++column) {
        const uint64_t positions[3] = {
            y.byte_offset + row * y.row_pitch + column * sample_bytes,
            uv.byte_offset + (row / 2) * uv.row_pitch +
                (column & ~1u) * sample_bytes,
            uv.byte_offset + (row / 2) * uv.row_pitch +
                ((column & ~1u) + 1) * sample_bytes};
        for (uint64_t offset : positions) {
          if (sample_bytes == 2) {
            ASSERT_EQ(output_memory_.bytes()[offset] & 63u, 0u);
          }
          for (uint32_t byte = 0; byte < sample_bytes; ++byte) {
            hash = (hash ^ output_memory_.bytes()[offset + byte]) *
                   UINT64_C(1099511628211);
          }
        }
      }
    }
    EXPECT_EQ(hash, fixture[(first + slot) % fixture.size()].pixel_hash)
        << "picture=" << first + slot;
  }
}

void MediaDecodeTest::TearDown() {
  if (last_submission_ != 0) {
    ASSERT_NO_FATAL_FAILURE(Join());
  }
  if (active_) {
    ASSERT_NO_FATAL_FAILURE(End());
  }
  for (auto* command : commands_) {
    if (command != nullptr) {
      ASSERT_EQ(media_->command_destroy(command), AMDF_STATUS_OK);
    }
  }
  if (control_ != nullptr) {
    ASSERT_EQ(media_->command_destroy(control_), AMDF_STATUS_OK);
  }
  if (session_ != nullptr) {
    ASSERT_EQ(media_->session_destroy(session_), AMDF_STATUS_OK);
  }
  if (queue_ != nullptr) {
    ASSERT_EQ(api_->kernel_queue_destroy(queue_), AMDF_STATUS_OK);
  }
  if (output_image_ != nullptr) {
    media_->image_destroy(output_image_);
  }
  if (reference_image_ != nullptr) {
    media_->image_destroy(reference_image_);
  }
  ASSERT_TRUE(input_.Release(api_));
  ASSERT_TRUE(output_memory_.Release(api_));
  ASSERT_TRUE(reference_memory_.Release(api_));
  if (device_ != nullptr) {
    ASSERT_EQ(api_->device_destroy(device_), AMDF_STATUS_OK);
  }
}

void MediaDecodeTest::RunPictures(std::span<const DecodeInput> fixture,
                                  uint32_t picture_count) {
  ASSERT_EQ(picture_count % 4, 0u);
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  ASSERT_NO_FATAL_FAILURE(Begin());
  for (uint32_t first = 0; first < picture_count; first += 4) {
    std::fill(output_memory_.bytes().begin(), output_memory_.bytes().end(),
              0xa5);
    ASSERT_NO_FATAL_FAILURE(
        Cache(output_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    for (uint32_t slot = 0; slot < 4; ++slot) {
      ASSERT_EQ(PreparePicture(fixture[(first + slot) % fixture.size()], slot),
                AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(Cache(input_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    for (auto* command : commands_) {
      ASSERT_EQ(Submit(command), AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(Join());
    ASSERT_NO_FATAL_FAILURE(CheckPixels(fixture, first));
  }
  ASSERT_NO_FATAL_FAILURE(End());
}

}  // namespace amdf::cts::media
