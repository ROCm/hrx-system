// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstring>
#include <utility>
#include <vector>

#include "amdf/media.h"
#include "gtest/gtest.h"
#include "libamdf/cts/media/h264_fixture.h"
#include "util/device_cache.h"
#include "util/mapped_memory.h"
#include "util/provider.h"

namespace {

class MediaH264Test : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
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
    ASSERT_EQ(
        api_->endpoint_enumerate(instance, count, summaries.data(), &count),
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
      ASSERT_EQ(media_->device_enumerate_codecs(device_, i, count,
                                                codecs.data(), &count),
                AMDF_STATUS_OK);
      if (std::any_of(codecs.begin(), codecs.end(), [](const auto& codec) {
            return codec.codec == AMDF_MEDIA_CODEC_H264 &&
                   codec.direction == AMDF_MEDIA_DIRECTION_DECODE;
          })) {
        family = i;
        break;
      }
    }
    if (family == info.queue_family_count) {
      GTEST_SKIP() << "no native H264 decode family";
    }
    const amdf_media_h264_configuration_t h264 = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION,
        .structure_size = sizeof(h264),
        .profile = AMDF_MEDIA_H264_PROFILE_MAIN,
        .level_idc = 13,
    };
    const amdf_media_configuration_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
        .structure_size = sizeof(query),
        .next = &h264,
        .queue_family_ordinal = family,
        .configuration = {AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                          AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 4},
    };
    configuration_.type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO;
    configuration_.structure_size = sizeof(configuration_);
    ASSERT_EQ(
        media_->device_query_configuration(device_, &query, &configuration_),
        AMDF_STATUS_OK);
    if (configuration_.support != AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
      GTEST_SKIP() << "exact H264 configuration is unsupported";
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
        .next = &h264,
        .configuration = query.configuration,
    };
    ASSERT_EQ(media_->session_create(queue_, &session_create, &session_),
              AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadH264Fixture(&fixture_));
    std::memset(&matrices_, 16, sizeof(matrices_));
    for (auto& entry : fixture_) {
      entry.picture.scaling_matrices = &matrices_;
      for (uint32_t i = 0; i < entry.picture.reference_count; ++i) {
        ASSERT_LT(entry.reference_slots[i], references_.size());
        entry.references[i].binding = {&references_[entry.reference_slots[i]],
                                       entry.reference_slots[i]};
      }
    }
  }

  void CreateCommand(amdf_media_command_t** out) {
    const amdf_media_command_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
        .structure_size = sizeof(create),
        .maximum_slice_count = 1,
        .maximum_reference_count = 3,
    };
    ASSERT_EQ(media_->command_create(queue_, &create, out), AMDF_STATUS_OK);
  }

  void CreateMemory(uint64_t length, CtsMappedMemory* memory) {
    const amdf_memory_device_access_t access = {
        device_,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
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

  void CreateFrames(const amdf_media_image_requirements_t& requirements,
                    CtsMappedMemory* memory, amdf_media_image_t** image,
                    std::array<amdf_media_linear_image_layout_t, 4>* layouts,
                    std::array<amdf_media_frame_t, 4>* frames) {
    auto description = requirements.description;
    description.array_layer_count = 4;
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
    for (uint32_t i = 0; i < frames->size(); ++i) {
      auto& layout = (*layouts)[i];
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
      auto& frame = (*frames)[i];
      frame.type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME;
      frame.structure_size = sizeof(frame);
      frame.format = description.format;
      frame.width = description.width;
      frame.height = description.height;
      if (GetParam()) {
        frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES;
        frame.storage.linear = {2, layout.planes};
      } else {
        frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE;
        frame.storage.image = {memory->memory, 256, *image, i};
      }
    }
  }

  void CreatePictures() {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(4 * 65536 + 256, &input_));
    ASSERT_NO_FATAL_FAILURE(CreateFrames(configuration_.output, &output_memory_,
                                         &output_image_, &output_layouts_,
                                         &outputs_));
    ASSERT_NO_FATAL_FAILURE(CreateFrames(configuration_.reference,
                                         &reference_memory_, &reference_image_,
                                         &reference_layouts_, &references_));
    for (auto& command : commands_) {
      ASSERT_NO_FATAL_FAILURE(CreateCommand(&command));
    }
    ASSERT_NO_FATAL_FAILURE(CreateCommand(&control_));
    ASSERT_NO_FATAL_FAILURE(
        Cache(reference_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
  }

  void Cache(CtsMappedMemory& memory, amdf_host_cache_operation_t operation) {
    ASSERT_EQ(api_->host_mapping_cache_control(memory.mapping, operation, 0,
                                               memory.info.byte_length),
              AMDF_STATUS_OK);
  }

  void PreparePicture(uint32_t index, uint32_t slot) {
    auto& entry = fixture_[index];
    const uint64_t offset = 256 + uint64_t(slot) * 65536;
    std::memcpy(input_.bytes().data() + offset, entry.bytes, entry.byte_length);
    const amdf_media_decode_picture_info_t decode = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
        .structure_size = sizeof(decode),
        .next = &entry.picture,
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
    ASSERT_EQ(media_->command_prepare(session_, commands_[slot], &prepare),
              AMDF_STATUS_OK);
  }

  void PrepareControl(amdf_media_command_operation_t operation) {
    const amdf_media_command_prepare_info_t prepare = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
        .structure_size = sizeof(prepare),
        .operation = operation,
    };
    ASSERT_EQ(media_->command_prepare(session_, control_, &prepare),
              AMDF_STATUS_OK);
  }

  amdf_status_t Submit(amdf_media_command_t* command) {
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

  void Join() {
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

  void Begin() {
    ASSERT_NO_FATAL_FAILURE(PrepareControl(AMDF_MEDIA_COMMAND_OPERATION_BEGIN));
    ASSERT_EQ(Submit(control_), AMDF_STATUS_OK);
    active_ = true;
  }

  void End() {
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

  void CheckGuards(
      CtsMappedMemory& memory,
      const std::array<amdf_media_linear_image_layout_t, 4>& layouts) {
    ASSERT_NO_FATAL_FAILURE(
        Cache(memory, AMDF_HOST_CACHE_OPERATION_INVALIDATE));
    std::vector<bool> writable(memory.info.byte_length, false);
    for (const auto& layout : layouts) {
      for (const auto& plane : layout.planes) {
        if (plane.byte_length == 0) {
          continue;
        }
        ASSERT_LE(plane.byte_offset + plane.byte_length, writable.size());
        std::fill(writable.begin() + plane.byte_offset,
                  writable.begin() + plane.byte_offset + plane.byte_length,
                  true);
      }
    }
    for (size_t i = 0; i < writable.size(); ++i) {
      if (!writable[i]) {
        ASSERT_EQ(memory.bytes()[i], 0xa5) << "guard=" << i;
      }
    }
  }

  void CheckPixels(uint32_t first) {
    ASSERT_NO_FATAL_FAILURE(CheckGuards(output_memory_, output_layouts_));
    ASSERT_NO_FATAL_FAILURE(CheckGuards(reference_memory_, reference_layouts_));
    for (uint32_t slot = 0; slot < 4; ++slot) {
      const auto& y = output_layouts_[slot].planes[0];
      const auto& uv = output_layouts_[slot].planes[1];
      uint64_t hash = UINT64_C(14695981039346656037);
      for (uint32_t row = 0; row < 256; ++row) {
        for (uint32_t column = 0; column < 256; ++column) {
          const uint64_t positions[3] = {
              y.byte_offset + row * y.row_pitch + column,
              uv.byte_offset + (row / 2) * uv.row_pitch + (column & ~1u),
              uv.byte_offset + (row / 2) * uv.row_pitch + (column & ~1u) + 1};
          for (uint64_t offset : positions) {
            hash = (hash ^ output_memory_.bytes()[offset]) *
                   UINT64_C(1099511628211);
          }
        }
      }
      EXPECT_EQ(hash, amdf::cts::media::kH264PixelHashes[(first + slot) % 12])
          << "picture=" << first + slot;
    }
  }

  void TearDown() override {
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

  // Borrowed process-lifetime core operations.
  const amdf_api_t* api_ = nullptr;
  // Borrowed optional media operations.
  const amdf_media_api_t* media_ = nullptr;
  // Case-owned independent native media domain.
  amdf_device_t* device_ = nullptr;
  // Application-owned ordered decode submissions.
  amdf_kernel_queue_t* queue_ = nullptr;
  // Explicit native state, begun and ended by this caller.
  amdf_media_session_t* session_ = nullptr;
  // Queried exact requirements, including separate reference storage.
  amdf_media_configuration_info_t configuration_ = {};
  // Greatest accepted point; retirement precedes all reuse and teardown.
  uint64_t last_submission_ = 0;
  // Whether an accepted BEGIN still requires application-authored END.
  bool active_ = false;
  // Separate reusable recording for BEGIN and END.
  amdf_media_command_t* control_ = nullptr;
  // Fixed picture recordings reused after batch retirement.
  std::array<amdf_media_command_t*, 4> commands_ = {};
  // Caller-owned compressed slots, each with 64 KiB capacity.
  CtsMappedMemory input_;
  // Caller-owned output storage reused after its pixel oracle consumes it.
  CtsMappedMemory output_memory_;
  // Caller-owned stable reference slots; no library reference scheduler.
  CtsMappedMemory reference_memory_;
  // Output identity placed at a nonzero offset with four array layers.
  amdf_media_image_t* output_image_ = nullptr;
  // Reference identity placed at a nonzero offset with four array layers.
  amdf_media_image_t* reference_image_ = nullptr;
  // Retained plane metadata for direct-plane preparation and host checking.
  std::array<amdf_media_linear_image_layout_t, 4> output_layouts_ = {};
  // Retained reference plane metadata and protected extents.
  std::array<amdf_media_linear_image_layout_t, 4> reference_layouts_ = {};
  // Borrowed output frames, with either native identities or direct planes.
  std::array<amdf_media_frame_t, 4> outputs_ = {};
  // Stable application reference frame bindings.
  std::array<amdf_media_frame_t, 4> references_ = {};
  // Parsed syntax and compressed bytes shared with Windows CTS.
  std::array<amdf::cts::media::H264FixturePicture, 12> fixture_ = {};
  // Explicit raster scaling lists used by every fixture picture.
  amdf_media_h264_scaling_matrices_t matrices_ = {};
};

TEST_P(MediaH264Test, ReusesRecordingsAndReferenceSlotsWithExactPixels) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  ASSERT_NO_FATAL_FAILURE(Begin());
  for (uint32_t first = 0; first < 24; first += 4) {
    std::fill(output_memory_.bytes().begin(), output_memory_.bytes().end(),
              0xa5);
    ASSERT_NO_FATAL_FAILURE(
        Cache(output_memory_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    for (uint32_t slot = 0; slot < 4; ++slot) {
      ASSERT_NO_FATAL_FAILURE(PreparePicture((first + slot) % 12, slot));
    }
    ASSERT_NO_FATAL_FAILURE(Cache(input_, AMDF_HOST_CACHE_OPERATION_FLUSH));
    for (auto* command : commands_) {
      ASSERT_EQ(Submit(command), AMDF_STATUS_OK);
    }
    ASSERT_NO_FATAL_FAILURE(Join());
    ASSERT_NO_FATAL_FAILURE(CheckPixels(first));
  }
  ASSERT_NO_FATAL_FAILURE(End());
}

TEST_P(MediaH264Test, LifecycleRequiresExplicitAcceptedBeginAndEnd) {
  ASSERT_NO_FATAL_FAILURE(CreatePictures());
  ASSERT_NO_FATAL_FAILURE(PreparePicture(0, 0));
  EXPECT_EQ(Submit(commands_[0]),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  ASSERT_NO_FATAL_FAILURE(Begin());
  ASSERT_NO_FATAL_FAILURE(Join());
  EXPECT_EQ(Submit(control_),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  // Release all recording borrows: the active session still owns its native
  // lifetime and cannot disappear without an ordered END.
  for (auto& command : commands_) {
    ASSERT_EQ(media_->command_destroy(std::exchange(command, nullptr)),
              AMDF_STATUS_OK);
  }
  ASSERT_EQ(media_->command_destroy(std::exchange(control_, nullptr)),
            AMDF_STATUS_OK);
  EXPECT_EQ(media_->session_destroy(session_),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  ASSERT_NO_FATAL_FAILURE(CreateCommand(&control_));
  ASSERT_NO_FATAL_FAILURE(End());
  ASSERT_NO_FATAL_FAILURE(CreateCommand(&commands_[0]));
  ASSERT_NO_FATAL_FAILURE(PreparePicture(0, 0));
  EXPECT_EQ(Submit(commands_[0]),
            amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
}

INSTANTIATE_TEST_SUITE_P(NativeAndLinear, MediaH264Test, ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& parameter) {
                           return parameter.param ? "Planes" : "Image";
                         });

}  // namespace
