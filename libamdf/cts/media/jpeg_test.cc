// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "amdf/media.h"
#include "gtest/gtest.h"
#include "libamdf/cts/media/jpeg_fixtures.h"
#include "util/device_cache.h"
#include "util/mapped_memory.h"
#include "util/native_event.h"
#include "util/provider.h"

namespace {

std::span<const uint8_t> Fixture(const std::string& name) {
  const auto* files = amdf_cts_jpeg_fixtures_create();
  for (size_t i = 0; i < amdf_cts_jpeg_fixtures_size(); ++i) {
    if (name == files[i].name) {
      return {reinterpret_cast<const uint8_t*>(files[i].data), files[i].size};
    }
  }
  ADD_FAILURE() << "missing fixture " << name;
  return {};
}

struct Shape {
  // Exact JPEG sample width, also the visible output width.
  uint32_t width;
  // Exact sample height; storage padding is queried independently.
  uint32_t height;
  // Selects direct planes instead of a retained native identity at preparation.
  bool linear;
};

class MediaJpegTest : public ::testing::TestWithParam<Shape> {
 protected:
  struct Channel {
    // Independently activated execution domain with its own native file/VM.
    amdf_device_t* device = nullptr;
    // Application-owned ordered queue; checked through last_submission.
    amdf_kernel_queue_t* queue = nullptr;
    // Exact JPEG session borrowing queue.
    amdf_media_session_t* session = nullptr;
    // Queried placement and compressed-input requirements.
    amdf_media_configuration_info_t configuration = {};
    // Fixed recording slots; the application decides when to reuse each.
    std::array<amdf_media_command_t*, 4> commands = {};
    // Greatest accepted point, including a partially successful publication.
    uint64_t last_submission = 0;
  };

  void SetUp() override {
    ASSERT_EQ(amdf_cts_provider_query_api()(AMDF_ABI_VERSION_LATEST,
                                            AMDF_ABI_VERSION_LATEST, &api_),
              AMDF_STATUS_OK);
    const void* extension = nullptr;
    const auto status = api_->query_extension(
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
    for (const auto& summary : summaries) {
      if (summary.engine_kind == AMDF_ENGINE_KIND_GPU &&
          GetCtsDeviceCache().IsGpuEndpointSelected(summary.id)) {
        ASSERT_EQ(GetCtsDeviceCache().OpenEndpoint(summary.id, &endpoint_),
                  AMDF_STATUS_OK);
        break;
      }
    }
    if (endpoint_ == nullptr) {
      GTEST_SKIP() << "no selected GPU endpoint";
    }
    ASSERT_NO_FATAL_FAILURE(CreateChannel());
    const Shape shape = GetParam();
    const std::string prefix = "jpeg_" + std::to_string(shape.width) + "x" +
                               std::to_string(shape.height);
    compressed_ = Fixture(prefix + ".jpg");
    reference_ = Fixture(prefix + ".nv12");
    inverse_compressed_ = Fixture(prefix + "_inverse.jpg");
    inverse_reference_ = Fixture(prefix + "_inverse.nv12");
    input_length_ =
        (std::max(compressed_.size(), inverse_compressed_.size()) + 127) &
        ~uint64_t(127);
    ASSERT_FALSE(compressed_.empty());
    ASSERT_EQ(reference_.size(), size_t(shape.width) * shape.height * 3 / 2);
  }

  void CreateChannel() {
    channels_.push_back(std::make_unique<Channel>());
    Channel& channel = *channels_.back();
    const amdf_media_device_create_info_t device_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_DEVICE_CREATE_INFO,
        .structure_size = sizeof(device_create),
    };
    const auto status =
        media_->device_create(endpoint_, &device_create, &channel.device);
    if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
      GTEST_SKIP() << "native media provider is unavailable on this endpoint";
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    const Shape shape = GetParam();
    const amdf_media_configuration_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
        .structure_size = sizeof(query),
        .configuration =
            {
                .codec = AMDF_MEDIA_CODEC_JPEG,
                .direction = AMDF_MEDIA_DIRECTION_DECODE,
                .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
                .coded_width = shape.width,
                .coded_height = shape.height,
            },
    };
    channel.configuration.type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO;
    channel.configuration.structure_size = sizeof(channel.configuration);
    ASSERT_EQ(media_->device_query_configuration(channel.device, &query,
                                                 &channel.configuration),
              AMDF_STATUS_OK);
    if (channel.configuration.support !=
        AMDF_MEDIA_CONFIGURATION_SUPPORT_SUPPORTED) {
      GTEST_SKIP() << "exact JPEG configuration is unsupported";
    }
    const amdf_media_kernel_queue_create_info_t queue_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_CREATE_INFO,
        .structure_size = sizeof(queue_create),
        .required_capabilities =
            AMDF_KERNEL_QUEUE_CAPABILITY_EXTERNAL_SYNCHRONIZATION,
        .maximum_pending_submission_count = 4,
        .maximum_wait_count = 1,
    };
    ASSERT_EQ(media_->kernel_queue_create(channel.device, &queue_create,
                                          &channel.queue),
              AMDF_STATUS_OK);
    const amdf_media_session_create_info_t session_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_SESSION_CREATE_INFO,
        .structure_size = sizeof(session_create),
        .configuration = query.configuration,
    };
    ASSERT_EQ(media_->session_create(channel.queue, &session_create,
                                     &channel.session),
              AMDF_STATUS_OK);
    const amdf_media_command_create_info_t command_create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_CREATE_INFO,
        .structure_size = sizeof(command_create),
        .maximum_slice_count = 1,
    };
    for (auto& command : channel.commands) {
      ASSERT_EQ(
          media_->command_create(channel.queue, &command_create, &command),
          AMDF_STATUS_OK);
    }
  }

  void CreateMemory(Channel& channel, uint64_t length, CtsMappedMemory** out) {
    auto memory = std::make_unique<CtsMappedMemory>();
    *out = memory.get();
    memories_.push_back(std::move(memory));
    const amdf_memory_device_access_t access = {
        channel.device,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal =
            channel.configuration.image_heap_memory_profile_ordinal,
        .access_count = 1,
        .required_flags =
            AMDF_MEMORY_FLAG_HOST_VISIBLE | AMDF_MEMORY_FLAG_SHAREABLE,
        .byte_length = length,
        .accesses = &access,
    };
    ASSERT_NO_FATAL_FAILURE((*out)->Create(
        api_, channel.configuration.image_heap_memory_scope, create));
  }

  void CreateInput(Channel& channel, CtsMappedMemory** out) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(channel, input_length_ + 256, out));
    std::fill((*out)->bytes().begin(), (*out)->bytes().end(), 0);
    std::memcpy((*out)->bytes().data() + 128, compressed_.data(),
                compressed_.size());
    ASSERT_EQ(api_->host_mapping_cache_control((*out)->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, (*out)->info.byte_length),
              AMDF_STATUS_OK);
  }

  void Import(Channel& consumer, amdf_memory_t* source, uint64_t offset,
              uint64_t length, amdf_memory_t** out) {
    amdf_memory_scope_t* scope = consumer.configuration.image_heap_memory_scope;
    const amdf_memory_device_access_t access = {
        consumer.device,
        {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE}};
    amdf_memory_scope_info_t scope_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_SCOPE_INFO,
        .structure_size = sizeof(scope_info),
    };
    ASSERT_EQ(api_->memory_scope_query_info(scope, &scope_info),
              AMDF_STATUS_OK);
    uint32_t ordinal = AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN;
    for (uint32_t i = 0; i < scope_info.memory_profile_count; ++i) {
      amdf_memory_profile_t profile = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
          .structure_size = sizeof(profile),
      };
      amdf_memory_access_capabilities_t capabilities = {
          .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
          .structure_size = sizeof(capabilities),
      };
      const auto status = api_->memory_scope_query_device_profile(
          scope, i, 1, &access, &profile, &capabilities);
      if (status == amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED)) {
        continue;
      }
      ASSERT_EQ(status, AMDF_STATUS_OK);
      if ((profile.roles & AMDF_MEMORY_PROFILE_ROLE_IMPORT) != 0) {
        ASSERT_EQ(profile.availability, AMDF_STATUS_OK);
        ordinal = i;
        break;
      }
    }
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const amdf_memory_export_info_t export_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
        .structure_size = sizeof(export_info),
        .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
        .byte_offset = offset,
        .byte_length = length,
    };
    amdf_external_memory_t external = {};
    ASSERT_EQ(api_->memory_export(source, &export_info, &external),
              AMDF_STATUS_OK);
    const amdf_memory_import_info_t import_info = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_IMPORT_INFO,
        .structure_size = sizeof(import_info),
        .memory_profile_ordinal = ordinal,
        .access_count = 1,
        .accesses = &access,
    };
    const auto status =
        api_->memory_import(scope, &import_info, &external, out, nullptr);
    api_->external_memory_release(&external);
    ASSERT_EQ(status, AMDF_STATUS_OK);
    imports_.push_back(*out);
  }

  void Bind(Channel& producer, Channel& consumer,
            amdf_kernel_queue_dependency_t** out) {
    const amdf_kernel_queue_dependency_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_CREATE_INFO,
        .structure_size = sizeof(create),
    };
    ASSERT_EQ(api_->kernel_queue_dependency_create(
                  producer.queue, consumer.queue, &create, out),
              AMDF_STATUS_OK);
    dependencies_.push_back(*out);
    amdf_kernel_queue_dependency_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_DEPENDENCY_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(api_->kernel_queue_dependency_query_info(*out, &info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.protocol,
              AMDF_KERNEL_QUEUE_DEPENDENCY_PROTOCOL_DRM_SYNCOBJ_TIMELINE);
    EXPECT_EQ(info.producer_signal_operation_count, 0u);
    EXPECT_EQ(info.consumer_wait_operation_count, 1u);
  }

  void CreateImage(Channel& channel, amdf_memory_t* memory, uint64_t offset,
                   uint32_t layers, amdf_media_image_t** out) {
    auto description = channel.configuration.output.description;
    description.array_layer_count = layers;
    const amdf_media_image_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory = memory,
        .description = description,
        .byte_offset = offset,
    };
    ASSERT_EQ(media_->image_create(channel.device, &create, out),
              AMDF_STATUS_OK);
    images_.push_back(*out);
  }

  void GetLayout(amdf_media_image_t* image, uint32_t layer,
                 amdf_media_linear_image_layout_t* out) {
    *out = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_LINEAR_IMAGE_LAYOUT,
        .structure_size = sizeof(*out),
    };
    const amdf_media_image_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_QUERY,
        .structure_size = sizeof(query),
        .array_layer = layer,
    };
    amdf_media_image_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_IMAGE_INFO,
        .structure_size = sizeof(info),
        .next = out,
    };
    ASSERT_EQ(media_->image_query_info(image, 0, &query, &info),
              AMDF_STATUS_OK);
    ASSERT_EQ(out->plane_count, 2u);
    ASSERT_EQ(out->array_layer, layer);
  }

  void Prepare(Channel& channel, uint32_t slot, amdf_memory_t* input,
               amdf_memory_t* output, amdf_media_image_t* image,
               uint32_t layer) {
    amdf_media_linear_image_layout_t layout;
    ASSERT_NO_FATAL_FAILURE(GetLayout(image, layer, &layout));
    const Shape shape = GetParam();
    amdf_media_frame_t frame = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
        .structure_size = sizeof(frame),
        .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
        .width = shape.width,
        .height = shape.height,
    };
    if (shape.linear) {
      frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES;
      frame.storage.linear = {2, layout.planes};
    } else {
      frame.storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_NATIVE_IMAGE;
      frame.storage.image = {output, 256, image, layer};
    }
    const amdf_media_decode_picture_info_t decode = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
        .structure_size = sizeof(decode),
        .compressed = {input, 0, 128, input_length_},
        .output = &frame,
    };
    const amdf_media_command_prepare_info_t prepare = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
        .structure_size = sizeof(prepare),
        .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
        .decode = &decode,
    };
    ASSERT_EQ(media_->command_prepare(channel.session, channel.commands[slot],
                                      &prepare),
              AMDF_STATUS_OK);
    amdf_media_command_result_info_t result = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
        .structure_size = sizeof(result),
    };
    EXPECT_EQ(media_->command_query_result(channel.commands[slot], &result),
              amdf_make_api_status(AMDF_STATUS_CODE_FAILED_PRECONDITION));
  }

  void Submit(Channel& channel, uint32_t slot,
              const amdf_kernel_queue_dependency_wait_t* wait = nullptr) {
    const amdf_media_kernel_queue_submission_info_t submit = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(submit),
        .command = channel.commands[slot],
        .wait_count = wait != nullptr ? 1u : 0u,
        .waits = wait,
    };
    uint64_t accepted = 0;
    const auto status =
        media_->kernel_queue_submit(channel.queue, &submit, &accepted);
    if (accepted != 0) {
      channel.last_submission = accepted;
    }
    ASSERT_EQ(status, AMDF_STATUS_OK);
    ASSERT_NE(accepted, 0u);
  }

  void Join(Channel& channel) {
    ASSERT_EQ(api_->kernel_queue_wait(channel.queue, channel.last_submission,
                                      UINT64_MAX, 0),
              AMDF_STATUS_OK);
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->kernel_queue_query_status(channel.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, channel.last_submission);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
  }

  void CheckPixels(CtsMappedMemory& output, amdf_media_image_t* image,
                   uint32_t layers, uint64_t source_offset = 0) {
    ASSERT_EQ(api_->host_mapping_cache_control(
                  output.mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                  output.info.byte_length),
              AMDF_STATUS_OK);
    const Shape shape = GetParam();
    std::vector<bool> writable(output.info.byte_length, false);
    for (uint32_t layer = 0; layer < layers; ++layer) {
      amdf_media_linear_image_layout_t layout;
      ASSERT_NO_FATAL_FAILURE(GetLayout(image, layer, &layout));
      size_t reference_offset = 0;
      for (uint32_t plane = 0; plane < 2; ++plane) {
        auto storage = layout.planes[plane];
        storage.byte_offset += source_offset;
        ASSERT_LT(storage.byte_offset, output.info.byte_length);
        ASSERT_LE(storage.byte_length,
                  output.info.byte_length - storage.byte_offset);
        const uint32_t rows = shape.height >> plane;
        // The entire returned plane extent belongs to the decoder, including
        // pitch and MCU padding. Guard only bytes outside those owned extents.
        std::fill(writable.begin() + storage.byte_offset,
                  writable.begin() + storage.byte_offset + storage.byte_length,
                  true);
        for (uint32_t row = 0; row < rows; ++row) {
          for (uint32_t x = 0; x < shape.width; ++x) {
            const size_t offset =
                storage.byte_offset + row * storage.row_pitch + x;
            const int expected =
                reference_[reference_offset + row * shape.width + x];
            ASSERT_LE(std::abs(int(output.bytes()[offset]) - expected), 1)
                << "layer=" << layer << " plane=" << plane << " x=" << x
                << " y=" << row;
          }
        }
        reference_offset += shape.width * rows;
      }
    }
    for (size_t i = 0; i < writable.size(); ++i) {
      if (!writable[i]) {
        ASSERT_EQ(output.bytes()[i], 0xa5) << "guard offset=" << i;
      }
    }
  }

  void TearDown() override {
    if (api_ == nullptr) {
      return;
    }
    for (auto& channel : channels_) {
      if (channel->last_submission != 0) {
        ASSERT_NO_FATAL_FAILURE(Join(*channel));
      }
    }
    if (event_.type != 0) {
      if (notification_pending_) {
        ASSERT_TRUE(amdf::cts::WaitNativeEvent(event_));
      }
      ASSERT_TRUE(amdf::cts::DestroyNativeEvent(&event_));
    }
    for (auto* dependency : dependencies_) {
      ASSERT_EQ(api_->kernel_queue_dependency_destroy(dependency),
                AMDF_STATUS_OK);
    }
    for (auto& channel : channels_) {
      for (auto* command : channel->commands) {
        if (command != nullptr) {
          ASSERT_EQ(media_->command_destroy(command), AMDF_STATUS_OK);
        }
      }
      if (channel->session != nullptr) {
        ASSERT_EQ(media_->session_destroy(channel->session), AMDF_STATUS_OK);
      }
      if (channel->queue != nullptr) {
        ASSERT_EQ(api_->kernel_queue_destroy(channel->queue), AMDF_STATUS_OK);
      }
    }
    for (auto* image : images_) {
      media_->image_destroy(image);
    }
    for (auto* memory : imports_) {
      ASSERT_EQ(api_->memory_destroy(memory, nullptr), AMDF_STATUS_OK);
    }
    for (auto& memory : memories_) {
      ASSERT_TRUE(memory->Release(api_));
    }
    for (auto& channel : channels_) {
      if (channel->device != nullptr) {
        ASSERT_EQ(api_->device_destroy(channel->device), AMDF_STATUS_OK);
      }
    }
  }

  // Negotiated tables borrowed from the process-lifetime CTS provider.
  const amdf_api_t* api_ = nullptr;
  // Optional media operations; no GPU compute provider is activated.
  const amdf_media_api_t* media_ = nullptr;
  // Endpoint borrowed from the corpus cache.
  amdf_endpoint_t* endpoint_ = nullptr;
  // Case-owned native execution domains in construction order.
  std::vector<std::unique_ptr<Channel>> channels_;
  // Case-owned host-visible backings and views, released after native uses.
  std::vector<std::unique_ptr<CtsMappedMemory>> memories_;
  // Imported logical attachments, released before their source backing.
  std::vector<amdf_memory_t*> imports_;
  // Image identities borrowing the above backings.
  std::vector<amdf_media_image_t*> images_;
  // Cold native queue bindings, released before either queue.
  std::vector<amdf_kernel_queue_dependency_t*> dependencies_;
  // Immutable complete JPEG file embedded in the executable.
  std::span<const uint8_t> compressed_;
  // Independent CPU-decoded NV12 samples with no padded rows.
  std::span<const uint8_t> reference_;
  // Distinct final pixels distinguish the two sides of the native queue chain.
  std::span<const uint8_t> inverse_compressed_;
  // CPU oracle for the inverse input, with the same exact geometry.
  std::span<const uint8_t> inverse_reference_;
  // Hardware-aligned complete compressed extent, excluding the prefix guard.
  uint64_t input_length_ = 0;
  // Caller-owned event-loop destination, closed only after its native delivery.
  amdf_native_event_t event_ = {};
  // One outstanding notification must be reconciled during teardown.
  bool notification_pending_ = false;
};

TEST_P(MediaJpegTest, ReusesFourRecordingsAcrossArrayLayers) {
  Channel& channel = *channels_[0];
  CtsMappedMemory* input = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateInput(channel, &input));
  auto description = channel.configuration.output.description;
  description.array_layer_count = 4;
  amdf_media_image_requirements_t requirements;
  ASSERT_EQ(media_->device_query_image_requirements(
                channel.device, &description, &requirements),
            AMDF_STATUS_OK);
  CtsMappedMemory* output = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(channel, requirements.byte_length + 512, &output));
  amdf_media_image_t* image = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateImage(channel, output->memory, 256, 4, &image));
  for (uint32_t generation = 0; generation < 4; ++generation) {
    std::fill(output->bytes().begin(), output->bytes().end(), 0xa5);
    ASSERT_EQ(api_->host_mapping_cache_control(output->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, output->info.byte_length),
              AMDF_STATUS_OK);
    for (uint32_t slot = 0; slot < 4; ++slot) {
      ASSERT_NO_FATAL_FAILURE(
          Prepare(channel, slot, input->memory, output->memory, image, slot));
      ASSERT_NO_FATAL_FAILURE(Submit(channel, slot));
      EXPECT_EQ(media_->command_destroy(channel.commands[slot]),
                amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
    }
    // No observation has advanced checked retirement. Submission cannot hide
    // an implicit wait/refresh or recycle an accepted recording behind us.
    const amdf_media_kernel_queue_submission_info_t duplicate = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_KERNEL_QUEUE_SUBMISSION_INFO,
        .structure_size = sizeof(duplicate),
        .command = channel.commands[0],
    };
    uint64_t accepted = 99;
    EXPECT_EQ(media_->kernel_queue_submit(channel.queue, &duplicate, &accepted),
              amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
    EXPECT_EQ(accepted, 0u);
    ASSERT_NO_FATAL_FAILURE(Join(channel));
    ASSERT_NO_FATAL_FAILURE(CheckPixels(*output, image, 4));
    for (uint32_t slot = 0; slot < 4; ++slot) {
      amdf_media_command_result_info_t result = {
          .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_RESULT_INFO,
          .structure_size = sizeof(result),
      };
      ASSERT_EQ(media_->command_query_result(channel.commands[slot], &result),
                AMDF_STATUS_OK);
      EXPECT_EQ(result.submission, generation * 4 + slot + 1);
      EXPECT_EQ(result.codec_result, AMDF_MEDIA_CODEC_RESULT_UNAVAILABLE);
    }
  }
  EXPECT_TRUE(std::equal(compressed_.begin(), compressed_.end(),
                         input->bytes().begin() + 128));
}

TEST_P(MediaJpegTest, IndependentImportsPreserveSubrangesAndNativeOwnership) {
  ASSERT_NO_FATAL_FAILURE(CreateChannel());
  if (IsSkipped()) {
    return;
  }
  Channel& source = *channels_[0];
  Channel& consumer = *channels_[1];
  CtsMappedMemory* input = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateInput(source, &input));
  amdf_memory_t* imported_input = nullptr;
  ASSERT_NO_FATAL_FAILURE(Import(consumer, input->memory, 0,
                                 input->info.byte_length, &imported_input));
  const uint64_t extent = consumer.configuration.output.byte_length + 512;
  CtsMappedMemory* output = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(source, 4096 + extent, &output));
  std::fill(output->bytes().begin(), output->bytes().end(), 0xa5);
  ASSERT_EQ(api_->host_mapping_cache_control(output->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             output->info.byte_length),
            AMDF_STATUS_OK);
  amdf_memory_t* first = nullptr;
  amdf_memory_t* second = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      Import(consumer, output->memory, 4096, extent, &first));
  ASSERT_NO_FATAL_FAILURE(
      Import(consumer, output->memory, 4096, extent, &second));
  amdf_memory_info_t info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_INFO,
      .structure_size = sizeof(info),
  };
  ASSERT_EQ(api_->memory_query_info(second, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.source_byte_offset, 4096u);
  EXPECT_EQ(info.byte_length, extent);
  EXPECT_TRUE(amdf_physical_memory_id_is_equal(
      &info.physical_backing_id, &output->info.physical_backing_id));
  const amdf_memory_export_info_t export_info = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_EXPORT_INFO,
      .structure_size = sizeof(export_info),
      .external_memory_type = AMDF_EXTERNAL_MEMORY_TYPE_DMA_BUF_FD,
      .byte_offset = 256,
      .byte_length = 512,
  };
  amdf_external_memory_t external = {};
  ASSERT_EQ(api_->memory_export(second, &export_info, &external),
            AMDF_STATUS_OK);
  EXPECT_EQ(external.source_byte_offset, 4352u);
  EXPECT_EQ(external.byte_length, 512u);
  api_->external_memory_release(&external);
  // PRIME returns one GEM handle for both imports. Releasing one attachment
  // must preserve the other attachment's handle, VA, and subsequent execution.
  const auto release = api_->memory_destroy(first, nullptr);
  imports_.erase(std::find(imports_.begin(), imports_.end(), first));
  ASSERT_EQ(release, AMDF_STATUS_OK);
  amdf_media_image_t* image = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateImage(consumer, second, 256, 1, &image));
  ASSERT_NO_FATAL_FAILURE(
      Prepare(consumer, 0, imported_input, second, image, 0));
  ASSERT_NO_FATAL_FAILURE(Submit(consumer, 0));
  ASSERT_NO_FATAL_FAILURE(Join(consumer));
  ASSERT_NO_FATAL_FAILURE(CheckPixels(*output, image, 1, 4096));
}

TEST_P(MediaJpegTest, NativeDependenciesOrderSharedPixelsWithoutHostJoins) {
  ASSERT_NO_FATAL_FAILURE(CreateChannel());
  if (IsSkipped()) {
    return;
  }
  Channel& producer = *channels_[0];
  Channel& consumer = *channels_[1];
  CtsMappedMemory* first_input = nullptr;
  CtsMappedMemory* last_input = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateInput(producer, &first_input));
  compressed_ = inverse_compressed_;
  reference_ = inverse_reference_;
  ASSERT_NO_FATAL_FAILURE(CreateInput(consumer, &last_input));
  CtsMappedMemory* output = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      producer, producer.configuration.output.byte_length + 512, &output));
  amdf_memory_t* imported = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      Import(consumer, output->memory, 0, output->info.byte_length, &imported));
  amdf_media_image_t* first_image = nullptr;
  amdf_media_image_t* last_image = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateImage(producer, output->memory, 256, 1, &first_image));
  ASSERT_NO_FATAL_FAILURE(CreateImage(consumer, imported, 256, 1, &last_image));
  amdf_kernel_queue_dependency_t* forward = nullptr;
  amdf_kernel_queue_dependency_t* backward = nullptr;
  ASSERT_NO_FATAL_FAILURE(Bind(producer, consumer, &forward));
  ASSERT_NO_FATAL_FAILURE(Bind(consumer, producer, &backward));
  const auto producer_site = output->DeviceSite(0, 0);
  amdf_memory_site_t consumer_site = producer_site;
  consumer_site.value.device.memory = imported;
  amdf_memory_pair_info_t pair = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
      .structure_size = sizeof(pair),
  };
  ASSERT_EQ(api_->memory_query_pair_info(&producer_site, &consumer_site, &pair),
            AMDF_STATUS_OK);
  ASSERT_TRUE(amdf::cts::CreateNativeEvent(&event_));
  for (uint32_t generation = 0; generation < 4; ++generation) {
    std::fill(output->bytes().begin(), output->bytes().end(), 0xa5);
    ASSERT_EQ(api_->host_mapping_cache_control(output->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, output->info.byte_length),
              AMDF_STATUS_OK);
    for (uint32_t slot = 0; slot < 4; ++slot) {
      ASSERT_NO_FATAL_FAILURE(Prepare(producer, slot, first_input->memory,
                                      output->memory, first_image, 0));
      ASSERT_NO_FATAL_FAILURE(
          Prepare(consumer, slot, last_input->memory, imported, last_image, 0));
      const amdf_kernel_queue_dependency_wait_t prior = {
          backward, consumer.last_submission};
      ASSERT_NO_FATAL_FAILURE(Submit(
          producer, slot, prior.producer_submission != 0 ? &prior : nullptr));
      const amdf_kernel_queue_dependency_wait_t decoded = {
          forward, producer.last_submission};
      ASSERT_NO_FATAL_FAILURE(Submit(consumer, slot, &decoded));
    }
    ASSERT_EQ(api_->kernel_queue_request_notification(
                  consumer.queue, consumer.last_submission, &event_),
              AMDF_STATUS_OK);
    notification_pending_ = true;
    ASSERT_TRUE(amdf::cts::WaitNativeEvent(event_));
    notification_pending_ = false;
    ASSERT_NO_FATAL_FAILURE(Join(consumer));
    // The final consumer fence covers all preceding GPU work. Checking the
    // producer's original errors is still required before its slots are reused.
    amdf_kernel_queue_status_t status = {
        .type = AMDF_STRUCTURE_TYPE_KERNEL_QUEUE_STATUS,
        .structure_size = sizeof(status),
    };
    ASSERT_EQ(api_->kernel_queue_refresh_status(producer.queue, &status),
              AMDF_STATUS_OK);
    ASSERT_EQ(status.retired_submission, producer.last_submission);
    ASSERT_EQ(status.terminal_status, AMDF_STATUS_OK);
    ASSERT_NO_FATAL_FAILURE(CheckPixels(*output, last_image, 1));
  }
  // A notification of an old checked point remains valid after slot reuse.
  ASSERT_EQ(api_->kernel_queue_request_notification(consumer.queue, 1, &event_),
            AMDF_STATUS_OK);
  notification_pending_ = true;
  ASSERT_TRUE(amdf::cts::WaitNativeEvent(event_));
  notification_pending_ = false;
}

TEST_P(MediaJpegTest, RejectsUnsupportedEnvelopesAndInvalidPlaneExtents) {
  Channel& channel = *channels_[0];
  for (amdf_memory_access_t permission :
       {AMDF_MEMORY_ACCESS_READ, AMDF_MEMORY_ACCESS_WRITE}) {
    const amdf_memory_device_access_t access = {channel.device,
                                                {.access = permission}};
    const amdf_memory_create_info_t create = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(create),
        .memory_profile_ordinal =
            channel.configuration.image_heap_memory_profile_ordinal,
        .access_count = 1,
        .byte_length = 4096,
        .accesses = &access,
    };
    amdf_memory_t* memory = nullptr;
    EXPECT_EQ(api_->memory_create(channel.configuration.image_heap_memory_scope,
                                  &create, &memory, nullptr),
              amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(memory, nullptr);
  }
  const auto original = channel.configuration.configuration;
  for (uint32_t predicate = 0; predicate < 3; ++predicate) {
    amdf_media_configuration_query_t query = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_QUERY,
        .structure_size = sizeof(query),
        .configuration = original,
    };
    const amdf_media_configuration_reason_t reasons[] = {
        AMDF_MEDIA_CONFIGURATION_REASON_FORMAT,
        AMDF_MEDIA_CONFIGURATION_REASON_DIMENSIONS,
        AMDF_MEDIA_CONFIGURATION_REASON_REFERENCE_STORAGE};
    if (predicate == 0) {
      query.configuration.format = AMDF_MEDIA_PIXEL_FORMAT_P010;
    }
    if (predicate == 1) {
      --query.configuration.coded_width;
    }
    if (predicate == 2) {
      query.configuration.reference_image_count = 1;
    }
    amdf_media_configuration_info_t info = {
        .type = AMDF_STRUCTURE_TYPE_MEDIA_CONFIGURATION_INFO,
        .structure_size = sizeof(info),
    };
    ASSERT_EQ(media_->device_query_configuration(channel.device, &query, &info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.support, AMDF_MEDIA_CONFIGURATION_SUPPORT_UNSUPPORTED);
    EXPECT_EQ(info.reason, reasons[predicate]);
    EXPECT_EQ(info.configuration.coded_width, query.configuration.coded_width);
    EXPECT_EQ(info.configuration.format, query.configuration.format);
    EXPECT_EQ(info.configuration.reference_image_count,
              query.configuration.reference_image_count);
  }
  CtsMappedMemory* input = nullptr;
  CtsMappedMemory* output = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateInput(channel, &input));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      channel, channel.configuration.output.byte_length + 512, &output));
  amdf_media_image_t* image = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateImage(channel, output->memory, 256, 1, &image));
  amdf_media_linear_image_layout_t layout;
  ASSERT_NO_FATAL_FAILURE(GetLayout(image, 0, &layout));
  amdf_media_frame_t frame = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_FRAME,
      .structure_size = sizeof(frame),
      .format = AMDF_MEDIA_PIXEL_FORMAT_NV12,
      .width = original.coded_width,
      .height = original.coded_height,
      .storage_kind = AMDF_MEDIA_FRAME_STORAGE_KIND_LINEAR_PLANES,
  };
  frame.storage.linear = {2, layout.planes};
  amdf_media_decode_picture_info_t decode = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_DECODE_PICTURE_INFO,
      .structure_size = sizeof(decode),
      .compressed = {input->memory, 0, 128, input_length_},
      .output = &frame,
  };
  const amdf_media_command_prepare_info_t prepare = {
      .type = AMDF_STRUCTURE_TYPE_MEDIA_COMMAND_PREPARE_INFO,
      .structure_size = sizeof(prepare),
      .operation = AMDF_MEDIA_COMMAND_OPERATION_PICTURE,
      .decode = &decode,
  };
  const auto good = layout.planes[0];
  --layout.planes[0].byte_length;
  EXPECT_EQ(
      media_->command_prepare(channel.session, channel.commands[0], &prepare),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  layout.planes[0] = good;
  --layout.planes[0].row_pitch;
  EXPECT_EQ(
      media_->command_prepare(channel.session, channel.commands[0], &prepare),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  layout.planes[0] = good;
  layout.planes[0].access_ordinal = 1;
  EXPECT_EQ(
      media_->command_prepare(channel.session, channel.commands[0], &prepare),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  layout.planes[0] = good;
  decode.compressed.memory = output->memory;
  EXPECT_EQ(
      media_->command_prepare(channel.session, channel.commands[0], &prepare),
      amdf_make_api_status(AMDF_STATUS_CODE_INVALID_ARGUMENT));
  decode.compressed.memory = input->memory;
  // A plane can describe a larger accessible range than its write footprint.
  // The disjoint luma/chroma write extents still fit their explicit pitches.
  layout.planes[0].byte_length = output->info.byte_length - good.byte_offset;
  std::fill(output->bytes().begin(), output->bytes().end(), 0xa5);
  ASSERT_EQ(api_->host_mapping_cache_control(output->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             output->info.byte_length),
            AMDF_STATUS_OK);
  ASSERT_EQ(
      media_->command_prepare(channel.session, channel.commands[0], &prepare),
      AMDF_STATUS_OK);
  ASSERT_NO_FATAL_FAILURE(Submit(channel, 0));
  ASSERT_NO_FATAL_FAILURE(Join(channel));
  ASSERT_NO_FATAL_FAILURE(CheckPixels(*output, image, 1));
}

INSTANTIATE_TEST_SUITE_P(
    NativeAndLinear, MediaJpegTest,
    ::testing::Values(Shape{128, 96, false}, Shape{80, 72, false},
                      Shape{128, 96, true}, Shape{80, 72, true}),
    [](const ::testing::TestParamInfo<Shape>& info) {
      return std::string(info.param.linear ? "Planes" : "Image") +
             std::to_string(info.param.width) + "x" +
             std::to_string(info.param.height);
    });

}  // namespace
