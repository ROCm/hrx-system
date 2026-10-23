// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "amdf/media/h264.h"

#include <array>
#include <cstring>

#include "amdf/media/gfx11.h"
#include "libamdf/cts/interop/gpu/media/decode_fixture.h"
#include "libamdf/cts/media/h264_fixture.h"

namespace {

class MediaH264InteropTest : public MediaDecodeInteropTest {
 protected:
  void SetUp() override {
    MediaDecodeInteropTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    h264_.type = AMDF_STRUCTURE_TYPE_MEDIA_H264_CONFIGURATION;
    h264_.structure_size = sizeof(h264_);
    h264_.profile = AMDF_MEDIA_H264_PROFILE_MAIN;
    h264_.level_idc = 13;
    ASSERT_NO_FATAL_FAILURE(
        CreateSession({AMDF_MEDIA_CODEC_H264, AMDF_MEDIA_DIRECTION_DECODE,
                       AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 4},
                      &h264_));
  }
  void RunH264(uint32_t frame_count, uint32_t recording_count) {
    std::array<amdf::cts::media::H264FixturePicture, 12> fixture = {};
    ASSERT_NO_FATAL_FAILURE(amdf::cts::media::LoadH264Fixture(&fixture));
    amdf_media_h264_scaling_matrices_t matrices;
    std::memset(&matrices, 16, sizeof(matrices));
    std::array<DecodePictureInput, 12> inputs = {};
    for (uint32_t index = 0; index < fixture.size(); ++index) {
      auto& entry = fixture[index];
      entry.picture.scaling_matrices = &matrices;
      for (uint32_t reference = 0; reference < entry.picture.reference_count;
           ++reference) {
        ASSERT_LT(entry.reference_slots[reference], reference_frames_.size());
        entry.references[reference].binding.slot_index =
            entry.reference_slots[reference];
        entry.references[reference].binding.frame =
            &reference_frames_[entry.reference_slots[reference]];
      }
      inputs[index] = {entry.bytes, entry.byte_length, entry.reference_slot,
                       &entry.picture,
                       amdf::cts::media::kH264PixelHashes[index]};
    }
    ASSERT_NO_FATAL_FAILURE(RunPictures(inputs, frame_count, recording_count));
  }
  // Parsed envelope kept live through exact admission and session creation.
  amdf_media_h264_configuration_t h264_ = {};
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
  ASSERT_NO_FATAL_FAILURE(RunH264(12, 12));
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
  const amdf_media_image_description_t descriptions[] = {
      {AMDF_MEDIA_PIXEL_FORMAT_NV12, 256, 256, 2,
       AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT},
      {AMDF_MEDIA_PIXEL_FORMAT_P010, 256, 256, 2,
       AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT},
      {AMDF_MEDIA_PIXEL_FORMAT_NV12, 520, 208, 9,
       AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT},
      {AMDF_MEDIA_PIXEL_FORMAT_P010, 520, 208, 9,
       AMDF_MEDIA_IMAGE_ROLE_DECODE_OUTPUT},
  };
  for (const auto& description : descriptions) {
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
    for (uint32_t layer = 0; layer < description.array_layer_count; ++layer) {
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
  ASSERT_NO_FATAL_FAILURE(RunH264(32, 4));
}

}  // namespace
