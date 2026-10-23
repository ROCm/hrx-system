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
#include "libamdf/cts/interop/gpu/media/reference_main.h"

namespace {

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
    std::array<FixturePicture, 12> fixture = {};
    ASSERT_NO_FATAL_FAILURE(LoadFixture(&fixture));
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
                       &entry.picture, kPixelHashes[index]};
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
  ASSERT_NO_FATAL_FAILURE(RunH264(32, 4));
}

}  // namespace
