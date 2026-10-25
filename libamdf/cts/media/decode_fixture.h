// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef AMDF_CTS_MEDIA_DECODE_FIXTURE_H_
#define AMDF_CTS_MEDIA_DECODE_FIXTURE_H_

#include <array>
#include <span>
#include <vector>

#include "amdf/media.h"
#include "gtest/gtest.h"
#include "util/mapped_memory.h"

namespace amdf::cts::media {

struct DecodeInput {
  // Borrowed compressed slice bytes.
  const uint8_t* bytes;
  // Native padded compressed extent.
  uint32_t byte_length;
  // Stable reconstructed output slot chosen by the caller.
  uint32_t reference_slot;
  // Borrowed typed public codec picture syntax.
  const void* picture;
  // Independent FNV-1a over every raster pixel's Y/U/V storage bytes.
  uint64_t pixel_hash;
};

// Actual API caller for native BEGIN/PICTURE/END codecs with host-visible
// linear outputs. Four recordings and four outputs are reused after batch
// retirement; all reference placement and scheduling remain in this caller.
class MediaDecodeTest : public ::testing::Test {
 protected:
  void SetUpMedia(const amdf_media_configuration_t& configuration,
                  const void* codec_configuration, bool linear_frames);
  void CreateCommand(amdf_media_command_t** out);
  void CreateMemory(uint64_t length, CtsMappedMemory* memory);
  void CreateFrames(const amdf_media_image_requirements_t& requirements,
                    CtsMappedMemory* memory, amdf_media_image_t** image,
                    std::span<amdf_media_linear_image_layout_t> layouts,
                    std::span<amdf_media_frame_t> frames,
                    bool* out_linear_layout, uint64_t* out_layer_byte_length);
  void CreatePictures();
  void Cache(CtsMappedMemory& memory, amdf_host_cache_operation_t operation);
  amdf_status_t PreparePicture(const DecodeInput& entry, uint32_t slot);
  void PrepareControl(amdf_media_command_operation_t operation);
  amdf_status_t Submit(amdf_media_command_t* command);
  void Join();
  void Begin();
  void End();
  void CheckGuards(CtsMappedMemory& memory,
                   std::span<const amdf_media_linear_image_layout_t> layouts);
  uint64_t HashFramePixels(
      const CtsMappedMemory& memory,
      const amdf_media_linear_image_layout_t& layout) const;
  void CheckFrameWritten(const CtsMappedMemory& memory,
                         const amdf_media_linear_image_layout_t& layout,
                         uint8_t initial_value);
  void CheckReferenceGuards();
  void CheckReferenceWritten(uint32_t slot, uint8_t initial_value);
  void CheckPixels(std::span<const DecodeInput> fixture, uint32_t first);
  void TearDown() override;
  void RunPictures(std::span<const DecodeInput> fixture,
                   uint32_t picture_count);

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
  // Reference identity placed at a nonzero offset with the configured slot
  // count.
  amdf_media_image_t* reference_image_ = nullptr;
  // Retained plane metadata for direct-plane preparation and host checking.
  std::array<amdf_media_linear_image_layout_t, 4> output_layouts_ = {};
  // Retained reference plane metadata and protected extents.
  std::vector<amdf_media_linear_image_layout_t> reference_layouts_ = {};
  // Whether the reference identity exposes direct linear planes.
  bool reference_linear_layout_ = false;
  // Physical byte extent of each decoder-private reference layer.
  uint64_t reference_layer_byte_length_ = 0;
  // Borrowed output frames, with either native identities or direct planes.
  std::array<amdf_media_frame_t, 4> outputs_ = {};
  // Stable application reference frame bindings.
  std::vector<amdf_media_frame_t> references_ = {};
  // Selected native codec queue family within this media device.
  uint32_t family_ = 0;
  // Whether the caller supplies direct planes for each linear native image.
  bool linear_frames_ = false;
};

}  // namespace amdf::cts::media

#endif  // AMDF_CTS_MEDIA_DECODE_FIXTURE_H_
