// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBAMDF_CTS_INTEROP_GPU_MEDIA_DECODE_FIXTURE_H_
#define LIBAMDF_CTS_INTEROP_GPU_MEDIA_DECODE_FIXTURE_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <wrl/client.h>

#include <array>
#include <span>
#include <vector>

#include "amdf/media.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

// One application-parsed input with an independent complete-image fingerprint.
struct DecodePictureInput {
  // Borrowed padded compressed bytes.
  const uint8_t* bytes;
  // Complete native input extent.
  uint32_t byte_length;
  // Application-selected reconstructed output slot.
  uint32_t reference_slot;
  // Borrowed codec-specific public picture parameters.
  const void* parameters;
  // FNV-1a over per-pixel Y/U/V component bytes from independent decode.
  uint64_t pixel_hash;
};

// Shared application ownership flow for D3D12 media and the native GPU KMQ.
class MediaDecodeInteropTest : public GpuCommandTest {
 protected:
  MediaDecodeInteropTest();
  void SetUp() override;
  void TearDown() override;
  void CreateSession(const amdf_media_configuration_t& configuration,
                     const void* codec_configuration);
  void CreateSharedMemory(uint64_t byte_length, amdf_memory_t** out_memory,
                          uint64_t* out_address);
  void CreateRecordings(uint32_t count);
  void SubmitGpu(const amdf_gpu_kernel_command_t& command,
                 const amdf_kernel_queue_dependency_wait_t* wait = nullptr);
  void RunPictures(std::span<const DecodePictureInput> fixture,
                   uint32_t frame_count, uint32_t recording_count);
  void CreateImage(amdf_memory_t* memory, uint64_t offset,
                   const amdf_media_image_description_t& description,
                   amdf_media_image_t** out_image);
  // Optional media table negotiated independently of compute.
  const amdf_media_api_t* media_ = nullptr;
  // Case-owned media activation on the compute endpoint.
  amdf_device_t* media_device_ = nullptr;
  // Exact native admission and storage requirements.
  amdf_media_configuration_info_t configuration_ = {};
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
  Microsoft::WRL::ComPtr<ID3D12InfoQueue> validation_;
  // Stable caller frame records; codec fixtures bind references before image
  // creation.
  std::array<amdf_media_frame_t, 17> reference_frames_ = {};
};

#endif  // LIBAMDF_CTS_INTEROP_GPU_MEDIA_DECODE_FIXTURE_H_
