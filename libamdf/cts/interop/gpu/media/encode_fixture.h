// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LIBAMDF_CTS_INTEROP_GPU_MEDIA_ENCODE_FIXTURE_H_
#define LIBAMDF_CTS_INTEROP_GPU_MEDIA_ENCODE_FIXTURE_H_

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <wrl/client.h>

#include <vector>

#include "amdf/media.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

struct EncodeExecutionCase {
  // Codec selected by the configuration and picture chains.
  amdf_media_codec_t codec;
  // Native input and reconstruction sample representation.
  amdf_media_pixel_format_t format;
  // Logical component precision used by codec syntax.
  uint32_t sample_bit_depth;
};

// Shared application ownership flow for a GPU producer, D3D12 video encoder,
// and GPU observer over caller-owned memory and native dependencies.
class MediaEncodeInteropTest : public GpuCommandTest {
 protected:
  MediaEncodeInteropTest();
  void SetUp() override;
  void TearDown() override;
  void RunPictures(const EncodeExecutionCase& test_case,
                   amdf_media_encode_rate_control_mode_t rate_control_mode,
                   uint32_t width, uint32_t height, uint32_t frame_count,
                   uint32_t recording_count);

 private:
  void CreateSharedMemory(amdf_memory_scope_t* scope, uint32_t profile_ordinal,
                          uint64_t byte_length, amdf_memory_t** out_memory,
                          uint64_t* out_address);
  void SubmitGpu(const amdf_gpu_kernel_command_t& command,
                 const amdf_kernel_queue_dependency_wait_t* wait = nullptr);

  // Optional typed media table negotiated independently of compute.
  const amdf_media_api_t* media_ = nullptr;
  // Case-owned media activation on the selected compute endpoint.
  amdf_device_t* media_device_ = nullptr;
  // Native encode family admitting every initial encode schema.
  uint32_t encode_family_ = 0;
  // Native video-encode ordering domain.
  amdf_kernel_queue_t* encode_queue_ = nullptr;
  // Separate native GPU KMQ producing and observing shared bytes.
  amdf_kernel_queue_t* gpu_queue_ = nullptr;
  // Cold GPU-to-encode dependency binding.
  amdf_kernel_queue_dependency_t* gpu_to_encode_ = nullptr;
  // Cold encode-to-GPU dependency binding.
  amdf_kernel_queue_dependency_t* encode_to_gpu_ = nullptr;
  // Last accepted encode prefix, retained through failure and teardown.
  uint64_t encode_point_ = 0;
  // Last accepted GPU prefix, retained through failure and teardown.
  uint64_t gpu_point_ = 0;
  // Application-owned backing destroyed after identities and last uses.
  std::vector<amdf_memory_t*> memories_;
  // Explicit placed identities destroyed before their backing.
  std::vector<amdf_media_image_t*> images_;
  // Fixed recording owners destroyed before their immutable session.
  std::vector<amdf_media_command_t*> commands_;
  // Active exact encode session.
  amdf_media_session_t* session_ = nullptr;
  // Explicit host view of caller-owned feedback.
  amdf_host_mapping_t* feedback_mapping_ = nullptr;
  // Direct pointer borrowed from feedback_mapping_.
  uint8_t* feedback_ = nullptr;
  // Native debug view kept through complete media teardown.
  Microsoft::WRL::ComPtr<ID3D12InfoQueue> validation_;
};

#endif  // LIBAMDF_CTS_INTEROP_GPU_MEDIA_ENCODE_FIXTURE_H_
