// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/aql/program.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"

namespace {

TEST(ProgramTest, HostPublicationUsesTaggedSamplesAcrossMixedPacketsAndWrap) {
  const iree_hal_amd_gpu_aql_dispatch_t dispatch = {
      .header = IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH | (1u << 16),
      .workgroup_size = {64, 1, 1},
      .grid_size = {64, 1, 1},
      .kernel_object = 0x10000,
      .kernarg_address = 0x20000,
  };
  std::array<iree_hal_amd_gpu_aql_packet_t, 4> source = {
      iree_hal_amd_gpu_aql_packet_dispatch(&dispatch),
      iree_hal_amd_gpu_aql_packet_indirect_buffer(
          IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED, 0x30000, 6, 0),
      iree_hal_amd_gpu_aql_packet_barrier(
          IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
              IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_AGENT,
          0, nullptr, 0),
      iree_hal_amd_gpu_aql_packet_dispatch(&dispatch),
  };
  source[0].words[14] = 1;
  source[3].words[14] = 2;
  const auto original = source;
  for (bool capture : {false, true}) {
    SCOPED_TRACE(capture);
    std::array<iree_hal_amd_gpu_aql_packet_t, 8> published = {};
    for (auto& packet : published) {
      packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    iree_hal_amd_gpu_aql_ring_t ring = {
        .packets = published.data(),
        .capacity = published.size(),
        .reserved = 10,
        .consumed = 6,
    };
    std::array<iree_hal_amd_gpu_aql_signal_t, 3> samples;
    memset(samples.data(), 0xA5, sizeof(samples));
    const auto guard = samples[2];
    const uint64_t sample_address = capture ? UINT64_C(0x123400001000) : 0;
    ASSERT_EQ(iree_hal_amd_gpu_aql_program_emit_host(
                  source.size(), source.data(), &ring, nullptr, 6,
                  sample_address, capture ? samples.data() : nullptr),
              AMDF_STATUS_OK);
    for (size_t i = 0; i < source.size(); ++i) {
      auto expected = source[i];
      expected.words[0] |= IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
      if (i == 0) {
        expected.words[0] |= IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED;
      }
      const uint64_t completion =
          capture && (i == 0 || i == 3)
              ? sample_address + (i == 3 ? sizeof(samples[0]) : 0)
              : 0;
      expected.words[14] = static_cast<uint32_t>(completion);
      expected.words[15] = static_cast<uint32_t>(completion >> 32);
      EXPECT_EQ(memcmp(&published[(6 + i) % published.size()], &expected,
                       sizeof(expected)),
                0);
    }
    const iree_hal_amd_gpu_aql_signal_t initialized = {.kind = 1, .value = 1};
    for (size_t i = 0; i < 2; ++i) {
      EXPECT_EQ(memcmp(&samples[i], capture ? &initialized : &guard,
                       sizeof(samples[i])),
                0);
    }
    EXPECT_EQ(memcmp(&samples[2], &guard, sizeof(guard)), 0);
    EXPECT_EQ(memcmp(source.data(), original.data(), sizeof(source)), 0);
  }
}

TEST(ProgramTest, HostPublicationStopsAtTheFirstNativeFailure) {
  std::array<iree_hal_amd_gpu_aql_packet_t, 4> published = {};
  for (auto& packet : published) {
    packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
  }
  uint32_t calls = 0;
  const amdf_api_t api = {
      .user_queue_mapping_flush =
          [](amdf_user_queue_mapping_t* mapping) {
            auto& calls = *reinterpret_cast<uint32_t*>(mapping);
            return ++calls == 2
                       ? amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST)
                       : AMDF_STATUS_OK;
          },
  };
  const iree_hal_amd_gpu_context_t context = {.api = &api};
  const iree_hal_amd_gpu_user_queue_t storage = {
      .context = &context,
      .mapping = reinterpret_cast<amdf_user_queue_mapping_t*>(&calls),
      .host_publish_executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
  };
  iree_hal_amd_gpu_aql_ring_t ring = {
      .packets = published.data(),
      .capacity = published.size(),
      .host_publish_executor = AMDF_CACHE_TRANSITION_EXECUTOR_HOST_API,
      .reserved = 3,
  };
  std::array<iree_hal_amd_gpu_aql_packet_t, 3> source;
  source.fill(iree_hal_amd_gpu_aql_packet_barrier(
      IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND, 0, nullptr, 0));
  EXPECT_EQ(iree_hal_amd_gpu_aql_program_emit_host(
                source.size(), source.data(), &ring, &storage, 0, 0, nullptr),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(calls, 2u);
  EXPECT_EQ(published[0].words[0],
            IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  for (size_t i = 1; i < published.size(); ++i) {
    EXPECT_EQ(published[i].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  }
}

}  // namespace
