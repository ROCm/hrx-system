// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/program.h"

#include <array>
#include <cstring>
#include <tuple>
#include <vector>

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

struct InstanceState {
  // Readable lifecycle boundary for failure attribution.
  const char* name;
  // Initial bytes have reached this resident allocation.
  bool initialized;
  // An accepted invocation already prepared this exact allocation.
  uint64_t last_epoch;
  // Zero identifies the permanent source; other ordinals identify clones.
  uint32_t ordinal;
  // The first block needs its initial bytes copied.
  bool copy;
  // Base-only relocations need their first application.
  bool relocate;
};

constexpr InstanceState kStates[] = {
    {"mapped initial", true, 0, 0, false, true},
    {"unmapped initial", false, 0, 0, true, true},
    {"new clone", false, 0, 1, true, true},
    {"reused initial", true, 37, 0, false, false},
    {"reused clone", true, 41, 1, false, false},
};

class ProgramPreparationTest
    : public ::testing::TestWithParam<
          std::tuple<uint32_t, uint32_t, bool, bool, bool>> {};

TEST_P(ProgramPreparationTest, CapturedPreparationMatchesEmittedPackets) {
  const auto [state_index, value_count, first, profiling, fixups] = GetParam();
  const auto& state = kStates[state_index];
  SCOPED_TRACE(state.name);
  const iree_hal_amd_gpu_aql_program_pool_t pool = {
      .blit = 0x100000,
      .fixup = 0x200000,
      .materialize = 0x300000,
      .materialize_profile = 0x400000,
  };
  iree_hal_amd_gpu_program_t program = {
      .upload_address = UINT64_C(0x1234560000),
      .byte_length = 4096,
      .fixup_count = fixups ? 129u : 0u,
      .fixup_offset = 4096,
  };
  iree_hal_amd_gpu_program_instance_t instance = {
      .address = UINT64_C(0x2345670000),
      .last_epoch = state.last_epoch,
      .ordinal = state.ordinal,
      .initialized = state.initialized,
      .reserved = true,
  };
  auto flags = iree_hal_amd_gpu_aql_program_prepare(&program, &instance,
                                                    value_count, first);
  if (profiling) {
    flags |= IREE_HAL_AMD_GPU_AQL_PROGRAM_PROFILE;
  }
  const bool copy = first && state.copy;
  const bool relocate = first && fixups && (state.relocate || value_count > 1);
  EXPECT_EQ((flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_FIRST) != 0, first);
  EXPECT_EQ((flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_INITIALIZE) != 0, copy);
  EXPECT_EQ((flags & IREE_HAL_AMD_GPU_AQL_PROGRAM_FIXUP) != 0, relocate);
  std::vector<uint64_t> kernels;
  if (copy) {
    kernels.push_back(pool.blit);
  }
  if (relocate) {
    kernels.push_back(pool.fixup);
  }
  kernels.push_back(profiling ? pool.materialize_profile : pool.materialize);
  const uint32_t prefix = iree_hal_amd_gpu_aql_program_prefix_count(flags);
  ASSERT_EQ(prefix, kernels.size());

  // Publication installs the next coordinate before native emission. The
  // captured preparation must still describe the reserved invocation.
  instance.initialized = true;
  instance.last_epoch = 99;
  std::array<iree_hal_amd_gpu_aql_packet_t, 8> packets = {};
  for (auto& packet : packets) {
    packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
  }
  iree_atomic_int64_t read_index = IREE_ATOMIC_VAR_INIT(6);
  iree_atomic_int64_t write_index = IREE_ATOMIC_VAR_INIT(6);
  uint64_t doorbell = 0;
  iree_hal_amd_gpu_aql_ring_t ring;
  iree_hal_amd_gpu_aql_ring_initialize(
      packets.data(), 0x500000, packets.size(), &read_index, &write_index,
      &doorbell, AMDF_CACHE_TRANSITION_EXECUTOR_NONE, &ring);
  ring.reserved = ring.consumed = 6;
  constexpr uint32_t kBlockPackets = 3;
  uint64_t position = 0;
  ASSERT_TRUE(iree_hal_amd_gpu_aql_ring_try_reserve(
      &ring, prefix + kBlockPackets + 1, &position));
  std::array<uint64_t, 32> upload = {};
  constexpr uint64_t kUploadAddress = 0x600000;
  ASSERT_EQ(
      iree_hal_amd_gpu_aql_program_emit(
          &pool, &program, &instance, 128, kBlockPackets, &ring, nullptr,
          position, kUploadAddress, 0x900008, 0x700000, upload.data(), flags),
      AMDF_STATUS_OK);
  for (uint32_t i = 0; i < prefix; ++i) {
    const auto& packet = packets[(position + i) % packets.size()];
    EXPECT_EQ(packet.words[8], static_cast<uint32_t>(kernels[i]));
    EXPECT_EQ(packet.words[9], static_cast<uint32_t>(kernels[i] >> 32));
    EXPECT_NE(packet.words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
    EXPECT_NE(packet.words[0] & IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED, 0u);
    EXPECT_EQ(packet.words[0] & (3u << 9),
              IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  }
  for (uint32_t i = 0; i <= kBlockPackets; ++i) {
    EXPECT_EQ(packets[(position + prefix + i) % packets.size()].words[0],
              IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  }
  const auto* arguments =
      reinterpret_cast<const iree_hal_amd_gpu_aql_program_arguments_t*>(
          upload.data());
  if (copy) {
    EXPECT_EQ(arguments->copy.source, program.upload_address);
    EXPECT_EQ(arguments->copy.target, instance.address);
    EXPECT_EQ(arguments->copy.length,
              state.ordinal
                  ? program.byte_length
                  : program.fixup_offset +
                        program.fixup_count * sizeof(iree_hal_amd_gpu_fixup_t));
  }
  if (relocate) {
    EXPECT_EQ(arguments->fixup.entries,
              program.upload_address + program.fixup_offset);
    EXPECT_EQ(arguments->fixup.target, instance.address);
    EXPECT_EQ(arguments->fixup.entry_count, program.fixup_count);
    EXPECT_EQ(arguments->fixup.values, 0x900008u);
  }
  const auto* materialize =
      first ? &arguments->materialize
            : reinterpret_cast<
                  const iree_hal_amd_gpu_aql_materialize_arguments_t*>(
                  upload.data());
  EXPECT_EQ(materialize->packets, instance.address + 128);
  EXPECT_EQ(materialize->first, (position + prefix) % packets.size());
  EXPECT_EQ(materialize->packet_count, kBlockPackets);
}

INSTANTIATE_TEST_SUITE_P(
    InstancePreparation, ProgramPreparationTest,
    ::testing::Combine(::testing::Range(0u, uint32_t{IREE_ARRAYSIZE(kStates)}),
                       ::testing::Values(1u, 2u), ::testing::Bool(),
                       ::testing::Bool(), ::testing::Bool()));

}  // namespace
