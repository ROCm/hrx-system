// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/command_buffer.h"

#include <array>
#include <memory>
#include <vector>

#include "iree/hal/device_spec.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Real public recording, binding validation/capture, and native projection.
// Only buffer backing is supplied by the fixture. Native carrier CTS separately
// qualifies packet execution and visibility on physical hardware.
class CommandBufferTest : public ::testing::Test {
 protected:
  struct Buffer {
    // Public identity used by the production resource set and binding capture.
    iree_hal_buffer_t base = {};
    // Committed native address, independently changed before invocation
    // resolve.
    iree_hal_buffer_native_binding_t binding = {};
  };

  struct Invocation {
    ~Invocation() {
      if (value.command_buffer) {
        iree_hal_amd_gpu_sdma_command_invocation_deinitialize(&value);
      }
    }
    iree_status_t Initialize(iree_hal_command_buffer_t* buffer,
                             iree_hal_buffer_binding_table_t bindings =
                                 iree_hal_buffer_binding_table_empty(),
                             iree_hal_queue_execute_flags_t flags = 0) {
      IREE_RETURN_IF_ERROR(
          iree_hal_command_buffer_validate_submission(buffer, bindings));
      const auto size =
          iree_hal_amd_gpu_sdma_command_buffer_invocation_size(buffer);
      storage.resize((size + 7) / 8);
      return iree_hal_amd_gpu_sdma_command_invocation_initialize(
          buffer, bindings, flags, storage.data(), &value);
    }
    iree_status_t Resolve(uint64_t address) {
      return iree_hal_amd_gpu_sdma_command_invocation_resolve(&value, 0,
                                                              address);
    }
    // Binding capture and dense values with independent invocation ownership.
    iree_hal_amd_gpu_sdma_command_invocation_t value = {};
    // Aligned host allocation exactly covering the recorded invocation layout.
    std::vector<uint64_t> storage;
    // Exclusive unpublished suffix; no state is written into the recording.
    iree_hal_amd_gpu_sdma_program_cursor_t cursor = {};
  };

  void SetUp() override {
    const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    const iree_hal_buffer_binding_layout_t layout = {
        sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 2, &layout, iree_allocator_system(), &contract_));
    contract_->scopes[1].interfaces =
        1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    contract_->scopes[1].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
    contract_->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
    spec_.role_flags = IREE_HAL_QUEUE_FAMILY_ROLE_FLAG_TRANSFER;
    spec_.flags = IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_COMMAND_TIMESTAMPS |
                  IREE_HAL_QUEUE_FAMILY_SPEC_FLAG_INDIRECT_COMMAND_TIMESTAMPS;
    family_.spec = &spec_;
    family_.memory = {.domain = this, .queue_scope_id = 1};
  }

  void TearDown() override {
    iree_hal_command_buffer_release(recording_);
    for (auto& buffer : buffers_) {
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer->base.resource.ref_count),
                1);
      iree_hal_buffer_release(&buffer->base);
    }
    iree_hal_memory_contract_release(contract_);
  }

  Buffer* MakeBuffer(uint64_t address, uint64_t length) {
    auto buffer = std::make_unique<Buffer>();
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy = [](iree_hal_buffer_t*) {},
    };
    iree_hal_buffer_initialize(
        {}, &buffer->base, length, 0, length, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable,
        &buffer->base);
    buffer->binding.device_address = address;
    buffer->base.memory = {.contract = contract_, .bindings = &buffer->binding};
    auto* result = buffer.get();
    buffers_.push_back(std::move(buffer));
    return result;
  }

  void Begin(
      uint32_t binding_capacity = 0, amdf_queue_format_features_t features = 0,
      amdf_queue_publication_modes_t mode = AMDF_QUEUE_PUBLICATION_MODE_USER) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_sdma_command_buffer_create(
        &family_, features,
        iree_hal_amd_gpu_sdma_barrier_profile(features, mode), 0,
        IREE_HAL_COMMAND_CATEGORY_TRANSFER, binding_capacity,
        iree_allocator_system(), &recording_));
    IREE_ASSERT_OK(iree_hal_command_buffer_begin(recording_));
  }

  void Barrier(iree_hal_barrier_flags_t flags = 0) {
    const iree_hal_barrier_t barrier = {
        .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
        .flags = flags,
    };
    IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &barrier));
  }

  std::vector<uint32_t> Emit(
      Invocation& invocation,
      iree_hal_amd_gpu_sdma_program_progress_t* progress) {
    std::array<uint32_t, IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS + 2> words;
    words.fill(0x95d3c7a1);
    uint32_t count = 0;
    *progress = iree_hal_amd_gpu_sdma_program_emit(
        iree_hal_amd_gpu_sdma_command_buffer_program(
            invocation.value.command_buffer),
        invocation.value.values, &invocation.cursor, words.data() + 1, &count);
    EXPECT_LE(count, IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS);
    EXPECT_EQ(words.front(), 0x95d3c7a1u);
    for (size_t i = count + 1; i < words.size(); ++i) {
      EXPECT_EQ(words[i], 0x95d3c7a1u);
    }
    return {words.begin() + 1, words.begin() + 1 + count};
  }

  static uint64_t Address(const uint32_t* words) {
    return uint64_t{words[0]} | (uint64_t{words[1]} << 32);
  }

  // Byte oracle for emitted COPY/FILL/NOP packets. Native ordering is asserted
  // separately from these values; this synchronous consumer cannot prove it.
  static void Apply(const std::vector<uint32_t>& words) {
    for (size_t offset = 0; offset < words.size();) {
      const auto* packet = words.data() + offset;
      switch (packet[0] & 0xff) {
        case 0:
          ++offset;
          break;
        case 1: {
          ASSERT_LE(offset + 7, words.size());
          memcpy(reinterpret_cast<void*>(Address(packet + 5)),
                 reinterpret_cast<const void*>(Address(packet + 3)),
                 uint64_t{packet[1]} + 1);
          offset += 7;
          break;
        }
        case 11: {
          ASSERT_LE(offset + 5, words.size());
          auto* target = reinterpret_cast<uint8_t*>(Address(packet + 1));
          for (uint32_t i = 0; i <= packet[4]; ++i) {
            target[i] = static_cast<uint8_t>(packet[3] >> ((i % 4) * 8));
          }
          offset += 5;
          break;
        }
        default:
          FAIL() << "unexpected native packet " << packet[0];
      }
    }
  }

  // Sealed queue-site guarantees shared by test backing allocations.
  iree_hal_memory_contract_t* contract_ = nullptr;
  // Exact transfer family, with no compute or public atomic capability.
  iree_hal_queue_family_spec_t spec_ = {};
  // Borrowed family owner of each recording.
  iree_hal_queue_family_t family_ = {};
  // Caller-owned recording, independently retained by accepted invocations.
  iree_hal_command_buffer_t* recording_ = nullptr;
  // Stable buffer identities, retained beyond their native test uses.
  std::vector<std::unique_ptr<Buffer>> buffers_;
};

TEST_F(CommandBufferTest, CapturedUpdatesAndNaturalFillEdgesPreserveAllBytes) {
  std::array<uint8_t, 2048> payload, output, expected;
  payload.fill(0x7b);
  expected = payload;
  output.fill(0x19);
  auto* target =
      MakeBuffer(reinterpret_cast<uintptr_t>(payload.data()), payload.size());
  auto* destination =
      MakeBuffer(reinterpret_cast<uintptr_t>(output.data()), output.size());
  ASSERT_NO_FATAL_FAILURE(Begin());
  struct Geometry {
    // Fill pattern width admitted by the public API.
    uint32_t width;
    // Natural alignment residue within a DWORD.
    uint32_t residue;
  };
  const Geometry geometries[] = {{1, 0}, {1, 1}, {1, 2}, {1, 3},
                                 {2, 0}, {2, 2}, {4, 0}};
  uint32_t slot = 0;
  for (const auto geometry : geometries) {
    for (uint32_t elements : {1u, 3u, 7u, 11u}) {
      const uint32_t offset = slot * 64 + geometry.residue;
      const uint32_t length = elements * geometry.width;
      const uint32_t pattern = 0xa5c17e39u ^ (slot * 0x05030709u);
      IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
          recording_, iree_hal_make_buffer_ref(&target->base, offset, length),
          &pattern, geometry.width, 0));
      for (uint32_t i = 0; i < length; ++i) {
        expected[offset + i] =
            static_cast<uint8_t>(pattern >> ((i % geometry.width) * 8));
      }
      ++slot;
    }
  }
  std::array<uint8_t, 16> update;
  for (uint32_t i = 0; i < update.size(); ++i) {
    update[i] = 0x51u + i * 7u;
  }
  memcpy(expected.data() + 1900, update.data(), update.size());
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      recording_, update.data(), 0,
      iree_hal_make_buffer_ref(&target->base, 1900, update.size()), 0));
  update.fill(0);  // Recording owns the original source bytes.
  ASSERT_NO_FATAL_FAILURE(Barrier());
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_, iree_hal_make_buffer_ref(&target->base, 0, payload.size()),
      iree_hal_make_buffer_ref(&destination->base, 0, output.size()), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  const auto* program =
      iree_hal_amd_gpu_sdma_command_buffer_program(recording_);
  alignas(64) std::array<uint8_t, 512> data;
  ASSERT_LE(program->data.data_length, data.size());
  memcpy(data.data(), program->data.data, program->data.data_length);
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(reinterpret_cast<uintptr_t>(data.data())));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  size_t spans = 0;
  do {
    auto words = Emit(invocation, &progress);
    ASSERT_NO_FATAL_FAILURE(Apply(words));
    ++spans;
  } while (progress != IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  EXPECT_GT(spans, 1u);
  EXPECT_EQ(payload, expected);
  EXPECT_EQ(output, expected);
}

TEST_F(CommandBufferTest,
       SharedWholeRangeRecordingHasIndependentLargeAndEmptyInvocations) {
  constexpr uint64_t kLength = (UINT64_C(5) << 30) + 31;
  auto* source = MakeBuffer(0, kLength + 128);
  auto* target = MakeBuffer(0, kLength + 128);
  ASSERT_NO_FATAL_FAILURE(Begin(2));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_,
      iree_hal_make_indirect_buffer_ref(0, 11, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(1, 13, IREE_HAL_WHOLE_BUFFER), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  std::array<iree_hal_buffer_binding_t, 2> large = {
      {{&source->base, 7, kLength + 11}, {&target->base, 9, kLength + 13}}};
  std::array<iree_hal_buffer_binding_t, 2> empty = {
      {{&source->base, 0, 11}, {&target->base, 0, 13}}};
  Invocation a, b;
  IREE_ASSERT_OK(a.Initialize(recording_, {large.size(), large.data()}));
  IREE_ASSERT_OK(b.Initialize(recording_, {empty.size(), empty.data()}));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, a.Resolve(0));
  // Backing becomes available only after dependency commitment.
  source->binding.device_address = UINT64_C(0x1234fffffff0);
  target->binding.device_address = UINT64_C(0x5678fffffff0);
  IREE_ASSERT_OK(a.Resolve(0));
  IREE_ASSERT_OK(b.Resolve(0));
  large[0].offset = large[1].offset = 0;  // Descriptors were captured.
  iree_hal_command_buffer_release(recording_);
  recording_ = nullptr;  // Both invocations own the recording and bindings.
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  EXPECT_TRUE(Emit(b, &progress).empty());
  EXPECT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  uint64_t copied = 0;
  size_t spans = 0;
  do {
    auto words = Emit(a, &progress);
    ASSERT_EQ(words.size() % 7, 0u);
    for (size_t offset = 0; offset < words.size(); offset += 7) {
      const auto* packet = words.data() + offset;
      EXPECT_EQ(packet[0], 1u);
      EXPECT_EQ(Address(packet + 3), UINT64_C(0x1234fffffff0) + 18 + copied);
      EXPECT_EQ(Address(packet + 5), UINT64_C(0x5678fffffff0) + 22 + copied);
      EXPECT_LE(packet[1] + 1, IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
      copied += uint64_t{packet[1]} + 1;
    }
    ++spans;
  } while (progress != IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  EXPECT_EQ(copied, kLength);
  EXPECT_GT(spans, 32u);
}

TEST_F(CommandBufferTest,
       WholeRangeEqualityRejectsMismatchWithoutLosingOwnership) {
  auto* source = MakeBuffer(0x1000, 256);
  auto* target = MakeBuffer(0x2000, 256);
  ASSERT_NO_FATAL_FAILURE(Begin(2));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_,
      iree_hal_make_indirect_buffer_ref(0, 3, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_indirect_buffer_ref(1, 5, IREE_HAL_WHOLE_BUFFER), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  const iree_hal_buffer_binding_t bindings[] = {{&source->base, 0, 128},
                                                {&target->base, 0, 128}};
  Invocation invocation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        invocation.Initialize(recording_, {2, bindings}));
}

TEST_F(CommandBufferTest,
       FixedPacketBlocksPreserveAddressesAcrossSpanBoundaries) {
  constexpr uint64_t kLength = (UINT64_C(3) << 30) + 7;
  auto* source = MakeBuffer(UINT64_C(0x1234fffffff1), kLength);
  auto* target = MakeBuffer(UINT64_C(0x5678fffffffb), kLength);
  ASSERT_NO_FATAL_FAILURE(Begin());
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_, iree_hal_make_buffer_ref(&source->base, 0, kLength),
      iree_hal_make_buffer_ref(&target->base, 0, kLength), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(0));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  uint64_t copied = 0;
  do {
    const auto words = Emit(invocation, &progress);
    ASSERT_EQ(words.size() % 7, 0u);
    for (size_t offset = 0; offset < words.size(); offset += 7) {
      const auto* packet = words.data() + offset;
      EXPECT_EQ(Address(packet + 3), source->binding.device_address + copied);
      EXPECT_EQ(Address(packet + 5), target->binding.device_address + copied);
      copied += uint64_t{packet[1]} + 1;
    }
  } while (progress != IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  EXPECT_EQ(copied, kLength);
}

TEST_F(CommandBufferTest, KernelCacheBoundaryCannotBecomeAnInStreamMarker) {
  const auto features = AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR;
  auto* source = MakeBuffer(0x1000, 64);
  auto* target = MakeBuffer(0x2000, 64);
  ASSERT_NO_FATAL_FAILURE(
      Begin(0, features, AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
  for (uint32_t i = 0; i < 2; ++i) {
    if (i) {
      ASSERT_NO_FATAL_FAILURE(
          Barrier(IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE));
    }
    IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
        recording_, iree_hal_make_buffer_ref(&source->base, i * 8, 8),
        iree_hal_make_buffer_ref(&target->base, i * 8, 8), 0));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(0));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  EXPECT_EQ(Emit(invocation, &progress),
            (std::vector<uint32_t>{1, 7, 0, 0x1000, 0, 0x2000, 0}));
  EXPECT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_BOUNDARY);
  // The queue must retire that native submission before requesting this span.
  EXPECT_EQ(Emit(invocation, &progress),
            (std::vector<uint32_t>{1, 7, 0, 0x1008, 0, 0x2008, 0}));
  EXPECT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
}

TEST_F(CommandBufferTest, WholeFillStreamsDwordBodyAndByteEdgesPastFourGiB) {
  constexpr uint64_t kLength = (UINT64_C(5) << 30) + 2;
  constexpr uint64_t kTarget = UINT64_C(0x1234fffffff3);
  auto* target = MakeBuffer(kTarget, kLength);
  ASSERT_NO_FATAL_FAILURE(Begin(1));
  const uint8_t pattern = 0xd9;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording_,
      iree_hal_make_indirect_buffer_ref(0, 0, IREE_HAL_WHOLE_BUFFER), &pattern,
      1, 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  const iree_hal_buffer_binding_t binding = {&target->base, 0, kLength};
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_, {1, &binding}));
  IREE_ASSERT_OK(invocation.Resolve(0x8000));
  uint64_t written = 0;
  uint32_t edges = 0;
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  do {
    const auto words = Emit(invocation, &progress);
    for (size_t offset = 0; offset < words.size();) {
      const auto* packet = words.data() + offset;
      if (packet[0] == 1) {
        ASSERT_LE(offset + 7, words.size());
        EXPECT_EQ(Address(packet + 3), 0x8000u + (written & 3));
        EXPECT_EQ(Address(packet + 5), kTarget + written);
        EXPECT_EQ(packet[1], 0u);
        written += uint64_t{packet[1]} + 1;
        offset += 7;
        ++edges;
      } else {
        ASSERT_LE(offset + 5, words.size());
        EXPECT_EQ(packet[0], 0x8000000bu);
        EXPECT_EQ(Address(packet + 1), kTarget + written);
        EXPECT_EQ(packet[3], 0xd9d9d9d9u);
        EXPECT_LE(packet[4] + 1, IREE_HAL_AMD_GPU_SDMA_MAX_PACKET_LENGTH);
        EXPECT_EQ((packet[4] + 1) % 4, 0u);
        written += uint64_t{packet[4]} + 1;
        offset += 5;
      }
    }
  } while (progress != IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  EXPECT_EQ(written, kLength);
  EXPECT_EQ(edges, 2u);
}

TEST_F(CommandBufferTest, UserCacheBoundariesStayInTheNativeStream) {
  const auto features = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
  auto* source = MakeBuffer(0x1000, 8);
  auto* target = MakeBuffer(0x2000, 8);
  ASSERT_NO_FATAL_FAILURE(Begin(0, features));
  const iree_hal_barrier_t acquire = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &acquire));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_, iree_hal_make_buffer_ref(&source->base, 0, 8),
      iree_hal_make_buffer_ref(&target->base, 0, 8), 0));
  ASSERT_NO_FATAL_FAILURE(Barrier(IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(0));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  EXPECT_EQ(
      Emit(invocation, &progress),
      (std::vector<uint32_t>{0x111, 0, 0xc3c00000, 0, 0, 1, 7, 0, 0x1000, 0,
                             0x2000, 0, 0, 0x111, 0, 0x80400000, 0, 0}));
  EXPECT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
}

TEST_F(CommandBufferTest,
       TimestampUsesPrivateAlignedSampleAndEightByteDelivery) {
  auto* target = MakeBuffer(UINT64_C(0x123400000000), 64);
  ASSERT_NO_FATAL_FAILURE(Begin(1, AMDF_GPU_SDMA_FORMAT_FEATURE_MEMORY_SCOPE));
  for (uint32_t i = 0; i < 2; ++i) {
    IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
        recording_, IREE_HAL_EXECUTION_STAGE_TRANSFER,
        IREE_HAL_EXECUTION_STAGE_TRANSFER,
        iree_hal_make_indirect_buffer_ref(0, 8 + i * 8, 8), 0));
  }
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  const iree_hal_buffer_binding_t binding = {&target->base, 0, 64};
  Invocation a, b;
  IREE_ASSERT_OK(a.Initialize(recording_, {1, &binding}));
  IREE_ASSERT_OK(b.Initialize(recording_, {1, &binding}));
  IREE_ASSERT_OK(a.Resolve(0x8000));
  IREE_ASSERT_OK(b.Resolve(0xc000));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  EXPECT_EQ(
      Emit(a, &progress),
      (std::vector<uint32_t>{
          0x0300020d, 0x8000, 0, 0, 1, 7, 0x0c0c0000, 0x8000, 0, 8,  0x1234,
          0x0300020d, 0x8000, 0, 0, 1, 7, 0x0c0c0000, 0x8000, 0, 16, 0x1234}));
  EXPECT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  const auto second = Emit(b, &progress);
  ASSERT_EQ(second.size(), 22u);
  EXPECT_EQ(second[1], 0xc000u);
  EXPECT_EQ(second[7], 0xc000u);
  EXPECT_EQ(second[12], 0xc000u);
  EXPECT_EQ(second[18], 0xc000u);
}

}  // namespace
