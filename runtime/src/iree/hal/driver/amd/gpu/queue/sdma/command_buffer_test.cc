// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/sdma/command_buffer.h"

#include <array>
#include <atomic>
#include <future>
#include <memory>
#include <vector>

#include "iree/hal/device_spec.h"
#include "iree/hal/driver/amd/gpu/context.h"
#include "iree/hal/driver/amd/gpu/queue/program.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Real public recording, binding validation/capture, and native projection.
// Only native memory and buffer backing are supplied by the fixture. Native
// carrier CTS separately qualifies packet execution and physical visibility.
class CommandBufferTest : public ::testing::Test {
 protected:
  struct Memory {
    // Native provider whose allocation/failure evidence this backing updates.
    CommandBufferTest* owner;
    // Host bytes with room for the requested native alignment.
    std::vector<uint8_t> bytes;
    // Aligned native extent exposed to the production memory owner.
    uint8_t* data;
    // Size of the exposed native extent, excluding alignment padding.
    uint64_t length;
    // Alignment admitted from the requested allocation geometry.
    uint64_t alignment;
    // Allocation flags reported independently from the requested host view.
    amdf_memory_flags_t flags;
  };

  void InitializeNativeMemory() {
    api_.memory_scope_query_info = [](amdf_memory_scope_t*,
                                      amdf_memory_scope_info_t* info) {
      info->kind = AMDF_MEMORY_SCOPE_KIND_SYSTEM;
      info->memory_profile_count = 1;
      return AMDF_STATUS_OK;
    };
    api_.memory_scope_query_device_profile =
        [](amdf_memory_scope_t*, uint32_t, uint32_t count,
           const amdf_memory_device_access_t* accesses,
           amdf_memory_profile_t* profile, amdf_memory_access_capabilities_t*) {
          EXPECT_EQ(count, 1u);
          EXPECT_EQ(accesses[0].requirements.access,
                    AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
          profile->roles = AMDF_MEMORY_PROFILE_ROLE_CREATE |
                           AMDF_MEMORY_PROFILE_ROLE_HOST_MAP;
          profile->supported_flags =
              AMDF_MEMORY_FLAG_DEVICE_LOCAL | AMDF_MEMORY_FLAG_HOST_VISIBLE;
          profile->host_mapping.supported_access =
              AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
          profile->allocation.minimum_alignment = 64;
          profile->allocation.byte_length_granularity = 64;
          profile->allocation.native_byte_length_granularity = 64;
          profile->allocation.maximum_byte_length = UINT32_MAX;
          return AMDF_STATUS_OK;
        };
    api_.memory_scope_query_pair_info =
        [](amdf_memory_scope_t*, const amdf_memory_profile_pair_query_t* query,
           amdf_memory_pair_info_t* pair) {
          EXPECT_EQ(query->producer.kind, AMDF_MEMORY_SITE_KIND_HOST);
          EXPECT_EQ(query->consumer.kind, AMDF_MEMORY_SITE_KIND_DEVICE);
          EXPECT_EQ(query->consumer.value.device.queue_family_ordinal, 1u);
          pair->release.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          pair->acquire.kind = AMDF_CACHE_TRANSITION_KIND_NONE;
          return AMDF_STATUS_OK;
        };
    api_.memory_create = [](amdf_memory_scope_t* scope,
                            const amdf_memory_create_info_t* info,
                            amdf_memory_t** out_memory,
                            amdf_memory_release_info_t* out_release_info) {
      auto* self = reinterpret_cast<CommandBufferTest*>(scope);
      if (out_release_info) {
        *out_release_info = {};
      }
      if (self->fail_allocate_) {
        return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
      }
      auto* memory = new Memory;
      memory->owner = self;
      memory->bytes.resize(info->byte_length + info->minimum_alignment);
      memory->data = reinterpret_cast<uint8_t*>(
          iree_align_uint64(reinterpret_cast<uintptr_t>(memory->bytes.data()),
                            info->minimum_alignment));
      memory->length = info->byte_length;
      memory->alignment = info->minimum_alignment;
      memory->flags = info->required_flags;
      ++self->allocations_;
      ++self->live_memory_;
      *out_memory = reinterpret_cast<amdf_memory_t*>(memory);
      return AMDF_STATUS_OK;
    };
    api_.memory_query_info = [](amdf_memory_t* handle,
                                amdf_memory_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->byte_length = info->native_allocation_byte_length = memory->length;
      info->alignment = memory->alignment;
      info->native_allocation_granularity = 64;
      info->flags = memory->flags;
      return AMDF_STATUS_OK;
    };
    api_.memory_query_address = [](amdf_memory_t* handle, uint32_t,
                                   amdf_memory_address_kind_t,
                                   uint64_t* out_address) {
      *out_address =
          reinterpret_cast<uintptr_t>(reinterpret_cast<Memory*>(handle)->data);
      return AMDF_STATUS_OK;
    };
    api_.memory_map = [](amdf_memory_t* handle, const amdf_memory_map_info_t*,
                         amdf_host_mapping_t** out_mapping) {
      EXPECT_NE(reinterpret_cast<Memory*>(handle)->flags &
                    AMDF_MEMORY_FLAG_HOST_VISIBLE,
                0u);
      *out_mapping = reinterpret_cast<amdf_host_mapping_t*>(handle);
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_query_info = [](amdf_host_mapping_t* handle,
                                      amdf_host_mapping_info_t* info) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      info->pointer = memory->data;
      info->byte_length = memory->length;
      info->properties = AMDF_HOST_MEMORY_PROPERTY_READ_CACHED |
                         AMDF_HOST_MEMORY_PROPERTY_WRITE_BACK;
      return AMDF_STATUS_OK;
    };
    api_.host_mapping_cache_control = [](amdf_host_mapping_t* handle,
                                         amdf_host_cache_operation_t operation,
                                         uint64_t offset, uint64_t length) {
      auto* memory = reinterpret_cast<Memory*>(handle);
      EXPECT_LE(offset + length, memory->length);
      EXPECT_EQ(operation, AMDF_HOST_CACHE_OPERATION_FLUSH);
      return memory->owner->publish_status_;
    };
    api_.host_mapping_destroy = [](amdf_host_mapping_t*) {
      return AMDF_STATUS_OK;
    };
    api_.memory_destroy = [](amdf_memory_t* handle,
                             amdf_memory_release_info_t* out_release_info) {
      if (out_release_info) {
        *out_release_info = {};
      }
      auto* memory = reinterpret_cast<Memory*>(handle);
      --memory->owner->live_memory_;
      delete memory;
      return AMDF_STATUS_OK;
    };
    api_.instance_destroy = [](amdf_instance_t*) { return AMDF_STATUS_OK; };
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &api_;
    fabric_->system_scope = reinterpret_cast<amdf_memory_scope_t*>(this);
    context_.api = &api_;
    context_.fabric = fabric_;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    IREE_ASSERT_OK(iree_hal_amd_gpu_program_pool_initialize(
        &context_, 1, 2, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        iree_allocator_system(), &pool_));
  }
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
    ASSERT_NO_FATAL_FAILURE(InitializeNativeMemory());
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
    iree_hal_amd_gpu_program_pool_deinitialize(&pool_);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    EXPECT_EQ(live_memory_, 0u);
    iree_hal_amd_fabric_release(fabric_);
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
        &family_, &pool_, features,
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
        invocation.value.values, &invocation.cursor,
        IREE_HAL_AMD_GPU_SDMA_MAX_COMMAND_WORDS, words.data() + 1, &count);
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
  // Native dependency supplying qualified memory and publication failures.
  amdf_api_t api_ = {};
  // Shared native allocation domain, joined before fixture teardown.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Device-owned memory callbacks used by the real program owner.
  iree_hal_amd_gpu_context_t context_ = {};
  // Qualified data-only program placement and independent invocation leases.
  iree_hal_amd_gpu_program_pool_t pool_ = {};
  // Native allocations observed on the recorder or allocation worker.
  std::atomic<uint32_t> allocations_ = 0;
  // Outstanding native allocations, including deferred releases.
  std::atomic<uint32_t> live_memory_ = 0;
  // Injected native allocation failure before a recording can be sealed.
  bool fail_allocate_ = false;
  // Native publication result returned while sealing captured data.
  amdf_status_t publish_status_ = AMDF_STATUS_OK;
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
  auto* storage = iree_hal_amd_gpu_sdma_command_buffer_storage(recording_);
  ASSERT_NE(storage, nullptr);
  EXPECT_EQ(storage->byte_length, program->data.data_length);
  EXPECT_EQ(memcmp(storage->contents.data, program->data.data,
                   program->data.data_length),
            0);
  const uint64_t frontier[2] = {};
  auto* instance = iree_hal_amd_gpu_program_try_acquire(storage, 0, frontier);
  ASSERT_NE(instance, nullptr);
  EXPECT_TRUE(instance->initialized);
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(instance->address));
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
  iree_hal_amd_gpu_program_cancel(storage, instance);
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
  EXPECT_EQ(iree_hal_amd_gpu_sdma_command_buffer_storage(recording_), nullptr);
  EXPECT_EQ(allocations_, 0u);
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
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &acquire));
  IREE_ASSERT_OK(iree_hal_command_buffer_copy_buffer(
      recording_, iree_hal_make_buffer_ref(&source->base, 0, 8),
      iree_hal_make_buffer_ref(&target->base, 0, 8), 0));
  const iree_hal_barrier_t release = {
      .source_stage_mask = IREE_HAL_EXECUTION_STAGE_TRANSFER,
      .target_stage_mask = IREE_HAL_EXECUTION_STAGE_HOST,
  };
  IREE_ASSERT_OK(iree_hal_command_buffer_barrier(recording_, &release));
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

TEST_F(CommandBufferTest, PartialRecordingKeepsItsReservedPlacement) {
  auto* target = MakeBuffer(0x1000, 32);
  ASSERT_NO_FATAL_FAILURE(Begin(0, AMDF_GPU_SDMA_FORMAT_FEATURE_SUBMISSION_GCR,
                                AMDF_QUEUE_PUBLICATION_MODE_KERNEL));
  IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
      recording_, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER,
      iree_hal_make_buffer_ref(&target->base, 8, 8), 0));
  ASSERT_NO_FATAL_FAILURE(Barrier(IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE));
  IREE_ASSERT_OK(iree_hal_command_buffer_timestamp(
      recording_, IREE_HAL_EXECUTION_STAGE_TRANSFER,
      IREE_HAL_EXECUTION_STAGE_TRANSFER,
      iree_hal_make_buffer_ref(&target->base, 16, 8), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  auto* storage = iree_hal_amd_gpu_sdma_command_buffer_storage(recording_);
  ASSERT_NE(storage, nullptr);
  uint64_t frontier[2] = {};
  auto* first = iree_hal_amd_gpu_program_try_acquire(storage, 0, frontier);
  ASSERT_NE(first, nullptr);
  Invocation invocation;
  IREE_ASSERT_OK(invocation.Initialize(recording_));
  IREE_ASSERT_OK(invocation.Resolve(first->address));
  iree_hal_amd_gpu_sdma_program_progress_t progress;
  ASSERT_EQ(Emit(invocation, &progress).size(), 11u);
  ASSERT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_BOUNDARY);
  // Even a later queue frontier cannot cover the unpublished suffix. Other
  // ready work needs a different placement while this invocation is parked.
  frontier[0] = UINT64_MAX;
  EXPECT_EQ(iree_hal_amd_gpu_program_try_acquire(storage, 0, frontier),
            nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_program_try_acquire(storage, 1, frontier),
            nullptr);
  iree_hal_amd_gpu_program_growth_t growth = {};
  std::promise<iree::Status> completed;
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_begin(
      storage, 1,
      {[](void* data, iree_status_t status) {
         static_cast<std::promise<iree::Status>*>(data)->set_value(
             iree::Status(std::move(status)));
       },
       &completed},
      &growth));
  auto result = completed.get_future().get();
  IREE_ASSERT_OK(iree_hal_amd_gpu_program_grow_end(&growth, result.release()));
  auto* independent =
      iree_hal_amd_gpu_program_try_acquire(storage, 1, frontier);
  ASSERT_NE(independent, nullptr);
  EXPECT_NE(independent->address, first->address);
  EXPECT_FALSE(independent->initialized);
  iree_hal_amd_gpu_program_cancel(storage, independent);
  ASSERT_EQ(Emit(invocation, &progress).size(), 11u);
  ASSERT_EQ(progress, IREE_HAL_AMD_GPU_SDMA_PROGRAM_END);
  iree_hal_amd_gpu_program_publish(storage, first, 0, 7);
  frontier[0] = 7;
  auto* successor = iree_hal_amd_gpu_program_try_acquire(storage, 0, frontier);
  EXPECT_EQ(successor, first);
  EXPECT_TRUE(successor->join_previous);
  iree_hal_amd_gpu_program_cancel(storage, successor);
  iree_hal_amd_gpu_program_release(storage, first);
}

TEST_F(CommandBufferTest, UnmappedCapturedDataKeepsAnOrderedUploadSource) {
  pool_.resident.host_access = 0;
  pool_.resident.required_flags &= ~AMDF_MEMORY_FLAG_HOST_VISIBLE;
  auto* target = MakeBuffer(0x1000, 32);
  ASSERT_NO_FATAL_FAILURE(Begin());
  const std::array<uint8_t, 8> bytes = {0x39, 0x72, 0xa1, 0x8d,
                                        0x16, 0xca, 0x57, 0xe3};
  IREE_ASSERT_OK(iree_hal_command_buffer_update_buffer(
      recording_, bytes.data(), 0,
      iree_hal_make_buffer_ref(&target->base, 0, bytes.size()), 0));
  IREE_ASSERT_OK(iree_hal_command_buffer_end(recording_));
  auto* storage = iree_hal_amd_gpu_sdma_command_buffer_storage(recording_);
  ASSERT_NE(storage, nullptr);
  EXPECT_NE(storage->upload, nullptr);
  EXPECT_FALSE(storage->initial->initialized);
  EXPECT_EQ(allocations_, 2u);
  const auto source = iree_hal_amd_gpu_program_source(storage);
  EXPECT_NE(source.address, storage->initial->address);
  EXPECT_EQ(memcmp(reinterpret_cast<void*>(source.address), bytes.data(),
                   bytes.size()),
            0);
}

TEST_F(CommandBufferTest, FailedNativeAllocationLeavesRecordingUnexecutable) {
  auto* target = MakeBuffer(0x1000, 32);
  ASSERT_NO_FATAL_FAILURE(Begin());
  const uint32_t pattern = 0x73d92ca1;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording_, iree_hal_make_buffer_ref(&target->base, 0, 32), &pattern, 4,
      0));
  fail_allocate_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_command_buffer_end(recording_));
  EXPECT_EQ(live_memory_, 0u);
  EXPECT_EQ(pool_.registry.head, nullptr);
  Invocation invocation;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        invocation.Initialize(recording_));
}

TEST_F(CommandBufferTest, FailedNativePublicationRetainsAndReleasesItsBacking) {
  auto* target = MakeBuffer(0x1000, 32);
  ASSERT_NO_FATAL_FAILURE(Begin());
  const uint32_t pattern = 0x73d92ca1;
  IREE_ASSERT_OK(iree_hal_command_buffer_fill_buffer(
      recording_, iree_hal_make_buffer_ref(&target->base, 0, 32), &pattern, 4,
      0));
  publish_status_ = amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INTERNAL,
                        iree_hal_command_buffer_end(recording_));
  EXPECT_EQ(live_memory_, 1u);
  EXPECT_EQ(pool_.registry.head, nullptr);
  iree_hal_command_buffer_release(recording_);
  recording_ = nullptr;
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(live_memory_, 0u);
}

}  // namespace
