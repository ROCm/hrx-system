// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>
#include <vector>

#include "iree/async/event.h"
#include "iree/async/proactor_platform.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/driver/amd/gpu/kernarg.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/hal/driver/amd/gpu/queue/byte_storage_pool.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Publication, capture ownership, and native timelines are real. Only the
// device-owned ring and completion allocations are supplied by the fixture.
class AqlPublicationTest
    : public ::testing::TestWithParam<std::tuple<int, bool, bool>> {
 protected:
  struct Signals {
    // Stable queue identity consumed by dependency admission.
    iree_hal_amd_gpu_signal_cache_t cache = {};
    // Allocation metadata shared by independent producer and reader owners.
    iree_hal_amd_gpu_signal_block_t block = {};
    // Host ownership for native completion descriptors.
    std::array<iree_hal_amd_gpu_signal_t, 128> signals = {};
    // Coherent native completion allocation.
    std::array<iree_hal_amd_gpu_aql_signal_t, 128> values = {};
    // Two-queue causal vectors owned by the allocation.
    std::array<uint64_t, 2 + 128 * 2> frontiers = {};
  };

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_event_native_initialize(&context_.kernel_event));
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    for (size_t i = 0; i < signals_.size(); ++i) {
      auto& signals = signals_[i];
      iree_hal_amd_gpu_signal_pool_initialize({}, &signals.cache.pool);
      iree_hal_amd_gpu_signal_block_initialize(
          signals.signals.data(), signals.signals.size(), 2,
          signals.frontiers.data(), &signals.block);
      const amdf_gpu_event_info_t event = {};
      iree_hal_amd_gpu_aql_signal_block_attach(
          &signals.cache.pool, &context_, &event, signals.values.data(),
          0x1000 + i * 0x10000, &signals.block);
    }
    queue_.base.queue_family = &family_;
    queue_.context = &context_;
    queue_.storage.context = &context_;
    queue_.options.signals = &signals_[0].cache;
    queue_.options.retirement_capacity = issued_.size();
    queue_.issued = issued_.data();
    queue_.producer.arguments = &arguments_;
    queue_.producer.frontier = frontier_.data();
    queue_.producer.queue_count = frontier_.size();
    for (auto& packet : packets_) {
      packet.words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
    }
    iree_hal_amd_gpu_aql_ring_initialize(
        packets_.data(), 0x10000, packets_.size(), &read_index_, &write_index_,
        &doorbell_, AMDF_CACHE_TRANSITION_EXECUTOR_NONE, &queue_.producer.ring);
    if (std::get<2>(GetParam())) {
      queue_.storage.capabilities =
          AMDF_USER_QUEUE_CAPABILITY_HOST_NON_TEMPORAL_PUBLICATION;
      queue_.storage.host_publish_executor =
          AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      queue_.storage.host_publish_fence = IREE_HAL_HOST_CACHE_FENCE_X86_SFENCE;
      queue_.producer.ring.host_publish_executor =
          AMDF_CACHE_TRANSITION_EXECUTOR_HOST_DIRECT;
      queue_.producer.argument_streaming = true;
    }
    iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(),
                                             &queue_.captures);
    function_.base.metadata = &metadata_;
    function_.base.kernarg = &layout_;
    function_.kernel_object = 0x123456789abcdef0;
    request_.function = &function_;
    request_.geometry.workgroup_size[0] = 64;
    request_.geometry.workgroup_size[1] = 1;
    request_.geometry.workgroup_size[2] = 1;
    for (int i = 0; i < 3; ++i) {
      request_.geometry.workgroup_count[i] = 1;
    }
    request_.flags = IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS;
  }

  void TearDown() override {
    ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
    for (auto* signal : other_producers_) {
      iree_hal_amd_gpu_signal_retire(signal, IREE_STATUS_OK);
      iree_hal_amd_gpu_signal_release(signal);
    }
    const uint64_t count =
        iree_atomic_load(&queue_.producer.count, iree_memory_order_acquire);
    for (uint64_t i = 0; i < count; ++i) {
      auto& action = issued_[i];
      iree_hal_amd_gpu_aql_submission_release_payload(action.submission);
      iree_hal_amd_gpu_signal_retire(action.signal, IREE_STATUS_OK);
      iree_hal_amd_gpu_signal_release(action.signal);
      iree_hal_amd_gpu_aql_submission_release(action.submission);
    }
    iree_hal_amd_gpu_kernarg_layout_free(cooperative_layout_);
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_amd_gpu_capture_pool_deinitialize(&queue_.captures);
    for (auto& signals : signals_) {
      iree_hal_amd_gpu_signal_block_deinitialize(&signals.block);
      iree_hal_amd_gpu_signal_pool_deinitialize(&signals.cache.pool);
    }
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(queue_.profile));
    if (control_contract_) {
      EXPECT_EQ(iree_atomic_ref_count_load(&control_buffer_.resource.ref_count),
                1);
      iree_hal_buffer_release(&control_buffer_);
      iree_hal_memory_contract_release(control_contract_);
    }
    iree_async_proactor_release(proactor_);
    iree_async_event_native_deinitialize(&context_.kernel_event);
  }

  void PrepareControlBuffer() {
    const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
    const iree_hal_buffer_binding_layout_t layout = {
        sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
    IREE_ASSERT_OK(iree_hal_memory_contract_create(
        this, 3, &layout, iree_allocator_system(), &control_contract_));
    control_contract_->scopes[1].interfaces =
        1u << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
    control_contract_->scopes[1]
        .bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
    control_contract_->scopes[1].atomic_operations = UINT32_MAX;
    control_contract_->zero_compute_atomic_operations[1] = UINT32_MAX;
    family_.memory = {.domain = this, .queue_scope_id = 1};
    static const iree_hal_buffer_vtable_t vtable = {
        .destroy = [](iree_hal_buffer_t*) {},
    };
    iree_hal_buffer_initialize(
        {}, &control_buffer_, 64, 0, 64, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
        IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_STORAGE, &vtable,
        &control_buffer_);
    control_buffer_.memory = {
        .contract = control_contract_,
        .bindings = &control_binding_,
    };
  }

  iree_hal_amd_gpu_profile_operation_t BeginProfile() {
    iree::hal::cts::TestProfileSinkInitialize(&sink_);
    const iree_hal_device_profiling_options_t options = {
        .data_families = IREE_HAL_DEVICE_PROFILING_DATA_QUEUE_EVENTS |
                         IREE_HAL_DEVICE_PROFILING_DATA_DISPATCH_EVENTS,
        .sink = iree::hal::cts::TestProfileSinkAsBase(&sink_),
    };
    IREE_CHECK_OK(iree_hal_amd_gpu_profile_begin(
        &options, 1, 2, 1000000, iree_allocator_system(), &queue_.profile));
    iree_hal_amd_gpu_profile_operation_t operation;
    iree_hal_amd_gpu_profile_operation_initialize(
        queue_.profile, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_DISPATCH, 0, 0, 1,
        &operation);
    operation.completion.count = 1;
    operation.completion.sample_stride = sizeof(iree_hal_amd_gpu_aql_signal_t);
    return operation;
  }

  iree_hal_semaphore_t* Semaphore() {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_semaphore_create(
        proactor_, nullptr, 0, 2, iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  void Publish(iree_hal_semaphore_t* output,
               std::vector<iree_hal_amd_gpu_aql_wait_t>& waits,
               bool reverse_scopes = false,
               uint32_t acquire = IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM,
               uint32_t release = IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
               bool release_before = false) {
    iree_hal_amd_gpu_aql_wait_counts_t counts;
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_admit_waits(
        &context_, &signals_[0].cache.pool, waits.size(), waits.data(), true,
        &counts, &wait_index, &kind));
    ASSERT_EQ(kind, IREE_HAL_AMD_GPU_AQL_WAIT_READY);
    uint32_t before = acquire | (release_before ? release : 0);
    uint32_t after = release_before ? 0 : release;
    if (reverse_scopes) {
      before |= IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
      after |= IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
    }
    uint64_t value = 1;
    iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
    iree_status_t status = iree_ok_status();
    EXPECT_EQ(iree_hal_amd_gpu_aql_publish(
                  &queue_, &request_, waits.size(), waits.data(), counts,
                  {1, &output, &value}, before, after, nullptr, nullptr,
                  &submission, &status),
              IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
    IREE_ASSERT_OK(status);
    ASSERT_NE(submission, nullptr);
  }

  uint64_t PacketAddress(size_t packet, size_t word) {
    return packets_[packet].words[word] |
           (uint64_t{packets_[packet].words[word + 1]} << 32);
  }

  // Real bounded profile stream used by timestamped publication.
  iree::hal::cts::TestProfileSink sink_;
  // Prepared native scalar target, retained by real control captures.
  iree_hal_buffer_t control_buffer_ = {};
  // Independent target-memory contract for direct native scalar operations.
  iree_hal_memory_contract_t* control_contract_ = nullptr;
  // Native allocation dependency supplies the scalar address.
  iree_hal_buffer_native_binding_t control_binding_ = {
      .device_address = 0x9000,
  };
  // Real host timeline infrastructure.
  iree_async_proactor_t* proactor_ = nullptr;
  // Native signal-domain identity, shared by both queues.
  iree_hal_amd_gpu_context_t context_ = {};
  // Queue family borrowed by publication.
  iree_hal_queue_family_t family_ = {};
  // Production queue publication and capture state.
  iree_hal_amd_gpu_aql_queue_t queue_ = {};
  // Native completion allocations for this queue and an independent producer.
  std::array<Signals, 2> signals_;
  // Retirement storage with the production inline capture extent.
  std::array<iree_hal_amd_gpu_aql_retirement_t, 128> issued_ = {};
  // Native packet allocation observed after publication.
  std::array<iree_hal_amd_gpu_aql_packet_t, 64> packets_ = {};
  // Firmware-owned packet consumption index.
  iree_atomic_int64_t read_index_ = 0;
  // Host-owned packet reservation index.
  iree_atomic_int64_t write_index_ = 0;
  // Native notification destination.
  uint64_t doorbell_ = 0;
  // Zero-argument native function needs no argument backing.
  iree_hal_amd_gpu_byte_storage_t arguments_ = {};
  // Queue's accepted causal prefix.
  std::array<uint64_t, 2> frontier_ = {};
  // Immutable zero-scratch function metadata.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Empty native argument layout.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 1};
  // Metadata-derived cooperative ABI, retained through captured work
  // retirement.
  iree_hal_amd_gpu_kernarg_layout_t* cooperative_layout_ = nullptr;
  // Placed descriptor used by actual dispatch encoding.
  iree_hal_amd_gpu_aql_function_t function_ = {};
  // Admitted ordinary raw-native launch.
  iree_hal_amd_gpu_aql_dispatch_request_t request_ = {};
  // Original owners of the external native producers.
  std::vector<iree_hal_amd_gpu_signal_t*> other_producers_;
  // Timelines kept alive until all issued captures retire.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_P(AqlPublicationTest, SharedCompletionPreservesEveryDispatchBoundary) {
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  const uint32_t header = (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                          IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                          IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                          IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
  EXPECT_EQ(packets_[0].words[0], header);
  EXPECT_EQ(PacketAddress(0, 14), issued_[0].signal->device_address);
  EXPECT_EQ(queue_.run.completion, nullptr);

  for (size_t i = 1; i < 5; ++i) {
    Publish(Semaphore(), waits);
    EXPECT_EQ(packets_[1].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
    EXPECT_EQ(packets_[i].words[0],
              i == 1 ? IREE_HAL_AMD_GPU_AQL_HEADER_INVALID : header);
    EXPECT_EQ(PacketAddress(i, 14), 0u);
    EXPECT_EQ(issued_[i].signal->host_value, issued_[1].signal->host_value);
    std::array<uint64_t, 2> frontier = {};
    iree_hal_amd_gpu_signal_join_frontier(issued_[i].signal, frontier.size(),
                                          frontier.data());
    EXPECT_EQ(frontier[0], i + 1);
    EXPECT_NE(issued_[i].signal->host_value, issued_[0].signal->host_value);
  }
  EXPECT_EQ(doorbell_, 0u);
  EXPECT_EQ(queue_.run.last - queue_.run.first + 1, 4u);
  EXPECT_EQ(queue_.run.position, 2);
  ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
  EXPECT_EQ(queue_.run.completion, nullptr);
  EXPECT_EQ(queue_.run.position, 0);
  EXPECT_EQ(queue_.producer.ring.reserved, 5u);
  EXPECT_EQ(doorbell_, 4u);
  for (size_t i = 1; i < 5; ++i) {
    EXPECT_EQ(packets_[i].words[0], header);
    EXPECT_EQ(PacketAddress(i, 14),
              i == 4 ? issued_[1].signal->device_address : 0u);
  }
}

TEST_P(AqlPublicationTest, HostPublicationAcquiresEachRunBeforePayloadScopes) {
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits, false, 0);
  const uint32_t system_header = (1u << 16) |
                                 IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                                 IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                                 IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                                 IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
  EXPECT_EQ(packets_[0].words[0], system_header);
  for (uint32_t acquire : {0u, 1u << 9, 2u << 9}) {
    const uint64_t first = queue_.producer.ring.reserved;
    const uint32_t payload_header = (system_header & ~(3u << 9)) | acquire;
    for (uint32_t i = 0; i < 4; ++i) {
      Publish(Semaphore(), waits, false, acquire);
      EXPECT_EQ(packets_[first].words[0], IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
      if (i) {
        EXPECT_EQ(packets_[first + i].words[0], payload_header);
      }
    }
    ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
    EXPECT_EQ(packets_[first].words[0], system_header);
    for (uint32_t i = 1; i < 4; ++i) {
      EXPECT_EQ(packets_[first + i].words[0], payload_header);
    }
    EXPECT_EQ(doorbell_, first + 3);
  }
}

TEST_P(AqlPublicationTest, FreshArgumentsRetainRequestedPayloadScopes) {
  queue_.producer.argument_cache_isolated = true;
  std::array<uint8_t, 4096> argument_bytes;
  argument_bytes.fill(0xCD);
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  // A nonempty record spans two ownership granules without touching padding.
  std::array<uint32_t, 33> constants;
  constants.fill(0x12345678);
  layout_.byte_length = std::get<1>(GetParam()) ? sizeof(constants) : 0;
  request_.constants =
      iree_make_const_byte_span(constants.data(), layout_.byte_length);
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  uint32_t record = 0;
  for (uint32_t acquire : {0u, 1u << 9, 2u << 9}) {
    for (uint32_t release : {0u, 1u << 11, 2u << 11}) {
      const uint64_t first = queue_.producer.ring.reserved;
      Publish(Semaphore(), waits, false, acquire, release);
      ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
      const uint32_t expected_acquire =
          layout_.byte_length && record == 0
              ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM
              : iree_max(acquire, IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT);
      EXPECT_EQ(packets_[first].words[0],
                (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                    IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | expected_acquire |
                    release);
      const size_t offset = record * 256;
      EXPECT_EQ(PacketAddress(first, 10),
                layout_.byte_length ? arguments_.address + offset : 0u);
      if (layout_.byte_length) {
        EXPECT_EQ(std::memcmp(argument_bytes.data() + offset, constants.data(),
                              sizeof(constants)),
                  0);
        EXPECT_TRUE(
            std::all_of(argument_bytes.begin() + offset + sizeof(constants),
                        argument_bytes.begin() + offset + 256,
                        [](uint8_t value) { return value == 0xCD; }));
      }
      ++record;
    }
  }
}

TEST_P(AqlPublicationTest, PublishedGroupsStayImmutableAcrossRingWrap) {
  // Begin near the ring's end with no outstanding native packets. Logical
  // retirement positions and native packet positions are independent domains.
  queue_.producer.ring.reserved = 60;
  queue_.producer.ring.consumed = 60;
  iree_atomic_store(&read_index_, 60, iree_memory_order_release);
  iree_atomic_store(&write_index_, 60, iree_memory_order_release);
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  const uint64_t first = queue_.producer.ring.reserved;
  const uint32_t acquire = std::get<1>(GetParam()) ? 1u << 9 : 0;
  const uint32_t release = 1u << 11;
  const uint32_t payload_header =
      (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
      IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | acquire | release;
  const uint32_t group_header = (payload_header & ~(3u << 9)) |
                                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
  iree_hal_semaphore_t* tail = nullptr;
  for (uint32_t i = 0; i < 9; ++i) {
    tail = Semaphore();
    Publish(tail, waits, false, acquire, release);
    EXPECT_EQ(doorbell_, i < 8 ? first - 1 : first + 7);
  }
  std::array<iree_hal_amd_gpu_aql_packet_t, 8> published;
  for (uint32_t i = 0; i < published.size(); ++i) {
    const uint32_t slot = (first + i) & (packets_.size() - 1);
    published[i] = packets_[slot];
    EXPECT_EQ(published[i].words[0], i ? payload_header : group_header);
    EXPECT_EQ(PacketAddress(slot, 14), 0u);
  }
  EXPECT_EQ(packets_[(first + 8) & (packets_.size() - 1)].words[0],
            IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
  for (uint32_t i = 9; i < 14; ++i) {
    tail = Semaphore();
    Publish(tail, waits, false, acquire, release);
    EXPECT_EQ(doorbell_, first + 7);
  }
  const uint64_t logical = queue_.producer.count;
  waits = {{.semaphore = tail, .value = 1}};
  request_.function = nullptr;
  Publish(Semaphore(), waits, false, 0,
          IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
  EXPECT_EQ(queue_.producer.ring.reserved, first + 14);
  EXPECT_EQ(doorbell_, first + 13);
  EXPECT_EQ(queue_.run.completion, nullptr);
  for (uint32_t i = 0; i < published.size(); ++i) {
    const uint32_t slot = (first + i) & (packets_.size() - 1);
    EXPECT_EQ(std::memcmp(&published[i], &packets_[slot], sizeof(published[i])),
              0);
  }
  for (uint32_t i = 8; i < 14; ++i) {
    const uint32_t slot = (first + i) & (packets_.size() - 1);
    uint32_t header = i == 8 ? group_header : payload_header;
    if (i == 13) {
      header =
          (header & ~(3u << 11)) | IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
    }
    EXPECT_EQ(packets_[slot].words[0], header);
    EXPECT_EQ(PacketAddress(slot, 14),
              i == 13 ? issued_[logical].signal->device_address : 0u);
  }
}

TEST_P(AqlPublicationTest, FullRunHasOnePhysicalCompletionAcrossGroups) {
  const bool isolated = std::get<1>(GetParam());
  queue_.producer.argument_cache_isolated = isolated;
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  // Firmware can reclaim the standalone packet before host retirement. This
  // permits the following full run to wrap without reusing its logical owner.
  packets_[0].words[0] = IREE_HAL_AMD_GPU_AQL_HEADER_INVALID;
  iree_atomic_store(&read_index_, 1, iree_memory_order_release);
  const uint32_t payload_header =
      (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
      IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED | (1u << 9) | (1u << 11);
  for (uint32_t i = 0; i < 64; ++i) {
    Publish(Semaphore(), waits, false, 1u << 9, 1u << 11);
    if (i < 63) {
      EXPECT_NE(queue_.run.completion, nullptr);
      EXPECT_EQ(doorbell_, isolated ? i : i < 8 ? 0 : (i / 8) * 8);
    }
    EXPECT_EQ(issued_[i + 1].signal->host_value, issued_[1].signal->host_value);
  }
  EXPECT_EQ(queue_.run.completion, nullptr);
  EXPECT_EQ(queue_.run.position, 0);
  EXPECT_EQ(queue_.producer.ring.reserved, 65u);
  EXPECT_EQ(doorbell_, 64u);
  for (uint32_t i = 0; i < 64; ++i) {
    const uint32_t slot = (i + 1) & (packets_.size() - 1);
    const uint32_t header =
        isolated || i % 8 ? payload_header
                          : (payload_header & ~(3u << 9)) |
                                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
    EXPECT_EQ(packets_[slot].words[0], header);
    EXPECT_EQ(PacketAddress(slot, 14),
              i == 63 ? issued_[1].signal->device_address : 0u);
  }
}

TEST_P(AqlPublicationTest, TerminalReleaseSharesTheUnpublishedFinalDispatch) {
  const int release_kind = std::get<0>(GetParam());
  const bool multiple_packets = std::get<1>(GetParam());
  queue_.producer.argument_cache_isolated = multiple_packets;
  const uint32_t terminal_release =
      release_kind ? IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM : 1u << 11;
  const uint32_t run_length = multiple_packets ? 3 : 1;
  std::array<uint8_t, 16 * IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT>
      argument_bytes = {};
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  const uint64_t constant = 0x123456789abcdef0;
  layout_.byte_length = sizeof(constant);
  request_.constants = iree_make_const_byte_span(&constant, sizeof(constant));
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);

  for (uint32_t release : {0u, 1u << 11, 2u << 11}) {
    const uint64_t first = queue_.producer.ring.reserved;
    iree_hal_semaphore_t* tail = nullptr;
    for (uint32_t i = 0; i < run_length; ++i) {
      tail = Semaphore();
      Publish(tail, waits, false, 0, release);
    }
    const uint64_t logical = queue_.producer.count;
    const uint64_t argument_end = arguments_.end;
    ASSERT_NE(queue_.run.completion, nullptr);
    EXPECT_EQ(
        packets_[first + (multiple_packets ? run_length - 1 : 0)].words[0],
        IREE_HAL_AMD_GPU_AQL_HEADER_INVALID);
    waits = {{.semaphore = tail, .value = 1}};
    request_.function = nullptr;
    Publish(Semaphore(), waits, false, 0, terminal_release, release_kind & 1);
    request_.function = &function_;
    waits.clear();

    EXPECT_EQ(queue_.producer.ring.reserved, first + run_length);
    EXPECT_EQ(arguments_.end, argument_end);
    EXPECT_EQ(queue_.producer.count, logical + 1);
    EXPECT_EQ(queue_.run.completion, nullptr);
    EXPECT_EQ(queue_.run.position, 0);
    EXPECT_EQ(doorbell_, first + run_length - 1);
    EXPECT_EQ(issued_[logical].submission->wait_count, 1u);
    EXPECT_EQ(issued_[logical].submission->native_wait_count, 0u);
    EXPECT_NE(issued_[logical].signal, issued_[logical - 1].signal);
    EXPECT_EQ(issued_[logical].signal->host_value,
              issued_[logical - 1].signal->host_value);
    for (uint64_t position : {logical - 1, logical}) {
      std::array<uint64_t, 2> frontier = {};
      iree_hal_amd_gpu_signal_join_frontier(issued_[position].signal,
                                            frontier.size(), frontier.data());
      EXPECT_EQ(frontier[0], position + 1);
    }
    for (uint32_t i = 0; i < run_length; ++i) {
      EXPECT_EQ(
          packets_[first + i].words[0],
          (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
              IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
              (multiple_packets ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT
               : i              ? 0
                                : IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM) |
              (i + 1 == run_length ? iree_max(release, terminal_release)
                                   : release));
      EXPECT_EQ(
          PacketAddress(first + i, 14),
          i + 1 == run_length ? issued_[logical].signal->device_address : 0u);
    }
  }
}

TEST_P(AqlPublicationTest, TerminalReleasePreservesForeignWaitsAndAcquires) {
  const int other_count = std::get<0>(GetParam());
  const bool acquire = std::get<1>(GetParam());
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits, false, 0, 0);
  auto* tail = Semaphore();
  Publish(tail, waits, false, 0, 0);
  waits = {{.semaphore = tail, .value = 1}};
  for (int i = 0; i < other_count; ++i) {
    auto* signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&signals_[1].cache.pool, 1);
    ASSERT_NE(signal, nullptr);
    other_producers_.push_back(signal);
    const uint64_t frontier[2] = {0, static_cast<uint64_t>(i + 1)};
    iree_hal_amd_gpu_signal_publish_frontier(signal, frontier);
    auto* semaphore = Semaphore();
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_semaphore_publish_signal(semaphore, 1, signal));
    waits.push_back({.semaphore = semaphore, .value = 1});
  }
  request_.function = nullptr;
  Publish(Semaphore(), waits, false,
          acquire ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM : 0);
  const bool separate = acquire || other_count;
  EXPECT_EQ(queue_.producer.ring.reserved,
            2u + (other_count + 4) / 5 + separate);
  EXPECT_EQ(queue_.run.completion, nullptr);
  EXPECT_EQ(frontier_[1], other_count);
  EXPECT_EQ(issued_[2].submission->native_wait_count, other_count);
  EXPECT_EQ(issued_[2].signal->host_value == issued_[1].signal->host_value,
            !separate);
  if (separate) {
    const uint64_t terminal = queue_.producer.ring.reserved - 1;
    EXPECT_EQ(packets_[terminal].words[0],
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                  (acquire ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM : 0) |
                  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
    EXPECT_EQ(PacketAddress(terminal, 14), issued_[2].signal->device_address);
  }
}

TEST_P(AqlPublicationTest, ProfiledBarriersRetainIndependentCompletions) {
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  Publish(Semaphore(), waits);
  auto operation = BeginProfile();
  iree_hal_amd_gpu_profile_operation_initialize(
      queue_.profile, 0, IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER, 0, 0, 0,
      &operation);
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {};
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  iree_status_t status = iree_ok_status();
  ASSERT_EQ(
      iree_hal_amd_gpu_aql_publish(&queue_, &request, 0, nullptr, {}, {}, 0,
                                   IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM,
                                   nullptr, &operation, &submission, &status),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(queue_.producer.ring.reserved, 3u);
  EXPECT_EQ(queue_.run.completion, nullptr);
  EXPECT_NE(issued_[2].signal->host_value, issued_[1].signal->host_value);
  EXPECT_NE(iree_hal_amd_gpu_aql_submission_profile(submission), nullptr);
  EXPECT_EQ(PacketAddress(2, 14), issued_[2].signal->device_address);
}

TEST_P(AqlPublicationTest, EmitsOnlyRequiredDependenciesAndCacheBoundaries) {
  const int other_count = std::get<0>(GetParam());
  const bool reverse_scopes = std::get<1>(GetParam());
  auto* first = Semaphore();
  auto* second = Semaphore();
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(first, waits);
  waits.push_back({.semaphore = first, .value = 1});
  Publish(second, waits);
  ASSERT_EQ(queue_.producer.ring.reserved, 2u);
  EXPECT_EQ(issued_[1].submission->wait_count, 1u);
  EXPECT_EQ(issued_[1].submission->native_wait_count, 0u);
  EXPECT_EQ(issued_[1].submission->waits[0].signal, issued_[0].signal);
  EXPECT_EQ(issued_[1].submission->pool, nullptr);

  waits = {{.semaphore = first, .value = 1}};
  for (int i = 0; i < other_count; ++i) {
    auto* signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&signals_[1].cache.pool, 1);
    ASSERT_NE(signal, nullptr);
    other_producers_.push_back(signal);
    const uint64_t frontier[2] = {0, static_cast<uint64_t>(i + 1)};
    iree_hal_amd_gpu_signal_publish_frontier(signal, frontier);
    auto* semaphore = Semaphore();
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_semaphore_publish_signal(semaphore, 1, signal));
    waits.push_back({.semaphore = semaphore, .value = 1});
  }
  waits.push_back({.semaphore = second, .value = 1});
  Publish(Semaphore(), waits, reverse_scopes);
  ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);

  auto* capture = issued_[2].submission;
  ASSERT_NE(capture, nullptr);
  EXPECT_EQ(capture->wait_count, other_count + 2u);
  EXPECT_EQ(capture->native_wait_count, other_count);
  EXPECT_EQ(frontier_[1], other_count);
  const size_t dependency_packets = (other_count + 4) / 5;
  EXPECT_EQ(queue_.producer.ring.reserved,
            3u + dependency_packets + (reverse_scopes ? 2u : 0u));
  size_t position = 2;
  for (int i = 0; i < other_count; i += 5, ++position) {
    EXPECT_EQ(packets_[position].words[0],
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED);
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(
          PacketAddress(position, 2 + 2 * j),
          i + j < other_count ? other_producers_[i + j]->device_address : 0u);
    }
  }
  if (reverse_scopes) {
    EXPECT_EQ(packets_[position++].words[0],
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                  IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
  }
  EXPECT_EQ(packets_[position].words[0],
            (1u << 16) | IREE_HAL_AMD_GPU_AQL_HEADER_DISPATCH |
                IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM |
                IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM);
  EXPECT_EQ(PacketAddress(position, 8), function_.kernel_object);
  EXPECT_EQ(PacketAddress(position++, 14),
            reverse_scopes ? 0u : issued_[2].signal->device_address);
  if (reverse_scopes) {
    EXPECT_EQ(packets_[position].words[0],
              IREE_HAL_AMD_GPU_AQL_HEADER_BARRIER_AND |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ORDERED |
                  IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
    EXPECT_EQ(PacketAddress(position++, 14), issued_[2].signal->device_address);
  }
  EXPECT_EQ(position, queue_.producer.ring.reserved);
  EXPECT_EQ(doorbell_, position - 1);
}

TEST_P(AqlPublicationTest, FullTransferStorageLeavesLaunchArgumentsAvailable) {
  queue_.producer.argument_cache_isolated = std::get<1>(GetParam());
  std::array<uint8_t, 4096> source;
  source.fill(0xA5);
  std::array<uint8_t, 4096> payload = {};
  constexpr size_t kSecondArgumentOffset =
      IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT;
  std::array<uint8_t, 2 * kSecondArgumentOffset> argument_bytes;
  argument_bytes.fill(0xCD);
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  iree_hal_amd_gpu_aql_program_pool_t programs = {.blit = 0x1234};
  queue_.options.programs = &programs;
  const iree_hal_amd_gpu_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD,
      .source = {.host = source.data()},
      .target = {.device = iree_hal_amd_gpu_buffer_reference_borrow(0x4000)},
      .length = source.size(),
  };
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_file(
      &queue_.captures, &operation, 0, 0, {}, &submission));
  iree_status_t status = iree_ok_status();
  EXPECT_EQ(iree_hal_amd_gpu_aql_publish_transfer(&queue_, submission, &status),
            IREE_HAL_AMD_GPU_AQL_PUBLISH_TRANSFERS);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(arguments_.end, 0u);
  EXPECT_EQ(queue_.producer.ring.reserved, 0u);

  iree_hal_amd_gpu_byte_storage_t transfers = {};
  transfers.memory.contents =
      iree_make_byte_span(payload.data(), payload.size());
  transfers.address = 0xABC000008000;
  transfers.capacity = payload.size();
  queue_.producer.transfers = &transfers;
  EXPECT_EQ(iree_hal_amd_gpu_aql_publish_transfer(&queue_, submission, &status),
            IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(payload, source);
  EXPECT_EQ(arguments_.end, 32u);
  EXPECT_EQ(transfers.end, 4096u);
  EXPECT_EQ(issued_[0].arguments, 32u);
  EXPECT_EQ(issued_[0].transfer.end, 4096u);
  EXPECT_EQ(PacketAddress(0, 10), arguments_.address);
  iree_hal_amd_gpu_blit_arguments_t blit;
  std::memcpy(&blit, argument_bytes.data(), sizeof(blit));
  EXPECT_EQ(blit.source, transfers.address);
  EXPECT_EQ(blit.target, 0x4000u);
  EXPECT_EQ(blit.length, source.size());

  // Neither transfer completion nor payload space is required by this launch.
  std::array<uint8_t, 32> launch_arguments;
  launch_arguments.fill(0x5A);
  layout_.byte_length = launch_arguments.size();
  request_.constants = iree_make_const_byte_span(launch_arguments.data(),
                                                 launch_arguments.size());
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits, false, 0);
  ASSERT_EQ(iree_hal_amd_gpu_aql_publish_run(&queue_), AMDF_STATUS_OK);
  EXPECT_EQ(arguments_.end, kSecondArgumentOffset + 32);
  EXPECT_EQ(transfers.end, 4096u);
  EXPECT_EQ(PacketAddress(2, 10), arguments_.address + kSecondArgumentOffset);
  EXPECT_EQ(packets_[0].words[0] & (3u << 9),
            IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  EXPECT_EQ(packets_[2].words[0] & (3u << 9),
            queue_.producer.argument_cache_isolated
                ? IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_AGENT
                : IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM);
  EXPECT_EQ(std::memcmp(argument_bytes.data() + kSecondArgumentOffset,
                        launch_arguments.data(), 32),
            0);
  EXPECT_TRUE(std::all_of(argument_bytes.begin() + sizeof(blit),
                          argument_bytes.begin() + kSecondArgumentOffset,
                          [](uint8_t value) { return value == 0xCD; }));
  EXPECT_TRUE(std::all_of(
      argument_bytes.begin() + kSecondArgumentOffset + launch_arguments.size(),
      argument_bytes.end(), [](uint8_t value) { return value == 0xCD; }));
  EXPECT_EQ(payload, source);
}

TEST_P(AqlPublicationTest, ScalarCommandsUseRetirementSlotsWithFullArguments) {
  ASSERT_NO_FATAL_FAILURE(PrepareControlBuffer());
  context_.target_info.topology.xcc_count = std::get<1>(GetParam()) ? 8 : 1;
  std::array<uint8_t, IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT> argument_bytes =
      {};
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  layout_.byte_length = argument_bytes.size();
  std::array<uint64_t, IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT / 8> constants;
  constants.fill(UINT64_C(0x123456789ABCDEF0));
  request_.constants =
      iree_make_const_byte_span(constants.data(), sizeof(constants));
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  ASSERT_EQ(arguments_.end, argument_bytes.size());

  std::vector<iree_hal_amd_gpu_control_t> controls;
  for (iree_hal_atomic_width_t width :
       {IREE_HAL_ATOMIC_WIDTH_32, IREE_HAL_ATOMIC_WIDTH_64}) {
    controls.push_back(
        iree_hal_amd_gpu_control_wait({.value = 7, .width = width}));
    controls.push_back(
        iree_hal_amd_gpu_control_store({.value = 11, .width = width}));
    controls.push_back(
        iree_hal_amd_gpu_control_rmw({.operand = 13, .width = width}));
  }
  controls.push_back(iree_hal_amd_gpu_control_timestamp());
  std::array<uint32_t, 512> words;
  words.fill(0xCDCDCDCD);
  iree_hal_amd_gpu_byte_storage_t commands = {};
  commands.memory.contents = iree_make_byte_span(words.data(), sizeof(words));
  commands.address = 0xABCD00008000;
  commands.capacity = sizeof(words);
  const uint32_t prefix = context_.target_info.topology.xcc_count > 1 ? 2 : 0;
  for (size_t i = 0; i < controls.size(); ++i) {
    SCOPED_TRACE(i);
    iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_control(
        &queue_.captures, nullptr, &family_, {}, {}, &control_buffer_, i * 8,
        controls[i], 0, 0, &submission));
    iree_status_t status = iree_ok_status();
    if (!i) {
      EXPECT_EQ(
          iree_hal_amd_gpu_aql_publish_control(&queue_, submission, &status),
          IREE_HAL_AMD_GPU_AQL_PUBLISH_COMMANDS);
      IREE_ASSERT_OK(status);
      EXPECT_EQ(queue_.producer.ring.reserved, 1u);
      EXPECT_EQ(arguments_.end, argument_bytes.size());
      queue_.commands = &commands;
    }
    ASSERT_EQ(
        iree_hal_amd_gpu_aql_publish_control(&queue_, submission, &status),
        IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
    IREE_ASSERT_OK(status);
    const size_t slot = i + 1;
    const size_t offset =
        slot * IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
    EXPECT_EQ(PacketAddress(slot, 2), commands.address + offset);
    const uint32_t length = iree_hal_amd_gpu_control_word_count(controls[i]);
    EXPECT_EQ(packets_[slot].words[4], (1u << 23) | (prefix + length));
    const size_t address_word =
        offset / 4 + prefix +
        iree_hal_amd_gpu_control_address_word(controls[i]);
    EXPECT_EQ(words[address_word], control_binding_.device_address + i * 8);
    EXPECT_EQ(words[address_word + 1], 0u);
    EXPECT_EQ(words[offset / 4 + prefix + length], 0xCDCDCDCDu);
    EXPECT_EQ(issued_[slot].arguments, argument_bytes.size());
    EXPECT_EQ(arguments_.end, argument_bytes.size());
  }
  EXPECT_EQ(
      std::memcmp(argument_bytes.data(), constants.data(), sizeof(constants)),
      0);
}

TEST_P(AqlPublicationTest, CooperativeCommandsAndKernargsHaveDistinctOwners) {
  context_.target_info.topology.xcc_count = std::get<1>(GetParam()) ? 8 : 1;
  std::array<uint8_t, 2 * IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT>
      argument_bytes = {};
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  const iree_hal_amd_gpu_metadata_argument_t argument = {
      .size = 8,
      .kind = IREE_HAL_AMD_GPU_METADATA_ARGUMENT_KIND_HIDDEN_MULTIGRID_SYNC_ARG,
  };
  metadata_.kernarg_segment_size = 8;
  metadata_.kernarg_segment_alignment = 8;
  metadata_.has_argument_metadata = true;
  metadata_.arguments = &argument;
  metadata_.argument_count = 1;
  IREE_ASSERT_OK(iree_hal_amd_gpu_kernarg_layout_allocate(
      &metadata_, iree_allocator_system(), &cooperative_layout_));
  function_.base.kernarg = cooperative_layout_;
  request_.flags |= IREE_HAL_DISPATCH_FLAG_COOPERATIVE;
  queue_.grid.memory = reinterpret_cast<amdf_memory_t*>(&queue_.grid);
  queue_.grid.address = 0x123400005000;
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  iree_status_t status = iree_ok_status();
  EXPECT_EQ(
      iree_hal_amd_gpu_aql_publish(&queue_, &request_, 0, nullptr, {}, {}, 0, 0,
                                   nullptr, nullptr, &submission, &status),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_COMMANDS);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(submission, nullptr);
  EXPECT_EQ(queue_.producer.ring.reserved, 0u);
  EXPECT_EQ(arguments_.end, 0u);

  std::array<uint32_t, 512> words;
  words.fill(0xCDCDCDCD);
  iree_hal_amd_gpu_byte_storage_t commands = {};
  commands.memory.contents = iree_make_byte_span(words.data(), sizeof(words));
  commands.address = 0xABCD00008000;
  commands.capacity = sizeof(words);
  queue_.commands = &commands;
  const uint32_t prefix = context_.target_info.topology.xcc_count > 1 ? 2 : 0;
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  for (size_t i = 0; i < 2; ++i) {
    request_.geometry.workgroup_count[0] = i + 2;
    Publish(Semaphore(), waits);
    const size_t offset = i * IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
    const size_t argument_offset = i * IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT;
    EXPECT_EQ(PacketAddress(i * 2, 2), commands.address + offset);
    EXPECT_EQ(PacketAddress(i * 2 + 1, 10),
              arguments_.address + argument_offset);
    EXPECT_EQ(packets_[i * 2].words[4], (1u << 23) | (prefix + 16));
    iree_hal_amd_gpu_grid_sync_t grid;
    std::memcpy(&grid, words.data() + offset / 4 + prefix + 4, sizeof(grid));
    EXPECT_EQ(grid.workgroup_count, i + 2);
    EXPECT_EQ(grid.total_workitem_count, (i + 2) * 64);
    uint64_t hidden_pointer = 0;
    std::memcpy(&hidden_pointer, argument_bytes.data() + argument_offset,
                sizeof(hidden_pointer));
    EXPECT_EQ(hidden_pointer, queue_.grid.address);
    EXPECT_EQ(arguments_.end, argument_offset + 8);
  }
  EXPECT_EQ(words[2 * IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE / 4],
            0xCDCDCDCDu);
}

TEST_P(AqlPublicationTest,
       CooperativeZeroArgumentDispatchDoesNotConsumeKernargs) {
  context_.target_info.topology.xcc_count = std::get<1>(GetParam()) ? 8 : 1;
  // Queue creation owns argument backing even when this dispatch has no ABI
  // bytes. Cooperative scheduling must neither consume nor write that backing.
  std::array<uint8_t, 64> argument_bytes;
  argument_bytes.fill(0xCD);
  const auto expected_arguments = argument_bytes;
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  request_.flags |= IREE_HAL_DISPATCH_FLAG_COOPERATIVE;
  queue_.grid.memory = reinterpret_cast<amdf_memory_t*>(&queue_.grid);
  queue_.grid.address = 0x123400005000;
  std::array<uint32_t, 512> words;
  words.fill(0xCDCDCDCD);
  iree_hal_amd_gpu_byte_storage_t commands = {};
  commands.memory.contents = iree_make_byte_span(words.data(), sizeof(words));
  commands.address = 0xABCD00008000;
  commands.capacity = sizeof(words);
  queue_.commands = &commands;
  const uint32_t prefix = context_.target_info.topology.xcc_count > 1 ? 2 : 0;
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  for (size_t i = 0; i < 2; ++i) {
    request_.geometry.workgroup_count[0] = i + 2;
    Publish(Semaphore(), waits);
    const size_t offset = i * IREE_HAL_AMD_GPU_AQL_INDIRECT_COMMAND_BYTE_STRIDE;
    EXPECT_EQ(PacketAddress(i * 2, 2), commands.address + offset);
    EXPECT_EQ(PacketAddress(i * 2 + 1, 10), 0u);
    EXPECT_EQ(packets_[i * 2].words[4], (1u << 23) | (prefix + 16));
    iree_hal_amd_gpu_grid_sync_t grid;
    std::memcpy(&grid, words.data() + offset / 4 + prefix + 4, sizeof(grid));
    EXPECT_EQ(grid.multi_grid_sync, 0u);
    EXPECT_EQ(grid.grid_ordinal, 0u);
    EXPECT_EQ(grid.grid_count, 1u);
    EXPECT_EQ(grid.preceding_workitem_count, 0u);
    EXPECT_EQ(grid.total_workitem_count, (i + 2) * 64);
    EXPECT_EQ(grid.single_grid_sync[0], 0u);
    EXPECT_EQ(grid.single_grid_sync[1], 0u);
    EXPECT_EQ(grid.workgroup_count, i + 2);
    EXPECT_EQ(grid.reserved, 0u);
    EXPECT_EQ(issued_[i].arguments, 0u);
    EXPECT_EQ(arguments_.end, 0u);
    EXPECT_EQ(argument_bytes, expected_arguments);
    EXPECT_EQ(words[offset / 4 + prefix + 16], 0xCDCDCDCDu);
  }
}

TEST_P(AqlPublicationTest, FullSamplesLeaveLaunchArgumentsAvailable) {
  auto operation = BeginProfile();
  constexpr size_t kSecondArgumentOffset =
      IREE_HAL_AMD_GPU_AQL_ARGUMENT_ALIGNMENT;
  std::array<uint8_t, 2 * kSecondArgumentOffset> argument_bytes;
  argument_bytes.fill(0xCD);
  arguments_.memory.contents =
      iree_make_byte_span(argument_bytes.data(), argument_bytes.size());
  arguments_.address = 0x7000;
  arguments_.capacity = argument_bytes.size();
  std::array<uint8_t, 32> constants;
  constants.fill(0xA5);
  layout_.byte_length = constants.size();
  request_.constants =
      iree_make_const_byte_span(constants.data(), constants.size());
  iree_hal_amd_gpu_aql_signal_t sample = {};
  iree_hal_amd_gpu_byte_storage_t generation = {};
  generation.memory.contents = iree_make_byte_span(&sample, sizeof(sample));
  generation.address = 0xABCD00009000;
  generation.capacity = sizeof(sample);
  iree_hal_amd_gpu_byte_storage_pool_t samples = {.current = &generation};
  queue_.samples = &samples;
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  iree_status_t status = iree_ok_status();
  ASSERT_EQ(
      iree_hal_amd_gpu_aql_publish(&queue_, &request_, 0, nullptr, {}, {}, 0, 0,
                                   nullptr, &operation, &submission, &status),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
  IREE_ASSERT_OK(status);
  ASSERT_NE(submission, nullptr);
  EXPECT_EQ(arguments_.end, 32u);
  EXPECT_EQ(generation.end, sizeof(sample));
  EXPECT_EQ(PacketAddress(0, 10), arguments_.address);
  EXPECT_EQ(PacketAddress(0, 14), generation.address);
  EXPECT_EQ(PacketAddress(1, 14), issued_[0].signal->device_address);
  EXPECT_EQ(sample.value, 1u);
  const auto* completion =
      &iree_hal_amd_gpu_aql_submission_profile(submission)->completion;
  EXPECT_EQ(completion->samples,
            reinterpret_cast<const iree_hal_amd_gpu_profile_sample_t*>(
                &sample.start_timestamp));
  EXPECT_EQ(completion->samples_end, sizeof(sample));

  submission = nullptr;
  EXPECT_EQ(
      iree_hal_amd_gpu_aql_publish(&queue_, &request_, 0, nullptr, {}, {}, 0, 0,
                                   nullptr, &operation, &submission, &status),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_CAPACITY);
  IREE_ASSERT_OK(status);
  EXPECT_EQ(submission, nullptr);
  EXPECT_EQ(arguments_.end, 32u);
  EXPECT_EQ(generation.end, sizeof(sample));
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits;
  Publish(Semaphore(), waits);
  EXPECT_EQ(arguments_.end, kSecondArgumentOffset + 32);
  EXPECT_EQ(generation.end, sizeof(sample));
  EXPECT_EQ(PacketAddress(2, 10), arguments_.address + kSecondArgumentOffset);
  EXPECT_EQ(std::memcmp(argument_bytes.data() + kSecondArgumentOffset,
                        constants.data(), 32),
            0);
}

TEST_P(AqlPublicationTest, EmptyDispatchKeepsMetadataWithoutNativeSamples) {
  auto operation = BeginProfile();
  request_.geometry.workgroup_count[0] = 0;
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  iree_status_t status = iree_ok_status();
  ASSERT_EQ(
      iree_hal_amd_gpu_aql_publish(&queue_, &request_, 0, nullptr, {}, {}, 0, 0,
                                   nullptr, &operation, &submission, &status),
      IREE_HAL_AMD_GPU_AQL_PUBLISH_DONE);
  IREE_ASSERT_OK(status);
  ASSERT_NE(submission, nullptr);
  EXPECT_EQ(arguments_.end, 0u);
  EXPECT_EQ(queue_.producer.ring.reserved, 1u);
  EXPECT_EQ(
      iree_hal_amd_gpu_aql_submission_profile(submission)->completion.samples,
      nullptr);
  EXPECT_EQ(iree_hal_amd_gpu_aql_submission_profile(submission)
                ->completion.samples_end,
            0u);
  EXPECT_EQ(PacketAddress(0, 14), issued_[0].signal->device_address);
}

INSTANTIATE_TEST_SUITE_P(NativeDependencies, AqlPublicationTest,
                         ::testing::Combine(::testing::Values(0, 1, 5, 6),
                                            ::testing::Bool(),
                                            ::testing::Bool()));

}  // namespace
