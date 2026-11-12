// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/publication.h"

#include <array>
#include <cstring>
#include <tuple>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/cts/util/profile_test_util.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/hal/driver/amd/gpu/queue/sample_storage.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Publication, capture ownership, and native timelines are real. Only the
// device-owned ring and completion allocations are supplied by the fixture.
class AqlPublicationTest
    : public ::testing::TestWithParam<std::tuple<int, bool>> {
 protected:
  struct Signals {
    // Stable queue identity consumed by dependency admission.
    iree_hal_amd_gpu_signal_cache_t cache = {};
    // Allocation metadata shared by independent producer and reader owners.
    iree_hal_amd_gpu_signal_block_t block = {};
    // Host ownership for sixteen native completion descriptors.
    std::array<iree_hal_amd_gpu_signal_t, 16> signals = {};
    // Coherent native completion allocation.
    std::array<iree_hal_amd_gpu_aql_signal_t, 16> values = {};
    // Two-queue causal vectors owned by the allocation.
    std::array<uint64_t, 2 + 16 * 2> frontiers = {};
  };

  void SetUp() override {
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
          0x1000 + i * 0x1000, &signals.block);
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
    iree_hal_amd_gpu_aql_ring_initialize(
        packets_.data(), 0x10000, packets_.size(), &read_index_, &write_index_,
        &doorbell_, AMDF_CACHE_TRANSITION_EXECUTOR_NONE, &queue_.producer.ring);
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
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    iree_hal_amd_gpu_capture_pool_deinitialize(&queue_.captures);
    for (auto& signals : signals_) {
      iree_hal_amd_gpu_signal_block_deinitialize(&signals.block);
      iree_hal_amd_gpu_signal_pool_deinitialize(&signals.cache.pool);
    }
    IREE_EXPECT_OK(iree_hal_amd_gpu_profile_end(queue_.profile));
    iree_async_proactor_release(proactor_);
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
               bool reverse_scopes = false) {
    iree_hal_amd_gpu_aql_wait_counts_t counts;
    iree_host_size_t wait_index = 0;
    iree_hal_amd_gpu_aql_wait_kind_t kind;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_admit_waits(
        &context_, &signals_[0].cache.pool, waits.size(), waits.data(), true,
        &counts, &wait_index, &kind));
    ASSERT_EQ(kind, IREE_HAL_AMD_GPU_AQL_WAIT_READY);
    uint32_t before = IREE_HAL_AMD_GPU_AQL_HEADER_ACQUIRE_SYSTEM;
    uint32_t after = IREE_HAL_AMD_GPU_AQL_HEADER_RELEASE_SYSTEM;
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
  std::array<iree_hal_amd_gpu_aql_retirement_t, 16> issued_ = {};
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
  // Placed descriptor used by actual dispatch encoding.
  iree_hal_amd_gpu_aql_function_t function_ = {};
  // Admitted ordinary raw-native launch.
  iree_hal_amd_gpu_aql_dispatch_request_t request_ = {};
  // Original owners of the external native producers.
  std::vector<iree_hal_amd_gpu_signal_t*> other_producers_;
  // Timelines kept alive until all issued captures retire.
  std::vector<iree_hal_semaphore_t*> semaphores_;
};

TEST_P(AqlPublicationTest, EmitsOnlyRequiredDependenciesAndCacheBoundaries) {
  auto [other_count, reverse_scopes] = GetParam();
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
  std::array<uint8_t, 4096> source;
  source.fill(0xA5);
  std::array<uint8_t, 4096> payload = {};
  std::array<uint8_t, 64> argument_bytes = {};
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
  Publish(Semaphore(), waits);
  EXPECT_EQ(arguments_.end, 64u);
  EXPECT_EQ(transfers.end, 4096u);
  EXPECT_EQ(PacketAddress(2, 10), arguments_.address + 32);
  EXPECT_EQ(
      std::memcmp(argument_bytes.data() + 32, launch_arguments.data(), 32), 0);
  EXPECT_EQ(payload, source);
}

TEST_P(AqlPublicationTest, FullSamplesLeaveLaunchArgumentsAvailable) {
  auto operation = BeginProfile();
  std::array<uint8_t, 64> argument_bytes = {};
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
  iree_hal_amd_gpu_sample_storage_t samples = {.current = &generation};
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
  EXPECT_EQ(arguments_.end, 64u);
  EXPECT_EQ(generation.end, sizeof(sample));
  EXPECT_EQ(PacketAddress(2, 10), arguments_.address + 32);
  EXPECT_EQ(std::memcmp(argument_bytes.data() + 32, constants.data(), 32), 0);
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
                                            ::testing::Bool()));

}  // namespace
