// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/submission.h"

#include <array>
#include <cstring>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/driver/amd/gpu/queue/aql/signal.h"
#include "iree/hal/driver/amd/gpu/queue/aql/transfer.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class AqlDependencyTest : public ::testing::Test {
 protected:
  struct NativeQueue {
    // Stable native queue identity and finite reader ownership.
    iree_hal_amd_gpu_signal_pool_t pool;
    // Native allocation metadata, attached until all readers return.
    iree_hal_amd_gpu_signal_block_t block;
    // Independent ownership for four emulated native completion cells.
    std::array<iree_hal_amd_gpu_signal_t, 4> signals = {};
    // Hardware-owned coherent completion words.
    std::array<uint64_t, 4> values = {};
    // Two-queue causal snapshots attached to the native allocation.
    std::array<uint64_t, 2 + 4 * 2> frontiers = {};
  };

  void SetUp() override {
    IREE_ASSERT_OK(iree_async_proactor_create_platform(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_));
    for (size_t i = 0; i < queues_.size(); ++i) {
      auto& queue = queues_[i];
      iree_hal_amd_gpu_signal_pool_initialize({}, &queue.pool);
      iree_hal_amd_gpu_signal_block_initialize(
          queue.signals.data(), queue.signals.size(), 2, queue.frontiers.data(),
          &queue.block);
      iree_hal_amd_gpu_signal_block_attach(
          &queue.pool, this, queue.values.data(), 0x1000 + i * 0x1000,
          sizeof(uint64_t), 0, &queue.block);
    }
  }

  void TearDown() override {
    for (auto& wait : waits_) {
      iree_hal_amd_gpu_signal_release(wait.signal);
    }
    for (auto* semaphore : semaphores_) {
      iree_hal_semaphore_release(semaphore);
    }
    for (auto* producer : producers_) {
      iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_OK);
      iree_hal_amd_gpu_signal_release(producer);
    }
    for (auto& queue : queues_) {
      iree_hal_amd_gpu_signal_block_deinitialize(&queue.block);
      iree_hal_amd_gpu_signal_pool_deinitialize(&queue.pool);
    }
    iree_async_proactor_release(proactor_);
  }

  iree_hal_semaphore_t* Semaphore(uint64_t initial_value = 0) {
    iree_hal_semaphore_t* semaphore = nullptr;
    IREE_CHECK_OK(
        iree_hal_amd_gpu_semaphore_create(proactor_, nullptr, initial_value, 2,
                                          iree_allocator_system(), &semaphore));
    semaphores_.push_back(semaphore);
    return semaphore;
  }

  iree_hal_amd_gpu_signal_t* AddNativeWait(size_t queue_index) {
    auto* producer =
        iree_hal_amd_gpu_signal_pool_try_acquire(&queues_[queue_index].pool, 1);
    IREE_ASSERT(producer);
    producers_.push_back(producer);
    const uint64_t frontier[2] = {queue_index == 0 ? 3u : 0u,
                                  queue_index == 1 ? 7u : 0u};
    iree_hal_amd_gpu_signal_publish_frontier(producer, frontier);
    auto* semaphore = Semaphore();
    IREE_CHECK_OK(
        iree_hal_amd_gpu_semaphore_publish_signal(semaphore, 1, producer));
    waits_.push_back({.semaphore = semaphore, .value = 1});
    return producer;
  }

  iree_status_t Admit(const void* domain) {
    return iree_hal_amd_gpu_aql_submission_admit_waits(
        domain, &queues_[0].pool, waits_.size(), waits_.data(), true, &counts_,
        &wait_index_, &kind_);
  }

  // Real timeline infrastructure shared by both native queues.
  iree_async_proactor_t* proactor_ = nullptr;
  // Native allocation dependencies; admission itself is production code.
  std::array<NativeQueue, 2> queues_;
  // Owned HAL timelines, released before their native allocation metadata.
  std::vector<iree_hal_semaphore_t*> semaphores_;
  // Original native producer references, independent of cached and wait refs.
  std::vector<iree_hal_amd_gpu_signal_t*> producers_;
  // Borrowed semaphore records and any reader references admitted into them.
  std::vector<iree_hal_amd_gpu_aql_wait_t> waits_;
  // Exact ownership and packet-operand extents produced by admission.
  iree_hal_amd_gpu_aql_wait_counts_t counts_ = {};
  // Unresolved record selected by deferred admission.
  iree_host_size_t wait_index_ = SIZE_MAX;
  // Resolution returned by the production admission path.
  iree_hal_amd_gpu_aql_wait_kind_t kind_ = IREE_HAL_AMD_GPU_AQL_WAIT_READY;
};

TEST_F(AqlDependencyTest, PartitionsCrossQueueOperandsWithoutDroppingReaders) {
  auto* same_first = AddNativeWait(0);
  waits_.push_back({.semaphore = Semaphore(1), .value = 1});
  auto* other_first = AddNativeWait(1);
  auto* same_second = AddNativeWait(0);
  auto* other_second = AddNativeWait(1);
  IREE_ASSERT_OK(Admit(this));
  EXPECT_EQ(kind_, IREE_HAL_AMD_GPU_AQL_WAIT_READY);
  EXPECT_EQ(counts_.retained_count, 4u);
  EXPECT_EQ(counts_.native_count, 2u);
  EXPECT_EQ(waits_[0].signal, other_first);
  EXPECT_EQ(waits_[1].signal, other_second);
  size_t same_count = 0;
  for (size_t i = counts_.native_count; i < waits_.size(); ++i) {
    if (waits_[i].signal) {
      EXPECT_TRUE(waits_[i].signal == same_first ||
                  waits_[i].signal == same_second);
      ++same_count;
    }
  }
  EXPECT_EQ(same_count, 2u);
  for (auto* signal : producers_) {
    EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 3);
  }
}

TEST_F(AqlDependencyTest, DeferredAdmissionReturnsEveryAcquiredReader) {
  AddNativeWait(0);
  AddNativeWait(1);
  waits_.push_back({.semaphore = Semaphore(), .value = 1});
  auto* deferred = waits_.back().semaphore;
  IREE_ASSERT_OK(Admit(this));
  EXPECT_EQ(kind_, IREE_HAL_AMD_GPU_AQL_WAIT_SUBMISSION);
  EXPECT_EQ(wait_index_, 2u);
  EXPECT_EQ(waits_[wait_index_].semaphore, deferred);
  EXPECT_EQ(counts_.retained_count, 0u);
  EXPECT_EQ(counts_.native_count, 0u);
  for (const auto& wait : waits_) {
    EXPECT_EQ(wait.signal, nullptr);
  }
  for (auto* signal : producers_) {
    EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 2);
  }
}

TEST_F(AqlDependencyTest, FailedAdmissionReturnsEveryAcquiredReader) {
  AddNativeWait(0);
  AddNativeWait(1);
  auto* failed = Semaphore();
  iree_hal_semaphore_fail(failed, iree_make_status(IREE_STATUS_ABORTED));
  waits_.push_back({.semaphore = failed, .value = 1});
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED, Admit(this));
  EXPECT_EQ(counts_.retained_count, 0u);
  EXPECT_EQ(counts_.native_count, 0u);
  for (const auto& wait : waits_) {
    EXPECT_EQ(wait.signal, nullptr);
  }
  for (auto* signal : producers_) {
    EXPECT_EQ(iree_atomic_ref_count_load(&signal->state.live.ref_count), 2);
  }
}

TEST_F(AqlDependencyTest, ForeignNativeDomainRequiresHostCompletion) {
  AddNativeWait(0);
  const uint64_t foreign_domain = 0;
  IREE_ASSERT_OK(Admit(&foreign_domain));
  EXPECT_EQ(kind_, IREE_HAL_AMD_GPU_AQL_WAIT_COMPLETION);
  EXPECT_EQ(wait_index_, 0u);
  EXPECT_EQ(counts_.retained_count, 0u);
  EXPECT_EQ(counts_.native_count, 0u);
  EXPECT_EQ(waits_[0].signal, nullptr);
}

struct Executable {
  // Real HAL resource whose payload dependency has no native work.
  iree_hal_executable_t base;
  // Destructor observation retained through the test.
  bool destroyed = false;
  static void Destroy(iree_hal_executable_t* base) {
    reinterpret_cast<Executable*>(base)->destroyed = true;
  }
};

class AqlSubmissionTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(),
                                             &pool_);
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&pool_);
  }
  // Actual bounded capture storage, shared by admission and retirement.
  iree_hal_amd_gpu_capture_pool_t pool_;
};

TEST_P(AqlSubmissionTest, AtomicTargetRetainsDeferredBackingUntilRetirement) {
  std::array<uint8_t, 16> bytes = {};
  bool destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, bytes.size(),
      iree_make_byte_span(bytes.data(), bytes.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &destroyed},
      iree_allocator_system(), &buffer));
  std::array<iree_hal_memory_scope_access_t, 2> scopes = {};
  scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  std::array<uint32_t, 2> zero_compute = {};
  iree_hal_memory_contract_t contract = {
      .scopes = scopes.data(),
      .zero_compute_atomic_operations = zero_compute.data(),
  };
  iree_hal_buffer_native_binding_t binding = {};
  buffer->memory = {.contract = &contract, .bindings = &binding};
  const iree_hal_queue_family_t family = {.memory = {.queue_scope_id = 1}};
  auto command = iree_hal_amd_gpu_control_store({
      .value = 123,
      .width = IREE_HAL_ATOMIC_WIDTH_64,
      .target_error_mode = static_cast<iree_hal_atomic_target_error_mode_t>(
          GetParam() ? IREE_HAL_ATOMIC_TARGET_ERROR_MODE_INCOMPATIBLE
                     : IREE_HAL_ATOMIC_TARGET_ERROR_MODE_DEFAULT),
  });
  iree_hal_amd_gpu_aql_submission_t* capture = nullptr;
  IREE_EXPECT_STATUS_IS(
      GetParam() ? IREE_STATUS_INCOMPATIBLE : IREE_STATUS_PERMISSION_DENIED,
      iree_hal_amd_gpu_aql_submission_control(&pool_, nullptr, &family, {}, {},
                                              buffer, 8, command, 0, 0,
                                              &capture));
  EXPECT_EQ(capture, nullptr);
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
  scopes[1].atomic_operations = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE << 8;
  command.flags |= IREE_HAL_ATOMIC_FLAG_ZERO_COMPUTE;
  IREE_EXPECT_STATUS_IS(
      GetParam() ? IREE_STATUS_INCOMPATIBLE : IREE_STATUS_PERMISSION_DENIED,
      iree_hal_amd_gpu_aql_submission_control(&pool_, nullptr, &family, {}, {},
                                              buffer, 8, command, 0, 0,
                                              &capture));
  EXPECT_EQ(capture, nullptr);
  EXPECT_EQ(iree_atomic_ref_count_load(&buffer->resource.ref_count), 1);
  zero_compute[1] = IREE_HAL_ATOMIC_OPERATION_FLAG_STORE << 8;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_control(
      &pool_, nullptr, &family, {}, {}, buffer, 8, command, 0, 0, &capture));
  uint64_t address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amd_gpu_control_resolve(
                            &capture->control.target, command, &address));
  binding.device_address = 0x1000;
  IREE_ASSERT_OK(iree_hal_amd_gpu_control_resolve(&capture->control.target,
                                                  command, &address));
  EXPECT_EQ(address, 0x1008u);
  iree_hal_buffer_release(buffer);
  EXPECT_FALSE(destroyed);
  iree_hal_amd_gpu_aql_submission_release(capture);
  EXPECT_TRUE(destroyed);
}

TEST_P(AqlSubmissionTest, CapturesInputsAndReleasesPayloadBeforeCompletion) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, nullptr, 0, 2, iree_allocator_system(), &semaphore));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_hal_amd_gpu_aql_function_t function = {};
  std::array<uint8_t, 16> bytes = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, bytes.size(),
      iree_make_byte_span(bytes.data(), bytes.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  auto reference = iree_hal_make_buffer_ref(buffer, 4, 8);
  uint32_t constant = 0x12345678;
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {
      .executable = &executable.base,
      .function = &function,
      .constants = iree_make_const_byte_span(&constant, sizeof(constant)),
      .bindings = {1, &reference},
  };
  uint64_t value = 4;
  iree_alignas(8) uint8_t inline_storage[496];
  iree_hal_amd_gpu_aql_submission_t* capture = nullptr;
  if (GetParam()) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
        &pool_, iree_make_byte_span(inline_storage, sizeof(inline_storage)),
        nullptr, &request, 0, nullptr, {}, {1, &semaphore, &value}, &capture));
    EXPECT_EQ(capture->pool, nullptr);
    EXPECT_EQ(capture->dispatch.constants.data_length, 0u);
  } else {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
        &pool_, nullptr, &request, {}, {1, &semaphore, &value}, 17, 23,
        &capture));
    constant = 0;
    uint32_t copied = 0;
    std::memcpy(&copied, capture->dispatch.constants.data, sizeof(copied));
    EXPECT_EQ(copied, 0x12345678u);
    EXPECT_EQ(capture->before, 17u);
    EXPECT_EQ(capture->after, 23u);
  }
  reference = {};
  value = 99;
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(semaphore);
  EXPECT_FALSE(executable.destroyed);
  EXPECT_FALSE(buffer_destroyed);
  EXPECT_EQ(capture->dispatch.bindings.values[0].offset, 4u);
  EXPECT_EQ(capture->signals.payload_values[0], 4u);
  iree_hal_amd_gpu_aql_submission_release_payload(capture);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
  uint64_t current = UINT64_MAX;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(capture->signals.semaphores[0], &current));
  EXPECT_EQ(current, 0u);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(capture->signals.semaphores[0], 4, nullptr));
  iree_hal_amd_gpu_aql_submission_release(capture);
  iree_async_proactor_release(proactor);
}

TEST_P(AqlSubmissionTest, DependencyReadersJoinProducerHostRetirement) {
  iree_hal_amd_gpu_signal_pool_t signals;
  iree_hal_amd_gpu_signal_pool_initialize({}, &signals);
  iree_hal_amd_gpu_aql_signal_t native;
  iree_hal_amd_gpu_signal_t owner = {};
  uint64_t frontier[2 + 1];
  iree_hal_amd_gpu_signal_block_t block;
  const amdf_gpu_event_info_t event = {};
  iree_hal_amd_gpu_signal_block_initialize(&owner, 1, 1, frontier, &block);
  iree_hal_amd_gpu_aql_signal_block_attach(&signals, this, &event, &native,
                                           0x1000, &block);
  auto* producer = iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1);
  iree_hal_amd_gpu_signal_retain(producer);
  const iree_hal_amd_gpu_aql_wait_t dependency = {.signal = producer};
  const iree_hal_amd_gpu_aql_dispatch_request_t barrier = {};
  iree_alignas(8) uint8_t inline_storage[496];
  iree_hal_amd_gpu_aql_submission_t* consumer = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
      &pool_,
      GetParam() ? iree_make_byte_span(inline_storage, sizeof(inline_storage))
                 : iree_byte_span_empty(),
      nullptr, &barrier, 1, &dependency,
      {.retained_count = 1, .native_count = 0}, {}, &consumer));
  EXPECT_EQ(consumer->wait_count, 1u);
  EXPECT_EQ(consumer->native_wait_count, 0u);
  EXPECT_EQ(consumer->waits[0].signal, producer);
  native.value = 0;  // Firmware has finished; the host still owns the payload.
  iree_status_code_t code = IREE_STATUS_OK;
  EXPECT_FALSE(
      iree_hal_amd_gpu_aql_submission_retirement_ready(consumer, &code));
  iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_signal_release(producer);
  EXPECT_TRUE(
      iree_hal_amd_gpu_aql_submission_retirement_ready(consumer, &code));
  EXPECT_EQ(code, IREE_STATUS_ABORTED);
  EXPECT_EQ(iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1), nullptr);
  iree_hal_amd_gpu_aql_submission_release(consumer);
  auto* reused = iree_hal_amd_gpu_signal_pool_try_acquire(&signals, 1);
  EXPECT_EQ(reused, producer);
  EXPECT_EQ(native.value, 1u);
  iree_hal_amd_gpu_signal_release(reused);
  iree_hal_amd_gpu_signal_block_deinitialize(&block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&signals);
}

TEST_P(AqlSubmissionTest, RejectionReleasesPayloadBeforeFailingPublicSignals) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, nullptr, 0, 2, iree_allocator_system(), &semaphore));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_hal_amd_gpu_aql_function_t function = {};
  struct Payload {
    // Public failure must remain unpublished while payload release runs.
    iree_hal_semaphore_t* semaphore;
    // Observes the actual resource release callback.
    bool destroyed = false;
  } payload{semaphore};
  std::array<uint8_t, 8> bytes = {};
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, bytes.size(),
      iree_make_byte_span(bytes.data(), bytes.size()),
      {[](void* user_data, iree_hal_buffer_t*) {
         auto* payload = static_cast<Payload*>(user_data);
         uint64_t value = UINT64_MAX;
         IREE_EXPECT_OK(iree_hal_semaphore_query(payload->semaphore, &value));
         EXPECT_EQ(value, 0u);
         payload->destroyed = true;
       },
       &payload},
      iree_allocator_system(), &buffer));
  const auto reference = iree_hal_make_buffer_ref(buffer, 0, bytes.size());
  const iree_hal_amd_gpu_aql_dispatch_request_t request = {
      .executable = &executable.base,
      .function = &function,
      .bindings = {1, &reference},
  };
  uint64_t one = 1;
  iree_alignas(8) uint8_t inline_storage[496];
  iree_hal_amd_gpu_aql_submission_t* capture = nullptr;
  if (GetParam()) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
        &pool_, iree_make_byte_span(inline_storage, sizeof(inline_storage)),
        nullptr, &request, 0, nullptr, {}, {1, &semaphore, &one}, &capture));
    EXPECT_EQ(capture->pool, nullptr);
  } else {
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
        &pool_, nullptr, &request, {}, {1, &semaphore, &one}, 0, 0, &capture));
  }
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  iree_hal_amd_gpu_aql_submission_reject(capture, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_aql_submission_reject(capture, IREE_STATUS_CANCELLED);
  uint64_t value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &value));
  EXPECT_EQ(value, 0u);
  EXPECT_FALSE(payload.destroyed);
  EXPECT_FALSE(executable.destroyed);
  iree_hal_amd_gpu_aql_submission_release(capture);
  EXPECT_TRUE(payload.destroyed);
  EXPECT_TRUE(executable.destroyed);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_ABORTED,
                        iree_hal_semaphore_query(semaphore, &value));
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST_P(AqlSubmissionTest, OptionalProfileSharesInlineAndPooledCapture) {
  const iree_hal_amd_gpu_aql_dispatch_request_t barrier = {};
  iree_alignas(8) uint8_t inline_storage[1024];
  iree_host_size_t unprofiled_size = 0;
  for (bool profiled : {false, true}) {
    iree_hal_amd_gpu_profile_operation_t operation = {};
    operation.event.submission_id = 42;
    operation.event.type = IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER;
    iree_hal_amd_gpu_aql_submission_t* capture = nullptr;
    if (GetParam()) {
      IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_retirement(
          &pool_, iree_make_byte_span(inline_storage, sizeof(inline_storage)),
          profiled ? &operation : nullptr, &barrier, 0, nullptr, {}, {},
          &capture));
      EXPECT_EQ(capture->pool, nullptr);
    } else {
      IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_capture(
          &pool_, profiled ? &operation : nullptr, &barrier, {}, {}, 0, 0,
          &capture));
      EXPECT_EQ(capture->pool, &pool_);
    }
    operation.event.submission_id = 0;
    auto* captured = iree_hal_amd_gpu_aql_submission_profile(capture);
    if (profiled) {
      ASSERT_NE(captured, nullptr);
      EXPECT_EQ(captured->event.submission_id, 42u);
      EXPECT_EQ(captured->event.type,
                IREE_HAL_PROFILE_QUEUE_EVENT_TYPE_BARRIER);
      EXPECT_EQ(capture->capture_size, unprofiled_size + sizeof(operation));
      EXPECT_EQ(static_cast<void*>(capture->waits),
                static_cast<void*>(captured + 1));
    } else {
      EXPECT_EQ(captured, nullptr);
      unprofiled_size = capture->capture_size;
      EXPECT_EQ(unprofiled_size, sizeof(*capture));
    }
    iree_hal_amd_gpu_aql_submission_release(capture);
  }
}

INSTANTIATE_TEST_SUITE_P(Storage, AqlSubmissionTest,
                         ::testing::Values(false, true));

TEST(AqlTransferSubmissionTest, RejectedSuffixWaitsForAcceptedHostWrites) {
  struct Buffer {
    // Real HAL resource retained by a captured transfer.
    iree_hal_buffer_t base;
    // Destruction observation independent of capture storage.
    bool destroyed = false;
    static void Destroy(iree_hal_buffer_t* base) {
      reinterpret_cast<Buffer*>(base)->destroyed = true;
    }
  };
  iree_hal_amd_gpu_capture_pool_t pool;
  iree_hal_amd_gpu_capture_pool_initialize(512, iree_allocator_system(), &pool);
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, nullptr, 0, 2, iree_allocator_system(), &semaphore));
  const iree_hal_buffer_vtable_t vtable = {.destroy = Buffer::Destroy};
  Buffer buffer = {};
  iree_hal_buffer_initialize(
      {}, &buffer.base, 8, 0, 8, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable,
      &buffer.base);
  // The native binding belongs to the dependency, not the capture under test.
  // Address commitment is deliberately deferred until native publication.
  const iree_hal_queue_family_t family = {.memory = {.queue_scope_id = 1}};
  std::array<iree_hal_memory_scope_access_t, 3> scopes = {};
  scopes[2].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  iree_hal_memory_contract_t contract = {.scopes = scopes.data()};
  buffer.base.memory.contract = &contract;
  std::array<uint8_t, 8> target = {};
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = &buffer.base,
              .target = target.data(),
              .length = target.size(),
          },
  };
  uint64_t value = 1;
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_transfer(
      &pool, nullptr, &family, {}, {1, &semaphore, &value}, 1, &operation, 0, 0,
      &submission));
  struct Completion {
    // File callbacks distinguish failure notification from ending borrows.
    bool retired = false;
    // Borrowed dependency must be destroyed before handing ownership back.
    Buffer* buffer;
  } completion = {.buffer = &buffer};
  submission->transfer.callback = {
      [](void* user_data, iree_status_t status,
         const iree_async_single_frontier_t* frontier, bool retired) {
        auto* completion = static_cast<Completion*>(user_data);
        EXPECT_TRUE(retired);
        EXPECT_TRUE(completion->buffer->destroyed);
        EXPECT_EQ(frontier, nullptr);
        EXPECT_EQ(iree_status_consume_code(status), IREE_STATUS_UNAVAILABLE);
        completion->retired = true;
      },
      &completion,
  };
  iree_hal_buffer_release(&buffer.base);
  // One accepted native piece retains the capture while a later, unpublished
  // suffix still owns its initial reference.
  iree_atomic_ref_count_inc(&submission->transfer.ref_count);
  iree_hal_amd_gpu_aql_submission_reject(submission, IREE_STATUS_UNAVAILABLE);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_FALSE(buffer.destroyed);
  EXPECT_FALSE(completion.retired);
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &observed));
  EXPECT_EQ(observed, 0u);
  EXPECT_EQ(submission->transfer.capture.operations[0].target.host,
            target.data());
  // The capture may report failure only after its last accepted piece has
  // stopped using the borrowed host destination.
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_TRUE(buffer.destroyed);
  EXPECT_TRUE(completion.retired);
  EXPECT_THAT(iree_hal_semaphore_query(semaphore, &observed),
              iree::testing::status::StatusIs(iree::StatusCode::kUnavailable));
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
  iree_hal_amd_gpu_capture_pool_deinitialize(&pool);
}

TEST_P(AqlSubmissionTest, FileCopiesJoinFailureAndRetirement) {
  struct Completion {
    // Early notification count, independent of ownership handoff.
    unsigned failures = 0;
    // Final ownership handoff count.
    unsigned retirements = 0;
    // Exact last accepted queue epoch returned with retirement.
    iree_async_single_frontier_t frontier = {};
  } completion;
  const iree_hal_amd_gpu_retirement_callback_t callback = {
      [](void* user_data, iree_status_t status,
         const iree_async_single_frontier_t* frontier, bool retired) {
        auto* completion = static_cast<Completion*>(user_data);
        EXPECT_EQ(iree_status_consume_code(status), IREE_STATUS_UNAVAILABLE);
        if (retired) {
          ++completion->retirements;
          ASSERT_NE(frontier, nullptr);
          completion->frontier = *frontier;
        } else {
          ++completion->failures;
          EXPECT_EQ(frontier, nullptr);
        }
      },
      &completion,
  };
  const iree_hal_amd_gpu_transfer_operation_t copy = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_COPY,
      .source =
          {
              .device = iree_hal_amd_gpu_buffer_reference_borrow(0x123456789),
          },
      .target =
          {
              .device = iree_hal_amd_gpu_buffer_reference_borrow(0x987654321),
          },
      .length = 2 * 1024 * 1024,
  };
  iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_submission_file(
      &pool_, GetParam() ? &copy : nullptr, 17, 23, callback, &submission));
  EXPECT_EQ(submission->entry.kind, IREE_HAL_AMD_GPU_AQL_ENTRY_TRANSFER);
  EXPECT_EQ(submission->transfer.capture.count, GetParam() ? 1u : 0u);
  EXPECT_EQ(submission->before, 17u);
  EXPECT_EQ(submission->after, 23u);
  if (GetParam()) {
    std::array<uint64_t, 4> arguments;
    iree_hal_amd_gpu_aql_transfer_chunk_t chunk;
    IREE_ASSERT_OK(iree_hal_amd_gpu_aql_transfer_prepare_chunk(
        &submission->transfer.capture, 0x2000, 1,
        iree_make_byte_span(arguments.data(), sizeof(arguments)), 0x3000,
        iree_byte_span_empty(), 0, &chunk));
    EXPECT_EQ(arguments, (std::array<uint64_t, 4>{0x123456789, 0x987654321,
                                                  2 * 1024 * 1024, 0}));
    EXPECT_EQ(chunk.packet_count, 1u);
    EXPECT_EQ(chunk.argument_length, sizeof(arguments));
    EXPECT_EQ(chunk.next.index, 1u);
    EXPECT_EQ(submission->transfer.capture.staging_capacity, 0u);
  }
  iree_atomic_ref_count_inc(&submission->transfer.ref_count);
  iree_hal_amd_gpu_aql_submission_fail(
      submission, iree_status_from_code(IREE_STATUS_UNAVAILABLE));
  iree_hal_amd_gpu_aql_submission_fail(
      submission, iree_status_from_code(IREE_STATUS_ABORTED));
  EXPECT_EQ(completion.failures, 1u);
  EXPECT_EQ(completion.retirements, 0u);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_EQ(completion.retirements, 0u);
  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, 3, 41);
  iree_hal_amd_gpu_aql_submission_set_frontier(submission, &frontier);
  iree_hal_amd_gpu_aql_submission_release(submission);
  EXPECT_EQ(completion.failures, 1u);
  EXPECT_EQ(completion.retirements, 1u);
  EXPECT_EQ(completion.frontier.entry_count, frontier.entry_count);
  EXPECT_EQ(completion.frontier.entries[0].axis, frontier.entries[0].axis);
  EXPECT_EQ(completion.frontier.entries[0].epoch, frontier.entries[0].epoch);
}

}  // namespace
