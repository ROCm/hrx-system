// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amd/gpu/queue/pm4/submission.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <thread>
#include <vector>

#include "iree/async/proactor_platform.h"
#include "iree/hal/drivers/amd/gpu/queue/pm4/dispatch.h"
#include "iree/hal/drivers/amd/gpu/queue/signal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SubmissionTest : public ::testing::TestWithParam<bool> {
 protected:
  void SetUp() override {
    function_.metadata = &metadata_;
    function_.kernarg = &layout_;
    iree_hal_amd_gpu_capture_pool_initialize(1024, iree_allocator_system(),
                                             &pool_);
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_pm4_barrier_profile_initialize({11, 0, 0}, &profile_));
  }
  void TearDown() override {
    iree_hal_amd_gpu_capture_pool_deinitialize(&pool_);
  }
  // Executable-owned image metadata borrowed by captured work.
  iree_hal_amd_gpu_metadata_kernel_t metadata_ = {};
  // Argument shape used to size captured constants and bindings.
  iree_hal_amd_gpu_kernarg_layout_t layout_ = {.alignment = 8};
  // Prepared function borrowed from the retained executable.
  iree_hal_amd_gpu_pm4_function_t function_ = {};
  // Real reusable capture storage.
  iree_hal_amd_gpu_capture_pool_t pool_;
  // Physical cache policy used to capture public barriers.
  iree_hal_amd_gpu_pm4_barrier_profile_t profile_;
};

// The executable is an ownership dependency; all capture/refcount logic is
// production code. No native driver is needed to observe its final release.
struct Executable {
  // Refcounted public object borrowed by the capture.
  iree_hal_executable_t base;
  // Final destructor observation, joined before this stack object disappears.
  bool destroyed = false;
  static void Destroy(iree_hal_executable_t* base) {
    reinterpret_cast<Executable*>(base)->destroyed = true;
  }
};

TEST_P(SubmissionTest, PayloadRetiresBeforePublicCompletion) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, 0, 4, iree_allocator_system(), &semaphore));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  std::array<uint8_t, 16> storage = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, storage.size(),
      iree_make_byte_span(storage.data(), storage.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  layout_.byte_length = 16;
  layout_.native_byte_length = 12;
  layout_.constant_byte_length = 4;
  layout_.binding_count = 1;
  uint64_t wait_value = 3, signal_value = 4;
  auto reference = iree_hal_make_buffer_ref(buffer, 4, 8);
  uint32_t constant = 0x12345678;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, GetParam(), {1, &semaphore, &wait_value},
      {1, &semaphore, &signal_value}, &executable.base, &function_, {}, {},
      iree_make_const_byte_span(&constant, sizeof(constant)), {1, &reference},
      &profile_, nullptr, 0, &capture));
  uint64_t native_value = 0;
  uint64_t native_frontier = 0;
  iree_hal_amd_gpu_signal_t native_signal = {};
  iree_hal_amd_gpu_signal_pool_t signal_pool;
  iree_hal_amd_gpu_signal_block_t signal_block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &signal_pool);
  iree_hal_amd_gpu_signal_block_initialize(&signal_pool, this, &native_value,
                                           0x1000, 8, 0, &native_signal, 1, 1,
                                           &native_frontier, &signal_block);
  capture->signal = iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool, 0);
  iree_hal_amd_gpu_pm4_submission_t* consumer = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, GetParam(), {1, &semaphore, &signal_value}, {}, nullptr, nullptr,
      {}, {}, {}, {}, &profile_, nullptr, 0, &consumer));
  // Native admission retains the producer's independent signal owner. GPU
  // completion alone must not let this consumer publish host completion.
  consumer->native_wait_count = 1;
  consumer->waits[0].signal = capture->signal;
  iree_hal_amd_gpu_signal_retain(consumer->waits[0].signal);
  native_value = 1;
  iree_status_code_t completion_code = IREE_STATUS_OK;
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      consumer, &completion_code));
  ASSERT_EQ(capture->allocation_size, pool_.block_size);
  const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
  ASSERT_GT(tail.data_length, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(tail.data) % iree_max_align_t, 0u);
  std::memset(tail.data, 0xA5, tail.data_length);
  constant = 0;
  reference = {};
  wait_value = signal_value = 99;
  EXPECT_EQ(capture->waits[0].value, 3u);
  EXPECT_EQ(capture->signals.payload_values[0], 4u);
  EXPECT_EQ(capture->dispatch.bindings.values[0].offset, 4u);
  EXPECT_EQ(capture->dispatch.bindings.values[0].length, 8u);
  uint32_t copied = 0;
  std::memcpy(&copied, capture->dispatch.constants.data, sizeof(copied));
  EXPECT_EQ(copied, 0x12345678u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(capture->signals.payload_values) %
                alignof(uint64_t),
            0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(capture->dispatch.bindings.values) %
                alignof(iree_hal_buffer_ref_t),
            0u);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  iree_hal_semaphore_release(semaphore);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_FALSE(executable.destroyed);
  EXPECT_FALSE(buffer_destroyed);
  // Native completion must release payloads before exposing its public
  // signal. The remaining reference now belongs exclusively to retirement.
  iree_hal_amd_gpu_pm4_submission_release_payload(capture);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      consumer, &completion_code));
  uint64_t completed_value = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(capture->signals.semaphores[0],
                                          &completed_value));
  EXPECT_EQ(completed_value, 0u);
  EXPECT_EQ(capture->signals.payload_values[0], 4u);
  EXPECT_EQ(capture->waits[0].value, 3u);
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(capture->signals.semaphores[0], 4, nullptr));
  iree_hal_amd_gpu_signal_retire(capture->signal, IREE_STATUS_OK);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  // Producer capture destruction does not discard its retirement fact while
  // the consumer still owns that dependency.
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      consumer, &completion_code));
  EXPECT_EQ(completion_code, IREE_STATUS_OK);
  iree_hal_amd_gpu_pm4_submission_release(consumer);
  auto* recycled = iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool, 0);
  ASSERT_NE(recycled, nullptr);
  EXPECT_FALSE(
      iree_hal_amd_gpu_signal_query_retirement(recycled, &completion_code));
  iree_hal_amd_gpu_signal_release(recycled);
  iree_hal_amd_gpu_signal_block_deinitialize(&signal_block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&signal_pool);
  iree_async_proactor_release(proactor);
}

TEST_P(SubmissionTest, RetirementOwnsNativeReadersWithoutCallerInputs) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
      proactor, 0, 4, iree_allocator_system(), &semaphore));
  uint64_t native_value = 0, native_frontier = 1;
  iree_hal_amd_gpu_signal_t native_signal = {};
  iree_hal_amd_gpu_signal_pool_t signal_pool;
  iree_hal_amd_gpu_signal_block_t signal_block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &signal_pool);
  iree_hal_amd_gpu_signal_block_initialize(&signal_pool, this, &native_value,
                                           0x1000, 8, 0, &native_signal, 1, 1,
                                           &native_frontier, &signal_block);
  auto* producer = iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool, 0);
  iree_hal_amd_gpu_signal_publish_frontier(producer, &native_frontier);
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_publish_signal(semaphore, 1, producer));
  iree_hal_amd_gpu_semaphore_wait_t acquired;
  IREE_ASSERT_OK(
      iree_hal_amd_gpu_semaphore_acquire_wait(semaphore, 1, this, &acquired));
  ASSERT_EQ(acquired.state, IREE_HAL_AMD_GPU_SEMAPHORE_WAIT_NATIVE);
  const iree_hal_amd_gpu_pm4_submission_wait_t waits[] = {
      {.semaphore = semaphore, .value = 0},
      {.semaphore = semaphore, .value = 1, .signal = acquired.signal},
  };
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  uint32_t constant = 42;
  layout_.constant_byte_length = sizeof(constant);
  layout_.byte_length = sizeof(constant);
  const iree_hal_amd_gpu_pm4_dispatch_request_t request = {
      .executable = &executable.base,
      .function = &function_,
      .constants = iree_make_const_byte_span(&constant, sizeof(constant)),
  };
  iree_hal_amd_gpu_profile_operation_t profile = {};
  iree_hal_amd_gpu_pm4_submission_t* retirement = nullptr;
  alignas(iree_hal_amd_gpu_pm4_submission_t) std::array<uint8_t, 512> storage;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_retirement(
      &pool_, iree_make_byte_span(storage.data(), storage.size()), &request,
      IREE_ARRAYSIZE(waits), waits, 1, {}, {}, {},
      GetParam() ? &profile : nullptr, &retirement));
  EXPECT_EQ(static_cast<void*>(retirement), storage.data());
  EXPECT_EQ(retirement->pool, nullptr);
  // Admission has consumed the semaphore objects and argument bytes. Only the
  // exact producer point and asynchronously used resources remain owned.
  EXPECT_EQ(retirement->wait_count, 1u);
  EXPECT_EQ(retirement->waits[0].semaphore, nullptr);
  EXPECT_EQ(retirement->waits[0].signal, acquired.signal);
  EXPECT_EQ(retirement->dispatch.constants.data, nullptr);
  EXPECT_EQ(retirement->dispatch.constants.data_length, 0u);
  iree_hal_semaphore_release(semaphore);
  iree_hal_executable_release(&executable.base);
  EXPECT_FALSE(executable.destroyed);
  iree_status_code_t completion_code = IREE_STATUS_OK;
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      retirement, &completion_code));
  iree_hal_amd_gpu_signal_retire(producer, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_signal_release(producer);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      retirement, &completion_code));
  EXPECT_EQ(completion_code, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_pm4_submission_release(retirement);
  EXPECT_TRUE(executable.destroyed);
  iree_hal_amd_gpu_signal_block_deinitialize(&signal_block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&signal_pool);
  iree_async_proactor_release(proactor);
}

TEST_P(SubmissionTest, InlineRetirementStorageReusesAfterLastOwner) {
  alignas(iree_hal_amd_gpu_pm4_submission_t) std::array<uint8_t, 512> storage;
  for (int iteration = 0; iteration < 2; ++iteration) {
    const iree_hal_queue_family_t family = {};
    const iree_hal_executable_vtable_t vtable = {
        .destroy = Executable::Destroy,
    };
    Executable executable;
    iree_hal_executable_initialize(&family, &vtable, &executable.base);
    const iree_hal_amd_gpu_pm4_dispatch_request_t request = {
        .executable = &executable.base,
        .function = &function_,
    };
    iree_hal_amd_gpu_profile_operation_t profile = {};
    iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_retirement(
        &pool_, iree_make_byte_span(storage.data(), storage.size()), &request,
        0, nullptr, 0, {}, {}, {}, GetParam() ? &profile : nullptr,
        &submission));
    ASSERT_EQ(static_cast<void*>(submission), storage.data());
    EXPECT_EQ(submission->pool, nullptr);
    EXPECT_EQ(submission->allocation_size, storage.size());
    iree_hal_executable_release(&executable.base);
    iree_hal_amd_gpu_pm4_submission_retain(submission);
    iree_hal_amd_gpu_pm4_submission_release(submission);
    EXPECT_FALSE(executable.destroyed);
    // Returning the final capture reference releases resources, after which
    // the owner can reuse exactly the same bytes for the next native slot.
    iree_hal_amd_gpu_pm4_submission_release(submission);
    EXPECT_TRUE(executable.destroyed);
  }
}

TEST_P(SubmissionTest, RetirementSpillsWhenInlineExtentIsTooSmall) {
  alignas(iree_hal_amd_gpu_pm4_submission_t) std::array<uint8_t, 448> storage;
  storage.fill(0xA5);
  const auto original = storage;
  std::array<uint8_t, 16> buffer_storage = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE, buffer_storage.size(),
      iree_make_byte_span(buffer_storage.data(), buffer_storage.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  std::array<iree_hal_buffer_ref_t, 16> references;
  references.fill(iree_hal_make_buffer_ref(buffer, 0, buffer_storage.size()));
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_hal_amd_gpu_pm4_dispatch_request_t request = {
      .executable = &executable.base,
      .function = &function_,
      .bindings = {references.size(), references.data()},
  };
  iree_hal_amd_gpu_profile_operation_t profile = {};
  iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_retirement(
      &pool_, iree_make_byte_span(storage.data(), storage.size()), &request, 0,
      nullptr, 0, {}, {}, {}, GetParam() ? &profile : nullptr, &submission));
  EXPECT_NE(static_cast<void*>(submission), storage.data());
  EXPECT_EQ(submission->pool, &pool_);
  EXPECT_GT(submission->capture_size, storage.size());
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  EXPECT_FALSE(executable.destroyed);
  EXPECT_FALSE(buffer_destroyed);
  iree_hal_amd_gpu_pm4_submission_release(submission);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
  EXPECT_EQ(storage, original);
}

TEST_P(SubmissionTest, CallerManagedResourcesStayBorrowedThroughRetirement) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  std::array<uint8_t, 16> storage = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_STORAGE |
          IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS,
      storage.size(), iree_make_byte_span(storage.data(), storage.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  const auto reference = iree_hal_make_buffer_ref(buffer, 0, storage.size());
  const auto* executable_resource =
      reinterpret_cast<const iree_hal_resource_t*>(&executable.base);
  const auto* buffer_resource =
      reinterpret_cast<const iree_hal_resource_t*>(buffer);
  // Deferred captures still copy descriptors, but neither a capture nor an
  // inline retirement owner needs resource reference traffic with this flag.
  const iree_hal_dispatch_flags_t parameter_modes[] = {
      0, IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
      IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS};
  for (bool deferred : {false, true}) {
    for (iree_hal_dispatch_flags_t parameters : parameter_modes) {
      const iree_hal_dispatch_flags_t flags =
          IREE_HAL_DISPATCH_FLAG_BORROW_RESOURCE_LIFETIMES | parameters;
      iree_hal_amd_gpu_pm4_submission_t* submission = nullptr;
      alignas(iree_hal_amd_gpu_pm4_submission_t) std::array<uint8_t, 512>
          retirement_storage;
      if (deferred) {
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
            &pool_, GetParam(), {}, {}, &executable.base, &function_, {},
            reference, {}, {1, &reference}, &profile_, nullptr, flags,
            &submission));
      } else {
        const iree_hal_amd_gpu_pm4_dispatch_request_t request = {
            .executable = &executable.base,
            .function = &function_,
            .parameters = reference,
            .bindings = {1, &reference},
            .flags = flags,
        };
        iree_hal_amd_gpu_profile_operation_t profile = {};
        IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_retirement(
            &pool_,
            iree_make_byte_span(retirement_storage.data(),
                                retirement_storage.size()),
            &request, 0, nullptr, 0, {}, {}, {},
            GetParam() ? &profile : nullptr, &submission));
      }
      EXPECT_EQ(iree_atomic_ref_count_load(&executable_resource->ref_count), 1);
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer_resource->ref_count), 1);
      iree_hal_amd_gpu_pm4_submission_release_payload(submission);
      iree_hal_amd_gpu_pm4_submission_release(submission);
      EXPECT_FALSE(executable.destroyed);
      EXPECT_FALSE(buffer_destroyed);
      EXPECT_EQ(iree_atomic_ref_count_load(&executable_resource->ref_count), 1);
      EXPECT_EQ(iree_atomic_ref_count_load(&buffer_resource->ref_count), 1);
    }
  }
  iree_hal_executable_release(&executable.base);
  iree_hal_buffer_release(buffer);
  EXPECT_TRUE(executable.destroyed);
  EXPECT_TRUE(buffer_destroyed);
}

TEST_P(SubmissionTest, FailedDependencyStillJoinsOtherProducers) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  std::array<iree_hal_semaphore_t*, 2> semaphores = {};
  for (auto& semaphore : semaphores) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_semaphore_create(
        proactor, 0, 4, iree_allocator_system(), &semaphore));
  }
  std::array<uint64_t, 2> values = {1, 1};
  std::array<uint64_t, 2> native_values = {};
  std::array<uint64_t, 2> native_frontiers = {};
  std::array<iree_hal_amd_gpu_signal_t, 2> native_signals = {};
  iree_hal_amd_gpu_signal_pool_t signal_pool;
  iree_hal_amd_gpu_signal_block_t signal_block;
  iree_hal_amd_gpu_signal_pool_initialize({}, &signal_pool);
  iree_hal_amd_gpu_signal_block_initialize(
      &signal_pool, this, native_values.data(), 0x1000, 8, 0,
      native_signals.data(), native_signals.size(), 1, native_frontiers.data(),
      &signal_block);
  std::array<iree_hal_amd_gpu_pm4_submission_t*, 2> producers = {};
  for (size_t i = 0; i < producers.size(); ++i) {
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &pool_, GetParam(), {}, {1, &semaphores[i], &values[i]}, nullptr,
        nullptr, {}, {}, {}, {}, &profile_, nullptr, 0, &producers[i]));
    producers[i]->signal =
        iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool, 0);
  }
  iree_hal_amd_gpu_pm4_submission_t* consumer = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, GetParam(), {2, semaphores.data(), values.data()}, {}, nullptr,
      nullptr, {}, {}, {}, {}, &profile_, nullptr, 0, &consumer));
  consumer->native_wait_count = producers.size();
  for (size_t i = 0; i < producers.size(); ++i) {
    consumer->waits[i].signal = producers[i]->signal;
    iree_hal_amd_gpu_signal_retain(consumer->waits[i].signal);
    iree_hal_semaphore_release(semaphores[i]);
  }

  // A failed producer's capture may disappear while its native readers still
  // need the outcome. Destruction must neither discard nor clear that failure.
  iree_hal_amd_gpu_signal_retire(producers[0]->signal, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_pm4_submission_release(producers[0]);
  iree_status_code_t completion_code = IREE_STATUS_OK;
  EXPECT_FALSE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      consumer, &completion_code));
  iree_hal_amd_gpu_signal_retire(producers[1]->signal, IREE_STATUS_OK);
  iree_hal_amd_gpu_pm4_submission_release(producers[1]);
  EXPECT_TRUE(iree_hal_amd_gpu_pm4_submission_retirement_ready(
      consumer, &completion_code));
  EXPECT_EQ(completion_code, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_pm4_submission_release(consumer);

  // Both slots can be reused, without inheriting either terminal outcome.
  std::array<iree_hal_amd_gpu_signal_t*, 2> recycled;
  for (auto& signal : recycled) {
    signal = iree_hal_amd_gpu_signal_pool_try_acquire(&signal_pool, 0);
    ASSERT_NE(signal, nullptr);
    EXPECT_FALSE(
        iree_hal_amd_gpu_signal_query_retirement(signal, &completion_code));
  }
  for (auto* signal : recycled) {
    iree_hal_amd_gpu_signal_release(signal);
  }
  iree_hal_amd_gpu_signal_block_deinitialize(&signal_block);
  iree_hal_amd_gpu_signal_pool_deinitialize(&signal_pool);
  iree_async_proactor_release(proactor);
}

TEST_P(SubmissionTest, IndirectParametersRetireWithoutAddingShaderBindings) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  for (auto flags : {IREE_HAL_DISPATCH_FLAG_STATIC_INDIRECT_PARAMETERS,
                     IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS}) {
    std::array<uint32_t, 4> storage = {0, 3, 2, 1};
    bool destroyed = false;
    iree_hal_buffer_t* parameters = nullptr;
    IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
        {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_READ,
        IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS, sizeof(storage),
        iree_make_byte_span(storage.data(), sizeof(storage)),
        {[](void* data, iree_hal_buffer_t*) {
           *static_cast<bool*>(data) = true;
         },
         &destroyed},
        iree_allocator_system(), &parameters));
    uint32_t constant = 0x12345678;
    auto reference = iree_hal_make_buffer_ref(parameters, 4, 12);
    iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &pool_, GetParam(), {}, {}, &executable.base, &function_, {}, reference,
        iree_make_const_byte_span(&constant, sizeof(constant)), {}, &profile_,
        nullptr, flags, &capture));
    iree_hal_buffer_release(parameters);
    EXPECT_FALSE(destroyed);
    reference = {};
    constant = 0;
    EXPECT_EQ(capture->dispatch.bindings.count, 0u);
    EXPECT_EQ(capture->dispatch.bindings.values[0].buffer, parameters);
    EXPECT_EQ(capture->dispatch.bindings.values[0].offset, 4u);
    EXPECT_EQ(capture->dispatch.bindings.values[0].length, 12u);
    const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
    std::memset(tail.data, 0xA5, tail.data_length);
    uint32_t captured_constant = 0;
    std::memcpy(&captured_constant, capture->dispatch.constants.data, 4);
    EXPECT_EQ(captured_constant, 0x12345678u);
    iree_hal_amd_gpu_pm4_submission_release_payload(capture);
    EXPECT_TRUE(destroyed);
    iree_hal_amd_gpu_pm4_submission_release(capture);
  }
  iree_hal_executable_release(&executable.base);
  EXPECT_TRUE(executable.destroyed);
}

TEST_P(SubmissionTest, OversizedCaptureUsesCompleteOwnedStorage) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  layout_.byte_length = layout_.native_byte_length = 4096;
  std::vector<uint8_t> constants(4096, 0x35);
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  const iree_hal_barrier_list_t empty = {};
  const iree_hal_queue_barriers_t barriers = {&empty, &empty};
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
      &pool_, GetParam(), {}, {}, &executable.base, &function_, {}, {},
      iree_make_const_byte_span(constants.data(), constants.size()), {},
      &profile_, &barriers, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
      &capture));
  EXPECT_EQ(
      capture->allocation_size,
      sizeof(*capture) +
          (GetParam() ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0) +
          constants.size());
  EXPECT_EQ(iree_hal_amd_gpu_pm4_submission_tail_storage(capture).data_length,
            0u);
  EXPECT_EQ(capture->before.cache_control, 0u);
  EXPECT_EQ(capture->after.cache_control, 0u);
  constants.assign(constants.size(), 0xAB);
  for (iree_host_size_t i = 0; i < capture->dispatch.constants.data_length;
       ++i) {
    ASSERT_EQ(capture->dispatch.constants.data[i], 0x35);
  }
  iree_hal_executable_release(&executable.base);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_TRUE(executable.destroyed);
}

TEST_P(SubmissionTest, TailStoragePreservesArgumentsAtBlockBoundaries) {
  const iree_hal_queue_family_t family = {};
  const iree_hal_executable_vtable_t vtable = {.destroy = Executable::Destroy};
  Executable executable;
  iree_hal_executable_initialize(&family, &vtable, &executable.base);
  const iree_host_size_t capacity =
      pool_.block_size - sizeof(iree_hal_amd_gpu_pm4_submission_t) -
      (GetParam() ? sizeof(iree_hal_amd_gpu_profile_operation_t) : 0);
  const std::array<iree_host_size_t, 8> sizes = {
      0, 1, 15, 16, 17, capacity - 1, capacity, capacity + 1};
  for (iree_host_size_t size : sizes) {
    layout_.byte_length = layout_.native_byte_length = size;
    std::vector<uint8_t> constants(size, 0x35);
    iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
    IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create(
        &pool_, GetParam(), {}, {}, &executable.base, &function_, {}, {},
        iree_make_const_byte_span(constants.data(), constants.size()), {},
        &profile_, nullptr, IREE_HAL_DISPATCH_FLAG_CUSTOM_DIRECT_ARGUMENTS,
        &capture));
    const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
    if (size >= capacity - 1) {
      EXPECT_EQ(tail.data_length, 0u);
    } else {
      ASSERT_GT(tail.data_length, 0u);
      EXPECT_EQ(reinterpret_cast<uintptr_t>(tail.data) % iree_max_align_t, 0u);
      std::memset(tail.data, 0xA5, tail.data_length);
    }
    for (iree_host_size_t i = 0; i < size; ++i) {
      EXPECT_EQ(capture->dispatch.constants.data[i], 0x35);
    }
    EXPECT_EQ(
        capture->allocation_size,
        size <= capacity
            ? pool_.block_size
            : sizeof(*capture) +
                  (GetParam() ? sizeof(iree_hal_amd_gpu_profile_operation_t)
                              : 0) +
                  size);
    iree_hal_amd_gpu_pm4_submission_release(capture);
  }
  iree_hal_executable_release(&executable.base);
  EXPECT_TRUE(executable.destroyed);
}

TEST_P(SubmissionTest, CaptureOverflowRetainsNoInputs) {
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_amd_gpu_pm4_submission_create(
          &pool_, GetParam(), {IREE_HOST_SIZE_MAX, nullptr, nullptr}, {},
          nullptr, nullptr, {}, {}, {}, {}, &profile_, nullptr, 0, &capture));
  EXPECT_EQ(capture, nullptr);
}

TEST_P(SubmissionTest, PreparedAddressChecksDomainUsageAndCommitment) {
  const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  const iree_hal_buffer_binding_layout_t binding_layout = {
      sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_contract_create(
      this, 2, &binding_layout, iree_allocator_system(), &contract));
  contract->scopes[1].interfaces = 1u
                                   << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  contract->scopes[1].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  contract->scopes[1].usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  iree_hal_queue_family_t family = {
      .memory = {.domain = this, .queue_scope_id = 0},
  };
  iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  iree_hal_buffer_t buffer = {};
  const iree_hal_buffer_vtable_t vtable = {
      .query_memory =
          [](const iree_hal_buffer_t* buffer) {
            ADD_FAILURE()
                << "GPU address resolution must not query cold backing";
            return buffer->memory;
          },
  };
  iree_hal_buffer_initialize({}, &buffer, 64, 0, 64,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                             IREE_HAL_MEMORY_ACCESS_READ,
                             IREE_HAL_BUFFER_USAGE_STORAGE, &vtable, &buffer);
  buffer.memory = {
      .contract = contract,
      .bindings = &binding,
      .binding_offset = 8,
  };
  auto reference = iree_hal_make_buffer_ref(&buffer, 4, 16);
  uint64_t address = 0;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_resolve_address(
      &family, IREE_HAL_MEMORY_SITE_PROGRAM, reference,
      IREE_HAL_BUFFER_USAGE_STORAGE, IREE_HAL_MEMORY_ACCESS_READ, &address));
  EXPECT_EQ(address, 0x100Cu);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, IREE_HAL_MEMORY_SITE_PROGRAM, reference,
                            IREE_HAL_BUFFER_USAGE_STORAGE,
                            IREE_HAL_MEMORY_ACCESS_WRITE, &address));
  buffer.allowed_usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, IREE_HAL_MEMORY_SITE_PROGRAM, reference,
                            IREE_HAL_BUFFER_USAGE_STORAGE, 0, &address));
  buffer.allowed_usage = IREE_HAL_BUFFER_USAGE_STORAGE;
  family.memory.domain = &family;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, IREE_HAL_MEMORY_SITE_PROGRAM, reference,
                            IREE_HAL_BUFFER_USAGE_STORAGE, 0, &address));
  family.memory.domain = this;
  binding.device_address = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION,
                        iree_hal_amd_gpu_pm4_submission_resolve_address(
                            &family, IREE_HAL_MEMORY_SITE_PROGRAM, reference,
                            IREE_HAL_BUFFER_USAGE_STORAGE, 0, &address));
  binding.device_address = 0x1000;
  reference.offset = 60;
  layout_.binding_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_amd_gpu_pm4_dispatch_validate_direct_bindings(
                            &layout_, {}, {1, &reference}, 0));
  iree_hal_memory_contract_release(contract);
}

// Records native notifications independently of the capture allocation.
struct FileRetirement {
  // Number of early logical failure reports.
  uint32_t failures = 0;
  // Number of final ownership handoffs.
  uint32_t retirements = 0;
  // First failure, preserved through the final callback.
  iree_status_code_t status_code = IREE_STATUS_OK;
  // Completed epoch copied before the callback's frontier borrow ends.
  iree_async_single_frontier_t frontier = {};
  // Optional capture owner destroyed immediately on final callback.
  iree_hal_amd_gpu_capture_pool_t* pool = nullptr;

  iree_hal_amd_gpu_retirement_callback_t callback() {
    return {[](void* user_data, iree_status_t status,
               const iree_async_single_frontier_t* frontier, bool retired) {
              auto* self = static_cast<FileRetirement*>(user_data);
              if (retired) {
                ++self->retirements;
                if (self->failures) {
                  EXPECT_EQ(iree_status_code(status), self->status_code);
                }
                if (frontier) {
                  self->frontier = *frontier;
                }
                if (self->pool) {
                  iree_hal_amd_gpu_capture_pool_deinitialize(self->pool);
                }
              } else {
                IREE_EXPECT_NOT_OK(iree_status_clone(status));
                EXPECT_EQ(frontier, nullptr);
                ++self->failures;
              }
              self->status_code = iree_status_code(status);
              iree_status_free(status);
            },
            this};
  }
};

TEST_P(SubmissionTest, FileRetirementTransfersTheFinalPublicationReference) {
  FileRetirement owner;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  const iree_hal_amd_gpu_pm4_barrier_t before = {.cache_control = 17};
  const iree_hal_amd_gpu_pm4_barrier_t after = {.cache_control = 29};
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_file(
      &pool_, nullptr, nullptr, &before, &after, owner.callback(), &capture));
  EXPECT_EQ(capture->entry.kind, IREE_HAL_AMD_GPU_PM4_ENTRY_BARRIER);
  EXPECT_EQ(capture->signals.count, 0u);
  EXPECT_EQ(capture->signal, nullptr);
  EXPECT_EQ(capture->before.cache_control, 17u);
  EXPECT_EQ(capture->after.cache_control, 29u);
  const auto tail = iree_hal_amd_gpu_pm4_submission_tail_storage(capture);
  std::memset(tail.data, 0xA5, tail.data_length);
  // Each nonterminal piece has a reference in addition to the unpublished
  // suffix. Retirement is ordered, so these pieces finish before the final
  // piece releases the payload and completes the file operation.
  if (GetParam()) {
    iree_hal_amd_gpu_pm4_submission_retain(capture);
    iree_hal_amd_gpu_pm4_submission_release(capture);
  }
  EXPECT_EQ(owner.retirements, 0u);
  // Final publication transfers the original reference to retirement without
  // another retain/release pair. Terminal retirement owns the capture alone.
  iree_async_single_frontier_t frontier;
  iree_async_single_frontier_initialize(&frontier, 7, 42);
  iree_hal_amd_gpu_pm4_submission_release_payload(capture);
  iree_hal_amd_gpu_pm4_submission_set_frontier(capture, &frontier);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_EQ(owner.failures, 0u);
  EXPECT_EQ(owner.retirements, 1u);
  EXPECT_EQ(owner.status_code, IREE_STATUS_OK);
  EXPECT_EQ(owner.frontier.entry_count, 1u);
  EXPECT_EQ(owner.frontier.entries[0].axis, 7u);
  EXPECT_EQ(owner.frontier.entries[0].epoch, 42u);
}

TEST_P(SubmissionTest, FileFailureDoesNotRetireAcceptedNativePieces) {
  FileRetirement owner;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_file(
      &pool_, nullptr, nullptr, nullptr, nullptr, owner.callback(), &capture));
  // Two accepted pieces hold references independently of the publishing owner.
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_fail(
      capture, iree_status_from_code(IREE_STATUS_ABORTED));
  EXPECT_EQ(owner.failures, 1u);
  EXPECT_EQ(owner.retirements, 0u);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  // Completion and a rejected suffix may both observe failure. Each owns its
  // reference until its failure callback and native payload accesses end.
  std::thread first([capture] {
    iree_hal_amd_gpu_pm4_submission_fail(
        capture, iree_status_from_code(IREE_STATUS_UNAVAILABLE));
    iree_hal_amd_gpu_pm4_submission_release(capture);
  });
  if (GetParam()) {
    first.join();
  }
  iree_hal_amd_gpu_pm4_submission_fail(
      capture, iree_status_from_code(IREE_STATUS_DATA_LOSS));
  iree_hal_amd_gpu_pm4_submission_release(capture);
  if (first.joinable()) {
    first.join();
  }
  EXPECT_EQ(owner.failures, 1u);
  EXPECT_EQ(owner.retirements, 1u);
  EXPECT_EQ(owner.status_code, IREE_STATUS_ABORTED);
  EXPECT_EQ(owner.frontier.entry_count, 0u);
}

TEST_P(SubmissionTest, RejectedFileSuffixWaitsForAcceptedHostAccesses) {
  FileRetirement owner;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_file(
      &pool_, nullptr, nullptr, nullptr, nullptr, owner.callback(), &capture));
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_reject(capture,
                                         IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_EQ(owner.failures, 0u);
  EXPECT_EQ(owner.retirements, 0u);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_EQ(owner.retirements, 0u);
  // A later reporter cannot replace the rejected suffix's first failure.
  iree_hal_amd_gpu_pm4_submission_reject(capture, IREE_STATUS_ABORTED);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_EQ(owner.failures, 0u);
  EXPECT_EQ(owner.retirements, 1u);
  EXPECT_EQ(owner.status_code, IREE_STATUS_RESOURCE_EXHAUSTED);
}

TEST_P(SubmissionTest, RejectedDownloadSuffixJoinsBeforeFailingPublicSignals) {
  iree_async_proactor_t* proactor = nullptr;
  IREE_ASSERT_OK(
      iree_async_proactor_create_platform(iree_async_proactor_options_default(),
                                          iree_allocator_system(), &proactor));
  iree_hal_semaphore_t* semaphore = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_semaphore_create(
      proactor, 0, 4, iree_allocator_system(), &semaphore));
  const uint16_t types[] = {IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS};
  const iree_hal_buffer_binding_layout_t binding_layout = {
      sizeof(iree_hal_buffer_native_binding_t), 1, 0, types};
  iree_hal_memory_contract_t* contract = nullptr;
  IREE_ASSERT_OK(iree_hal_memory_contract_create(
      this, 2, &binding_layout, iree_allocator_system(), &contract));
  contract->scopes[0].interfaces = 1u
                                   << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS;
  contract->scopes[0].bindings[IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS] = 0;
  contract->scopes[0].usage = IREE_HAL_BUFFER_USAGE_TRANSFER;
  const iree_hal_queue_family_t family = {
      .memory = {.domain = this, .queue_scope_id = 0},
  };
  const iree_hal_buffer_native_binding_t binding = {.device_address = 0x1000};
  const iree_hal_buffer_vtable_t vtable = {
      .destroy = [](iree_hal_buffer_t*) {},
  };
  iree_hal_buffer_t buffer = {};
  iree_hal_buffer_initialize({}, &buffer, 8, 0, 8,
                             IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
                             IREE_HAL_MEMORY_ACCESS_READ,
                             IREE_HAL_BUFFER_USAGE_TRANSFER, &vtable, &buffer);
  buffer.memory = {.contract = contract, .bindings = &binding};
  std::array<uint8_t, 8> output = {};
  const iree_hal_transfer_operation_t operation = {
      .type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD,
      .download =
          {
              .source_buffer = &buffer,
              .target = output.data(),
              .length = output.size(),
          },
  };
  uint64_t value = 1;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_transfer(
      &pool_, GetParam(), &family, {}, {1, &semaphore, &value}, 1, &operation,
      &profile_, nullptr, &capture));
  ASSERT_TRUE(capture->transfer.host_completion_required);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_retain(capture);
  iree_hal_amd_gpu_pm4_submission_reject(capture,
                                         IREE_STATUS_RESOURCE_EXHAUSTED);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  uint64_t observed = UINT64_MAX;
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &observed));
  EXPECT_EQ(observed, 0u);
  // Accepted pieces keep the borrowed host destination live through their
  // final copy, even though the unpublished suffix has already been rejected.
  std::fill_n(output.data(), 4, 0x17);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  IREE_ASSERT_OK(iree_hal_semaphore_query(semaphore, &observed));
  EXPECT_EQ(observed, 0u);
  std::fill_n(output.data() + 4, 4, 0x29);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_semaphore_query(semaphore, &observed));
  EXPECT_EQ(output, (std::array<uint8_t, 8>{0x17, 0x17, 0x17, 0x17, 0x29, 0x29,
                                            0x29, 0x29}));
  iree_hal_buffer_release(&buffer);
  iree_hal_memory_contract_release(contract);
  iree_hal_semaphore_release(semaphore);
  iree_async_proactor_release(proactor);
}

TEST_P(SubmissionTest, FileCallbackMayDestroyCapturePool) {
  iree_hal_amd_gpu_capture_pool_t pool;
  iree_hal_amd_gpu_capture_pool_initialize(GetParam() ? 1024 : 64,
                                           iree_allocator_system(), &pool);
  FileRetirement owner;
  owner.pool = &pool;
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_ASSERT_OK(iree_hal_amd_gpu_pm4_submission_create_file(
      &pool, nullptr, nullptr, nullptr, nullptr, owner.callback(), &capture));
  iree_hal_amd_gpu_pm4_submission_fail(
      capture, iree_status_from_code(IREE_STATUS_CANCELLED));
  EXPECT_EQ(owner.retirements, 0u);
  iree_hal_amd_gpu_pm4_submission_release(capture);
  EXPECT_EQ(owner.retirements, 1u);
  EXPECT_EQ(owner.status_code, IREE_STATUS_CANCELLED);
}

TEST_P(SubmissionTest, RejectedFileCaptureDoesNotAcquireCallbackOwnership) {
  FileRetirement owner;
  std::array<uint8_t, 16> contents = {};
  bool buffer_destroyed = false;
  iree_hal_buffer_t* buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_heap_buffer_wrap(
      {}, IREE_HAL_MEMORY_TYPE_HOST_LOCAL, IREE_HAL_MEMORY_ACCESS_ALL,
      IREE_HAL_BUFFER_USAGE_TRANSFER, contents.size(),
      iree_make_byte_span(contents.data(), contents.size()),
      {[](void* data, iree_hal_buffer_t*) { *static_cast<bool*>(data) = true; },
       &buffer_destroyed},
      iree_allocator_system(), &buffer));
  // A host-only buffer has no qualified native transfer binding. Its caller
  // must handle rejection; no callback can subsequently release that owner.
  iree_hal_transfer_operation_t transfer = {};
  if (GetParam()) {
    transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_UPLOAD;
    transfer.upload = {contents.data(), buffer, 0, contents.size()};
  } else {
    transfer.type = IREE_HAL_TRANSFER_OPERATION_TYPE_DOWNLOAD;
    transfer.download = {buffer, 0, contents.data(), contents.size()};
  }
  const iree_hal_queue_family_t family = {};
  iree_hal_amd_gpu_pm4_submission_t* capture = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_PERMISSION_DENIED,
                        iree_hal_amd_gpu_pm4_submission_create_file(
                            &pool_, &family, &transfer, nullptr, nullptr,
                            owner.callback(), &capture));
  EXPECT_EQ(capture, nullptr);
  EXPECT_EQ(owner.failures, 0u);
  EXPECT_EQ(owner.retirements, 0u);
  iree_hal_buffer_release(buffer);
  EXPECT_TRUE(buffer_destroyed);
}

INSTANTIATE_TEST_SUITE_P(CaptureStorage, SubmissionTest, ::testing::Bool());

}  // namespace
