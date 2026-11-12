// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amd/gpu/queue/aql/growth.h"

#include <semaphore>
#include <tuple>

#include "iree/hal/driver/amd/fabric/memory_backend_test_base.h"
#include "iree/hal/driver/amd/gpu/queue/backing.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Only native backing is substituted. Real captures, native acquisition, and
// result publication remain under the caller's exclusive publisher ownership.
class GrowthTest : public ::testing::TestWithParam<
                       std::tuple<uint64_t, iree_status_code_t>> {
 protected:
  void SetUp() override {
    native_.user_data = this;
    native_.required_flags = 0;
    native_.api.instance_destroy = [](amdf_instance_t*) {
      return AMDF_STATUS_OK;
    };
    native_.api.memory_query_address = [](amdf_memory_t* handle, uint32_t,
                                          amdf_memory_address_kind_t,
                                          uint64_t* out_address) {
      auto* memory = reinterpret_cast<Native::Memory*>(handle);
      auto* self = static_cast<GrowthTest*>(memory->owner->user_data);
      *out_address = std::get<0>(self->GetParam());
      return AMDF_STATUS_OK;
    };
    if (std::get<1>(GetParam()) == IREE_STATUS_RESOURCE_EXHAUSTED) {
      native_.create_failure =
          amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    }
    IREE_ASSERT_OK(iree_allocator_malloc(iree_allocator_system(),
                                         sizeof(*fabric_),
                                         reinterpret_cast<void**>(&fabric_)));
    iree_atomic_ref_count_init(&fabric_->ref_count);
    iree_slim_mutex_initialize(&fabric_->allocation_mutex);
    fabric_->host_allocator = iree_allocator_system();
    fabric_->api = &native_.api;
    context_.fabric = fabric_;
    context_.api = &native_.api;
    IREE_ASSERT_OK(
        iree_hal_amd_gpu_context_prepare_memory(&context_, UINT32_MAX));
    memory_.scope = native_.scope();
    memory_.host_access =
        AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
    memory_.profile.allocation.minimum_alignment = 8;
    memory_.profile.allocation.byte_length_granularity = 8;
    memory_.profile.allocation.maximum_byte_length = UINT32_MAX;
    iree_hal_amd_gpu_capture_pool_initialize(4096, iree_allocator_system(),
                                             &captures_);
    iree_hal_amd_gpu_publisher_initialize(&publisher_);
    growth_.context = &context_;
    growth_.memory = &memory_;
    growth_.captures = &captures_;
    growth_.publisher = &publisher_;
    growth_.publication = &publication_;
    source_ = {
        .publish =
            [](void* user_data) {
              auto* self = static_cast<GrowthTest*>(user_data);
              if (!iree_hal_amd_gpu_publisher_acquire_service(
                      &self->publisher_)) {
                return false;
              }
              // The caller consumes the result before its final handoff.
              EXPECT_EQ(iree_hal_amd_gpu_publisher_next(&self->publisher_),
                        nullptr);
              const bool pending = iree_hal_amd_gpu_publisher_handoff(
                  &self->publisher_, nullptr);
              self->idle_.release();
              return pending;
            },
        .user_data = this,
    };
    IREE_ASSERT_OK(iree_hal_amd_gpu_publication_initialize(
        1, &source_, {}, iree_allocator_system(), &publication_));
  }

  void TearDown() override {
    iree_hal_amd_gpu_context_join_memory(&context_);
    while (!iree_hal_amd_gpu_publisher_is_idle(&publisher_)) {
      idle_.acquire();
    }
    iree_hal_amd_gpu_publication_deinitialize(&publication_);
    EXPECT_EQ(growth_.commands, nullptr);
    EXPECT_EQ(
        iree_atomic_load(&growth_.pending_count, iree_memory_order_acquire), 0);
    iree_hal_amd_gpu_publisher_deinitialize(&publisher_);
    iree_hal_amd_gpu_capture_pool_deinitialize(&captures_);
    iree_hal_amd_gpu_context_deinitialize_memory(&context_);
    EXPECT_EQ(native_.live_memories, 0u);
    iree_hal_amd_fabric_release(fabric_);
  }

  iree_hal_amd_gpu_aql_submission_t* Capture() {
    const iree_hal_amd_gpu_aql_dispatch_request_t request = {};
    iree_hal_amd_gpu_aql_submission_t* submission = nullptr;
    IREE_CHECK_OK(iree_hal_amd_gpu_aql_submission_capture(
        &captures_, nullptr, &request, {}, {}, 0, 0, &submission));
    return submission;
  }

  using Native = iree::hal::amd::fabric::FakeNative;
  // Allocation dependency with observable native ownership and failure.
  Native native_;
  // Real owner of the NUMA allocation service.
  iree_hal_amd_fabric_t* fabric_ = nullptr;
  // Real native acquisition and callback join.
  iree_hal_amd_gpu_context_t context_ = {};
  // Prepared control-memory policy borrowed by acquisition.
  iree_hal_amd_fabric_memory_options_t memory_ = {};
  // Reusable production capture storage.
  iree_hal_amd_gpu_capture_pool_t captures_ = {};
  // Exact queue's caller and deferred publication ownership.
  iree_hal_amd_gpu_publisher_t publisher_ = {};
  // Production native growth state under test.
  iree_hal_amd_gpu_aql_growth_context_t growth_ = {};
  // Real device service receiving native completion notifications.
  iree_hal_amd_gpu_publication_t publication_ = {};
  // Borrowed callback joining stale readiness after caller publication.
  iree_hal_amd_gpu_publication_source_t source_ = {};
  // Explicit service handoff completion, independent of elapsed time.
  std::counting_semaphore<> idle_{0};
};

TEST_P(GrowthTest, CommandAcquisitionReturnsTheCompleteWaitingCohort) {
  ASSERT_TRUE(iree_hal_amd_gpu_publisher_try_acquire(&publisher_));
  auto* first = Capture();
  auto* last = Capture();
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_growth_begin(
      &growth_, IREE_HAL_AMD_GPU_AQL_ENTRY_COMMANDS, first, 4096));
  IREE_ASSERT_OK(iree_hal_amd_gpu_aql_growth_begin(
      &growth_, IREE_HAL_AMD_GPU_AQL_ENTRY_COMMANDS, last, 4096));
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(native_.create_count, 1);
  auto* entry = iree_hal_amd_gpu_publisher_next(&publisher_);
  ASSERT_NE(entry, nullptr);
  auto* result_entry =
      iree_containerof(entry, iree_hal_amd_gpu_aql_entry_t, publication);
  EXPECT_EQ(result_entry->kind, IREE_HAL_AMD_GPU_AQL_ENTRY_COMMANDS);
  iree_hal_amd_gpu_aql_growth_result_t result;
  const auto expected = std::get<1>(GetParam());
  IREE_EXPECT_STATUS_IS(
      expected, iree_hal_amd_gpu_aql_growth_end(result_entry, 0, &result));
  EXPECT_EQ(result.first, &first->entry.publication);
  EXPECT_EQ(result.last, &last->entry.publication);
  EXPECT_EQ(iree_atomic_load(&result.first->next, iree_memory_order_relaxed),
            reinterpret_cast<intptr_t>(result.last));
  EXPECT_EQ(iree_atomic_load(&result.last->next, iree_memory_order_relaxed), 0);
  if (expected == IREE_STATUS_OK) {
    ASSERT_NE(result.bytes, nullptr);
    EXPECT_EQ(result.bytes->address, std::get<0>(GetParam()));
    EXPECT_EQ(result.bytes->capacity, 4096u);
    iree_hal_amd_gpu_byte_storage_free(&context_, result.bytes);
  } else {
    EXPECT_EQ(result.bytes, nullptr);
  }
  iree_hal_amd_gpu_aql_submission_release_payload(first);
  iree_hal_amd_gpu_aql_submission_release(first);
  iree_hal_amd_gpu_aql_submission_release_payload(last);
  iree_hal_amd_gpu_aql_submission_release(last);
  if (iree_hal_amd_gpu_publisher_handoff(&publisher_, nullptr)) {
    iree_hal_amd_gpu_publication_notify(&publication_, 0);
  }
  iree_hal_amd_gpu_context_join_memory(&context_);
  EXPECT_EQ(native_.destroy_count,
            expected == IREE_STATUS_RESOURCE_EXHAUSTED ? 0 : 1);
}

INSTANTIATE_TEST_SUITE_P(
    NativeResults, GrowthTest,
    ::testing::Values(
        std::make_tuple((UINT64_C(1) << 48) - 4096, IREE_STATUS_OK),
        std::make_tuple((UINT64_C(1) << 48) - 4088, IREE_STATUS_OUT_OF_RANGE),
        std::make_tuple(UINT64_C(0x10000), IREE_STATUS_RESOURCE_EXHAUSTED)));

}  // namespace
