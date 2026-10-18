// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <future>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/pool_wait.h"

namespace iree::hal::cts {
namespace {

class AmdgpuVirtualBufferTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    ASSERT_NE(transfer_queue_, nullptr);
    access_ = {
        .family = iree_hal_queue_family(transfer_queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    scope_ = {1, &access_, {}};
    any_ = {iree_hal_device_group_memory_domain(device_group_),
            IREE_HAL_MEMORY_SCOPE_ANY};
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        device_group_, scope_, iree_allocator_system(), &capabilities_));
    if (!capabilities_.max_byte_length) {
      GTEST_SKIP() << "ROCr does not expose native virtual memory";
    }
    auto source_access = access_;
    source_access.interfaces = UINT64_C(1)
                               << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
    iree_hal_slab_pool_options_t options = {};
    IREE_ASSERT_OK(iree_hal_slab_pool_create(
        device_group_, {1, &source_access, {}}, &options,
        iree_allocator_system(), pool_.out()));
    const auto requirements = iree_hal_pool_query_mapping_requirements(pool_);
    EXPECT_EQ(requirements.flags, IREE_HAL_POOL_MAPPING_FLAG_WHOLE_ALLOCATION);
    ASSERT_NE(requirements.minimum_granularity, 0u);
    granularity_ = std::max(capabilities_.mapping_granularity,
                            requirements.recommended_granularity);
  }

  void TearDown() override {
    pool_.reset();
    CtsTestBase::TearDown();
  }

  iree_status_t Reserve(iree_device_size_t count,
                        iree_hal_buffer_t** out_buffer) {
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = count * granularity_,
        .mapping_granularity = granularity_,
    };
    return iree_hal_virtual_buffer_create(device_group_, scope_, &options,
                                          iree_allocator_system(), out_buffer);
  }

  void ExpectPattern(iree_hal_buffer_t* buffer, iree_device_size_t offset,
                     uint32_t pattern) {
    std::array<uint32_t, 16> actual = {};
    IREE_ASSERT_OK(
        DownloadBufferData(buffer, offset, actual.data(), sizeof(actual)));
    std::array<uint32_t, 16> expected;
    expected.fill(pattern);
    EXPECT_EQ(actual, expected);
  }

  // Exact source and target access domain for these native transactions.
  iree_hal_pool_family_access_t access_ = {};
  // Borrowed family array used only by cold constructors.
  iree_hal_pool_scope_t scope_ = {};
  // Native address geometry independent of physical allocations.
  iree_hal_virtual_buffer_capabilities_t capabilities_ = {};
  // Wildcard in this sealed group, not the empty invalid scope.
  iree_hal_memory_scope_t any_ = {};
  // Owned physical source retaining the selected native allocation route.
  Ref<iree_hal_pool_t> pool_;
  // Common source/target byte unit selected from both native requirements.
  iree_device_size_t granularity_ = 0;
};

TEST_P(AmdgpuVirtualBufferTest, QueuedBackingMapsAndProducesBytesAcrossHoles) {
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(Reserve(3, target.out()));
  const std::array<iree_hal_pool_reservation_request_t, 2> requests = {
      iree_hal_make_pool_reservation_request(granularity_, 1),
      iree_hal_make_pool_reservation_request(granularity_, 1)};
  std::array<iree_hal_buffer_t*, 2> sources = {};
  SemaphoreList allocated(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_alloca(
      transfer_queue_, iree_hal_semaphore_list_empty(), allocated, pool_,
      requests.size(), requests.data(), sources.data()));
  std::array<iree_hal_memory_mapping_t, 2> mappings = {};
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = target,
        .scope = any_,
        .target_range = {2 * i * granularity_, granularity_},
        .source = sources[i],
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_, allocated, mapped,
                                    mappings.size(), mappings.data(), 0));
  for (size_t i = 0; i < mappings.size(); ++i) {
    const uint32_t pattern = 0xA0B0C000u + static_cast<uint32_t>(i);
    SemaphoreList filled(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_fill(
        transfer_queue_, mapped, filled, target, 2 * i * granularity_, 64,
        &pattern, sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(filled, iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
    ExpectPattern(target, 2 * i * granularity_, pattern);
    ExpectPattern(sources[i], 0, pattern);
    mappings[i].source = nullptr;
    mappings[i].access = IREE_HAL_MEMORY_ACCESS_NONE;
    mappings[i].operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  SemaphoreList unmapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_,
                                    iree_hal_semaphore_list_empty(), unmapped,
                                    mappings.size(), mappings.data(), 0));
  SemaphoreList returned(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, unmapped, returned,
                                         sources.size(), sources.data()));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(returned, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  for (auto* source : sources) {
    iree_hal_buffer_release(source);
  }
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, any_));
}

TEST_P(AmdgpuVirtualBufferTest, CheckedDetachRemovesAliasedMappingsAndHoles) {
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(Reserve(3, target.out()));
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(granularity_, 1),
      iree_infinite_timeout(), source.out()));
  const std::array<iree_hal_memory_mapping_t, 2> mappings = {{
      {
          .target = target,
          .scope = any_,
          .target_range = {0, granularity_},
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
      {
          .target = target,
          .scope = any_,
          .target_range = {2 * granularity_, granularity_},
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
  }};
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_,
                                    iree_hal_semaphore_list_empty(), mapped,
                                    mappings.size(), mappings.data(), 0));
  const uint32_t pattern = 0xFEDC1234;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(transfer_queue_, mapped, filled, target, 0,
                                     64, &pattern, sizeof(pattern), nullptr,
                                     IREE_HAL_FILL_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(filled, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPattern(target, 2 * granularity_, pattern);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, any_));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, any_));
  source.reset();
}

TEST_P(AmdgpuVirtualBufferTest, ChildProtectionAndCompleteUnbind) {
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(Reserve(5, target.out()));
  Ref<iree_hal_buffer_t> child;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(target, granularity_, 3 * granularity_,
                                         iree_allocator_system(), child.out()));
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool_, iree_hal_make_pool_reservation_request(3 * granularity_, 1),
      iree_infinite_timeout(), source.out()));
  iree_hal_memory_mapping_t mapping = {
      .target = child,
      .scope = any_,
      .target_range = {0, 3 * granularity_},
      .source = source,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
  };
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_,
                                    iree_hal_semaphore_list_empty(), mapped, 1,
                                    &mapping, 0));
  const uint32_t pattern = 0xDA7AFEED;
  SemaphoreList filled(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_fill(
      transfer_queue_, mapped, filled, child, 0, 3 * granularity_, &pattern,
      sizeof(pattern), nullptr, IREE_HAL_FILL_FLAG_NONE));
  mapping.access = IREE_HAL_MEMORY_ACCESS_READ;
  mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
  SemaphoreList protected_reads(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_, filled, protected_reads, 1,
                                    &mapping, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      protected_reads, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPattern(child, granularity_, pattern);
  mapping.access = IREE_HAL_MEMORY_ACCESS_ALL;
  SemaphoreList protected_writes(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_, protected_reads,
                                    protected_writes, 1, &mapping, 0));
  mapping.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND;
  mapping.source = nullptr;
  mapping.source_offset = 0;
  mapping.access = IREE_HAL_MEMORY_ACCESS_NONE;
  SemaphoreList unmapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(transfer_queue_, protected_writes, unmapped,
                                    1, &mapping, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(unmapped, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, any_));
  source.reset();
}

TEST_P(AmdgpuVirtualBufferTest, PartialBatchFailureCanDetachEveryTarget) {
  // A failed queue timeline belongs to this private device, not the shared CTS
  // device used by successful workloads.
  DeviceCreateContext context;
  IREE_ASSERT_OK(context.Initialize(iree_allocator_system()));
  Ref<iree_hal_device_t> device;
  IREE_ASSERT_OK(iree_hal_driver_create_default_device(
      driver_, context.params(), iree_allocator_system(), device.out()));
  Ref<iree_hal_device_group_t> group;
  IREE_ASSERT_OK(iree_hal_device_group_create_from_device(
      device, context.frontier_tracker(), iree_allocator_system(),
      group.out()));
  iree_hal_queue_t* queue = iree_hal_device_queue(device, 0, 0);
  const iree_hal_pool_family_access_t access = {
      .family = iree_hal_queue_family(queue),
      .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
  };
  const iree_hal_pool_scope_t scope = {1, &access, {}};
  const iree_hal_memory_scope_t any = {
      iree_hal_device_group_memory_domain(group), IREE_HAL_MEMORY_SCOPE_ANY};
  auto source_access = access;
  source_access.interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  const iree_hal_slab_pool_options_t pool_options = {};
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(group, {1, &source_access, {}}, &pool_options,
                                iree_allocator_system(), pool.out()));
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(2 * granularity_, 1),
      iree_infinite_timeout(), source.out()));
  Ref<iree_hal_buffer_t> targets[2];
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = 2 * granularity_,
      .mapping_granularity = granularity_,
  };
  for (auto& target : targets) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
        group, scope, &options, iree_allocator_system(), target.out()));
  }
  // The common batch sorts by native root. Put a complete valid mapping first,
  // then a logical source slice unsupported by ROCr's whole-handle rule.
  iree_hal_buffer_t* first = targets[0];
  iree_hal_buffer_t* second = targets[1];
  if (reinterpret_cast<uintptr_t>(first) >
      reinterpret_cast<uintptr_t>(second)) {
    std::swap(first, second);
  }
  const iree_hal_memory_mapping_t mappings[] = {
      {
          .target = first,
          .scope = any,
          .target_range = {0, 2 * granularity_},
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
      {
          .target = second,
          .scope = any,
          .target_range = {0, granularity_},
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
  };
  SemaphoreList mapped(device, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(queue, iree_hal_semaphore_list_empty(),
                                    mapped, IREE_ARRAYSIZE(mappings), mappings,
                                    0));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_semaphore_list_wait(mapped, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  for (auto& target : targets) {
    IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, any));
    target.reset();
  }
  source.reset();
  // Final view release returns logical ownership immediately. Capacity is
  // published only after the native allocation worker has freed the handle.
  iree_hal_pool_wait_t* capacity = nullptr;
  IREE_ASSERT_OK(
      iree_hal_pool_wait_allocate(pool, iree_allocator_system(), &capacity));
  iree_status_t status = iree_ok_status();
  while (iree_status_is_ok(status)) {
    iree_hal_pool_wait_prepare(capacity);
    iree_hal_pool_stats_t stats;
    iree_hal_pool_query_stats(pool, &stats);
    if (stats.bytes_committed == 0) {
      iree_hal_pool_wait_abort(capacity);
      break;
    }
    std::promise<iree_status_code_t> completed;
    auto future = completed.get_future();
    iree_hal_pool_wait_commit(
        capacity, iree_infinite_timeout(),
        {[](void* user_data, iree_status_t status) {
           static_cast<std::promise<iree_status_code_t>*>(user_data)->set_value(
               iree_status_consume_code(status));
         },
         &completed});
    status = iree_status_from_code(future.get());
  }
  iree_hal_pool_wait_free(capacity);
  IREE_EXPECT_OK(status);
}

TEST_P(AmdgpuVirtualBufferTest, NativeCapabilitiesRejectUnrepresentableScopes) {
  EXPECT_EQ(capabilities_.operations,
            (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_BIND) |
                (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND) |
                (1u << IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT));
  auto scope = scope_;
  scope.host.access = IREE_HAL_MEMORY_ACCESS_READ;
  scope.host.modes = IREE_HAL_MAPPING_MODE_SCOPED;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      device_group_, scope, iree_allocator_system(), &capabilities));
  EXPECT_EQ(capabilities.max_byte_length, 0u);
}

CTS_REGISTER_TEST_SUITE(AmdgpuVirtualBufferTest);

}  // namespace
}  // namespace iree::hal::cts
