// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/driver/amd/gpu/cts/test_device.h"
#include "iree/hal/driver/amd/gpu/cts/testdata.h"
#include "iree/hal/driver/amd/gpu/queue/aql/state.h"
#include "iree/hal/driver/amd/gpu/queue/pm4/state.h"
#include "iree/hal/slab_pool.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amd::gpu::cts {
namespace {

using iree::hal::cts::Ref;

class TransferStorageTest
    : public GpuNativeTest,
      public ::testing::WithParamInterface<amdf_queue_command_type_t> {
 protected:
  amdf_queue_command_type_t command_type() const override { return GetParam(); }
};

TEST_P(TransferStorageTest, GrowthPreservesLaunchStorageAndOutput) {
  options_.queue.ring_byte_length = 4096;
  options_.queue.argument_capacity = 4096;
  options_.queue.retirement_capacity = 4;
  iree_hal_executable_load_params_t load;
  iree_hal_executable_load_params_initialize(&load);
  IREE_ASSERT_OK(LookupExecutableData("amd_gpu_code", context_->target_info,
                                      IREE_SV("transform.bin"),
                                      &load.executable_data));
  Create();
  if (HasFatalFailure() || IsSkipped()) {
    return;
  }
  const bool aql = GetParam() == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL;
  const auto arguments = [aql](iree_hal_queue_t* queue) {
    return aql ? reinterpret_cast<iree_hal_amd_gpu_aql_queue_t*>(queue)
                     ->producer.arguments
               : reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queue)
                     ->producer.arguments;
  };
  const auto transfers = [aql](iree_hal_queue_t* queue) {
    return aql ? reinterpret_cast<iree_hal_amd_gpu_aql_queue_t*>(queue)
                     ->producer.transfers
               : reinterpret_cast<iree_hal_amd_gpu_pm4_queue_t*>(queue)
                     ->producer.transfers.current;
  };
  std::array<iree_hal_amd_gpu_byte_storage_t*, 2> initial_arguments = {};
  for (uint32_t i = 0; i < queues_.size(); ++i) {
    initial_arguments[i] = arguments(queues_[i]);
    EXPECT_EQ(initial_arguments[i]->capacity, 4096u);
    EXPECT_EQ(transfers(queues_[i]), nullptr);
    RecordProperty(
        "queue" + std::to_string(i) + "_kernarg_native_bytes",
        std::to_string(
            initial_arguments[i]->memory.info.native_allocation_byte_length));
  }

  const auto* family = iree_hal_queue_family(queues_[0]);
  const iree_hal_pool_family_access_t access = {
      .family = family,
      .usage = IREE_HAL_BUFFER_USAGE_STORAGE | IREE_HAL_BUFFER_USAGE_TRANSFER,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &access};
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  Ref<iree_hal_pool_t> pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      group_, scope, &pool_options, iree_allocator_system(), pool.out()));
  // Three increasing demands require initial payload acquisition and two real
  // replacements. The final demand also crosses the bounded download window.
  constexpr std::array<size_t, 3> kByteLengths = {256, 8192,
                                                  2 * 1024 * 1024 + 4};
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(kByteLengths.back(), 4),
      iree_infinite_timeout(), buffer.out()));
  Ref<iree_hal_executable_t> executable;
  const auto* target =
      &iree_hal_device_spec_executables(iree_hal_device_spec(device_))
           ->targets[0];
  IREE_ASSERT_OK(
      iree_hal_executable_load(family, target, &load, executable.out()));
  iree_hal_executable_function_t function;
  IREE_ASSERT_OK(iree_hal_executable_lookup_function_by_name(
      executable, IREE_SV("aql_transform"), &function));
  auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
  config.workgroup_size[0] = 64;
  config.workgroup_size[1] = config.workgroup_size[2] = 1;
  std::vector<uint32_t> source(kByteLengths.back() / sizeof(uint32_t));
  std::vector<uint32_t> observed(source.size() + 2);
  uint64_t submitted = 0;
  const auto join = [&submitted](iree_hal_semaphore_t* semaphore) {
    if (submitted) {
      IREE_EXPECT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                             iree_infinite_timeout(),
                                             IREE_ASYNC_WAIT_FLAG_NONE));
    }
    iree_hal_semaphore_release(semaphore);
  };
  iree_hal_semaphore_t* semaphore = nullptr;
  CreateSemaphore(&semaphore);
  std::unique_ptr<iree_hal_semaphore_t, decltype(join)> timeline(semaphore,
                                                                 join);
  ASSERT_FALSE(HasFatalFailure());
  for (uint32_t i = 0; i < queues_.size(); ++i) {
    auto* queue = queues_[i];
    uint64_t payload_capacity = 0;
    uint32_t acquisitions = 0;
    // Repeating from the smaller requests exercises reuse of the last
    // generation, including wrap, without any further payload acquisition.
    for (uint32_t round = 0; round < 2; ++round) {
      for (size_t byte_length : kByteLengths) {
        SCOPED_TRACE(i);
        SCOPED_TRACE(round);
        SCOPED_TRACE(byte_length);
        const size_t word_count = byte_length / sizeof(uint32_t);
        for (size_t j = 0; j < word_count; ++j) {
          source[j] = 0xCAFE0000u ^ (j * 17u) ^ (round * 97u) ^ i;
        }
        std::fill(observed.begin(), observed.end(), 0xDEADBEEFu);
        uint64_t next = submitted + 1;
        IREE_ASSERT_OK(iree_hal_queue_upload(queue, {}, {1, &semaphore, &next},
                                             source.data(), buffer, 0,
                                             byte_length, nullptr));
        submitted = next++;
        const uint32_t constants[] = {static_cast<uint32_t>(word_count),
                                      11 + round};
        const auto binding = iree_hal_make_buffer_ref(buffer, 0, byte_length);
        const iree_hal_buffer_ref_t bindings[] = {binding, binding};
        config.workgroup_count[0] = (word_count + 63) / 64;
        IREE_ASSERT_OK(iree_hal_queue_dispatch(
            queue, {1, &semaphore, &submitted}, {1, &semaphore, &next},
            executable, function, config,
            iree_make_const_byte_span(constants, sizeof(constants)),
            {2, bindings}, nullptr, IREE_HAL_DISPATCH_FLAG_NONE));
        submitted = next++;
        IREE_ASSERT_OK(iree_hal_queue_download(
            queue, {1, &semaphore, &submitted}, {1, &semaphore, &next}, buffer,
            0, observed.data() + 1, byte_length, nullptr));
        submitted = next;
        IREE_ASSERT_OK(iree_hal_semaphore_wait(semaphore, submitted,
                                               iree_infinite_timeout(),
                                               IREE_ASYNC_WAIT_FLAG_NONE));
        for (size_t j = 0; j < word_count; ++j) {
          source[j] = source[j] * 3u + constants[1];
        }
        EXPECT_EQ(observed.front(), 0xDEADBEEFu);
        EXPECT_EQ(observed[word_count + 1], 0xDEADBEEFu);
        EXPECT_TRUE(std::equal(source.begin(), source.begin() + word_count,
                               observed.begin() + 1));

        // No submissions remain. Their native completion and final host copy
        // synchronize these observations of the publisher-owned generations.
        EXPECT_EQ(arguments(queue), initial_arguments[i]);
        EXPECT_EQ(arguments(queue)->capacity, 4096u);
        const auto* payload = transfers(queue);
        ASSERT_NE(payload, nullptr);
        EXPECT_NE(payload->memory.handle, arguments(queue)->memory.handle);
        if (payload->capacity > payload_capacity) {
          ++acquisitions;
          payload_capacity = payload->capacity;
          RecordProperty(
              "queue" + std::to_string(i) + "_payload" +
                  std::to_string(acquisitions) + "_native_bytes",
              std::to_string(
                  payload->memory.info.native_allocation_byte_length));
        }
        EXPECT_GE(payload_capacity, byte_length);
      }
      EXPECT_EQ(acquisitions, 3u);
      IREE_ASSERT_OK(iree_hal_device_trim(device_));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(Protocol, TransferStorageTest,
                         ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                                           AMDF_QUEUE_COMMAND_TYPE_GPU_AQL),
                         [](const auto& info) {
                           return info.param == AMDF_QUEUE_COMMAND_TYPE_GPU_AQL
                                      ? "Aql"
                                      : "Pm4";
                         });

}  // namespace
}  // namespace iree::hal::amd::gpu::cts
