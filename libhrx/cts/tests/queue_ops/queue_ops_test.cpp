// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

#include "hrx_test_fixture.hpp"

namespace {

hrx_status_t hostCallIncrement(void* user_data) {
  auto* value = static_cast<int*>(user_data);
  *value += 1;
  return hrx_ok_status();
}

struct QueueTransferResources {
  ~QueueTransferResources() {
    if (completion_value) {
      ReportHrxTestFailure(
          "queue transfer completion",
          hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
    }
    hrx().buffer_release(target);
    hrx().buffer_release(source);
    hrx().semaphore_release(completion);
    hrx().queue_release(queue);
  }

  // Acquired queue retained until all submitted operations complete.
  hrx_queue_t queue = nullptr;
  // Transfer source retained through the final completion.
  hrx_buffer_t source = nullptr;
  // Transfer destination retained through the final completion.
  hrx_buffer_t target = nullptr;
  // Timeline joining operations across both exact queues.
  hrx_semaphore_t completion = nullptr;
  // Latest accepted submission's completion, including assertion cleanup.
  uint64_t completion_value = 0;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture,
                 "Queue acquisition preserves explicit properties",
                 "[queue_ops][acquire]") {
  const uint32_t family = hrx().queue_get_family_ordinal(transfer_queue_);
  hrx_queue_family_info_t info;
  REQUIRE_OK(hrx().device_query_queue_family(device_, family, &info));
  if (!info.supports_acquisition) {
    SKIP("Queue family does not support dynamic acquisition");
  }
  REQUIRE(info.priority_count > 0);
  std::vector<uint32_t> ordinals(info.execution_resource_count);
  for (uint32_t i = 0; i < ordinals.size(); ++i) {
    hrx_queue_execution_resource_info_t resource;
    REQUIRE_OK(hrx().device_query_queue_execution_resource(device_, family, i,
                                                           &resource));
    REQUIRE(resource.execution_unit_count > 0);
    REQUIRE(resource.first_execution_unit_ordinal < info.execution_unit_count);
    hrx_queue_execution_resource_group_info_t group;
    REQUIRE_OK(hrx().device_query_queue_execution_resource_group(
        device_, family, resource.group_ordinal, &group));
    REQUIRE(group.minimum_selected_resource_count <=
            info.execution_resource_count);
    ordinals[i] = i;
  }
  const hrx_queue_params_t params = {
      .priority = 0,
      .placement = (info.placements & HRX_QUEUE_PLACEMENT_CAPABILITY_HOST_LOCAL)
                       ? HRX_QUEUE_PLACEMENT_HOST_LOCAL
                       : HRX_QUEUE_PLACEMENT_DEFAULT,
      .execution_resources = {ordinals.size(), ordinals.data()},
  };
  QueueTransferResources resources;
  REQUIRE_OK(hrx().queue_acquire(device_, family, &params, &resources.queue));
  hrx_queue_params_t achieved;
  REQUIRE_OK(hrx().queue_query_params(resources.queue, &achieved));
  REQUIRE(achieved.priority == params.priority);
  if (params.placement != HRX_QUEUE_PLACEMENT_DEFAULT) {
    REQUIRE(achieved.placement == params.placement);
  }
  REQUIRE(achieved.features == HRX_QUEUE_FEATURE_NONE);
  // An explicit complete set canonicalizes to the all-resources representation.
  REQUIRE(achieved.execution_resources.count == 0);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "Explicit queues exchange semaphore dependencies",
                 "[queue_ops][acquire][sync]") {
  const uint32_t family = hrx().queue_get_family_ordinal(transfer_queue_);
  hrx_queue_family_info_t info;
  REQUIRE_OK(hrx().device_query_queue_family(device_, family, &info));
  if (!info.supports_acquisition) {
    SKIP("Queue family does not support dynamic acquisition");
  }
  QueueTransferResources resources;
  REQUIRE_OK(hrx().queue_acquire(device_, family, nullptr, &resources.queue));
  REQUIRE(resources.queue != transfer_queue_);
  REQUIRE(hrx().queue_get_device(resources.queue) == device_);
  REQUIRE(hrx().queue_get_family_ordinal(resources.queue) == family);
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &resources.completion));
  const hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_TRANSFER,
  };
  constexpr size_t kWordCount = 256;
  constexpr size_t kByteLength = kWordCount * sizeof(uint32_t);
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      hrx().device_allocator(device_), params, kByteLength, &resources.source));
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      hrx().device_allocator(device_), params, kByteLength, &resources.target));
  const uint32_t pattern = 0x1234ABCD;
  uint64_t wait_value = 0;
  uint64_t signal_value = 1;
  const hrx_semaphore_list_t waits = {&resources.completion, &wait_value, 1};
  const hrx_semaphore_list_t signals = {&resources.completion, &signal_value,
                                        1};
  const hrx_barrier_list_t malformed = {.count = 1};
  const hrx_queue_barriers_t invalid = {.before = &malformed};
  hrx_status_t rejected =
      hrx().queue_barrier(resources.queue, nullptr, &signals, &invalid);
  REQUIRE(hrx().status_code(rejected) == HRX_STATUS_INVALID_ARGUMENT);
  hrx().status_ignore(rejected);
  uint64_t observed = UINT64_MAX;
  REQUIRE_OK(hrx().semaphore_query(resources.completion, &observed));
  REQUIRE(observed == 0);

  // Each independently tracked block contributes its own dependency. Eight
  // records exercise descriptor capture beyond the common inline capacity.
  const hrx_memory_barrier_t memory = {
      .source_scope = HRX_ACCESS_SCOPE_TRANSFER_WRITE,
      .target_scope = HRX_ACCESS_SCOPE_TRANSFER_READ,
  };
  std::array<hrx_buffer_barrier_t, 8> ranges = {};
  std::array<hrx_barrier_t, 8> releases = {};
  std::array<hrx_barrier_t, 8> acquires = {};
  for (size_t i = 0; i < ranges.size(); ++i) {
    ranges[i] = {
        .source_scope = HRX_ACCESS_SCOPE_TRANSFER_WRITE,
        .target_scope = HRX_ACCESS_SCOPE_TRANSFER_READ,
        .buffer_ref = {resources.source, i * (kByteLength / ranges.size()),
                       kByteLength / ranges.size()},
    };
    releases[i] = {
        .source_stage_mask = HRX_EXECUTION_STAGE_TRANSFER,
        .flags = HRX_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
        .memory_barrier_count = 1,
        .memory_barriers = &memory,
        .buffer_barrier_count = 1,
        .buffer_barriers = &ranges[i],
    };
    acquires[i] = releases[i];
    acquires[i].source_stage_mask = 0;
    acquires[i].target_stage_mask = HRX_EXECUTION_STAGE_TRANSFER;
    acquires[i].flags = HRX_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE;
  }
  const hrx_barrier_list_t empty = {};
  const hrx_barrier_list_t release_list = {releases.size(), releases.data()};
  const hrx_barrier_list_t acquire_list = {acquires.size(), acquires.data()};
  const hrx_queue_barriers_t fill_barriers = {&empty, &release_list};
  const hrx_queue_barriers_t copy_barriers = {&acquire_list, nullptr};
  const hrx_queue_barriers_t empty_barriers = {&empty, &empty};
  REQUIRE_OK(hrx().queue_fill(resources.queue, nullptr, &signals,
                              resources.source, 0, kByteLength, &pattern,
                              sizeof(pattern), &fill_barriers));
  resources.completion_value = signal_value;
  wait_value = signal_value++;
  REQUIRE_OK(hrx().queue_copy(transfer_queue_, &waits, &signals,
                              resources.source, 0, resources.target, 0,
                              kByteLength, &copy_barriers));
  resources.completion_value = signal_value;
  wait_value = signal_value++;
  REQUIRE_OK(
      hrx().queue_barrier(resources.queue, &waits, &signals, &empty_barriers));
  resources.completion_value = signal_value;
  REQUIRE_OK(
      hrx().semaphore_wait(resources.completion, signal_value, UINT64_MAX));
  std::vector<uint32_t> result(kWordCount);
  REQUIRE_OK(hrx().synchronous_d2h(resources.queue, resources.target, 0,
                                   result.data(), kByteLength));
  for (uint32_t value : result) {
    REQUIRE(value == pattern);
  }
}

TEST_CASE_METHOD(HrxTestFixture, "Queue host call", "[queue_ops][host_call]") {
  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));

  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  int callback_count = 0;
  REQUIRE_OK(hrx().queue_host_call(
      FindProvisionedQueue(device_, HRX_QUEUE_ROLE_HOST_CALL), nullptr,
      &signal_list, hostCallIncrement, &callback_count));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));
  REQUIRE(callback_count == 1);

  hrx().semaphore_release(done);
}

TEST_CASE_METHOD(HrxTestFixture, "Queue fill signals completion semaphore",
                 "[queue_ops][fill][sync]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_TRANSFER,
  };

  constexpr size_t kWordCount = 4096;
  constexpr uint32_t kPattern = 0xA5C33CA5u;
  hrx_buffer_t buf = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      alloc, params, kWordCount * sizeof(uint32_t), &buf));

  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));
  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  REQUIRE_OK(hrx().queue_fill(transfer_queue_, nullptr, &signal_list, buf, 0,
                              kWordCount * sizeof(uint32_t), &kPattern,
                              sizeof(kPattern), nullptr));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));

  std::vector<uint32_t> readback(kWordCount);
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, buf, 0, readback.data(),
                                   readback.size() * sizeof(uint32_t)));
  for (uint32_t value : readback) {
    REQUIRE(value == kPattern);
  }

  hrx().semaphore_release(done);
  hrx().buffer_release(buf);
}

TEST_CASE_METHOD(HrxTestFixture, "Queue copy signals completion semaphore",
                 "[queue_ops][copy][sync]") {
  hrx_allocator_t alloc = hrx().device_allocator(device_);

  hrx_buffer_params_t params = {
      .type = HRX_MEMORY_TYPE_DEVICE_LOCAL,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_TRANSFER,
  };

  constexpr size_t kByteCount = 16 * 1024;
  hrx_buffer_t src = nullptr;
  hrx_buffer_t dst = nullptr;
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, kByteCount, &src));
  REQUIRE_OK(hrx().allocator_allocate_buffer(alloc, params, kByteCount, &dst));

  std::vector<uint8_t> expected(kByteCount);
  for (size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<uint8_t>((i * 13u + 7u) & 0xFFu);
  }
  REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, expected.data(), src, 0,
                                   expected.size()));

  hrx_semaphore_t done = nullptr;
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &done));
  hrx_semaphore_t signal_semaphores[] = {done};
  uint64_t signal_values[] = {1};
  hrx_semaphore_list_t signal_list = {
      signal_semaphores,
      signal_values,
      1,
  };

  REQUIRE_OK(hrx().queue_copy(transfer_queue_, nullptr, &signal_list, src, 0,
                              dst, 0, kByteCount, nullptr));
  REQUIRE_OK(hrx().semaphore_wait(done, 1, UINT64_MAX));

  std::vector<uint8_t> readback(kByteCount);
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, dst, 0, readback.data(),
                                   readback.size()));
  REQUIRE(std::memcmp(readback.data(), expected.data(), expected.size()) == 0);

  hrx().semaphore_release(done);
  hrx().buffer_release(dst);
  hrx().buffer_release(src);
}
