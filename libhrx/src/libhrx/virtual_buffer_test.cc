// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <vector>

#include "hrx_internal.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class CpuVirtualBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_initialize(0)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_cpu_device_get(0, &device_)));
  }

  void TearDown() override {
    IREE_EXPECT_OK(hrx_status_to_iree(hrx_cpu_shutdown()));
  }

  // Borrowed device in the initialized CPU runtime's sealed group.
  hrx_device_t device_ = nullptr;
};

TEST_F(CpuVirtualBufferTest, BatchedAliasesCaptureDependenciesAndSources) {
  // Exercise inline and overflow translation through the same native lifecycle.
  for (size_t count : {2, 9}) {
    SCOPED_TRACE(count);
    const hrx_virtual_buffer_scope_t scope = {
        .device_count = 1,
        .devices = &device_,
        .host_access = HRX_MEMORY_ACCESS_ALL,
    };
    const hrx_virtual_buffer_source_t source_class = {
        .device = device_,
        .memory_type = HRX_MEMORY_TYPE_HOST_LOCAL,
    };
    hrx_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
        scope, {1, &source_class}, &capabilities)));
    ASSERT_GT(capabilities.max_byte_length, 0u);
    EXPECT_EQ(0xFu, capabilities.operations);
    EXPECT_NE(capabilities.supported_flags &
                  HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
              0u);
    hrx_memory_mapping_requirements_t requirements;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_allocator_query_mapping_requirements(
        hrx_device_allocator(device_), HRX_MEMORY_TYPE_HOST_LOCAL,
        &requirements)));
    ASSERT_GT(requirements.minimum_granularity, 0u);
    const size_t unit = std::max({capabilities.reservation_granularity,
                                  capabilities.mapping_granularity,
                                  requirements.recommended_granularity});
    const hrx_virtual_buffer_options_t options = {
        .byte_length = 2 * unit,
        .min_alignment = 2 * unit,
        .mapping_granularity = unit,
        .flags = HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
    };
    std::vector<hrx_buffer_t> reservations(count);
    for (auto& reservation : reservations) {
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_create(
          scope, {1, &source_class}, &options, &reservation)));
      hrx_virtual_buffer_info_t info;
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_virtual_buffer_query_info(reservation, &info)));
      EXPECT_EQ(unit, info.mapping_granularity);
    }
    const hrx_buffer_params_t params = {
        .type = HRX_MEMORY_TYPE_HOST_LOCAL,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
    };
    hrx_buffer_t source = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_allocator_allocate_buffer(
        hrx_device_allocator(device_), params, 2 * unit, &source)));
    std::vector<uint32_t> contents(2 * unit / sizeof(uint32_t), 0xAABBCCDD);
    std::fill_n(contents.begin(), unit / sizeof(uint32_t), 0x10203040);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_h2d(device_, contents.data(), source, 0, 2 * unit)));

    hrx_semaphore_t gate = nullptr;
    hrx_semaphore_t completion = nullptr;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_create(device_, 0, &gate)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_semaphore_create(device_, 0, &completion)));
    uint64_t gate_value = 1;
    uint64_t completed = 1;
    hrx_semaphore_list_t gate_list = {
        .semaphores = &gate,
        .values = &gate_value,
        .count = 1,
    };
    hrx_semaphore_list_t signal_list = {
        .semaphores = &completion,
        .values = &completed,
        .count = 1,
    };
    std::vector<hrx_memory_mapping_t> mappings(count);
    for (size_t i = 0; i < count; ++i) {
      mappings[i] = {
          .target = reservations[i],
          .target_offset = unit,
          .byte_length = unit,
          .source = source,
          .source_offset = unit,
          .access = HRX_MEMORY_ACCESS_ALL,
          .operation = HRX_MEMORY_MAPPING_OPERATION_BIND,
      };
    }
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_map(device_, 0, &gate_list, &signal_list, count,
                      mappings.data(), HRX_QUEUE_MAP_FLAG_NONE)));
    std::fill(mappings.begin(), mappings.end(), hrx_memory_mapping_t{});
    uint64_t observed = UINT64_MAX;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_semaphore_query(completion, &observed)));
    EXPECT_EQ(0u, observed);
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate, gate_value)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));

    auto first = (reservations.front());
    auto last = (reservations.back());
    void* device_pointer = nullptr;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_buffer_get_device_ptr(first, &device_pointer)));
    EXPECT_EQ(0u, reinterpret_cast<uintptr_t>(device_pointer) % (2 * unit));
    const uint32_t pattern = 0x55667788;
    uint64_t predecessor = completed++;
    const hrx_semaphore_list_t wait_list = {
        .semaphores = &completion,
        .values = &predecessor,
        .count = 1,
    };
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_fill(device_, 0, &wait_list, &signal_list, first, unit,
                       sizeof(pattern), &pattern, sizeof(pattern))));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    uint32_t result = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(device_, last, unit, &result, sizeof(result))));
    EXPECT_EQ(pattern, result);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(device_, source, unit, &result, sizeof(result))));
    EXPECT_EQ(pattern, result);

    hrx_memory_scope_t queue_scope, program_scope, host_scope;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_device_query_memory_scope(
        device_, 0, HRX_MEMORY_SITE_QUEUE, &queue_scope)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_device_query_memory_scope(
        device_, 0, HRX_MEMORY_SITE_PROGRAM, &program_scope)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_device_query_memory_scope(
        device_, 0, HRX_MEMORY_SITE_HOST, &host_scope)));
    uint32_t queue_projection, program_projection, host_projection;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_resolve_projection(
        reservations.front(), queue_scope, &queue_projection)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_resolve_projection(
        reservations.front(), program_scope, &program_projection)));
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_resolve_projection(
        reservations.front(), host_scope, &host_projection)));
    EXPECT_EQ(queue_projection, program_projection);
    EXPECT_EQ(queue_projection, host_projection);

    hrx_memory_mapping_t replacement = {
        .target = reservations.front(),
        .scope = program_scope,
        .target_offset = unit,
        .byte_length = unit,
        .source = source,
        .access = HRX_MEMORY_ACCESS_NONE,
        .operation = HRX_MEMORY_MAPPING_OPERATION_REPLACE,
        .previous_source = source,
        .previous_source_offset = unit,
    };
    predecessor = completed++;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_map(device_, 0, &wait_list, &signal_list, 1, &replacement,
                      HRX_QUEUE_MAP_FLAG_BORROW_BUFFERS)));
    replacement.operation = HRX_MEMORY_MAPPING_OPERATION_PROTECT;
    replacement.previous_source = nullptr;
    replacement.previous_source_offset = 0;
    replacement.access = HRX_MEMORY_ACCESS_READ;
    predecessor = completed++;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_map(device_, 0, &wait_list, &signal_list, 1, &replacement,
                      HRX_QUEUE_MAP_FLAG_NONE)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(device_, first, unit, &result, sizeof(result))));
    EXPECT_EQ(0x10203040u, result);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(device_, last, unit, &result, sizeof(result))));
    EXPECT_EQ(pattern, result);

    for (size_t i = 0; i < count; ++i) {
      mappings[i] = {
          .target = reservations[i],
          .target_offset = unit,
          .byte_length = unit,
          .operation = HRX_MEMORY_MAPPING_OPERATION_UNBIND,
          .previous_source = source,
          .previous_source_offset = i == 0 ? 0 : unit,
      };
    }
    predecessor = completed++;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_queue_map(device_, 0, &wait_list, &signal_list, count,
                      mappings.data(), HRX_QUEUE_MAP_FLAG_NONE)));
    predecessor = completed++;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_queue_map(device_, 0, &wait_list, &signal_list,
                                         0, nullptr, HRX_QUEUE_MAP_FLAG_NONE)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    hrx_buffer_retain(first);
    for (auto reservation : reservations) {
      hrx_buffer_release(reservation);
    }
    size_t retained_size = 0;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_buffer_get_size(first, &retained_size)));
    EXPECT_EQ(2 * unit, retained_size);
    hrx_buffer_release(first);
    hrx_buffer_release(source);
    hrx_semaphore_release(completion);
    hrx_semaphore_release(gate);
  }
}

TEST_F(CpuVirtualBufferTest, ValidatesConstructionBeforeAcquiringAddresses) {
  hrx_virtual_buffer_capabilities_t capabilities;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(
          hrx_virtual_buffer_query_capabilities({}, {}, &capabilities)));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  std::array<hrx_device_t, 2> repeated = {device_, device_};
  const hrx_virtual_buffer_scope_t scope = {
      .device_count = repeated.size(),
      .devices = repeated.data(),
  };
  const hrx_virtual_buffer_source_t source_class = {
      .device = device_,
      .memory_type = HRX_MEMORY_TYPE_HOST_LOCAL,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
          scope, {1, &source_class}, &capabilities)));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  hrx_buffer_t reservation = nullptr;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        hrx_status_to_iree(hrx_virtual_buffer_create(
                            scope, {1, &source_class}, nullptr, &reservation)));
  EXPECT_EQ(nullptr, reservation);

  const hrx_virtual_buffer_scope_t valid_scope = {
      .device_count = 1,
      .devices = &device_,
  };
  const std::array<hrx_virtual_buffer_source_t, 2> incomplete_sources = {
      source_class,
      hrx_virtual_buffer_source_t{},
  };
  const hrx_virtual_buffer_options_t options = {.byte_length = 65536};
  const std::array<hrx_virtual_buffer_sources_t, 4> invalid_sources = {
      hrx_virtual_buffer_sources_t{},
      hrx_virtual_buffer_sources_t{1, nullptr},
      hrx_virtual_buffer_sources_t{1, &incomplete_sources[1]},
      hrx_virtual_buffer_sources_t{incomplete_sources.size(),
                                   incomplete_sources.data()},
  };
  for (const auto& sources : invalid_sources) {
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
            valid_scope, sources, &capabilities)));
    EXPECT_EQ(0u, capabilities.max_byte_length);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          hrx_status_to_iree(hrx_virtual_buffer_create(
                              valid_scope, sources, &options, &reservation)));
    EXPECT_EQ(nullptr, reservation);
  }
  hrx_buffer_release(nullptr);
}

TEST_F(CpuVirtualBufferTest, HostRegistrationCannotBecomeADetach) {
  const hrx_virtual_buffer_scope_t scope = {
      .device_count = 1,
      .devices = &device_,
  };
  const hrx_virtual_buffer_source_t source_class = {
      .device = device_,
      .memory_type = HRX_MEMORY_TYPE_HOST_LOCAL,
  };
  hrx_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
      scope, {1, &source_class}, &capabilities)));
  const hrx_virtual_buffer_options_t options = {
      .byte_length = capabilities.reservation_granularity,
  };
  hrx_buffer_t reservation = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_create(
      scope, {1, &source_class}, &options, &reservation)));
  std::vector<uint8_t> storage(options.byte_length);
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_host_memory_register(device_, storage.data(), storage.size(), 0)));
  hrx_buffer_t registration = nullptr;
  size_t offset = 0;
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_buffer_lookup(device_, storage.data(), &registration, &offset)));
  const hrx_memory_mapping_t mapping = {
      .target = reservation,
      .byte_length = options.byte_length,
      .source = registration,
      .access = HRX_MEMORY_ACCESS_NONE,
      .operation = HRX_MEMORY_MAPPING_OPERATION_REPLACE,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      hrx_status_to_iree(hrx_queue_map(device_, 0, nullptr, nullptr, 1,
                                       &mapping, HRX_QUEUE_MAP_FLAG_NONE)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_host_memory_unregister(device_, storage.data())));
  hrx_buffer_release(reservation);
}

}  // namespace
