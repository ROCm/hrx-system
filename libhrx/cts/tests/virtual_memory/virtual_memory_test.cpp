// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>

#include "hrx_test_fixture.hpp"

namespace {

class VirtualMemoryResources {
 public:
  explicit VirtualMemoryResources(hrx_device_t device) : device(device) {}

  ~VirtualMemoryResources() {
    if (host_mapped) {
      hrx_status_t status = hrx().buffer_unmap(reservation);
      CHECK(hrx_status_is_ok(status));
      hrx().status_ignore(status);
    }
    bool detached = true;
    if (reservation) {
      hrx_status_t status = hrx().virtual_buffer_detach(reservation, {});
      detached = hrx_status_is_ok(status);
      CHECK(detached);
      hrx().status_ignore(status);
      hrx().buffer_release(reservation);
    }
    // Failed removal cannot authorize recycling possibly mapped backing.
    if (detached) {
      for (auto source : sources) {
        hrx().buffer_release(source);
      }
    }
    hrx().semaphore_release(completion);
  }

  void Map(size_t count, const hrx_memory_mapping_t* mappings) {
    ++completion_value;
    const hrx_semaphore_list_t signals = {
        .semaphores = &completion,
        .values = &completion_value,
        .count = 1,
    };
    REQUIRE_OK(hrx().queue_map(device, 0, nullptr, &signals, count, mappings,
                               HRX_QUEUE_MAP_FLAG_NONE));
    REQUIRE_OK(hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
  }

  // Borrowed execution device that outlives these resources.
  hrx_device_t device;
  // Ordinary virtual buffer retaining the address interval.
  hrx_buffer_t reservation = nullptr;
  // Ordinary prepared backing retained until checked removal succeeds.
  std::array<hrx_buffer_t, 2> sources = {};
  // Completion for synchronous test observation of each mapping batch.
  hrx_semaphore_t completion = nullptr;
  // Last submitted completion value.
  uint64_t completion_value = 0;
  // Whether the reservation has a live scoped public host mapping.
  bool host_mapped = false;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "query reservation capabilities",
                 "[virtual_memory][query]") {
  const hrx_virtual_buffer_scope_t scope = {
      .device_count = 1,
      .devices = &device_,
  };
  const hrx_virtual_buffer_source_t source_class = {
      .device = device_,
      .memory_type = (is_cpu() ? HRX_MEMORY_TYPE_HOST_LOCAL
                               : HRX_MEMORY_TYPE_DEVICE_LOCAL),
  };
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(scope, {1, &source_class},
                                                     &capabilities));
  if (capabilities.max_byte_length) {
    CHECK(capabilities.reservation_granularity > 0);
    CHECK(capabilities.mapping_granularity > 0);
    CHECK(capabilities.operations & (1u << HRX_MEMORY_MAPPING_OPERATION_BIND));
    CHECK(capabilities.operations &
          (1u << HRX_MEMORY_MAPPING_OPERATION_UNBIND));
  } else {
    const hrx_virtual_buffer_options_t options = {.byte_length = 65536};
    hrx_buffer_t reservation = nullptr;
    hrx_status_t status = hrx().virtual_buffer_create(scope, {1, &source_class},
                                                      &options, &reservation);
    CHECK(!hrx_status_is_ok(status));
    CHECK(reservation == nullptr);
    hrx().status_ignore(status);
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "batched virtual aliases share ordinary backing",
                 "[virtual_memory][lifecycle]") {
  const hrx_virtual_buffer_scope_t scope = {
      .device_count = 1,
      .devices = &device_,
  };
  const hrx_virtual_buffer_source_t source_class = {
      .device = device_,
      .memory_type = (is_cpu() ? HRX_MEMORY_TYPE_HOST_LOCAL
                               : HRX_MEMORY_TYPE_DEVICE_LOCAL),
  };
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(scope, {1, &source_class},
                                                     &capabilities));
  if (!capabilities.max_byte_length) {
    SKIP("The selected device has no virtual buffer backend.");
  }
  const hrx_memory_type_t memory_type =
      is_cpu() ? HRX_MEMORY_TYPE_HOST_LOCAL : HRX_MEMORY_TYPE_DEVICE_LOCAL;
  hrx_memory_mapping_requirements_t requirements;
  REQUIRE_OK(hrx().allocator_query_mapping_requirements(
      hrx().device_allocator(device_), memory_type, &requirements));
  const size_t unit = std::max({capabilities.reservation_granularity,
                                capabilities.mapping_granularity,
                                requirements.recommended_granularity});
  REQUIRE(unit > 0);
  VirtualMemoryResources resources(device_);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(scope, {1, &source_class}, &options,
                                         &resources.reservation));
  const hrx_buffer_params_t params = {
      .type = memory_type,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
  };
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      hrx().device_allocator(device_), params, unit, &resources.sources[0]));
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &resources.completion));
  std::array<hrx_memory_mapping_t, 2> mappings;
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = resources.reservation,
        .target_offset = i * unit,
        .byte_length = unit,
        .source = resources.sources[0],
        .access = HRX_MEMORY_ACCESS_ALL,
        .operation = HRX_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  resources.Map(mappings.size(), mappings.data());
  hrx_buffer_t view = resources.reservation;
  const std::array<uint32_t, 4> expected = {7, 13, 19, 23};
  REQUIRE_OK(hrx().synchronous_h2d(device_, expected.data(), view, 0,
                                   sizeof(expected)));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(device_, view, unit, observed.data(),
                                   sizeof(observed)));
  CHECK(observed == expected);
  REQUIRE_OK(hrx().synchronous_d2h(device_, resources.sources[0], 0,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  if (capabilities.operations & (1u << HRX_MEMORY_MAPPING_OPERATION_PROTECT)) {
    for (auto& mapping : mappings) {
      mapping.operation = HRX_MEMORY_MAPPING_OPERATION_PROTECT;
      mapping.access = HRX_MEMORY_ACCESS_READ;
    }
    resources.Map(mappings.size(), mappings.data());
    REQUIRE_OK(hrx().synchronous_d2h(device_, view, unit, observed.data(),
                                     sizeof(observed)));
    CHECK(observed == expected);
  }
  for (auto& mapping : mappings) {
    mapping.previous_source = mapping.source;
    mapping.previous_source_offset = mapping.source_offset;
    mapping.source = nullptr;
    mapping.source_offset = 0;
    mapping.access = HRX_MEMORY_ACCESS_NONE;
    mapping.operation = HRX_MEMORY_MAPPING_OPERATION_UNBIND;
  }
  resources.Map(mappings.size(), mappings.data());
}

TEST_CASE_METHOD(HrxTestFixture,
                 "common host and program aliases share one pointer",
                 "[virtual_memory][common_address]") {
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
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(scope, {1, &source_class},
                                                     &capabilities));
  if (!capabilities.max_byte_length ||
      !(capabilities.supported_flags &
        HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS)) {
    SKIP("The selected scope cannot share host and program addresses.");
  }
  const hrx_memory_type_t memory_type =
      HRX_MEMORY_TYPE_HOST_LOCAL | HRX_MEMORY_TYPE_HOST_COHERENT;
  hrx_memory_mapping_requirements_t requirements;
  REQUIRE_OK(hrx().allocator_query_mapping_requirements(
      hrx().device_allocator(device_), memory_type, &requirements));
  const size_t unit = std::max({capabilities.reservation_granularity,
                                capabilities.mapping_granularity,
                                requirements.recommended_granularity});
  REQUIRE(unit > 0);
  VirtualMemoryResources resources(device_);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
      .flags = HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(scope, {1, &source_class}, &options,
                                         &resources.reservation));
  const hrx_buffer_params_t params = {
      .type = memory_type,
      .access = HRX_MEMORY_ACCESS_ALL,
      .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
  };
  REQUIRE_OK(hrx().allocator_allocate_buffer(
      hrx().device_allocator(device_), params, unit, &resources.sources[0]));
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &resources.completion));
  std::array<hrx_memory_mapping_t, 2> mappings;
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = resources.reservation,
        .target_offset = i * unit,
        .byte_length = unit,
        .source = resources.sources[0],
        .access = HRX_MEMORY_ACCESS_ALL,
        .operation = HRX_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  resources.Map(mappings.size(), mappings.data());
  void* program_address = nullptr;
  REQUIRE_OK(
      hrx().buffer_get_device_ptr(resources.reservation, &program_address));
  void* host_address = nullptr;
  REQUIRE_OK(hrx().buffer_map(resources.reservation,
                              HRX_MAP_READ | HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS,
                              0, 2 * unit, &host_address));
  resources.host_mapped = true;
  REQUIRE(host_address == program_address);
  const std::array<uint32_t, 4> expected = {11, 17, 29, 47};
  std::memcpy(host_address, expected.data(), sizeof(expected));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(device_, resources.reservation, unit,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  const std::array<uint32_t, 4> update = {53, 71, 89, 97};
  REQUIRE_OK(hrx().synchronous_h2d(
      device_, update.data(), resources.reservation, unit, sizeof(update)));
  std::memcpy(observed.data(), host_address, sizeof(observed));
  CHECK(observed == update);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "mixed backing preserves scoped host and device access",
                 "[virtual_memory][mixed_backing]") {
  const hrx_virtual_buffer_scope_t scope = {
      .device_count = 1,
      .devices = &device_,
      .host_access = HRX_MEMORY_ACCESS_ALL,
  };
  const std::array<hrx_virtual_buffer_source_t, 2> sources = {{
      {
          .device = device_,
          .memory_type = is_cpu() ? HRX_MEMORY_TYPE_HOST_LOCAL
                                  : HRX_MEMORY_TYPE_DEVICE_LOCAL,
      },
      {
          .device = device_,
          .memory_type = HRX_MEMORY_TYPE_HOST_LOCAL,
      },
  }};
  const hrx_virtual_buffer_sources_t accepted = {sources.size(),
                                                 sources.data()};
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(
      hrx().virtual_buffer_query_capabilities(scope, accepted, &capabilities));
  if (!capabilities.max_byte_length) {
    SKIP("The selected scope cannot map both declared backing classes.");
  }
  size_t unit = std::max(capabilities.reservation_granularity,
                         capabilities.mapping_granularity);
  for (const auto& source : sources) {
    hrx_memory_mapping_requirements_t requirements;
    REQUIRE_OK(hrx().allocator_query_mapping_requirements(
        hrx().device_allocator(source.device), source.memory_type,
        &requirements));
    unit = std::max(unit, requirements.recommended_granularity);
  }
  REQUIRE(unit > 0);
  VirtualMemoryResources resources(device_);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 3 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(scope, accepted, &options,
                                         &resources.reservation));
  for (size_t i = 0; i < sources.size(); ++i) {
    const hrx_buffer_params_t params = {
        .type = sources[i].memory_type,
        .access = HRX_MEMORY_ACCESS_ALL,
        .usage = HRX_BUFFER_USAGE_DEFAULT | HRX_BUFFER_USAGE_MAPPING_SOURCE,
    };
    REQUIRE_OK(hrx().allocator_allocate_buffer(
        hrx().device_allocator(device_), params, unit, &resources.sources[i]));
  }
  REQUIRE_OK(hrx().semaphore_create(device_, 0, &resources.completion));
  hrx_memory_scope_t device_scope;
  REQUIRE_OK(hrx().device_query_memory_scope(
      device_, 0, HRX_MEMORY_SITE_PROGRAM, &device_scope));
  std::array<hrx_memory_mapping_t, 3> mappings;
  for (size_t i = 0; i < mappings.size(); ++i) {
    mappings[i] = {
        .target = resources.reservation,
        .scope = i == 0 ? device_scope : hrx_memory_scope_t{},
        .target_offset = i * unit,
        .byte_length = unit,
        .source = resources.sources[i == 0 ? 0 : 1],
        .access = HRX_MEMORY_ACCESS_ALL,
        .operation = HRX_MEMORY_MAPPING_OPERATION_BIND,
    };
  }
  resources.Map(mappings.size(), mappings.data());
  void* host_address = nullptr;
  REQUIRE_OK(hrx().buffer_map(resources.reservation,
                              HRX_MAP_READ | HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS,
                              unit, 2 * unit, &host_address));
  resources.host_mapped = true;
  const std::array<uint32_t, 4> expected = {41, 43, 47, 53};
  std::memcpy(host_address, expected.data(), sizeof(expected));
  ++resources.completion_value;
  const hrx_semaphore_list_t signals = {
      .semaphores = &resources.completion,
      .values = &resources.completion_value,
      .count = 1,
  };
  REQUIRE_OK(hrx().queue_copy(device_, 0, nullptr, &signals,
                              resources.reservation, 2 * unit,
                              resources.reservation, 0, sizeof(expected)));
  REQUIRE_OK(hrx().semaphore_wait(resources.completion,
                                  resources.completion_value, UINT64_MAX));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(device_, resources.reservation, 0,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  const std::array<uint32_t, 4> update = {59, 61, 67, 71};
  REQUIRE_OK(hrx().synchronous_h2d(
      device_, update.data(), resources.reservation, 2 * unit, sizeof(update)));
  std::memcpy(observed.data(), host_address, sizeof(observed));
  CHECK(observed == update);
}
