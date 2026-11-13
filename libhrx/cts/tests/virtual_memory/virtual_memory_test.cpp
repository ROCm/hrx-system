// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <vector>

#include "hrx_test_fixture.hpp"

namespace {

class VirtualMemoryResources {
 public:
  explicit VirtualMemoryResources(hrx_device_t device)
      : device(device),
        group(hrx().device_get_device_group(device)),
        queue(FindProvisionedQueue(device, HRX_QUEUE_ROLE_TRANSFER)) {}

  ~VirtualMemoryResources() {
    bool retired = true;
    if (completion_value) {
      hrx_status_t status =
          hrx().semaphore_wait(completion, completion_value, UINT64_MAX);
      retired = hrx_status_is_ok(status);
      ReportHrxTestFailure("retire virtual memory work", status);
    }
    if (host_mapped) {
      ReportHrxTestFailure("unmap virtual buffer",
                           hrx().buffer_unmap(reservation));
    }
    bool detached = true;
    if (reservation) {
      hrx_status_t status = hrx().virtual_buffer_detach(reservation, {});
      detached = hrx_status_is_ok(status);
      ReportHrxTestFailure("detach virtual buffer", status);
      hrx().buffer_release(reservation);
    }
    // A failed completion or removal cannot authorize recycling mapped backing.
    if (retired && detached) {
      for (auto source : sources) {
        hrx().buffer_release(source);
      }
    }
    for (auto pool : pools) {
      hrx().mem_pool_release(pool);
    }
    hrx().semaphore_release(completion);
  }

  void Initialize() {
    hrx_accelerator_type_t type;
    REQUIRE_OK(hrx().device_get_type(device, &type));
    const auto interface = type == HRX_ACCELERATOR_CPU
                               ? HRX_BUFFER_INTERFACE_HOST
                               : HRX_BUFFER_INTERFACE_DEVICE_ADDRESS;
    for (uint32_t ordinal = 0;
         ordinal < hrx().device_queue_family_count(device); ++ordinal) {
      hrx_queue_family_info_t info;
      REQUIRE_OK(hrx().device_query_queue_family(device, ordinal, &info));
      hrx_buffer_usage_t usage = 0;
      if (info.roles & HRX_QUEUE_ROLE_TRANSFER) {
        usage |= HRX_BUFFER_USAGE_TRANSFER;
      }
      if (info.roles & HRX_QUEUE_ROLE_DISPATCH) {
        usage |= HRX_BUFFER_USAGE_DISPATCH;
      }
      families.push_back({
          .device = device,
          .family_ordinal = ordinal,
          .usage = usage,
          .interfaces = HRX_BUFFER_INTERFACE_BIT(interface),
      });
    }
    REQUIRE_OK(hrx().semaphore_create(device, 0, &completion));
  }

  hrx_pool_scope_t Scope(bool host_access = false) const {
    hrx_pool_scope_t scope = {
        .family_count = families.size(),
        .families = families.data(),
    };
    if (host_access) {
      scope.host = {
          .access = HRX_MEMORY_ACCESS_ALL,
          .modes = HRX_MAPPING_MODE_SCOPED | HRX_MAPPING_MODE_PERSISTENT,
      };
    }
    return scope;
  }

  void AddSource(bool host) {
    auto source_families = families;
    for (auto& family : source_families) {
      family.interfaces |=
          HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_MAPPING_SOURCE);
      if (host) {
        family.requirements |= HRX_POOL_ACCESS_REQUIRE_COHERENT_WITH_HOST;
      }
    }
    auto scope = Scope(host);
    scope.families = source_families.data();
    hrx_slab_pool_options_t options = {};
    if (!host) {
      for (size_t i = 0; i < hrx().device_group_topology_node_count(group);
           ++i) {
        hrx_topology_node_t node;
        REQUIRE_OK(hrx().device_group_query_topology_node(group, i, &node));
        if (node.kind == HRX_TOPOLOGY_NODE_KIND_PHYSICAL_DEVICE &&
            hrx().device_group_get_device(group, node.device_ordinal) ==
                device) {
          options.placement = {HRX_POOL_PLACEMENT_PREFERRED, node.ordinal};
          break;
        }
      }
    }
    REQUIRE(source_count < pools.size());
    REQUIRE_OK(
        hrx().slab_pool_create(group, scope, &options, &pools[source_count]));
    ++source_count;
  }

  hrx_virtual_buffer_sources_t Accepted() const {
    return {source_count, pools.data()};
  }

  void Allocate(size_t index, size_t size) {
    uint64_t next_value = completion_value + 1;
    const hrx_semaphore_list_t signals = {&completion, &next_value, 1};
    const hrx_pool_reservation_request_t request = {size, 0};
    REQUIRE_OK(hrx().queue_alloca(queue, nullptr, &signals, pools[index], 1,
                                  &request, &sources[index]));
    completion_value = next_value;
    REQUIRE_OK(hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
  }

  void Map(size_t count, const hrx_memory_mapping_t* mappings) {
    uint64_t next_value = completion_value + 1;
    const hrx_semaphore_list_t signals = {&completion, &next_value, 1};
    REQUIRE_OK(hrx().queue_map(queue, nullptr, &signals, count, mappings,
                               HRX_QUEUE_MAP_FLAG_NONE));
    completion_value = next_value;
    REQUIRE_OK(hrx().semaphore_wait(completion, completion_value, UINT64_MAX));
  }

  // Borrowed device and its sealed owner retained by the process fixture.
  hrx_device_t device;
  // Borrowed sealed owner for both reservation and backing construction.
  hrx_device_group_t group;
  // Borrowed execution queue whose device outlives these resources.
  hrx_queue_t queue;
  // Exact execution sites, borrowed by construction calls only.
  std::vector<hrx_pool_family_access_t> families;
  // Ordinary virtual buffer retaining the address interval.
  hrx_buffer_t reservation = nullptr;
  // Explicit source pools retained independently of their captured contracts.
  std::array<hrx_mem_pool_t, 2> pools = {};
  // Number of populated source pools.
  size_t source_count = 0;
  // Ordinary prepared backing retained until checked removal succeeds.
  std::array<hrx_buffer_t, 2> sources = {};
  // Completion for synchronous test observation of each accepted operation.
  hrx_semaphore_t completion = nullptr;
  // Last successfully accepted signal, including allocation and mapping.
  uint64_t completion_value = 0;
  // Whether the reservation has a live scoped public host mapping.
  bool host_mapped = false;
};

}  // namespace

TEST_CASE_METHOD(HrxTestFixture, "query reservation capabilities",
                 "[virtual_memory][query]") {
  VirtualMemoryResources resources(device_);
  resources.Initialize();
  resources.AddSource(is_cpu());
  const auto scope = resources.Scope();
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(
      resources.group, scope, resources.Accepted(), &capabilities));
  hrx_pool_stats_t stats;
  hrx().mem_pool_query_stats(resources.pools[0], &stats);
  CHECK(stats.reserve_count == 0);
  if (capabilities.max_byte_length) {
    CHECK(capabilities.reservation_granularity > 0);
    CHECK(capabilities.mapping_granularity > 0);
    CHECK(capabilities.operations & (1u << HRX_MEMORY_MAPPING_OPERATION_BIND));
    CHECK(capabilities.operations &
          (1u << HRX_MEMORY_MAPPING_OPERATION_UNBIND));
  } else {
    const hrx_virtual_buffer_options_t options = {.byte_length = 65536};
    hrx_buffer_t reservation = nullptr;
    hrx_status_t status = hrx().virtual_buffer_create(
        resources.group, scope, resources.Accepted(), &options, &reservation);
    CHECK(!hrx_status_is_ok(status));
    CHECK(reservation == nullptr);
    hrx().status_ignore(status);
  }
}

TEST_CASE_METHOD(HrxTestFixture,
                 "batched virtual aliases share ordinary backing",
                 "[virtual_memory][lifecycle]") {
  VirtualMemoryResources resources(device_);
  resources.Initialize();
  resources.AddSource(is_cpu());
  const auto scope = resources.Scope();
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(
      resources.group, scope, resources.Accepted(), &capabilities));
  if (!capabilities.max_byte_length) {
    SKIP("The selected device has no virtual buffer backend.");
  }
  const auto requirements =
      hrx().mem_pool_query_mapping_requirements(resources.pools[0]);
  const size_t unit = std::max({capabilities.reservation_granularity,
                                capabilities.mapping_granularity,
                                requirements.recommended_granularity});
  REQUIRE(unit > 0);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(resources.group, scope,
                                         resources.Accepted(), &options,
                                         &resources.reservation));
  resources.Allocate(0, unit);
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
  REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, expected.data(), view, 0,
                                   sizeof(expected)));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, view, unit, observed.data(),
                                   sizeof(observed)));
  CHECK(observed == expected);
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, resources.sources[0], 0,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  if (capabilities.operations & (1u << HRX_MEMORY_MAPPING_OPERATION_PROTECT)) {
    for (auto& mapping : mappings) {
      mapping.operation = HRX_MEMORY_MAPPING_OPERATION_PROTECT;
      mapping.access = HRX_MEMORY_ACCESS_READ;
    }
    resources.Map(mappings.size(), mappings.data());
    REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, view, unit,
                                     observed.data(), sizeof(observed)));
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
  VirtualMemoryResources resources(device_);
  resources.Initialize();
  resources.AddSource(true);
  const auto scope = resources.Scope(true);
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(
      resources.group, scope, resources.Accepted(), &capabilities));
  if (!capabilities.max_byte_length ||
      !(capabilities.supported_flags &
        HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS)) {
    SKIP("The selected scope cannot share host and program addresses.");
  }
  const auto requirements =
      hrx().mem_pool_query_mapping_requirements(resources.pools[0]);
  const size_t unit = std::max({capabilities.reservation_granularity,
                                capabilities.mapping_granularity,
                                requirements.recommended_granularity});
  REQUIRE(unit > 0);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 2 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
      .flags = HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(resources.group, scope,
                                         resources.Accepted(), &options,
                                         &resources.reservation));
  resources.Allocate(0, unit);
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
  hrx_memory_scope_t program_scope;
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      device_, hrx().queue_get_family_ordinal(transfer_queue_),
      HRX_MEMORY_SITE_PROGRAM, &program_scope));
  hrx_accelerator_type_t accelerator_type;
  REQUIRE_OK(hrx().device_get_type(device_, &accelerator_type));
  uint64_t program_address = 0;
  REQUIRE_OK(
      hrx().buffer_query_address(resources.reservation, program_scope,
                                 accelerator_type == HRX_ACCELERATOR_CPU
                                     ? HRX_BUFFER_INTERFACE_HOST
                                     : HRX_BUFFER_INTERFACE_DEVICE_ADDRESS,
                                 &program_address));
  void* host_address = nullptr;
  REQUIRE_OK(hrx().buffer_map(resources.reservation,
                              HRX_MAP_READ | HRX_MAP_WRITE | HRX_MAP_MAY_ALIAS,
                              0, 2 * unit, &host_address));
  resources.host_mapped = true;
  REQUIRE(reinterpret_cast<uintptr_t>(host_address) == program_address);
  const std::array<uint32_t, 4> expected = {11, 17, 29, 47};
  std::memcpy(host_address, expected.data(), sizeof(expected));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, resources.reservation, unit,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  const std::array<uint32_t, 4> update = {53, 71, 89, 97};
  REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, update.data(),
                                   resources.reservation, unit,
                                   sizeof(update)));
  std::memcpy(observed.data(), host_address, sizeof(observed));
  CHECK(observed == update);
}

TEST_CASE_METHOD(HrxTestFixture,
                 "mixed backing preserves scoped host and device access",
                 "[virtual_memory][mixed_backing]") {
  VirtualMemoryResources resources(device_);
  resources.Initialize();
  resources.AddSource(is_cpu());
  resources.AddSource(true);
  const auto scope = resources.Scope(true);
  const auto accepted = resources.Accepted();
  hrx_virtual_buffer_capabilities_t capabilities;
  REQUIRE_OK(hrx().virtual_buffer_query_capabilities(resources.group, scope,
                                                     accepted, &capabilities));
  if (!capabilities.max_byte_length) {
    SKIP("The selected scope cannot map both declared backing classes.");
  }
  size_t unit = std::max(capabilities.reservation_granularity,
                         capabilities.mapping_granularity);
  for (size_t i = 0; i < resources.source_count; ++i) {
    const auto requirements =
        hrx().mem_pool_query_mapping_requirements(resources.pools[i]);
    unit = std::max(unit, requirements.recommended_granularity);
  }
  REQUIRE(unit > 0);
  const hrx_virtual_buffer_options_t options = {
      .byte_length = 3 * unit,
      .min_alignment = unit,
      .mapping_granularity = unit,
  };
  REQUIRE_OK(hrx().virtual_buffer_create(resources.group, scope, accepted,
                                         &options, &resources.reservation));
  for (size_t i = 0; i < resources.source_count; ++i) {
    resources.Allocate(i, unit);
  }
  hrx_memory_scope_t device_scope;
  const hrx_queue_t dispatch_queue =
      FindProvisionedQueue(device_, HRX_QUEUE_ROLE_DISPATCH);
  REQUIRE_OK(hrx().device_query_queue_family_memory_scope(
      device_, hrx().queue_get_family_ordinal(dispatch_queue),
      HRX_MEMORY_SITE_PROGRAM, &device_scope));
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
  uint64_t copy_value = resources.completion_value + 1;
  const hrx_semaphore_list_t signals = {
      .semaphores = &resources.completion,
      .values = &copy_value,
      .count = 1,
  };
  REQUIRE_OK(hrx().queue_copy(
      transfer_queue_, nullptr, &signals, resources.reservation, 2 * unit,
      resources.reservation, 0, sizeof(expected), nullptr));
  resources.completion_value = copy_value;
  REQUIRE_OK(hrx().semaphore_wait(resources.completion,
                                  resources.completion_value, UINT64_MAX));
  std::array<uint32_t, 4> observed = {};
  REQUIRE_OK(hrx().synchronous_d2h(transfer_queue_, resources.reservation, 0,
                                   observed.data(), sizeof(observed)));
  CHECK(observed == expected);
  const std::array<uint32_t, 4> update = {59, 61, 67, 71};
  REQUIRE_OK(hrx().synchronous_h2d(transfer_queue_, update.data(),
                                   resources.reservation, 2 * unit,
                                   sizeof(update)));
  std::memcpy(observed.data(), host_address, sizeof(observed));
  CHECK(observed == update);
}
