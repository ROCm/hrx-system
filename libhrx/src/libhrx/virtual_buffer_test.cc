// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <vector>

#include "hrx_internal.h"
#include "hrx_runtime_cxx.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

// Owns an accepted mapping chain, including assertion exits before the host
// gate opens. Retire the chain and remove aliases before returning any backing.
struct VirtualBufferBatch {
  explicit VirtualBufferBatch(size_t count) : reservations(count) {}
  ~VirtualBufferBatch() {
    if (gate) {
      uint64_t value = 0;
      IREE_EXPECT_OK(hrx_status_to_iree(hrx_semaphore_query(gate, &value)));
      if (!value) {
        IREE_EXPECT_OK(hrx_status_to_iree(hrx_semaphore_signal(gate, 1)));
      }
    }
    bool retired = true;
    if (accepted) {
      auto status = hrx_semaphore_wait(completion, accepted, UINT64_MAX);
      retired = hrx_status_is_ok(status);
      IREE_EXPECT_OK(hrx_status_to_iree(status));
    }
    bool detached = true;
    for (auto reservation : reservations) {
      if (reservation) {
        auto status = hrx_virtual_buffer_detach(reservation, {});
        detached &= hrx_status_is_ok(status);
        IREE_EXPECT_OK(hrx_status_to_iree(status));
        hrx_buffer_release(reservation);
      }
    }
    if (retired && detached) {
      hrx_buffer_release(source);
    }
    hrx_semaphore_release(completion);
    hrx_semaphore_release(gate);
  }
  iree_status_t Record(hrx_status_t status, uint64_t value) {
    if (hrx_status_is_ok(status)) {
      accepted = value;
    }
    return hrx_status_to_iree(status);
  }
  // Virtual targets whose mappings must be removed before source release.
  std::vector<hrx_buffer_t> reservations;
  // Physical owner retained until all aliases are detached.
  hrx_buffer_t source = nullptr;
  // Host-controlled dependency opened even on assertion failure.
  hrx_semaphore_t gate = nullptr;
  // Native chain completion, retained through the final accepted operation.
  hrx_semaphore_t completion = nullptr;
  // Last successfully enqueued completion value, never an attempted value.
  uint64_t accepted = 0;
};

class CpuVirtualBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_runtime_create(
        nullptr, hrx_host_allocator_system(), runtime_.for_output())));
    const hrx_device_create_params_t params = {.uri = "task://"};
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_device_group_create(runtime_, 1, &params, group_.for_output())));
    device_ = hrx_device_group_get_device(group_, 0);
    family_ = {
        .device = device_,
        .family_ordinal = 0,
        .usage = HRX_BUFFER_USAGE_TRANSFER | HRX_BUFFER_USAGE_DISPATCH,
        .interfaces = HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_HOST),
    };
    scope_ = {
        .family_count = 1,
        .families = &family_,
        .host =
            {
                .access = HRX_MEMORY_ACCESS_ALL,
                .modes = HRX_MAPPING_MODE_SCOPED | HRX_MAPPING_MODE_PERSISTENT,
            },
    };
    auto source_family = family_;
    source_family.interfaces |=
        HRX_BUFFER_INTERFACE_BIT(HRX_BUFFER_INTERFACE_MAPPING_SOURCE);
    auto source_scope = scope_;
    source_scope.families = &source_family;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_slab_pool_create(
        group_, source_scope, nullptr, pool_.for_output())));
  }

  iree_status_t AllocateBacking(size_t size, hrx_buffer_t* out_buffer) {
    hrx::runtime::semaphore_ptr ready;
    IREE_RETURN_IF_ERROR(hrx_status_to_iree(
        hrx_semaphore_create(device_, 0, ready.for_output())));
    hrx_semaphore_t semaphore = ready;
    uint64_t value = 1;
    const hrx_semaphore_list_t signals = {&semaphore, &value, 1};
    const hrx_pool_reservation_request_t request = {size, 0};
    IREE_RETURN_IF_ERROR(hrx_status_to_iree(
        hrx_queue_alloca(hrx_device_get_queue(device_, 0, 0), nullptr, &signals,
                         pool_, 1, &request, out_buffer)));
    return hrx_status_to_iree(hrx_semaphore_wait(ready, value, UINT64_MAX));
  }

  // Shared execution services retained through the group's lifetime.
  hrx::runtime::runtime_ptr runtime_;
  // Explicit immutable device set used by reservations and sources.
  hrx::runtime::device_group_ptr group_;
  // Borrowed device in the sealed group.
  hrx_device_t device_ = nullptr;
  // Exact family whose execution interfaces the virtual addresses expose.
  hrx_pool_family_access_t family_ = {};
  // Virtual access contract borrowing family_.
  hrx_pool_scope_t scope_ = {};
  // Actual source owner whose captured mapping contract is admitted.
  hrx::runtime::mem_pool_ptr pool_;
};

TEST_F(CpuVirtualBufferTest, MemoryScopeRequiresAnExplicitValidSite) {
  hrx_memory_scope_t scope;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
          device_, UINT32_MAX, HRX_MEMORY_SITE_PROGRAM, &scope)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
          device_, 0, HRX_MEMORY_SITE_HOST, &scope)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
          device_, 0, static_cast<hrx_memory_site_t>(UINT32_MAX), &scope)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_device_query_host_memory_scope(nullptr, &scope)));
}

TEST_F(CpuVirtualBufferTest, BatchedAliasesCaptureDependenciesAndSources) {
  // Exercise inline and overflow translation through the same native lifecycle.
  for (size_t count : {2, 9}) {
    SCOPED_TRACE(count);
    const hrx_pool_scope_t scope = scope_;
    std::vector<hrx_mem_pool_t> source_pools(count, pool_);
    const hrx_virtual_buffer_sources_t accepted_sources = {count,
                                                           source_pools.data()};
    hrx_virtual_buffer_capabilities_t capabilities;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
        group_, scope, accepted_sources, &capabilities)));
    ASSERT_GT(capabilities.max_byte_length, 0u);
    EXPECT_EQ(0xFu, capabilities.operations);
    EXPECT_NE(capabilities.supported_flags &
                  HRX_VIRTUAL_BUFFER_FLAG_REQUIRE_COMMON_ADDRESS,
              0u);
    const auto requirements = hrx_mem_pool_query_mapping_requirements(pool_);
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
    VirtualBufferBatch batch(count);
    auto& reservations = batch.reservations;
    for (auto& reservation : reservations) {
      IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_create(
          group_, scope, accepted_sources, &options, &reservation)));
      hrx_virtual_buffer_info_t info;
      IREE_ASSERT_OK(hrx_status_to_iree(
          hrx_virtual_buffer_query_info(reservation, &info)));
      EXPECT_EQ(unit, info.mapping_granularity);
    }
    hrx_pool_stats_t before_allocation;
    hrx_mem_pool_query_stats(pool_, &before_allocation);
    EXPECT_EQ(0u, before_allocation.reservation_count);
    auto& source = batch.source;
    IREE_ASSERT_OK(AllocateBacking(2 * unit, &source));
    std::vector<uint32_t> contents(2 * unit / sizeof(uint32_t), 0xAABBCCDD);
    std::fill_n(contents.begin(), unit / sizeof(uint32_t), 0x10203040);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_h2d(hrx_device_get_queue(device_, 0, 0),
                            contents.data(), source, 0, 2 * unit)));

    auto& gate = batch.gate;
    auto& completion = batch.completion;
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
    IREE_ASSERT_OK(
        batch.Record(hrx_queue_map(hrx_device_get_queue(device_, 0, 0),
                                   &gate_list, &signal_list, count,
                                   mappings.data(), HRX_QUEUE_MAP_FLAG_NONE),
                     completed));
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
    hrx_memory_scope_t address_scope;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
            device_, 0, HRX_MEMORY_SITE_PROGRAM, &address_scope)));
    uint64_t device_pointer = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(hrx_buffer_query_address(
        first, address_scope, HRX_BUFFER_INTERFACE_HOST, &device_pointer)));
    EXPECT_EQ(0u, device_pointer % (2 * unit));
    const uint32_t pattern = 0x55667788;
    uint64_t predecessor = completed++;
    const hrx_semaphore_list_t wait_list = {
        .semaphores = &completion,
        .values = &predecessor,
        .count = 1,
    };
    IREE_ASSERT_OK(
        batch.Record(hrx_queue_fill(hrx_device_get_queue(device_, 0, 0),
                                    &wait_list, &signal_list, first, unit,
                                    sizeof(pattern), &pattern, sizeof(pattern)),
                     completed));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    uint32_t result = 0;
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), last, unit,
                            &result, sizeof(result))));
    EXPECT_EQ(pattern, result);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), source, unit,
                            &result, sizeof(result))));
    EXPECT_EQ(pattern, result);

    hrx_memory_scope_t queue_scope, program_scope, host_scope;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
            device_, 0, HRX_MEMORY_SITE_QUEUE, &queue_scope)));
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_device_query_queue_family_memory_scope(
            device_, 0, HRX_MEMORY_SITE_PROGRAM, &program_scope)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_device_query_host_memory_scope(device_, &host_scope)));
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
    IREE_ASSERT_OK(
        batch.Record(hrx_queue_map(hrx_device_get_queue(device_, 0, 0),
                                   &wait_list, &signal_list, 1, &replacement,
                                   HRX_QUEUE_MAP_FLAG_BORROW_BUFFERS),
                     completed));
    replacement.operation = HRX_MEMORY_MAPPING_OPERATION_PROTECT;
    replacement.previous_source = nullptr;
    replacement.previous_source_offset = 0;
    replacement.access = HRX_MEMORY_ACCESS_READ;
    predecessor = completed++;
    IREE_ASSERT_OK(batch.Record(
        hrx_queue_map(hrx_device_get_queue(device_, 0, 0), &wait_list,
                      &signal_list, 1, &replacement, HRX_QUEUE_MAP_FLAG_NONE),
        completed));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), first, unit,
                            &result, sizeof(result))));
    EXPECT_EQ(0x10203040u, result);
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_synchronous_d2h(hrx_device_get_queue(device_, 0, 0), last, unit,
                            &result, sizeof(result))));
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
    IREE_ASSERT_OK(
        batch.Record(hrx_queue_map(hrx_device_get_queue(device_, 0, 0),
                                   &wait_list, &signal_list, count,
                                   mappings.data(), HRX_QUEUE_MAP_FLAG_NONE),
                     completed));
    predecessor = completed++;
    IREE_ASSERT_OK(batch.Record(
        hrx_queue_map(hrx_device_get_queue(device_, 0, 0), &wait_list,
                      &signal_list, 0, nullptr, HRX_QUEUE_MAP_FLAG_NONE),
        completed));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
    auto retained = hrx::runtime::buffer_ptr::borrow(first);
    for (auto& reservation : reservations) {
      hrx_buffer_release(reservation);
      reservation = nullptr;
    }
    size_t retained_size = 0;
    IREE_ASSERT_OK(
        hrx_status_to_iree(hrx_buffer_get_size(first, &retained_size)));
    EXPECT_EQ(2 * unit, retained_size);
    retained.reset();
    predecessor = completed++;
    IREE_ASSERT_OK(
        batch.Record(hrx_queue_dealloca(hrx_device_get_queue(device_, 0, 0),
                                        &wait_list, &signal_list, 1, &source),
                     completed));
    IREE_ASSERT_OK(hrx_status_to_iree(
        hrx_semaphore_wait(completion, completed, UINT64_MAX)));
  }
}

TEST_F(CpuVirtualBufferTest, ValidatesConstructionBeforeAcquiringAddresses) {
  hrx_virtual_buffer_capabilities_t capabilities;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(nullptr, {}, {},
                                                               &capabilities)));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  std::array<hrx_pool_family_access_t, 2> repeated = {family_, family_};
  const hrx_pool_scope_t scope = {
      .family_count = repeated.size(),
      .families = repeated.data(),
  };
  const hrx_mem_pool_t source_pool = pool_;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
          group_, scope, {1, &source_pool}, &capabilities)));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  hrx_buffer_t reservation = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_virtual_buffer_create(
          group_, scope, {1, &source_pool}, nullptr, &reservation)));
  EXPECT_EQ(nullptr, reservation);

  const hrx_pool_scope_t valid_scope = scope_;
  const std::array<hrx_mem_pool_t, 2> incomplete_sources = {pool_, nullptr};
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
            group_, valid_scope, sources, &capabilities)));
    EXPECT_EQ(0u, capabilities.max_byte_length);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INVALID_ARGUMENT,
        hrx_status_to_iree(hrx_virtual_buffer_create(
            group_, valid_scope, sources, &options, &reservation)));
    EXPECT_EQ(nullptr, reservation);
  }
  hrx::runtime::device_group_ptr foreign_group;
  const hrx_device_create_params_t task = {.uri = "task://"};
  IREE_ASSERT_OK(hrx_status_to_iree(
      hrx_device_group_create(runtime_, 1, &task, foreign_group.for_output())));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
          foreign_group, {}, {1, &source_pool}, &capabilities)));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  hrx_buffer_release(nullptr);
}

TEST_F(CpuVirtualBufferTest, HostRegistrationCannotBecomeADetach) {
  const hrx_pool_scope_t scope = scope_;
  const hrx_mem_pool_t source_pool = pool_;
  hrx_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_query_capabilities(
      group_, scope, {1, &source_pool}, &capabilities)));
  const hrx_virtual_buffer_options_t options = {
      .byte_length = capabilities.reservation_granularity,
  };
  hrx_buffer_t reservation = nullptr;
  IREE_ASSERT_OK(hrx_status_to_iree(hrx_virtual_buffer_create(
      group_, scope, {1, &source_pool}, &options, &reservation)));
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
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        hrx_status_to_iree(hrx_queue_map(
                            hrx_device_get_queue(device_, 0, 0), nullptr,
                            nullptr, 1, &mapping, HRX_QUEUE_MAP_FLAG_NONE)));
  IREE_ASSERT_OK(
      hrx_status_to_iree(hrx_host_memory_unregister(device_, storage.data())));
  hrx_buffer_release(reservation);
}

}  // namespace
