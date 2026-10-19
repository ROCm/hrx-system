// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstring>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/vulkan/buffer.h"
#include "iree/hal/drivers/vulkan/queue.h"
#include "iree/hal/drivers/vulkan/queue_test_util.h"

namespace iree::hal::cts {
namespace {

class VulkanSparseBindingTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    sparse_queue_ = reinterpret_cast<iree_hal_vulkan_queue_t*>(
        FindVulkanSparseBindingQueue(device_));
    family_ = {
        .family = iree_hal_queue_family(transfer_queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE,
        .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_VULKAN_BUFFER,
    };
    IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
        device_group_, Scope(), iree_allocator_system(), &capabilities_));
    if (!capabilities_.max_byte_length) {
      GTEST_SKIP() << "Vulkan sparse residency and aliasing are unavailable";
    }
    ASSERT_NE(nullptr, sparse_queue_);
  }

  void TearDown() override {
    for (auto* reservation : reservations_) {
      iree_hal_buffer_release(reservation);
    }
    CtsTestBase::TearDown();
  }

  iree_hal_pool_scope_t Scope() const {
    return {.family_count = 1, .families = &family_};
  }

  iree_status_t Reserve(iree_device_size_t length,
                        iree_hal_buffer_t** out_buffer,
                        iree_device_size_t alignment = 0) {
    const iree_hal_virtual_buffer_options_t options = {
        .byte_length = length,
        .min_alignment = alignment,
    };
    iree_hal_buffer_t* reservation = nullptr;
    IREE_RETURN_IF_ERROR(
        iree_hal_virtual_buffer_create(device_group_, Scope(), &options,
                                       iree_allocator_system(), &reservation));
    reservations_.push_back(reservation);
    *out_buffer = reservation;
    return iree_ok_status();
  }

  void ExpectPage(iree_hal_buffer_t* buffer, iree_device_size_t offset,
                  uint32_t first_word) {
    const auto bytes = ReadBufferBytes(buffer, offset, 256);
    ASSERT_EQ(256u, bytes.size());
    for (size_t i = 0; i < bytes.size() / sizeof(uint32_t); ++i) {
      uint32_t actual = 0;
      memcpy(&actual, bytes.data() + i * sizeof(uint32_t), sizeof(actual));
      EXPECT_EQ(first_word + i, actual) << "word " << i;
    }
  }

  // Borrowed queue owned by the shared CTS device.
  iree_hal_vulkan_queue_t* sparse_queue_ = nullptr;
  // Owned virtual buffers released after completed native operations.
  std::vector<iree_hal_buffer_t*> reservations_;
  // Complete execution scope used for address geometry and construction.
  iree_hal_pool_family_access_t family_ = {};
  // Cold address-only geometry, queried without creating native resources.
  iree_hal_virtual_buffer_capabilities_t capabilities_ = {};
};

TEST_P(VulkanSparseBindingTest, DeferredMultiBufferBindingsAliasAndReplace) {
  iree_device_size_t granularity = capabilities_.mapping_granularity;
  const iree_hal_pool_family_access_t family = {
      .family = iree_hal_queue_family(transfer_queue_),
      .usage = family_.usage,
      .interfaces = UINT64_C(1) << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
  };
  const iree_hal_pool_scope_t scope = {.family_count = 1, .families = &family};
  iree_hal_slab_pool_options_t options;
  iree_hal_slab_pool_options_initialize(&options);
  Ref<iree_hal_pool_t> source_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(device_group_, scope, &options,
                                           iree_allocator_system(),
                                           source_pool.out()));
  const auto source_requirements =
      iree_hal_pool_query_mapping_requirements(source_pool);
  ASSERT_GT(source_requirements.minimum_granularity, 0u);
  EXPECT_EQ(source_requirements.minimum_granularity,
            source_requirements.recommended_granularity);
  EXPECT_EQ(0u, source_requirements.flags);
  granularity = iree_max(granularity, source_requirements.minimum_granularity);
  ASSERT_GE(granularity, 256u);

  std::array<iree_hal_buffer_t*, 2> targets = {};
  IREE_ASSERT_OK(Reserve(3 * granularity, &targets[0]));
  IREE_ASSERT_OK(Reserve(2 * granularity, &targets[1]));
  std::array<VkBuffer, 2> handles = {};
  uint32_t memory_type_bits = UINT32_MAX;
  for (size_t i = 0; i < targets.size(); ++i) {
    VkDeviceMemory unused_memory = VK_NULL_HANDLE;
    IREE_ASSERT_OK(
        iree_hal_vulkan_buffer_handle(targets[i], &unused_memory, &handles[i]));
    VkMemoryRequirements requirements = {};
    iree_vkGetBufferMemoryRequirements(IREE_VULKAN_DEVICE(&sparse_queue_->syms),
                                       sparse_queue_->logical_device,
                                       handles[i], &requirements);
    memory_type_bits &= requirements.memoryTypeBits;
    ASSERT_EQ(0u, granularity % requirements.alignment);
  }

  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool,
      iree_hal_make_pool_reservation_request(5 * granularity, 2 * granularity),
      iree_infinite_timeout(), backing.out()));
  // A pooled view supplies pages without owning another physical allocation.
  // The nonzero view origin must survive both provider wrapping and subspans.
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(backing, granularity, 4 * granularity,
                                         iree_allocator_system(),
                                         source.out()));
  iree_hal_memory_scope_t memory_scope;
  const iree_hal_memory_site_t site = {
      .kind = IREE_HAL_MEMORY_SITE_QUEUE,
      .family = family.family,
  };
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(device_group_, site,
                                                            &memory_scope));
  iree_hal_buffer_native_binding_slot_t slot;
  IREE_ASSERT_OK(iree_hal_pool_resolve_binding(
      source_pool, memory_scope, IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE,
      &slot));
  const auto binding = iree_hal_buffer_native_binding(source, slot);
  ASSERT_NE(nullptr, binding.mapping_source.descriptor);
  const auto native_source = iree_hal_vulkan_buffer_mapping_source_info(
      binding.mapping_source.descriptor);
  EXPECT_EQ(sparse_queue_->logical_device, native_source.logical_device);
  ASSERT_NE(0u, memory_type_bits & (1u << native_source.memory_type_index));
  const VkDeviceMemory source_memory = native_source.memory;
  ASSERT_TRUE(source_memory != VK_NULL_HANDLE);
  const VkDeviceSize source_offset = binding.mapping_source.offset;
  ASSERT_GE(source_offset, granularity);
  ASSERT_EQ(0u, source_offset % granularity);
  ASSERT_GE(native_source.byte_length, source_offset + 4 * granularity);

  std::vector<uint32_t> input(4 * granularity / sizeof(uint32_t));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0x12340000u + i;
  }
  SemaphoreList uploaded(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_update(
      transfer_queue_, {}, uploaded, input.data(), 0, source, 0,
      input.size() * sizeof(uint32_t), nullptr, IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(uploaded, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));

  // Two target resources share three physical pages. The first target has a
  // hole between its bindings; the second aliases the first source page.
  std::array<VkSparseMemoryBind, 4> binds = {{
      {
          .resourceOffset = 0,
          .size = granularity,
          .memory = source_memory,
          .memoryOffset = source_offset,
      },
      {
          .resourceOffset = 2 * granularity,
          .size = granularity,
          .memory = source_memory,
          .memoryOffset = source_offset + 2 * granularity,
      },
      {
          .resourceOffset = 0,
          .size = granularity,
          .memory = source_memory,
          .memoryOffset = source_offset + 3 * granularity,
      },
      {
          .resourceOffset = granularity,
          .size = granularity,
          .memory = source_memory,
          .memoryOffset = source_offset,
      },
  }};
  std::array<VkSparseBufferMemoryBindInfo, 2> buffers = {{
      {.buffer = handles[0], .bindCount = 2, .pBinds = binds.data()},
      {.buffer = handles[1], .bindCount = 2, .pBinds = binds.data() + 2},
  }};
  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_vulkan_queue_submit_sparse_bind(
      sparse_queue_, gate, mapped, buffers.size(), buffers.data()));
  // The caller's complete descriptor graph expires before its prerequisite.
  buffers = {};
  binds = {};
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(mapped, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(targets[0], 0, input[0]);
  ExpectPage(targets[0], 2 * granularity,
             input[2 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], 0, input[3 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], granularity, input[0]);

  std::array<uint32_t, 64> replacement;
  for (size_t i = 0; i < replacement.size(); ++i) {
    replacement[i] = 0xCAFE0000u + i;
  }
  SemaphoreList written(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_update(
      transfer_queue_, mapped, written, replacement.data(), 0, targets[0], 0,
      sizeof(replacement), nullptr, IREE_HAL_UPDATE_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(written, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(source, 0, replacement[0]);
  ExpectPage(targets[1], granularity, replacement[0]);

  binds[0] = {
      .resourceOffset = 0,
      .size = granularity,
      .memory = source_memory,
      .memoryOffset = source_offset + granularity,
  };
  binds[1] = {
      .resourceOffset = granularity,
      .size = granularity,
      .memory = source_memory,
      .memoryOffset = source_offset + 2 * granularity,
  };
  buffers = {{{.buffer = handles[0], .bindCount = 1, .pBinds = &binds[0]},
              {.buffer = handles[1], .bindCount = 1, .pBinds = &binds[1]}}};
  SemaphoreList replaced(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_vulkan_queue_submit_sparse_bind(
      sparse_queue_, written, replaced, buffers.size(), buffers.data()));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(replaced, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(targets[0], 0, input[granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], granularity,
             input[2 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[0], 2 * granularity,
             input[2 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], 0, input[3 * granularity / sizeof(uint32_t)]);
  ExpectPage(source, 0, replacement[0]);

  binds = {{{.resourceOffset = 0, .size = granularity},
            {.resourceOffset = 2 * granularity, .size = granularity},
            {.resourceOffset = 0, .size = granularity},
            {.resourceOffset = granularity, .size = granularity}}};
  buffers = {
      {{.buffer = handles[0], .bindCount = 2, .pBinds = binds.data()},
       {.buffer = handles[1], .bindCount = 2, .pBinds = binds.data() + 2}}};
  SemaphoreList detached(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_vulkan_queue_submit_sparse_bind(
      sparse_queue_, replaced, detached, buffers.size(), buffers.data()));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(detached, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(VulkanSparseBindingTest, ReservationScopeAndGeometryQualification) {
  iree_hal_pool_scope_t scope = Scope();
  scope.host.access = IREE_HAL_MEMORY_ACCESS_ALL;
  scope.host.modes = IREE_HAL_MAPPING_MODE_PERSISTENT;
  iree_hal_virtual_buffer_capabilities_t capabilities;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_capabilities(
      device_group_, scope, iree_allocator_system(), &capabilities));
  EXPECT_EQ(0u, capabilities.max_byte_length);
  iree_hal_virtual_buffer_options_t options = {
      .byte_length = capabilities_.reservation_granularity,
  };
  iree_hal_buffer_t* reservation = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_virtual_buffer_create(device_group_, scope, &options,
                                     iree_allocator_system(), &reservation));
  EXPECT_EQ(nullptr, reservation);
  options.byte_length += 1;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_virtual_buffer_create(device_group_, Scope(), &options,
                                     iree_allocator_system(), &reservation));
  EXPECT_EQ(nullptr, reservation);
}

TEST_P(VulkanSparseBindingTest, CoarseMappingUnitsPreserveExactTail) {
  auto source_family = family_;
  source_family.interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  const iree_hal_slab_pool_options_t pool_options = {};
  Ref<iree_hal_pool_t> source_pool;
  IREE_ASSERT_OK(iree_hal_slab_pool_create(
      device_group_, {1, &source_family, {}}, &pool_options,
      iree_allocator_system(), source_pool.out()));
  const auto requirements =
      iree_hal_pool_query_mapping_requirements(source_pool);
  const auto tail = iree_max(capabilities_.reservation_granularity,
                             requirements.minimum_granularity);
  const auto unit = 2 * tail;
  const auto length = unit + tail;
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source_pool, iree_hal_make_pool_reservation_request(length, tail),
      iree_infinite_timeout(), source.out()));
  const iree_hal_virtual_buffer_options_t options = {
      .byte_length = length,
      .mapping_granularity = unit,
  };
  Ref<iree_hal_buffer_t> target;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_create(
      device_group_, Scope(), &options, iree_allocator_system(), target.out()));
  const iree_hal_memory_scope_t all = {
      iree_hal_device_group_memory_domain(device_group_),
      IREE_HAL_MEMORY_SCOPE_ANY};
  iree_hal_memory_mapping_t mapping = {
      .target = target,
      .scope = all,
      .target_range = {0, length},
      .source = source,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
  };
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(
      iree_hal_queue_map(&sparse_queue_->base, {}, mapped, 1, &mapping, 0));
  std::vector<uint32_t> input(length / sizeof(uint32_t));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0x12340000u + i;
  }
  SemaphoreList uploaded(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_upload(transfer_queue_, mapped, uploaded,
                                       input.data(), source, 0, length,
                                       nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(uploaded, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(target, unit, input[unit / sizeof(uint32_t)]);
  mapping.target_range = {unit, tail};
  SemaphoreList replaced(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, uploaded, replaced, 1,
                                    &mapping, 0));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(replaced, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(target, 0, input[0]);
  ExpectPage(target, unit, input[0]);
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(target, all));
}

TEST_P(VulkanSparseBindingTest, QueueMapCapturesPendingAllocations) {
  iree_hal_pool_family_access_t source_family = family_;
  source_family.interfaces = UINT64_C(1)
                             << IREE_HAL_BUFFER_INTERFACE_MAPPING_SOURCE;
  const iree_hal_pool_scope_t source_scope = {
      .family_count = 1,
      .families = &source_family,
  };
  iree_hal_slab_pool_options_t pool_options;
  iree_hal_slab_pool_options_initialize(&pool_options);
  Ref<iree_hal_pool_t> source_pool;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(device_group_, source_scope, &pool_options,
                                iree_allocator_system(), source_pool.out()));
  const auto source_geometry =
      iree_hal_pool_query_mapping_requirements(source_pool);
  const auto granularity = iree_max(capabilities_.mapping_granularity,
                                    source_geometry.minimum_granularity);
  const auto alignment = iree_min(4 * granularity, capabilities_.max_alignment);
  std::array<iree_hal_buffer_t*, 2> targets = {};
  IREE_ASSERT_OK(Reserve(3 * granularity, &targets[0], alignment));
  IREE_ASSERT_OK(Reserve(2 * granularity, &targets[1], alignment));
  for (auto* target : targets) {
    const auto address =
        iree_hal_vulkan_buffer_native(target)
            ->bindings[IREE_HAL_VULKAN_BUFFER_BINDING_DEVICE_ADDRESS]
            .device_address +
        target->memory.binding_offset;
    EXPECT_EQ(0u, address & (alignment - 1));
  }
  iree_hal_memory_scope_t scope;
  const iree_hal_memory_site_t site = {
      .kind = IREE_HAL_MEMORY_SITE_QUEUE,
      .family = family_.family,
  };
  IREE_ASSERT_OK(
      iree_hal_device_group_resolve_memory_scope(device_group_, site, &scope));
  uint32_t projection = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations_[0], scope, &projection));
  EXPECT_EQ(0u, projection);
  iree_hal_memory_scope_t program_scope;
  const iree_hal_memory_site_t program_site = {
      .kind = IREE_HAL_MEMORY_SITE_PROGRAM,
      .family = family_.family,
  };
  IREE_ASSERT_OK(iree_hal_device_group_resolve_memory_scope(
      device_group_, program_site, &program_scope));
  uint32_t program_projection = UINT32_MAX;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_resolve_projection(
      reservations_[0], program_scope, &program_projection));
  EXPECT_EQ(projection, program_projection);
  EXPECT_EQ(IREE_HAL_MEMORY_MAPPING_OPERATIONS_ALL, capabilities_.operations);

  SemaphoreList gate(device_, {0}, {1});
  SemaphoreList allocated(device_, {0}, {1});
  const auto request =
      iree_hal_make_pool_reservation_request(5 * granularity, 2 * granularity);
  Ref<iree_hal_buffer_t> allocation;
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, gate, allocated,
                                       source_pool, 1, &request,
                                       allocation.out()));
  // Both the source view and mapping array exist before backing is staged.
  Ref<iree_hal_buffer_t> source;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(allocation, granularity, 4 * granularity,
                              iree_allocator_system(), source.out()));
  std::array<iree_hal_memory_mapping_t, 4> mappings = {{
      {
          .target = reservations_[0],
          .scope = {.domain = scope.domain, .id = IREE_HAL_MEMORY_SCOPE_ANY},
          .target_range =
              {
                  .byte_length = granularity,
              },
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_BIND,
      },
      {
          .target = reservations_[0],
          .scope = scope,
          .target_range =
              {
                  .byte_offset = 2 * granularity,
                  .byte_length = granularity,
              },
          .source = source,
          .source_offset = 2 * granularity,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
      {
          .target = reservations_[1],
          .scope = scope,
          .target_range =
              {
                  .byte_length = granularity,
              },
          .source = source,
          .source_offset = 3 * granularity,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
      {
          .target = reservations_[1],
          .scope = scope,
          .target_range =
              {
                  .byte_offset = granularity,
                  .byte_length = granularity,
              },
          .source = source,
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
      },
  }};
  auto unsupported = mappings[0];
  unsupported.access = IREE_HAL_MEMORY_ACCESS_READ;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_UNAVAILABLE,
      iree_hal_queue_map(&sparse_queue_->base, {}, {}, 1, &unsupported,
                         IREE_HAL_QUEUE_MAP_FLAG_NONE));
  SemaphoreList mapped(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, allocated, mapped,
                                    mappings.size(), mappings.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  mappings = {};
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate.semaphores[0], 1, nullptr));

  std::vector<uint32_t> input(4 * granularity / sizeof(uint32_t));
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = 0x56780000u + i;
  }
  SemaphoreList uploaded(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_upload(
      transfer_queue_, mapped, uploaded, input.data(), source, 0,
      input.size() * sizeof(uint32_t), nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(uploaded, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(targets[0], 0, input[0]);
  ExpectPage(targets[0], 2 * granularity,
             input[2 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], 0, input[3 * granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], granularity, input[0]);

  std::array<uint32_t, 64> output;
  for (size_t i = 0; i < output.size(); ++i) {
    output[i] = 0xFACE0000u + i;
  }
  SemaphoreList written(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_upload(transfer_queue_, uploaded, written,
                                       output.data(), targets[0], 0,
                                       sizeof(output), nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(written, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(source, 0, output[0]);
  ExpectPage(targets[1], granularity, output[0]);

  // Revoke access without retaining a native installed-source ledger. The
  // later protection operation supplies the same source explicitly.
  auto protection = unsupported;
  protection.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_PROTECT;
  protection.access = IREE_HAL_MEMORY_ACCESS_NONE;
  SemaphoreList revoked(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, written, revoked, 1,
                                    &protection,
                                    IREE_HAL_QUEUE_MAP_FLAG_BORROW_BUFFERS));
  protection.access = IREE_HAL_MEMORY_ACCESS_ALL;
  SemaphoreList restored(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, revoked, restored, 1,
                                    &protection, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(restored, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(targets[0], 0, output[0]);

  auto replacement = protection;
  replacement.operation = IREE_HAL_MEMORY_MAPPING_OPERATION_REPLACE;
  replacement.source_offset = granularity;
  SemaphoreList replaced(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, restored, replaced, 1,
                                    &replacement,
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(replaced, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  ExpectPage(targets[0], 0, input[granularity / sizeof(uint32_t)]);
  ExpectPage(targets[1], granularity, output[0]);

  // Checked full removal covers both mappings and holes while preserving the
  // ordinary buffer and its native resource for subsequent queue operations.
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkBuffer handle = VK_NULL_HANDLE;
  IREE_ASSERT_OK(iree_hal_vulkan_buffer_handle(targets[0], &memory, &handle));
  IREE_ASSERT_OK(iree_hal_virtual_buffer_detach(targets[0], scope));
  VkBuffer retained_handle = VK_NULL_HANDLE;
  IREE_ASSERT_OK(
      iree_hal_vulkan_buffer_handle(targets[0], &memory, &retained_handle));
  EXPECT_EQ(retained_handle, handle);
  iree_hal_virtual_buffer_info_t info;
  IREE_ASSERT_OK(iree_hal_virtual_buffer_query_info(targets[0], &info));
  EXPECT_EQ(info.mapping_granularity, capabilities_.mapping_granularity);

  // General detach may span installed pages and holes. Backing is released
  // only after the entire batch has detached and all readers have completed.
  std::array<iree_hal_memory_mapping_t, 2> detach = {{
      {
          .target = reservations_[0],
          .scope = scope,
          .target_range = {.byte_length = 3 * granularity},
      },
      {
          .target = reservations_[1],
          .scope = scope,
          .target_range = {.byte_length = 2 * granularity},
          .operation = IREE_HAL_MEMORY_MAPPING_OPERATION_UNBIND,
      },
  }};
  SemaphoreList detached(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, replaced, detached,
                                    detach.size(), detach.data(),
                                    IREE_HAL_QUEUE_MAP_FLAG_NONE));
  SemaphoreList deallocated(device_, {0}, {1});
  auto* raw_allocation = allocation.get();
  IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, detached, deallocated,
                                         1, &raw_allocation));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  source.reset();
  allocation.reset();
  SemaphoreList empty(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_map(&sparse_queue_->base, deallocated, empty, 0,
                                    nullptr, IREE_HAL_QUEUE_MAP_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(empty, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
}

CTS_REGISTER_TEST_SUITE(VulkanSparseBindingTest);

}  // namespace
}  // namespace iree::hal::cts
