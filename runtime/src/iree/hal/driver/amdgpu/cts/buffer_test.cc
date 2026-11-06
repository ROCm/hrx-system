// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/driver/amdgpu/buffer.h"

#include <array>
#include <cstring>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/memory/tlsf_pool.h"

namespace iree::hal::cts {

static constexpr iree_hal_buffer_native_binding_slot_t kDeviceAddressSlot = {
    IREE_HAL_AMDGPU_BUFFER_BINDING_DEVICE_ADDRESS,
    IREE_HAL_BUFFER_INTERFACE_DEVICE_ADDRESS,
};

class AmdgpuBufferTest : public CtsTestBase<> {
 protected:
  void SetUp() override {
    CtsTestBase::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    dispatch_queue_ =
        QueueForCommandCategories(IREE_HAL_COMMAND_CATEGORY_DISPATCH);
    ASSERT_NE(dispatch_queue_, nullptr);
    ASSERT_NE(transfer_queue_, nullptr);
    LoadExecutableOrSkipUnsupported(
        "command_buffer_dispatch_constants_bindings_test.bin", &executable_);
  }

  void TearDown() override {
    iree_hal_executable_release(executable_);
    executable_ = nullptr;
    CtsTestBase::TearDown();
  }

  // Borrowed queue from the shared CTS device.
  iree_hal_queue_t* dispatch_queue_ = nullptr;
  // Native executable retained for this test.
  iree_hal_executable_t* executable_ = nullptr;
};

TEST_P(AmdgpuBufferTest, NativeBindingsExecuteFromInteriorPoolRanges) {
  const iree_hal_buffer_usage_t usage =
      IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE |
      IREE_HAL_BUFFER_USAGE_DISPATCH_INDIRECT_PARAMETERS;
  std::array<iree_hal_pool_family_access_t, 2> families = {};
  families[0].family = iree_hal_queue_family(transfer_queue_);
  families[0].usage = usage;
  families[1].family = iree_hal_queue_family(dispatch_queue_);
  families[1].usage = usage;
  const iree_hal_pool_scope_t scope = {
      families[0].family == families[1].family ? 1u : 2u, families.data(), {}};
  iree_hal_slab_pool_options_t source_options;
  iree_hal_slab_pool_options_initialize(&source_options);
  Ref<iree_hal_pool_t> source;
  IREE_ASSERT_OK(
      iree_hal_slab_pool_create(device_group_, scope, &source_options,
                                iree_allocator_system(), source.out()));
  Ref<iree_hal_buffer_t> backing;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      source, iree_hal_make_pool_reservation_request(8192, /*min_alignment=*/1),
      iree_infinite_timeout(), backing.out()));
  iree_hal_tlsf_pool_options_t options = {};
  options.tlsf_options.frontier_capacity = 4;
  Ref<iree_hal_pool_t> arena_pool;
  IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
      backing, 256, 7168, &options, iree_allocator_system(), arena_pool.out()));
  for (bool queued_arena : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "queued arena: " << queued_arena);
    Ref<iree_hal_buffer_t> arena;
    if (queued_arena) {
      const iree_hal_pool_reservation_request_t request =
          iree_hal_make_pool_reservation_request(/*allocation_size=*/4096,
                                                 /*min_alignment=*/1);
      SemaphoreList allocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_alloca(
          transfer_queue_, iree_hal_semaphore_list_empty(), allocated,
          arena_pool, 1, &request, arena.out()));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          allocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    } else {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          arena_pool,
          iree_hal_make_pool_reservation_request(4096, /*min_alignment=*/1),
          iree_infinite_timeout(), arena.out()));
    }
    Ref<iree_hal_pool_t> pool;
    IREE_ASSERT_OK(iree_hal_tlsf_pool_create_from_buffer(
        arena, 256, 3584, &options, iree_allocator_system(), pool.out()));

    enum class DispatchMode {
      kQueue,
      kRecorded,
      kReusable,
      kBindingTable,
      kIndirectParameters,
    };
    for (bool queued : {false, true}) {
      SCOPED_TRACE(queued);
      for (auto mode : {DispatchMode::kQueue, DispatchMode::kRecorded,
                        DispatchMode::kReusable, DispatchMode::kBindingTable,
                        DispatchMode::kIndirectParameters}) {
        SCOPED_TRACE(static_cast<int>(mode));
        const std::array<uint32_t, 8> input = {11, 22, 1, 2, 30, 400, 77, 88};
        const std::array<uint32_t, 8> initial_output = {11, 22, 99, 99,
                                                        99, 99, 77, 88};
        const std::array<uint32_t, 3> workgroups = {1, 1, 1};
        std::array<Ref<iree_hal_buffer_t>, 3> roots;
        std::array<Ref<iree_hal_buffer_t>, 2> views;
        SemaphoreList gate(device_, {0}, {1});
        SemaphoreList allocated(device_, {0}, {1});
        if (queued) {
          std::array<iree_hal_pool_reservation_request_t, 3> requests = {};
          for (auto& request : requests) {
            request.allocation_size = sizeof(input);
            request.min_alignment = 1;
          }
          iree_hal_buffer_t* buffers[3] = {};
          IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, gate, allocated,
                                               pool, requests.size(),
                                               requests.data(), buffers));
          for (size_t i = 0; i < roots.size(); ++i) {
            roots[i].reset(buffers[i]);
          }
        } else {
          for (auto& root : roots) {
            IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
                pool,
                iree_hal_make_pool_reservation_request(sizeof(input),
                                                       /*min_alignment=*/1),
                iree_infinite_timeout(), root.out()));
          }
        }
        // The unsignaled gate keeps queued storage uncommitted while both
        // levels of ordinary views capture their immutable binding-table
        // pointer.
        for (size_t i = 0; i < views.size(); ++i) {
          Ref<iree_hal_buffer_t> outer;
          IREE_ASSERT_OK(iree_hal_buffer_subspan(
              roots[i], sizeof(uint32_t), 6 * sizeof(uint32_t),
              iree_allocator_system(), outer.out()));
          IREE_ASSERT_OK(iree_hal_buffer_subspan(
              outer, sizeof(uint32_t), 4 * sizeof(uint32_t),
              iree_allocator_system(), views[i].out()));
          EXPECT_EQ(views[i].get()->memory.bindings,
                    roots[i].get()->memory.bindings);
          iree_hal_buffer_mapping_t mapping = {};
          IREE_EXPECT_STATUS_IS(
              IREE_STATUS_PERMISSION_DENIED,
              iree_hal_buffer_map_range(views[i], IREE_HAL_MAPPING_MODE_SCOPED,
                                        IREE_HAL_MEMORY_ACCESS_READ,
                                        IREE_HAL_BUFFER_MAP_FLAG_NONE, 0,
                                        IREE_HAL_WHOLE_BUFFER, &mapping));
        }
        IREE_ASSERT_OK(iree_hal_semaphore_list_signal(gate, nullptr));
        const iree_hal_semaphore_list_t ready =
            queued ? static_cast<iree_hal_semaphore_list_t>(allocated)
                   : iree_hal_semaphore_list_empty();
        SemaphoreList uploaded(device_, {0, 0, 0}, {1, 1, 1});
        const void* upload_data[] = {input.data(), initial_output.data(),
                                     workgroups.data()};
        const iree_device_size_t upload_lengths[] = {
            sizeof(input), sizeof(initial_output), sizeof(workgroups)};
        for (size_t i = 0; i < roots.size(); ++i) {
          const iree_hal_semaphore_list_t done = {1, &uploaded.semaphores[i],
                                                  &uploaded.payload_values[i]};
          IREE_ASSERT_OK(iree_hal_queue_upload(
              transfer_queue_, ready, done, upload_data[i], roots[i], 0,
              upload_lengths[i], /*barriers=*/NULL));
        }

        iree_hal_buffer_ref_t refs[2];
        refs[0] = iree_hal_make_buffer_ref(views[0], 0, 4 * sizeof(uint32_t));
        refs[1] = iree_hal_make_buffer_ref(views[1], 0, 4 * sizeof(uint32_t));
        const iree_hal_buffer_ref_list_t bindings = {2, refs};
        const uint32_t constants[] = {3, 10};
        const auto constant_data =
            iree_make_const_byte_span(constants, sizeof(constants));
        SemaphoreList executed(device_, {0}, {1});
        if (mode == DispatchMode::kRecorded ||
            mode == DispatchMode::kReusable ||
            mode == DispatchMode::kBindingTable) {
          Ref<iree_hal_command_buffer_t> commands;
          iree_hal_buffer_binding_t entries[2];
          auto table = iree_hal_buffer_binding_table_empty();
          if (mode == DispatchMode::kBindingTable) {
            refs[0] =
                iree_hal_make_indirect_buffer_ref(0, 0, 4 * sizeof(uint32_t));
            refs[1] =
                iree_hal_make_indirect_buffer_ref(1, 0, 4 * sizeof(uint32_t));
            entries[0] = {views[0], 0, 4 * sizeof(uint32_t)};
            entries[1] = {views[1], 0, 4 * sizeof(uint32_t)};
            table = {2, entries};
          }
          if (mode == DispatchMode::kRecorded ||
              mode == DispatchMode::kReusable) {
            // Static recording captures native identity after allocation.
            // Binding-table recording only captures slots and needs no wait.
            IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
                ready, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
          }
          IREE_ASSERT_OK(CreateCommandBuffer(
              mode == DispatchMode::kReusable
                  ? IREE_HAL_COMMAND_BUFFER_MODE_DEFAULT
                  : IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
              IREE_HAL_COMMAND_CATEGORY_DISPATCH, table.count, commands.out()));
          IREE_ASSERT_OK(iree_hal_command_buffer_begin(commands));
          IREE_ASSERT_OK(iree_hal_command_buffer_dispatch(
              commands, executable_, iree_hal_executable_function_from_index(0),
              iree_hal_make_static_dispatch_config(1, 1, 1), constant_data,
              bindings, IREE_HAL_DISPATCH_FLAG_NONE));
          IREE_ASSERT_OK(iree_hal_command_buffer_end(commands));
          IREE_ASSERT_OK(iree_hal_queue_execute(
              dispatch_queue_, uploaded, executed, commands, table,
              IREE_HAL_QUEUE_EXECUTE_FLAG_NONE));
        } else {
          auto config = iree_hal_make_static_dispatch_config(1, 1, 1);
          iree_hal_dispatch_flags_t flags = IREE_HAL_DISPATCH_FLAG_NONE;
          if (mode == DispatchMode::kIndirectParameters) {
            config.workgroup_count_ref =
                iree_hal_make_buffer_ref(roots[2], 0, sizeof(workgroups));
            flags |= IREE_HAL_DISPATCH_FLAG_DYNAMIC_INDIRECT_PARAMETERS;
          }
          IREE_ASSERT_OK(iree_hal_queue_dispatch(
              dispatch_queue_, uploaded, executed, executable_,
              iree_hal_executable_function_from_index(0), config, constant_data,
              bindings, /*barriers=*/NULL, flags));
        }
        std::array<uint32_t, 8> result = {};
        SemaphoreList downloaded(device_, {0}, {1});
        IREE_ASSERT_OK(iree_hal_queue_download(
            transfer_queue_, executed, downloaded, roots[1], 0, result.data(),
            sizeof(result), /*barriers=*/NULL));
        IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
            downloaded, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        EXPECT_THAT(result,
                    ::testing::ElementsAre(11, 22, 13, 16, 100, 1210, 77, 88));
        for (size_t i = 0; i < views.size(); ++i) {
          const auto root_binding =
              iree_hal_buffer_native_binding(roots[i], kDeviceAddressSlot);
          const auto view_binding =
              iree_hal_buffer_native_binding(views[i], kDeviceAddressSlot);
          EXPECT_NE(root_binding.device_address, 0u);
          EXPECT_EQ(view_binding.device_address,
                    root_binding.device_address + 2 * sizeof(uint32_t));
          EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(views[i]),
                    iree_hal_amdgpu_buffer_atomic_memory_cells(backing));
        }
        if (queued) {
          iree_hal_buffer_t* buffers[] = {roots[0], roots[1], roots[2]};
          SemaphoreList deallocated(device_, {0}, {1});
          IREE_ASSERT_OK(
              iree_hal_queue_dealloca(transfer_queue_, downloaded, deallocated,
                                      IREE_ARRAYSIZE(buffers), buffers));
          IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
              deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
        }
      }
    }
    pool.reset();
    if (queued_arena) {
      iree_hal_buffer_t* buffers[] = {arena};
      SemaphoreList deallocated(device_, {0}, {1});
      IREE_ASSERT_OK(iree_hal_queue_dealloca(
          transfer_queue_, iree_hal_semaphore_list_empty(), deallocated,
          IREE_ARRAYSIZE(buffers), buffers));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
  }
}

TEST(AmdgpuBufferNativeTest, HostMappingAndExportUseIndependentHostAddress) {
  std::array<uint8_t, 256> host_storage = {};
  // The wrapper treats this agent address as opaque and never dereferences it.
  constexpr uint64_t kAgentAddress = 0x100000;
  int release_count = 0;
  const iree_hal_buffer_release_callback_t release_callback = {
      [](void* user_data, iree_hal_buffer_t*) {
        ++*static_cast<int*>(user_data);
      },
      &release_count,
  };
  Ref<iree_hal_buffer_t> root;
  IREE_ASSERT_OK(iree_hal_amdgpu_buffer_create(
      nullptr, iree_hal_buffer_placement_undefined(),
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
      IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_MAPPING,
      IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL, host_storage.size(),
      host_storage.size(), reinterpret_cast<void*>(kAgentAddress),
      host_storage.data(), release_callback, iree_allocator_system(),
      root.out()));
  Ref<iree_hal_buffer_t> outer;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(root, 32, 128, iree_allocator_system(),
                                         outer.out()));
  Ref<iree_hal_buffer_t> view;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(outer, 16, 64, iree_allocator_system(),
                                         view.out()));
  EXPECT_EQ(
      iree_hal_buffer_native_binding(view, kDeviceAddressSlot).device_address,
      kAgentAddress + 48);
  iree_byte_span_t host_span = iree_byte_span_empty();
  IREE_ASSERT_OK(iree_hal_buffer_native_host_span(view, 4, 16, &host_span));
  EXPECT_EQ(host_span.data, host_storage.data() + 52);
  EXPECT_EQ(host_span.data_length, 16u);
  iree_hal_buffer_mapping_t mapping = {};
  IREE_ASSERT_OK(iree_hal_buffer_map_range(
      view, IREE_HAL_MAPPING_MODE_SCOPED, IREE_HAL_MEMORY_ACCESS_WRITE,
      IREE_HAL_BUFFER_MAP_FLAG_NONE, 4, 16, &mapping));
  EXPECT_EQ(mapping.contents.data, host_span.data);
  memset(mapping.contents.data, 0xAB, mapping.contents.data_length);
  IREE_ASSERT_OK(iree_hal_buffer_unmap_range(&mapping));
  for (size_t i = 0; i < host_storage.size(); ++i) {
    EXPECT_EQ(host_storage[i], i >= 52 && i < 68 ? 0xAB : 0);
  }
  iree_hal_external_storage_t exported = {};
  iree_hal_buffer_range_t logical_range = {};
  IREE_ASSERT_OK(iree_hal_buffer_export(
      view, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
      &exported, &logical_range));
  EXPECT_EQ(exported.payload.host_pointer, host_storage.data());
  EXPECT_EQ(exported.byte_length, host_storage.size());
  EXPECT_EQ(logical_range.byte_offset, 48u);
  EXPECT_EQ(logical_range.byte_length, 64u);
  iree_hal_external_storage_release(&exported);
  IREE_ASSERT_OK(iree_hal_buffer_export(
      view, iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER),
      iree_hal_make_external_storage_class(
          IREE_HAL_EXTERNAL_STORAGE_TYPE_DEVICE_ADDRESS),
      &exported, &logical_range));
  EXPECT_EQ(exported.payload.device_address, kAgentAddress);
  EXPECT_EQ(logical_range.byte_offset, 48u);
  EXPECT_EQ(logical_range.byte_length, 64u);
  iree_hal_external_storage_release(&exported);
  EXPECT_EQ(iree_hal_amdgpu_buffer_atomic_memory_cells(view),
            IREE_HAL_AMDGPU_ATOMIC_MEMORY_CELL_FLAGS_ALL);
  view.reset();
  outer.reset();
  root.reset();
  EXPECT_EQ(release_count, 1);
}

CTS_REGISTER_EXECUTABLE_TEST_SUITE(AmdgpuBufferTest);

}  // namespace iree::hal::cts
