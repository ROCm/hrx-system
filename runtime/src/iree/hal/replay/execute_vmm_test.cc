// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <vector>

#include "iree/async/frontier_tracker.h"
#include "iree/hal/replay/execute.h"
#include "iree/hal/replay/file_reader.h"
#include "iree/hal/replay/recorder.h"
#include "iree/hal/testing/mock_device.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

typedef struct VmmAllocatorState {
  // Number of granularity queries received by this allocator.
  iree_host_size_t query_count = 0;
  // Number of successful reservation calls received by this allocator.
  iree_host_size_t reserve_count = 0;
  // Number of successful reservation release calls received by this allocator.
  iree_host_size_t release_count = 0;
  // Number of successful physical allocation calls received by this allocator.
  iree_host_size_t physical_allocate_count = 0;
  // Number of successful physical free calls received by this allocator.
  iree_host_size_t physical_free_count = 0;
  // Number of successful mapping calls received by this allocator.
  iree_host_size_t map_count = 0;
  // Number of successful unmapping calls received by this allocator.
  iree_host_size_t unmap_count = 0;
  // Number of successful protection calls received by this allocator.
  iree_host_size_t protect_count = 0;
  // Number of successful advice calls received by this allocator.
  iree_host_size_t advise_count = 0;
  // Live virtual reservation, or NULL after its consuming release.
  iree_hal_buffer_t* virtual_buffer = nullptr;
  // Live opaque physical allocation, or NULL after its consuming free.
  iree_hal_physical_memory_t* physical_memory = nullptr;
  // Whether the live physical allocation is mapped into the reservation.
  bool is_mapped = false;
  // Queue affinity from the most recent reservation.
  iree_hal_queue_affinity_t reserve_queue_affinity = 0;
  // Byte size from the most recent reservation.
  iree_device_size_t reserve_size = 0;
  // Parameters from the most recent physical allocation.
  iree_hal_buffer_params_t physical_params = {};
  // Byte size from the most recent physical allocation.
  iree_device_size_t physical_size = 0;
  // Virtual byte offset from the most recent mapping.
  iree_device_size_t map_virtual_offset = 0;
  // Physical byte offset from the most recent mapping.
  iree_device_size_t map_physical_offset = 0;
  // Byte size from the most recent mapping.
  iree_device_size_t map_size = 0;
  // Virtual byte offset from the most recent unmapping.
  iree_device_size_t unmap_virtual_offset = 0;
  // Byte size from the most recent unmapping.
  iree_device_size_t unmap_size = 0;
  // Queue affinity from the most recent protection update.
  iree_hal_queue_affinity_t protect_queue_affinity = 0;
  // Execution access scope from the most recent protection update.
  iree_hal_virtual_memory_access_scope_t protect_access_scope = 0;
  // Memory protection from the most recent protection update.
  iree_hal_memory_protection_t protection = 0;
  // Queue affinity from the most recent memory advice call.
  iree_hal_queue_affinity_t advise_queue_affinity = 0;
  // Memory advice from the most recent memory advice call.
  iree_hal_memory_advice_t advice = 0;
} VmmAllocatorState;

typedef struct VmmTestPhysicalMemory {
  // Host allocator used for this test handle's lifetime.
  iree_allocator_t host_allocator;
} VmmTestPhysicalMemory;

typedef struct VmmTestAllocator {
  // HAL resource header for allocator lifetime management.
  iree_hal_resource_t resource;
  // Host allocator used for this allocator's lifetime.
  iree_allocator_t host_allocator;
  // Heap allocator used only to materialize reservation-shaped buffers.
  iree_hal_allocator_t* heap_allocator;
  // Externally owned observation state for the test lifetime.
  VmmAllocatorState* state;
} VmmTestAllocator;

extern const iree_hal_allocator_vtable_t vmm_test_allocator_vtable;

static VmmTestAllocator* vmm_test_allocator_cast(
    iree_hal_allocator_t* base_allocator) {
  IREE_HAL_ASSERT_TYPE(base_allocator, &vmm_test_allocator_vtable);
  return reinterpret_cast<VmmTestAllocator*>(base_allocator);
}

static void vmm_test_allocator_destroy(iree_hal_allocator_t* base_allocator) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->host_allocator;
  iree_hal_allocator_release(allocator->heap_allocator);
  iree_allocator_free(host_allocator, allocator);
}

static iree_allocator_t vmm_test_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  const VmmTestAllocator* allocator =
      reinterpret_cast<const VmmTestAllocator*>(base_allocator);
  return allocator->host_allocator;
}

static bool vmm_test_allocator_supports_virtual_memory(
    iree_hal_allocator_t* base_allocator) {
  return true;
}

static iree_status_t vmm_test_allocator_query_granularity(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  ++allocator->state->query_count;
  *out_minimum_page_size = 4096;
  *out_recommended_page_size = 65536;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_affinity_t queue_affinity, iree_device_size_t size,
    iree_hal_buffer_t** out_virtual_buffer) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (state->virtual_buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test allocator already has a reservation");
  }
  iree_hal_buffer_params_t params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*.type=*/
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.queue_affinity=*/queue_affinity,
      /*.min_alignment=*/0,
  };
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator->heap_allocator, params, size, out_virtual_buffer));
  state->virtual_buffer = *out_virtual_buffer;
  state->reserve_queue_affinity = queue_affinity;
  state->reserve_size = size;
  ++state->reserve_count;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_release(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test reservation handle mismatch");
  }
  if (state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test reservation is still mapped");
  }
  state->virtual_buffer = nullptr;
  ++state->release_count;
  iree_hal_buffer_release(virtual_buffer);
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_physical_memory_allocate(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (state->physical_memory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test allocator already has physical memory");
  }
  VmmTestPhysicalMemory* physical_memory = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, sizeof(*physical_memory),
                            reinterpret_cast<void**>(&physical_memory)));
  physical_memory->host_allocator = host_allocator;
  state->physical_memory =
      reinterpret_cast<iree_hal_physical_memory_t*>(physical_memory);
  state->physical_params = params;
  state->physical_size = size;
  ++state->physical_allocate_count;
  *out_physical_memory = state->physical_memory;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_physical_memory_free(
    iree_hal_allocator_t* base_allocator,
    iree_hal_physical_memory_t* physical_memory) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (physical_memory != state->physical_memory) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "test physical memory handle mismatch");
  }
  if (state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test physical memory is still mapped");
  }
  VmmTestPhysicalMemory* test_physical_memory =
      reinterpret_cast<VmmTestPhysicalMemory*>(physical_memory);
  iree_allocator_t host_allocator = test_physical_memory->host_allocator;
  state->physical_memory = nullptr;
  ++state->physical_free_count;
  iree_allocator_free(host_allocator, test_physical_memory);
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_map(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer ||
      physical_memory != state->physical_memory || state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test mapping state mismatch");
  }
  state->is_mapped = true;
  state->map_virtual_offset = virtual_offset;
  state->map_physical_offset = physical_offset;
  state->map_size = size;
  ++state->map_count;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test unmapping state mismatch");
  }
  state->is_mapped = false;
  state->unmap_virtual_offset = virtual_offset;
  state->unmap_size = size;
  ++state->unmap_count;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_protect(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test protection state mismatch");
  }
  state->protect_queue_affinity = queue_affinity;
  state->protect_access_scope = access_scope;
  state->protection = protection;
  ++state->protect_count;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_advise(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_affinity_t queue_affinity, iree_hal_memory_advice_t advice) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test advice state mismatch");
  }
  state->advise_queue_affinity = queue_affinity;
  state->advice = advice;
  ++state->advise_count;
  return iree_ok_status();
}

const iree_hal_allocator_vtable_t vmm_test_allocator_vtable = {
    /*.destroy=*/vmm_test_allocator_destroy,
    /*.host_allocator=*/vmm_test_allocator_host_allocator,
    /*.trim=*/nullptr,
    /*.query_statistics=*/nullptr,
    /*.query_memory_heaps=*/nullptr,
    /*.query_buffer_compatibility=*/nullptr,
    /*.allocate_buffer=*/nullptr,
    /*.deallocate_buffer=*/nullptr,
    /*.import_buffer=*/nullptr,
    /*.export_buffer=*/nullptr,
    /*.supports_virtual_memory=*/vmm_test_allocator_supports_virtual_memory,
    /*.virtual_memory_query_granularity=*/
    vmm_test_allocator_query_granularity,
    /*.virtual_memory_reserve=*/vmm_test_allocator_virtual_memory_reserve,
    /*.virtual_memory_release=*/vmm_test_allocator_virtual_memory_release,
    /*.physical_memory_allocate=*/
    vmm_test_allocator_physical_memory_allocate,
    /*.physical_memory_free=*/vmm_test_allocator_physical_memory_free,
    /*.virtual_memory_map=*/vmm_test_allocator_virtual_memory_map,
    /*.virtual_memory_unmap=*/vmm_test_allocator_virtual_memory_unmap,
    /*.virtual_memory_protect=*/vmm_test_allocator_virtual_memory_protect,
    /*.virtual_memory_advise=*/vmm_test_allocator_virtual_memory_advise,
};

static iree_status_t CreateVmmTestAllocator(
    VmmAllocatorState* state, iree_allocator_t host_allocator,
    iree_hal_allocator_t** out_allocator) {
  *out_allocator = nullptr;
  iree_hal_allocator_t* heap_allocator = nullptr;
  IREE_RETURN_IF_ERROR(
      iree_hal_allocator_create_heap(IREE_SV("replay-vmm-test"), host_allocator,
                                     host_allocator, &heap_allocator));

  VmmTestAllocator* allocator = nullptr;
  iree_status_t status = iree_allocator_malloc(
      host_allocator, sizeof(*allocator), reinterpret_cast<void**>(&allocator));
  if (!iree_status_is_ok(status)) {
    iree_hal_allocator_release(heap_allocator);
    return status;
  }
  iree_hal_resource_initialize(&vmm_test_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->heap_allocator = heap_allocator;
  allocator->state = state;
  *out_allocator = reinterpret_cast<iree_hal_allocator_t*>(allocator);
  return iree_ok_status();
}

static iree_hal_device_group_t* CreateVmmTestDeviceGroup(
    VmmAllocatorState* state) {
  iree_hal_allocator_t* allocator = nullptr;
  IREE_CHECK_OK(
      CreateVmmTestAllocator(state, iree_allocator_system(), &allocator));

  iree_hal_mock_device_options_t options;
  iree_hal_mock_device_options_initialize(&options);
  options.identifier = IREE_SV("replay-vmm-test");
  options.device_allocator = allocator;
  iree_hal_device_t* device = nullptr;
  IREE_CHECK_OK(
      iree_hal_mock_device_create(&options, iree_allocator_system(), &device));
  iree_hal_allocator_release(allocator);

  iree_async_frontier_tracker_t* frontier_tracker = nullptr;
  IREE_CHECK_OK(iree_async_frontier_tracker_create(
      iree_async_frontier_tracker_options_default(), iree_allocator_system(),
      &frontier_tracker));
  iree_hal_device_group_builder_t builder;
  iree_hal_device_group_builder_initialize(&builder, frontier_tracker);
  iree_async_frontier_tracker_release(frontier_tracker);
  IREE_CHECK_OK(iree_hal_device_group_builder_add_device(&builder, device));

  iree_hal_device_group_t* group = nullptr;
  IREE_CHECK_OK(iree_hal_device_group_builder_finalize(
      &builder, iree_allocator_system(), &group));
  iree_hal_device_release(device);
  return group;
}

static iree_hal_replay_recorder_t* CreateHostAllocationRecorder(
    std::vector<uint8_t>* storage) {
  iree_io_file_handle_t* file_handle = nullptr;
  IREE_CHECK_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(storage->data(), storage->size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &file_handle));
  iree_hal_replay_recorder_t* recorder = nullptr;
  IREE_CHECK_OK(iree_hal_replay_recorder_create(
      file_handle, nullptr, iree_allocator_system(), &recorder));
  iree_io_file_handle_release(file_handle);
  return recorder;
}

static iree_const_byte_span_t GetCapturedFileContents(
    const std::vector<uint8_t>& storage) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t record_offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage.data(), storage.size()), &file_header,
      &record_offset));
  return iree_make_const_byte_span(
      storage.data(), static_cast<iree_host_size_t>(file_header.file_length));
}

static void TruncateCapturedFileAfterOperation(
    std::vector<uint8_t>* storage,
    iree_hal_replay_operation_code_t operation_code) {
  iree_hal_replay_file_header_t file_header;
  iree_host_size_t record_offset = 0;
  IREE_CHECK_OK(iree_hal_replay_file_parse_header(
      iree_make_const_byte_span(storage->data(), storage->size()), &file_header,
      &record_offset));
  const iree_const_byte_span_t file_contents = iree_make_const_byte_span(
      storage->data(), static_cast<iree_host_size_t>(file_header.file_length));
  while (record_offset < file_contents.data_length) {
    iree_hal_replay_file_record_t record;
    iree_host_size_t next_record_offset = 0;
    IREE_CHECK_OK(iree_hal_replay_file_parse_record(
        file_contents, record_offset, &record, &next_record_offset));
    if (record.header.record_type ==
            IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION &&
        record.header.operation_code == operation_code) {
      file_header.file_length = next_record_offset;
      memcpy(storage->data(), &file_header, sizeof(file_header));
      return;
    }
    record_offset = next_record_offset;
  }
  FAIL() << "captured replay operation was not found";
}

TEST(ReplayVmmTest, PreservesCompleteVirtualMemoryLifecycle) {
  constexpr iree_hal_queue_affinity_t kQueueAffinity = 1;
  constexpr iree_device_size_t kReservationSize = 16384;
  constexpr iree_device_size_t kVirtualOffset = 4096;
  constexpr iree_device_size_t kPhysicalOffset = 8192;
  constexpr iree_device_size_t kMappingSize = 4096;
  const iree_hal_buffer_params_t physical_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.queue_affinity=*/kQueueAffinity,
      /*.min_alignment=*/4096,
  };

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  VmmAllocatorState source_state;
  iree_hal_device_group_t* source_group =
      CreateVmmTestDeviceGroup(&source_state);
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(
      iree_hal_device_group_device_at(wrapped_group, 0));
  iree_device_size_t minimum_page_size = 0;
  iree_device_size_t recommended_page_size = 0;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_query_granularity(
      allocator, physical_params, &minimum_page_size, &recommended_page_size));
  EXPECT_EQ(minimum_page_size, 4096u);
  EXPECT_EQ(recommended_page_size, 65536u);

  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kQueueAffinity, kReservationSize, &virtual_buffer));
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, physical_params, kReservationSize, iree_allocator_system(),
      &physical_memory));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, kVirtualOffset, physical_memory,
      kPhysicalOffset, kMappingSize));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_protect(
      allocator, virtual_buffer, kVirtualOffset, kMappingSize, kQueueAffinity,
      IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
      IREE_HAL_MEMORY_PROTECTION_READ_WRITE));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_advise(
      allocator, virtual_buffer, kVirtualOffset, kMappingSize, kQueueAffinity,
      IREE_HAL_MEMORY_ADVICE_WILL_NEED));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, kVirtualOffset, kMappingSize));
  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));

  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);

  VmmAllocatorState replay_state;
  iree_hal_device_group_t* replay_group =
      CreateVmmTestDeviceGroup(&replay_state);
  const iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_ASSERT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);

  EXPECT_EQ(replay_state.query_count, 0u);
  EXPECT_EQ(replay_state.reserve_count, 1u);
  EXPECT_EQ(replay_state.release_count, 1u);
  EXPECT_EQ(replay_state.physical_allocate_count, 1u);
  EXPECT_EQ(replay_state.physical_free_count, 1u);
  EXPECT_EQ(replay_state.map_count, 1u);
  EXPECT_EQ(replay_state.unmap_count, 1u);
  EXPECT_EQ(replay_state.protect_count, 1u);
  EXPECT_EQ(replay_state.advise_count, 1u);
  EXPECT_EQ(replay_state.virtual_buffer, nullptr);
  EXPECT_EQ(replay_state.physical_memory, nullptr);
  EXPECT_FALSE(replay_state.is_mapped);
  EXPECT_EQ(replay_state.reserve_queue_affinity, kQueueAffinity);
  EXPECT_EQ(replay_state.reserve_size, kReservationSize);
  EXPECT_EQ(replay_state.physical_params.usage, physical_params.usage);
  EXPECT_EQ(replay_state.physical_params.access, physical_params.access);
  EXPECT_EQ(replay_state.physical_params.type, physical_params.type);
  EXPECT_EQ(replay_state.physical_params.queue_affinity,
            physical_params.queue_affinity);
  EXPECT_EQ(replay_state.physical_params.min_alignment,
            physical_params.min_alignment);
  EXPECT_EQ(replay_state.physical_size, kReservationSize);
  EXPECT_EQ(replay_state.map_virtual_offset, kVirtualOffset);
  EXPECT_EQ(replay_state.map_physical_offset, kPhysicalOffset);
  EXPECT_EQ(replay_state.map_size, kMappingSize);
  EXPECT_EQ(replay_state.unmap_virtual_offset, kVirtualOffset);
  EXPECT_EQ(replay_state.unmap_size, kMappingSize);
  EXPECT_EQ(replay_state.protect_queue_affinity, kQueueAffinity);
  EXPECT_EQ(replay_state.protect_access_scope,
            IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE);
  EXPECT_EQ(replay_state.protection, IREE_HAL_MEMORY_PROTECTION_READ_WRITE);
  EXPECT_EQ(replay_state.advise_queue_affinity, kQueueAffinity);
  EXPECT_EQ(replay_state.advice, IREE_HAL_MEMORY_ADVICE_WILL_NEED);
}

TEST(ReplayVmmTest, CleansUpLiveMappingsAtEndOfReplay) {
  constexpr iree_hal_queue_affinity_t kQueueAffinity = 1;
  constexpr iree_device_size_t kReservationSize = 16384;
  constexpr iree_device_size_t kVirtualOffset = 4096;
  constexpr iree_device_size_t kPhysicalOffset = 8192;
  constexpr iree_device_size_t kMappingSize = 4096;
  const iree_hal_buffer_params_t physical_params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      /*.type=*/IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      /*.queue_affinity=*/kQueueAffinity,
      /*.min_alignment=*/4096,
  };

  std::vector<uint8_t> storage(65536, 0);
  iree_hal_replay_recorder_t* recorder = CreateHostAllocationRecorder(&storage);
  VmmAllocatorState source_state;
  iree_hal_device_group_t* source_group =
      CreateVmmTestDeviceGroup(&source_state);
  iree_hal_device_group_t* wrapped_group = nullptr;
  IREE_ASSERT_OK(iree_hal_replay_wrap_device_group(
      recorder, source_group, iree_allocator_system(), &wrapped_group));

  iree_hal_allocator_t* allocator = iree_hal_device_allocator(
      iree_hal_device_group_device_at(wrapped_group, 0));
  iree_hal_buffer_t* virtual_buffer = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_reserve(
      allocator, kQueueAffinity, kReservationSize, &virtual_buffer));
  iree_hal_physical_memory_t* physical_memory = nullptr;
  IREE_ASSERT_OK(iree_hal_allocator_physical_memory_allocate(
      allocator, physical_params, kReservationSize, iree_allocator_system(),
      &physical_memory));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_map(
      allocator, virtual_buffer, kVirtualOffset, physical_memory,
      kPhysicalOffset, kMappingSize));
  IREE_ASSERT_OK(iree_hal_allocator_virtual_memory_unmap(
      allocator, virtual_buffer, kVirtualOffset, kMappingSize));
  IREE_ASSERT_OK(
      iree_hal_allocator_physical_memory_free(allocator, physical_memory));
  IREE_ASSERT_OK(
      iree_hal_allocator_virtual_memory_release(allocator, virtual_buffer));
  IREE_ASSERT_OK(iree_hal_replay_recorder_close(recorder));

  iree_hal_device_group_release(wrapped_group);
  iree_hal_device_group_release(source_group);
  iree_hal_replay_recorder_release(recorder);

  TruncateCapturedFileAfterOperation(
      &storage, IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP);

  VmmAllocatorState replay_state;
  iree_hal_device_group_t* replay_group =
      CreateVmmTestDeviceGroup(&replay_state);
  const iree_hal_replay_execute_options_t options =
      iree_hal_replay_execute_options_default();
  IREE_ASSERT_OK(iree_hal_replay_execute_file(GetCapturedFileContents(storage),
                                              replay_group, &options,
                                              iree_allocator_system()));
  iree_hal_device_group_release(replay_group);

  EXPECT_EQ(replay_state.reserve_count, 1u);
  EXPECT_EQ(replay_state.release_count, 1u);
  EXPECT_EQ(replay_state.physical_allocate_count, 1u);
  EXPECT_EQ(replay_state.physical_free_count, 1u);
  EXPECT_EQ(replay_state.map_count, 1u);
  EXPECT_EQ(replay_state.unmap_count, 1u);
  EXPECT_EQ(replay_state.virtual_buffer, nullptr);
  EXPECT_EQ(replay_state.physical_memory, nullptr);
  EXPECT_FALSE(replay_state.is_mapped);
}

}  // namespace
