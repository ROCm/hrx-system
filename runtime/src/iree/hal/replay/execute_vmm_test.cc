// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "execute_operation.h"
#include "execute_state.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

static constexpr iree_hal_replay_object_id_t kAllocatorId = 1;
static constexpr iree_hal_replay_object_id_t kVirtualBufferId = 2;
static constexpr iree_hal_replay_object_id_t kPhysicalMemoryId = 3;
static constexpr iree_hal_queue_family_affinity_t kQueueFamilyAffinity =
    UINT64_C(1) << 2;
static constexpr iree_device_size_t kReservationSize = 16384;
static constexpr iree_device_size_t kVirtualOffset = 4096;
static constexpr iree_device_size_t kPhysicalOffset = 8192;
static constexpr iree_device_size_t kMappingSize = 4096;

typedef struct VmmAllocatorState {
  bool fail_reserve = false;
  bool fail_release = false;
  bool fail_physical_allocate = false;
  bool fail_physical_free = false;
  bool fail_map = false;
  bool fail_unmap = false;
  bool fail_protect = false;
  bool fail_advise = false;
  iree_host_size_t reserve_attempt_count = 0;
  iree_host_size_t release_attempt_count = 0;
  iree_host_size_t physical_allocate_attempt_count = 0;
  iree_host_size_t physical_free_attempt_count = 0;
  iree_host_size_t map_attempt_count = 0;
  iree_host_size_t unmap_attempt_count = 0;
  iree_host_size_t protect_attempt_count = 0;
  iree_host_size_t advise_attempt_count = 0;
  iree_hal_buffer_t* virtual_buffer = nullptr;
  iree_hal_physical_memory_t* physical_memory = nullptr;
  bool is_mapped = false;
  iree_hal_queue_family_affinity_t reserve_queue_family_affinity = 0;
  iree_device_size_t reserve_size = 0;
  iree_hal_buffer_params_t physical_params = {};
  iree_device_size_t physical_size = 0;
  iree_device_size_t map_virtual_offset = 0;
  iree_device_size_t map_physical_offset = 0;
  iree_device_size_t map_size = 0;
  iree_device_size_t unmap_virtual_offset = 0;
  iree_device_size_t unmap_size = 0;
  iree_hal_queue_family_affinity_t protect_queue_family_affinity = 0;
  iree_hal_virtual_memory_access_scope_t protect_access_scope = 0;
  iree_hal_memory_protection_t protection = 0;
  iree_hal_queue_family_affinity_t advise_queue_family_affinity = 0;
  iree_hal_memory_advice_t advice = 0;
} VmmAllocatorState;

typedef struct VmmTestPhysicalMemory {
  iree_allocator_t host_allocator;
} VmmTestPhysicalMemory;

typedef struct VmmTestAllocator {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  iree_hal_allocator_t* heap_allocator;
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
  return reinterpret_cast<const VmmTestAllocator*>(base_allocator)
      ->host_allocator;
}

static bool vmm_test_allocator_supports_virtual_memory(
    iree_hal_allocator_t* base_allocator) {
  (void)base_allocator;
  return true;
}

static iree_status_t vmm_test_allocator_query_granularity(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t* out_minimum_page_size,
    iree_device_size_t* out_recommended_page_size) {
  (void)base_allocator;
  (void)params;
  *out_minimum_page_size = 4096;
  *out_recommended_page_size = 65536;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_reserve(
    iree_hal_allocator_t* base_allocator,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_device_size_t size, iree_hal_buffer_t** out_virtual_buffer) {
  VmmTestAllocator* allocator = vmm_test_allocator_cast(base_allocator);
  VmmAllocatorState* state = allocator->state;
  ++state->reserve_attempt_count;
  if (state->fail_reserve) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected reserve failure");
  }
  if (state->virtual_buffer) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test reservation is already live");
  }
  iree_hal_buffer_params_t params = {
      /*.usage=*/IREE_HAL_BUFFER_USAGE_STORAGE,
      /*.access=*/IREE_HAL_MEMORY_ACCESS_ALL,
      /*.type=*/IREE_HAL_MEMORY_TYPE_HOST_LOCAL,
      /*.queue_family_affinity=*/IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*.min_alignment=*/4096,
  };
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator->heap_allocator, params, size, out_virtual_buffer));
  state->virtual_buffer = *out_virtual_buffer;
  state->reserve_queue_family_affinity = queue_family_affinity;
  state->reserve_size = size;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_release(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->release_attempt_count;
  if (state->fail_release) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected release failure");
  }
  if (virtual_buffer != state->virtual_buffer || state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test reservation state mismatch");
  }
  state->virtual_buffer = nullptr;
  iree_hal_buffer_release(virtual_buffer);
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_physical_memory_allocate(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_params_t params,
    iree_device_size_t size, iree_allocator_t host_allocator,
    iree_hal_physical_memory_t** out_physical_memory) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->physical_allocate_attempt_count;
  if (state->fail_physical_allocate) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected physical allocation failure");
  }
  if (state->physical_memory) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test physical allocation is already live");
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
  *out_physical_memory = state->physical_memory;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_physical_memory_free(
    iree_hal_allocator_t* base_allocator,
    iree_hal_physical_memory_t* physical_memory) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->physical_free_attempt_count;
  if (state->fail_physical_free) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected physical free failure");
  }
  if (physical_memory != state->physical_memory || state->is_mapped) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test physical allocation state mismatch");
  }
  VmmTestPhysicalMemory* test_physical_memory =
      reinterpret_cast<VmmTestPhysicalMemory*>(physical_memory);
  state->physical_memory = nullptr;
  iree_allocator_free(test_physical_memory->host_allocator,
                      test_physical_memory);
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_map(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset,
    iree_hal_physical_memory_t* physical_memory,
    iree_device_size_t physical_offset, iree_device_size_t size) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->map_attempt_count;
  if (state->fail_map) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected map failure");
  }
  if (virtual_buffer != state->virtual_buffer ||
      physical_memory != state->physical_memory || state->is_mapped ||
      virtual_offset + size > state->reserve_size ||
      physical_offset + size > state->physical_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test mapping state mismatch");
  }
  state->is_mapped = true;
  state->map_virtual_offset = virtual_offset;
  state->map_physical_offset = physical_offset;
  state->map_size = size;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_unmap(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->unmap_attempt_count;
  if (state->fail_unmap) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected unmap failure");
  }
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test unmapping state mismatch");
  }
  state->is_mapped = false;
  state->unmap_virtual_offset = virtual_offset;
  state->unmap_size = size;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_protect(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_virtual_memory_access_scope_t access_scope,
    iree_hal_memory_protection_t protection) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->protect_attempt_count;
  if (state->fail_protect) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected protect failure");
  }
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test protection state mismatch");
  }
  state->protect_queue_family_affinity = queue_family_affinity;
  state->protect_access_scope = access_scope;
  state->protection = protection;
  return iree_ok_status();
}

static iree_status_t vmm_test_allocator_virtual_memory_advise(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* virtual_buffer,
    iree_device_size_t virtual_offset, iree_device_size_t size,
    iree_hal_queue_family_affinity_t queue_family_affinity,
    iree_hal_memory_advice_t advice) {
  VmmAllocatorState* state = vmm_test_allocator_cast(base_allocator)->state;
  ++state->advise_attempt_count;
  if (state->fail_advise) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "injected advise failure");
  }
  if (virtual_buffer != state->virtual_buffer || !state->is_mapped ||
      virtual_offset != state->map_virtual_offset || size != state->map_size) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "test advice state mismatch");
  }
  state->advise_queue_family_affinity = queue_family_affinity;
  state->advice = advice;
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
  memset(allocator, 0, sizeof(*allocator));
  iree_hal_resource_initialize(&vmm_test_allocator_vtable,
                               &allocator->resource);
  allocator->host_allocator = host_allocator;
  allocator->heap_allocator = heap_allocator;
  allocator->state = state;
  *out_allocator = reinterpret_cast<iree_hal_allocator_t*>(allocator);
  return iree_ok_status();
}

class OperationRecord {
 public:
  template <typename Payload>
  void Reset(iree_hal_replay_operation_code_t operation_code,
             iree_hal_replay_payload_type_t payload_type,
             iree_hal_replay_object_id_t object_id,
             iree_hal_replay_object_id_t related_object_id,
             const Payload& payload, bool unaligned = false) {
    const iree_host_size_t payload_offset = unaligned ? 1 : 0;
    payload_storage_.assign(payload_offset + sizeof(payload), 0);
    memcpy(payload_storage_.data() + payload_offset, &payload, sizeof(payload));
    memset(&record_, 0, sizeof(record_));
    record_.header.record_type = IREE_HAL_REPLAY_FILE_RECORD_TYPE_OPERATION;
    record_.header.operation_code = operation_code;
    record_.header.payload_type = payload_type;
    record_.header.object_id = object_id;
    record_.header.related_object_id = related_object_id;
    record_.header.status_code = IREE_STATUS_OK;
    record_.payload = iree_make_const_byte_span(
        payload_storage_.data() + payload_offset, sizeof(payload));
  }

  const iree_hal_replay_file_record_t* get() const { return &record_; }

 private:
  std::vector<uint8_t> payload_storage_;
  iree_hal_replay_file_record_t record_ = {};
};

class ReplayVmmExecutionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    options_ = iree_hal_replay_execute_options_default();
    IREE_ASSERT_OK(iree_hal_replay_executor_initialize(
        &executor_, iree_const_byte_span_empty(),
        IREE_HAL_REPLAY_FILE_VERSION_MINOR, /*object_capacity=*/8,
        /*device_group=*/nullptr, &options_, iree_allocator_system()));
    initialized_ = true;
    iree_hal_allocator_t* allocator = nullptr;
    IREE_ASSERT_OK(
        CreateVmmTestAllocator(&state_, iree_allocator_system(), &allocator));
    iree_hal_replay_object_entry_t entry = {};
    entry.value.allocator = allocator;
    IREE_ASSERT_OK(iree_hal_replay_executor_store(
        &executor_, kAllocatorId, IREE_HAL_REPLAY_OBJECT_TYPE_ALLOCATOR,
        entry));
  }

  void TearDown() override {
    if (!initialized_) {
      return;
    }
    state_.fail_release = false;
    state_.fail_physical_free = false;
    state_.fail_unmap = false;
    iree_status_ignore(iree_hal_replay_executor_deinitialize(&executor_));
  }

  iree_status_t Replay(const OperationRecord& record) {
    return iree_hal_replay_executor_replay_operation(&executor_, record.get());
  }

  void Reserve(OperationRecord* record, bool unaligned = false) {
    const iree_hal_replay_allocator_virtual_memory_reserve_payload_t payload = {
        /*.queue_family_affinity=*/kQueueFamilyAffinity,
        /*.size=*/kReservationSize,
    };
    record->Reset(
        IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
        kAllocatorId, kVirtualBufferId, payload, unaligned);
    IREE_ASSERT_OK(Replay(*record));
  }

  void AllocatePhysical(OperationRecord* record, bool unaligned = false) {
    iree_hal_replay_allocator_physical_memory_allocate_payload_t payload = {};
    payload.allocation.allocation_size = kReservationSize;
    payload.allocation.queue_family_affinity = kQueueFamilyAffinity;
    payload.allocation.min_alignment = 4096;
    payload.allocation.usage = IREE_HAL_BUFFER_USAGE_STORAGE;
    payload.allocation.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
    payload.allocation.access =
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE;
    record->Reset(
        IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
        IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
        kAllocatorId, kPhysicalMemoryId, payload, unaligned);
    IREE_ASSERT_OK(Replay(*record));
  }

  void Map(OperationRecord* record, bool unaligned = false) {
    const iree_hal_replay_allocator_virtual_memory_map_payload_t payload = {
        /*.virtual_buffer_id=*/kVirtualBufferId,
        /*.physical_memory_id=*/kPhysicalMemoryId,
        /*.virtual_offset=*/kVirtualOffset,
        /*.physical_offset=*/kPhysicalOffset,
        /*.size=*/kMappingSize,
    };
    record->Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
                  IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
                  kAllocatorId, kVirtualBufferId, payload, unaligned);
    IREE_ASSERT_OK(Replay(*record));
  }

  VmmAllocatorState state_;
  iree_hal_replay_execute_options_t options_ = {};
  iree_hal_replay_executor_t executor_ = {};
  bool initialized_ = false;
};

TEST_F(ReplayVmmExecutionTest, ReplaysCompletePackedLifecycle) {
  OperationRecord record;
  Reserve(&record, /*unaligned=*/true);
  AllocatePhysical(&record, /*unaligned=*/true);
  Map(&record, /*unaligned=*/true);

  ASSERT_EQ(1u, executor_.virtual_memory_mapping_count);
  EXPECT_EQ(kVirtualBufferId,
            executor_.virtual_memory_mappings[0].virtual_buffer_id);
  EXPECT_EQ(kPhysicalMemoryId,
            executor_.virtual_memory_mappings[0].physical_memory_id);
  EXPECT_EQ(kVirtualOffset,
            executor_.virtual_memory_mappings[0].virtual_offset);
  EXPECT_EQ(kPhysicalOffset,
            executor_.virtual_memory_mappings[0].physical_offset);
  EXPECT_EQ(kMappingSize, executor_.virtual_memory_mappings[0].size);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
            executor_.objects[kVirtualBufferId].type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY,
            executor_.objects[kPhysicalMemoryId].type);

  const iree_hal_replay_allocator_virtual_memory_protect_payload_t
      protect_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset,
          /*.size=*/kMappingSize,
          /*.queue_family_affinity=*/kQueueFamilyAffinity,
          /*.access_scope=*/IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL,
          /*.reserved0=*/0,
          /*.protection=*/IREE_HAL_MEMORY_PROTECTION_READ_WRITE,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
               kAllocatorId, kVirtualBufferId, protect_payload,
               /*unaligned=*/true);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(kQueueFamilyAffinity, state_.protect_queue_family_affinity);
  EXPECT_EQ(IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_ALL,
            state_.protect_access_scope);
  EXPECT_EQ(IREE_HAL_MEMORY_PROTECTION_READ_WRITE, state_.protection);

  const iree_hal_replay_allocator_virtual_memory_advise_payload_t
      advise_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset,
          /*.size=*/kMappingSize,
          /*.queue_family_affinity=*/kQueueFamilyAffinity,
          /*.advice=*/IREE_HAL_MEMORY_ADVICE_WILL_NEED,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_ADVISE,
               kAllocatorId, kVirtualBufferId, advise_payload,
               /*unaligned=*/true);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(kQueueFamilyAffinity, state_.advise_queue_family_affinity);
  EXPECT_EQ(IREE_HAL_MEMORY_ADVICE_WILL_NEED, state_.advice);

  const iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap_payload =
      {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset,
          /*.size=*/kMappingSize,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               kAllocatorId, kVirtualBufferId, unmap_payload,
               /*unaligned=*/true);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(0u, executor_.virtual_memory_mapping_count);

  const iree_hal_replay_allocator_physical_memory_free_payload_t free_payload =
      {
          /*.physical_memory_id=*/kPhysicalMemoryId,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               kAllocatorId, kPhysicalMemoryId, free_payload,
               /*unaligned=*/true);
  IREE_ASSERT_OK(Replay(record));
  const iree_hal_replay_allocator_virtual_memory_release_payload_t
      release_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               kAllocatorId, kVirtualBufferId, release_payload,
               /*unaligned=*/true);
  IREE_ASSERT_OK(Replay(record));

  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
            executor_.objects[kVirtualBufferId].type);
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
            executor_.objects[kPhysicalMemoryId].type);
  EXPECT_EQ(kQueueFamilyAffinity, state_.reserve_queue_family_affinity);
  EXPECT_EQ(kReservationSize, state_.reserve_size);
  EXPECT_EQ(kQueueFamilyAffinity, state_.physical_params.queue_family_affinity);
  EXPECT_EQ(kReservationSize, state_.physical_size);
  EXPECT_EQ(kVirtualOffset, state_.map_virtual_offset);
  EXPECT_EQ(kPhysicalOffset, state_.map_physical_offset);
  EXPECT_EQ(kMappingSize, state_.map_size);
  EXPECT_EQ(kVirtualOffset, state_.unmap_virtual_offset);
  EXPECT_EQ(kMappingSize, state_.unmap_size);
}

TEST_F(ReplayVmmExecutionTest, RejectsInvalidIdsTypesAndRanges) {
  OperationRecord record;
  Reserve(&record);
  AllocatePhysical(&record);

  iree_hal_replay_allocator_virtual_memory_map_payload_t map_payload = {
      /*.virtual_buffer_id=*/kVirtualBufferId,
      /*.physical_memory_id=*/kPhysicalMemoryId,
      /*.virtual_offset=*/kVirtualOffset,
      /*.physical_offset=*/kPhysicalOffset,
      /*.size=*/0,
  };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               kAllocatorId, kVirtualBufferId, map_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(0u, state_.map_attempt_count);

  map_payload.size = kMappingSize;
  map_payload.physical_memory_id = kVirtualBufferId;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               kAllocatorId, kVirtualBufferId, map_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Replay(record));
  EXPECT_EQ(0u, state_.map_attempt_count);

  map_payload.physical_memory_id = kPhysicalMemoryId;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               kAllocatorId, kPhysicalMemoryId, map_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(0u, state_.map_attempt_count);

  Map(&record);
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               kAllocatorId, kVirtualBufferId, map_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(1u, state_.map_attempt_count);

  const iree_hal_replay_allocator_virtual_memory_unmap_payload_t
      uncovered_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset + kMappingSize,
          /*.size=*/kMappingSize,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               kAllocatorId, kVirtualBufferId, uncovered_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(0u, state_.unmap_attempt_count);

  iree_hal_replay_allocator_virtual_memory_protect_payload_t protect_payload = {
      /*.virtual_buffer_id=*/kVirtualBufferId,
      /*.virtual_offset=*/kVirtualOffset,
      /*.size=*/kMappingSize,
      /*.queue_family_affinity=*/kQueueFamilyAffinity,
      /*.access_scope=*/IREE_HAL_VIRTUAL_MEMORY_ACCESS_SCOPE_DEVICE,
      /*.reserved0=*/1,
      /*.protection=*/IREE_HAL_MEMORY_PROTECTION_READ,
  };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_PROTECT,
               kAllocatorId, kVirtualBufferId, protect_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DATA_LOSS, Replay(record));
  EXPECT_EQ(0u, state_.protect_attempt_count);

  const iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap_payload =
      {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset,
          /*.size=*/kMappingSize,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               kAllocatorId, kVirtualBufferId, unmap_payload);
  IREE_ASSERT_OK(Replay(record));
}

TEST_F(ReplayVmmExecutionTest, BackendFailuresRetainRetryableObjects) {
  OperationRecord record;
  state_.fail_reserve = true;
  const iree_hal_replay_allocator_virtual_memory_reserve_payload_t
      reserve_payload = {
          /*.queue_family_affinity=*/kQueueFamilyAffinity,
          /*.size=*/kReservationSize,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RESERVE,
               kAllocatorId, kVirtualBufferId, reserve_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
            executor_.objects[kVirtualBufferId].type);
  state_.fail_reserve = false;
  Reserve(&record);

  state_.fail_physical_allocate = true;
  iree_hal_replay_allocator_physical_memory_allocate_payload_t
      allocate_payload = {};
  allocate_payload.allocation.allocation_size = kReservationSize;
  record.Reset(
      IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_ALLOCATE,
      kAllocatorId, kPhysicalMemoryId, allocate_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_NONE,
            executor_.objects[kPhysicalMemoryId].type);
  state_.fail_physical_allocate = false;
  AllocatePhysical(&record);

  state_.fail_map = true;
  const iree_hal_replay_allocator_virtual_memory_map_payload_t map_payload = {
      /*.virtual_buffer_id=*/kVirtualBufferId,
      /*.physical_memory_id=*/kPhysicalMemoryId,
      /*.virtual_offset=*/kVirtualOffset,
      /*.physical_offset=*/kPhysicalOffset,
      /*.size=*/kMappingSize,
  };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_MAP,
               kAllocatorId, kVirtualBufferId, map_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  EXPECT_EQ(0u, executor_.virtual_memory_mapping_count);
  state_.fail_map = false;

  const iree_hal_replay_allocator_physical_memory_free_payload_t free_payload =
      {
          /*.physical_memory_id=*/kPhysicalMemoryId,
      };
  state_.fail_physical_free = true;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               kAllocatorId, kPhysicalMemoryId, free_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_PHYSICAL_MEMORY,
            executor_.objects[kPhysicalMemoryId].type);
  state_.fail_physical_free = false;
  IREE_ASSERT_OK(Replay(record));

  const iree_hal_replay_allocator_virtual_memory_release_payload_t
      release_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
      };
  state_.fail_release = true;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               kAllocatorId, kVirtualBufferId, release_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  EXPECT_EQ(IREE_HAL_REPLAY_OBJECT_TYPE_BUFFER,
            executor_.objects[kVirtualBufferId].type);
  state_.fail_release = false;
  IREE_ASSERT_OK(Replay(record));
}

TEST_F(ReplayVmmExecutionTest, FailedUnmapRetainsDependenciesUntilRetry) {
  OperationRecord record;
  Reserve(&record);
  AllocatePhysical(&record);
  Map(&record);

  const iree_hal_replay_allocator_virtual_memory_unmap_payload_t unmap_payload =
      {
          /*.virtual_buffer_id=*/kVirtualBufferId,
          /*.virtual_offset=*/kVirtualOffset,
          /*.size=*/kMappingSize,
      };
  state_.fail_unmap = true;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               kAllocatorId, kVirtualBufferId, unmap_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, Replay(record));
  ASSERT_EQ(1u, executor_.virtual_memory_mapping_count);

  const iree_hal_replay_allocator_physical_memory_free_payload_t free_payload =
      {
          /*.physical_memory_id=*/kPhysicalMemoryId,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               kAllocatorId, kPhysicalMemoryId, free_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Replay(record));
  EXPECT_EQ(0u, state_.physical_free_attempt_count);

  const iree_hal_replay_allocator_virtual_memory_release_payload_t
      release_payload = {
          /*.virtual_buffer_id=*/kVirtualBufferId,
      };
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               kAllocatorId, kVirtualBufferId, release_payload);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_FAILED_PRECONDITION, Replay(record));
  EXPECT_EQ(0u, state_.release_attempt_count);

  state_.fail_unmap = false;
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_UNMAP,
               kAllocatorId, kVirtualBufferId, unmap_payload);
  IREE_ASSERT_OK(Replay(record));
  EXPECT_EQ(0u, executor_.virtual_memory_mapping_count);

  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_PHYSICAL_MEMORY_FREE,
               kAllocatorId, kPhysicalMemoryId, free_payload);
  IREE_ASSERT_OK(Replay(record));
  record.Reset(IREE_HAL_REPLAY_OPERATION_CODE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               IREE_HAL_REPLAY_PAYLOAD_TYPE_ALLOCATOR_VIRTUAL_MEMORY_RELEASE,
               kAllocatorId, kVirtualBufferId, release_payload);
  IREE_ASSERT_OK(Replay(record));
}

TEST_F(ReplayVmmExecutionTest, FailedCleanupRetainsMappedResources) {
  OperationRecord record;
  Reserve(&record);
  AllocatePhysical(&record);
  Map(&record);
  state_.fail_unmap = true;

  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        iree_hal_replay_executor_deinitialize(&executor_));
  EXPECT_TRUE(state_.is_mapped);
  EXPECT_NE(nullptr, state_.virtual_buffer);
  EXPECT_NE(nullptr, state_.physical_memory);
  EXPECT_EQ(0u, state_.physical_free_attempt_count);
  EXPECT_EQ(0u, state_.release_attempt_count);
  EXPECT_EQ(1u, executor_.virtual_memory_mapping_count);

  state_.fail_unmap = false;
  IREE_ASSERT_OK(iree_hal_replay_executor_deinitialize(&executor_));
  initialized_ = false;
  EXPECT_FALSE(state_.is_mapped);
  EXPECT_EQ(nullptr, state_.virtual_buffer);
  EXPECT_EQ(nullptr, state_.physical_memory);
  EXPECT_EQ(1u, state_.physical_free_attempt_count);
  EXPECT_EQ(1u, state_.release_attempt_count);
}

}  // namespace
