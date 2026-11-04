// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/memory_reservation.h"

#include <sys/ioctl.h>

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/linux/device.h"
#include "libamdf/src/gpu/umd/dxg/linux/kernel_queue.h"
#include "libamdf/src/gpu/umd/dxg/linux/memory.h"

namespace {

constexpr NTSTATUS kStatusPending = static_cast<NTSTATUS>(0x00000103u);
constexpr NTSTATUS kStatusNoMemory = static_cast<NTSTATUS>(0xC0000017u);

enum class Operation {
  kReserve,
  kMap,
  kWait,
  kQueryDeviceState,
  kFree,
  kUpdate,
  kOrder,
};

struct ReserveRecord {
  uint64_t base_address;
  uint64_t minimum_address;
  uint64_t byte_length;
};

struct MapRecord {
  uint64_t address;
  uint32_t allocation_handle;
  uint64_t source_byte_offset;
  uint64_t byte_length;
  uint32_t write;
  uint32_t execute;
  uint32_t no_access;
};

struct FreeRecord {
  uint64_t address;
  uint64_t byte_length;
};

struct FakeReservationState {
  NTSTATUS reserve_status = 0;
  NTSTATUS map_status = kStatusPending;
  uint64_t automatic_address = 0x410000;
  uint64_t next_paging_fence = 7;
  volatile uint64_t paging_progress = 0;
  volatile uint64_t queue_progress = 0;
  uint64_t mapping_predecessor = 0;
  uint64_t ordering_point = 0;
  uint32_t queue_destroy_count = 0;
  D3DDDI_UPDATEGPUVIRTUALADDRESS_OPERATION mapping_operation = {};
  std::vector<Operation> operations;
  std::vector<ReserveRecord> reserves;
  std::vector<MapRecord> maps;
  std::vector<uint64_t> waits;
  std::vector<FreeRecord> frees;
};

FakeReservationState* current_state = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;

amdf_dxg_provider_result_t FakeCreateQueue(
    amdf_dxg_provider_gpu_adapter_t* adapter,
    const amdf_dxg_provider_gpu_kernel_queue_create_info_t* create_info,
    amdf_dxg_provider_gpu_kernel_queue_t** out_queue,
    amdf_dxg_provider_gpu_kernel_queue_info_t* out_info,
    uint32_t* out_native_status) {
  auto* state = reinterpret_cast<FakeReservationState*>(adapter);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->command_type,
            AMDF_DXG_PROVIDER_GPU_QUEUE_COMMAND_TYPE_PM4);
  *out_info = {};
  out_info->command_buffer_alignment = 4;
  out_info->maximum_command_buffer_byte_length = 4096;
  out_info->progress_fence_handle = 0x60;
  out_info->progress_fence_pointer = &state->queue_progress;
  out_info->progress_fence_device_address = 0x10000;
  out_info->context_handle = 0x70;
  out_info->queue_handle = 0x80;
  *out_queue = reinterpret_cast<amdf_dxg_provider_gpu_kernel_queue_t*>(state);
  *out_native_status = 0;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

amdf_dxg_provider_result_t FakeDestroyQueue(
    amdf_dxg_provider_gpu_kernel_queue_t* queue, uint32_t* out_native_status) {
  ++reinterpret_cast<FakeReservationState*>(queue)->queue_destroy_count;
  *out_native_status = 0;
  return AMDF_DXG_PROVIDER_RESULT_SUCCESS;
}

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x08: {
      current_state->operations.push_back(Operation::kReserve);
      auto* reserve = static_cast<D3DDDI_RESERVEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(reserve->hAdapter, 0x08u);
      current_state->reserves.push_back(
          {reserve->BaseAddress, reserve->MinimumAddress, reserve->Size});
      if (current_state->reserve_status != 0) {
        return static_cast<uint32_t>(current_state->reserve_status);
      }
      reserve->VirtualAddress = reserve->BaseAddress != 0
                                    ? reserve->BaseAddress
                                    : current_state->automatic_address;
      return 0;
    }
    case 0x0C: {
      current_state->operations.push_back(Operation::kMap);
      auto* map = static_cast<D3DDDI_MAPGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(map->hPagingQueue, 0x40u);
      current_state->maps.push_back({
          .address = map->BaseAddress,
          .allocation_handle = map->hAllocation,
          .source_byte_offset = map->OffsetInPages * 4096,
          .byte_length = map->SizeInPages * 4096,
          .write = map->Protection.Write,
          .execute = map->Protection.Execute,
          .no_access = map->Protection.NoAccess,
      });
      if (current_state->map_status != kStatusPending &&
          current_state->map_status != 0) {
        return static_cast<uint32_t>(current_state->map_status);
      }
      map->VirtualAddress = map->BaseAddress;
      map->PagingFenceValue = current_state->next_paging_fence++;
      return static_cast<uint32_t>(current_state->map_status);
    }
    case 0x0E: {
      current_state->operations.push_back(Operation::kQueryDeviceState);
      auto* query = static_cast<D3DKMT_GETDEVICESTATE*>(arguments);
      EXPECT_EQ(query->hDevice, 0x10u);
      EXPECT_EQ(query->StateType, D3DKMT_DEVICESTATE_EXECUTION);
      query->ExecutionState = D3DKMT_DEVICEEXECUTION_ACTIVE;
      return 0;
    }
    case 0x20: {
      current_state->operations.push_back(Operation::kFree);
      const auto* release =
          static_cast<const D3DKMT_FREEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(release->hAdapter, 0x08u);
      current_state->frees.push_back({release->BaseAddress, release->Size});
      return 0;
    }
    case 0x3A: {
      current_state->operations.push_back(Operation::kWait);
      const auto* wait =
          static_cast<const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*>(
              arguments);
      EXPECT_EQ(wait->hDevice, 0x10u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x50u);
      const uint64_t target = wait->FenceValueArray[0];
      current_state->waits.push_back(target);
      current_state->paging_progress = target;
      return 0;
    }
    case 0x39: {
      current_state->operations.push_back(Operation::kUpdate);
      const auto* update =
          static_cast<D3DKMT_UPDATEGPUVIRTUALADDRESS*>(arguments);
      EXPECT_EQ(update->hDevice, 0x10u);
      EXPECT_EQ(update->hContext, 0x70u);
      EXPECT_EQ(update->hFenceObject, 0x60u);
      EXPECT_EQ(update->NumOperations, 1u);
      current_state->mapping_predecessor = update->FenceValue;
      current_state->mapping_operation = update->Operations[0];
      return 0;
    }
    case 0x36: {
      current_state->operations.push_back(Operation::kOrder);
      const auto* wait =
          static_cast<D3DKMT_SUBMITWAITFORSYNCOBJECTSTOHWQUEUE*>(arguments);
      EXPECT_EQ(wait->hHwQueue, 0x80u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 0x60u);
      current_state->ordering_point = wait->FenceValueArray[0];
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return static_cast<uint32_t>(kStatusNoMemory);
  }
}

class LinuxDxgMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_connection = &connection_;
    allocator_ = amdf_allocator_system();
    connection_.descriptor = 1;
    kmt_.connection = &connection_;
    kmt_.host_allocator = &allocator_;
    provider_.gpu_kernel_queue_create = FakeCreateQueue;
    provider_.gpu_kernel_queue_destroy = FakeDestroyQueue;
    device_ = {
        .host_allocator = allocator_,
        .adapter = 0x08,
        .adapter_luid = UINT64_C(0x1122334455667788),
        .physical_adapter_index = 3,
        .memory_capabilities =
            {
                .virtual_address_bit_count = 48,
                .read_only_memory_supported = 1,
                .no_execute_memory_supported = 1,
            },
        .logical_device =
            {
                .kmt = &kmt_,
                .native = 0x10,
            },
        .paging_queue =
            {
                .kmt = &kmt_,
                .native = 0x40,
                .synchronization_object = 0x50,
                .progress_fence = &state_.paging_progress,
            },
    };
    device_.logical_device.provider_adapter.api = &provider_;
    device_.logical_device.provider_adapter.native =
        reinterpret_cast<amdf_dxg_provider_gpu_adapter_t*>(&state_);
  }

  void TearDown() override {
    current_connection = nullptr;
    current_state = nullptr;
  }

  amdf_gpu_umd_memory_reservation_t* CreateReservation(
      uint64_t minimum_alignment = 0, uint64_t common_address = 0) {
    amdf_gpu_umd_device_t* devices[] = {&device_};
    amdf_gpu_umd_memory_reservation_t* reservation = nullptr;
    EXPECT_EQ(amdf_gpu_umd_memory_reservation_create(
                  allocator_, 1, devices, 65536, minimum_alignment,
                  common_address, &reservation),
              AMDF_STATUS_OK);
    return reservation;
  }

  FakeReservationState state_;
  amdf_allocator_t allocator_ = {};
  amdf_linux_dxg_connection_t connection_ = {};
  amdf_linux_dxg_kmt_t kmt_ = {};
  amdf_dxg_provider_api_t provider_ = {};
  amdf_gpu_umd_device_t device_ = {};
};

TEST_F(LinuxDxgMemoryReservationTest,
       PublishesCapabilitiesAndMappingDomainIdentity) {
  amdf_memory_reservation_capabilities_t capabilities = {};
  ASSERT_EQ(amdf_gpu_umd_query_memory_reservation_capabilities(&device_, 0,
                                                               &capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(capabilities.supported_access, AMDF_MEMORY_ACCESS_READ |
                                               AMDF_MEMORY_ACCESS_WRITE |
                                               AMDF_MEMORY_ACCESS_EXECUTE);
  EXPECT_EQ(capabilities.required_access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(capabilities.reservation_granularity, 65536u);
  EXPECT_EQ(capabilities.mapping_granularity, 4096u);
  EXPECT_EQ(capabilities.maximum_address, (UINT64_C(1) << 48) - 1);

  amdf_gpu_umd_device_t alias = device_;
  EXPECT_TRUE(amdf_gpu_umd_devices_share_mapping_domain(&device_, &alias));
  alias.physical_adapter_index = 4;
  EXPECT_FALSE(amdf_gpu_umd_devices_share_mapping_domain(&device_, &alias));
}

TEST_F(LinuxDxgMemoryReservationTest,
       OwnsAlignedReservationThroughPublishedKmtOperations) {
  amdf_gpu_umd_memory_reservation_t* reservation =
      CreateReservation(UINT64_C(2) * 1024 * 1024);
  ASSERT_NE(reservation, nullptr);
  ASSERT_EQ(state_.reserves.size(), 1u);
  EXPECT_EQ(state_.reserves[0].base_address, 0u);
  EXPECT_EQ(state_.reserves[0].minimum_address, 65536u);
  EXPECT_EQ(state_.reserves[0].byte_length, UINT64_C(2) * 1024 * 1024);

  uint64_t address = 0;
  uint32_t mapping_access_ordinal = UINT32_MAX;
  uint64_t mapping_granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_umd_memory_reservation_query_access(
      reservation, 0, &address, &mapping_access_ordinal, &mapping_granularity,
      &required_access);
  EXPECT_EQ(address, 0x600000u);
  EXPECT_EQ(mapping_access_ordinal, 0u);
  EXPECT_EQ(mapping_granularity, 4096u);
  EXPECT_EQ(required_access, AMDF_MEMORY_ACCESS_READ);

  ASSERT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kReserve, Operation::kMap, Operation::kWait,
                Operation::kQueryDeviceState, Operation::kFree}));
  ASSERT_EQ(state_.maps.size(), 1u);
  EXPECT_EQ(state_.maps[0].address, 0x600000u);
  EXPECT_EQ(state_.maps[0].allocation_handle, 0u);
  EXPECT_EQ(state_.maps[0].source_byte_offset, 0u);
  EXPECT_EQ(state_.maps[0].byte_length, 65536u);
  EXPECT_EQ(state_.maps[0].write, 0u);
  EXPECT_EQ(state_.maps[0].execute, 0u);
  EXPECT_EQ(state_.maps[0].no_access, 1u);
  EXPECT_EQ(state_.waits, (std::vector<uint64_t>{7}));
  ASSERT_EQ(state_.frees.size(), 1u);
  EXPECT_EQ(state_.frees[0].address, 0x410000u);
  EXPECT_EQ(state_.frees[0].byte_length, UINT64_C(2) * 1024 * 1024);
}

TEST_F(LinuxDxgMemoryReservationTest,
       TranslatesAllocationOffsetsAndProtections) {
  const amdf_gpu_dxg_memory_device_t memory_device =
      amdf_gpu_linux_dxg_memory_device(&device_);
  uint64_t mapped_address = 0;
  uint64_t paging_fence = 0;
  ASSERT_EQ(memory_device.native_api->map_allocation(
                memory_device.native_context, memory_device.paging_queue_handle,
                0x900000, 0x20, 8192, 4096,
                AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                &mapped_address, &paging_fence),
            AMDF_STATUS_OK);
  EXPECT_EQ(mapped_address, 0x900000u);
  EXPECT_EQ(paging_fence, 7u);
  ASSERT_EQ(state_.maps.size(), 1u);
  EXPECT_EQ(state_.maps[0].address, 0x900000u);
  EXPECT_EQ(state_.maps[0].allocation_handle, 0x20u);
  EXPECT_EQ(state_.maps[0].source_byte_offset, 8192u);
  EXPECT_EQ(state_.maps[0].byte_length, 4096u);
  EXPECT_EQ(state_.maps[0].write, 0u);
  EXPECT_EQ(state_.maps[0].execute, 1u);
  EXPECT_EQ(state_.maps[0].no_access, 0u);
}

TEST_F(LinuxDxgMemoryReservationTest,
       PublishesQueuedUnmapBeforeTheFollowingCommandPoint) {
  amdf_gpu_umd_memory_reservation_t* reservation = CreateReservation();
  ASSERT_NE(reservation, nullptr);
  amdf_gpu_umd_kernel_queue_t* queue = nullptr;
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_create(
                &device_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4, &queue),
            AMDF_STATUS_OK);
  ASSERT_EQ(amdf_gpu_umd_kernel_queue_mapping_capacity(queue), 128u);

  const amdf_gpu_umd_mapping_update_t update = {
      .target_byte_offset = 8192,
      .byte_length = 4096,
  };
  bool mutation_started = false;
  bool accepted = false;
  ASSERT_EQ(
      amdf_gpu_umd_memory_reservation_update_on_queue(
          reservation, queue, 0, 1, &update, 19, &mutation_started, &accepted),
      AMDF_STATUS_OK);
  EXPECT_TRUE(mutation_started);
  EXPECT_TRUE(accepted);
  EXPECT_EQ(state_.mapping_predecessor, 18u);
  EXPECT_EQ(state_.ordering_point, 19u);
  EXPECT_EQ(state_.mapping_operation.OperationType,
            D3DDDI_UPDATEGPUVIRTUALADDRESS_UNMAP);
  EXPECT_EQ(state_.mapping_operation.Unmap.BaseAddress, 0x412000u);
  EXPECT_EQ(state_.mapping_operation.Unmap.SizeInBytes, 4096u);
  EXPECT_EQ(state_.mapping_operation.Unmap.Protection.NoAccess, 1u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kReserve, Operation::kUpdate,
                                    Operation::kOrder}));

  ASSERT_EQ(amdf_gpu_umd_kernel_queue_destroy(queue), AMDF_STATUS_OK);
  EXPECT_EQ(state_.queue_destroy_count, 1u);
  ASSERT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kReserve, Operation::kUpdate, Operation::kOrder,
                Operation::kMap, Operation::kWait, Operation::kQueryDeviceState,
                Operation::kFree}));
}

TEST_F(LinuxDxgMemoryReservationTest, PreservesAnExactCommonAddress) {
  constexpr uint64_t common_address = 0x800000;
  amdf_gpu_umd_memory_reservation_t* reservation =
      CreateReservation(UINT64_C(2) * 1024 * 1024, common_address);
  ASSERT_NE(reservation, nullptr);
  ASSERT_EQ(state_.reserves.size(), 1u);
  EXPECT_EQ(state_.reserves[0].base_address, common_address);
  EXPECT_EQ(state_.reserves[0].minimum_address, 65536u);
  EXPECT_EQ(state_.reserves[0].byte_length, 65536u);

  uint64_t address = 0;
  uint32_t mapping_access_ordinal = 0;
  uint64_t mapping_granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_umd_memory_reservation_query_access(
      reservation, 0, &address, &mapping_access_ordinal, &mapping_granularity,
      &required_access);
  EXPECT_EQ(address, common_address);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  ASSERT_EQ(state_.frees.size(), 1u);
  EXPECT_EQ(state_.frees[0].address, common_address);
  EXPECT_EQ(state_.frees[0].byte_length, 65536u);
}

TEST_F(LinuxDxgMemoryReservationTest,
       NativeReservationFailurePreservesTheCallerOutput) {
  state_.reserve_status = kStatusNoMemory;
  amdf_gpu_umd_device_t* devices[] = {&device_};
  auto* sentinel = reinterpret_cast<amdf_gpu_umd_memory_reservation_t*>(1);
  EXPECT_EQ(amdf_gpu_umd_memory_reservation_create(allocator_, 1, devices,
                                                   65536, 0, 0, &sentinel),
            amdf_linux_dxg_kmt_make_status(kStatusNoMemory));
  EXPECT_EQ(sentinel, reinterpret_cast<void*>(1));
  EXPECT_EQ(state_.operations, (std::vector<Operation>{Operation::kReserve}));
}

}  // namespace
