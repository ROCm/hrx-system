// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/memory_reservation.h"

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/memory_profile.h"

namespace {

enum class Operation {
  kReserve,
  kMap,
  kWait,
  kFree,
};

struct ReservationRequest {
  uint64_t base_address;
  uint64_t minimum_address;
  uint64_t byte_length;
};

struct MappingRequest {
  uint64_t address;
  uint32_t allocation_handle;
  uint64_t source_byte_offset;
  uint64_t byte_length;
  amdf_memory_access_t access;
};

struct ReleaseRequest {
  uint64_t address;
  uint64_t byte_length;
};

struct FakeState {
  uint32_t reserve_count = 0;
  uint64_t next_paging_fence = 1;
  uint32_t failing_map_ordinal = UINT32_MAX;
  uint32_t map_count = 0;
  std::vector<Operation> operations;
  std::vector<ReservationRequest> reservations;
  std::vector<MappingRequest> mappings;
  std::vector<uint64_t> wait_targets;
  std::vector<ReleaseRequest> releases;
};

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL
FakeQueryAllocationLayout(amdf_dxg_bridge_gpu_adapter_t* adapter,
                          uint64_t byte_length, uint32_t* out_allocation_count,
                          uint64_t* out_maximum_allocation_byte_length) {
  EXPECT_NE(adapter, nullptr);
  EXPECT_EQ(byte_length, 65536u);
  *out_allocation_count = 2;
  *out_maximum_allocation_byte_length = 32768;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL FakeCreateAllocations(
    amdf_dxg_bridge_gpu_adapter_t* adapter,
    const amdf_dxg_bridge_gpu_allocation_create_info_t* create_info,
    uint32_t allocation_handle_capacity, uint32_t* out_allocation_handles,
    uint32_t* out_resource_handle, uint32_t* out_allocation_count,
    uint32_t* out_native_status) {
  EXPECT_NE(adapter, nullptr);
  EXPECT_EQ(create_info->device_handle, 0x10u);
  EXPECT_EQ(create_info->domain, AMDF_DXG_BRIDGE_GPU_ALLOCATION_DOMAIN_LOCAL);
  EXPECT_EQ(create_info->byte_length, 65536u);
  EXPECT_EQ(create_info->placement_device_address, 0x100000u);
  EXPECT_EQ(create_info->host_pointer, nullptr);
  EXPECT_EQ(allocation_handle_capacity, 2u);
  out_allocation_handles[0] = 0x20;
  out_allocation_handles[1] = 0x21;
  *out_resource_handle = 0;
  *out_allocation_count = 2;
  *out_native_status = 0;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_status_t FakeReserveAddress(void* context, uint32_t adapter_handle,
                                 uint64_t base_address,
                                 uint64_t minimum_address, uint64_t byte_length,
                                 uint64_t* out_address) {
  auto* state = static_cast<FakeState*>(context);
  state->operations.push_back(Operation::kReserve);
  state->reservations.push_back({base_address, minimum_address, byte_length});
  EXPECT_EQ(adapter_handle, 0x08u);
  if (base_address != 0) {
    *out_address = base_address;
  } else {
    *out_address = state->reserve_count == 0 ? 0x100000 : 0x410000;
  }
  ++state->reserve_count;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeMapAllocation(void* context, uint32_t paging_queue_handle,
                                uint64_t address, uint32_t allocation_handle,
                                uint64_t source_byte_offset,
                                uint64_t byte_length,
                                amdf_memory_access_t access,
                                uint64_t* out_address,
                                uint64_t* out_paging_fence) {
  auto* state = static_cast<FakeState*>(context);
  state->operations.push_back(Operation::kMap);
  state->mappings.push_back(
      {address, allocation_handle, source_byte_offset, byte_length, access});
  EXPECT_EQ(paging_queue_handle, 0x40u);
  const uint32_t ordinal = state->map_count++;
  if (ordinal == state->failing_map_ordinal) {
    return amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
  }
  *out_address = address;
  *out_paging_fence = state->next_paging_fence++;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeMakeResident(void* context, uint32_t paging_queue_handle,
                               uint32_t allocation_count,
                               const uint32_t* allocation_handles,
                               uint32_t* out_resident_allocation_count,
                               uint64_t* out_paging_fence) {
  auto* state = static_cast<FakeState*>(context);
  EXPECT_EQ(paging_queue_handle, 0x40u);
  EXPECT_EQ(allocation_count, 2u);
  EXPECT_EQ(allocation_handles[0], 0x20u);
  EXPECT_EQ(allocation_handles[1], 0x21u);
  *out_resident_allocation_count = allocation_count;
  *out_paging_fence = state->next_paging_fence++;
  return AMDF_STATUS_OK;
}

amdf_status_t FakeWaitForPaging(void* context, uint32_t device_handle,
                                uint32_t synchronization_object_handle,
                                const volatile uint64_t* progress_fence,
                                uint64_t target_value) {
  auto* state = static_cast<FakeState*>(context);
  state->operations.push_back(Operation::kWait);
  state->wait_targets.push_back(target_value);
  EXPECT_EQ(device_handle, 0x10u);
  EXPECT_EQ(synchronization_object_handle, 0x50u);
  EXPECT_NE(progress_fence, nullptr);
  return AMDF_STATUS_OK;
}

amdf_gpu_dxg_memory_release_result_t FakeDestroyAllocations(
    void* context, uint32_t device_handle, uint32_t resource_handle,
    uint32_t allocation_count, const uint32_t* allocation_handles,
    bool synchronous) {
  (void)context;
  EXPECT_EQ(device_handle, 0x10u);
  EXPECT_EQ(resource_handle, 0u);
  EXPECT_EQ(allocation_count, 2u);
  EXPECT_EQ(allocation_handles[0], 0x20u);
  EXPECT_EQ(allocation_handles[1], 0x21u);
  EXPECT_FALSE(synchronous);
  return {
      .status = AMDF_STATUS_OK,
      .ownership = AMDF_GPU_DXG_MEMORY_OWNERSHIP_CONSUMED,
  };
}

amdf_status_t FakeFreeAddress(void* context, uint32_t adapter_handle,
                              uint64_t address, uint64_t byte_length) {
  auto* state = static_cast<FakeState*>(context);
  state->operations.push_back(Operation::kFree);
  state->releases.push_back({address, byte_length});
  EXPECT_EQ(adapter_handle, 0x08u);
  return AMDF_STATUS_OK;
}

amdf_status_t FakeInvalidateAllocation(void* context, uint32_t device_handle,
                                       uint32_t allocation_handle,
                                       uint64_t byte_offset,
                                       uint64_t byte_length) {
  (void)context;
  (void)device_handle;
  (void)allocation_handle;
  (void)byte_offset;
  (void)byte_length;
  return AMDF_STATUS_OK;
}

amdf_gpu_dxg_memory_reservation_device_t DescribeDevice(const void* context,
                                                        uint32_t ordinal) {
  const auto* devices =
      static_cast<const amdf_gpu_dxg_memory_reservation_device_t*>(context);
  return devices[ordinal];
}

class DxgMemoryReservationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    bridge_api_.gpu_allocation_query_layout = FakeQueryAllocationLayout;
    bridge_api_.gpu_allocation_create = FakeCreateAllocations;
    bridge_adapter_ = {
        .api = &bridge_api_,
        .native = reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(&state_),
    };
    native_api_ = {
        .reserve_address = FakeReserveAddress,
        .map_allocation = FakeMapAllocation,
        .make_resident = FakeMakeResident,
        .wait_for_paging = FakeWaitForPaging,
        .destroy_allocations = FakeDestroyAllocations,
        .free_address = FakeFreeAddress,
        .invalidate_allocation = FakeInvalidateAllocation,
    };
    device_ = {
        .host_allocator = amdf_allocator_system(),
        .native_api = &native_api_,
        .native_context = &state_,
        .bridge_adapter = &bridge_adapter_,
        .adapter_handle = 0x08,
        .adapter_luid = UINT64_C(0x1122334455667788),
        .device_handle = 0x10,
        .paging_queue_handle = 0x40,
        .paging_synchronization_object_handle = 0x50,
        .paging_progress_fence = &paging_progress_,
    };
    memory_capabilities_ = {
        .virtual_address_bit_count = 48,
        .read_only_memory_supported = 1,
        .no_execute_memory_supported = 1,
    };
    reservation_device_ = {
        .native_api = device_.native_api,
        .native_context = device_.native_context,
        .adapter_handle = device_.adapter_handle,
        .adapter_luid = device_.adapter_luid,
        .physical_adapter_index = 3,
        .virtual_address_bit_count =
            memory_capabilities_.virtual_address_bit_count,
        .read_only_memory_supported =
            memory_capabilities_.read_only_memory_supported,
        .no_execute_memory_supported =
            memory_capabilities_.no_execute_memory_supported,
        .device_handle = device_.device_handle,
        .paging_queue_handle = device_.paging_queue_handle,
        .paging_synchronization_object_handle =
            device_.paging_synchronization_object_handle,
        .paging_progress_fence = device_.paging_progress_fence,
    };
  }

  void TearDown() override {
    if (source_ != nullptr) {
      EXPECT_EQ(amdf_gpu_dxg_memory_destroy(source_), AMDF_STATUS_OK);
    }
  }

  void PrepareSource() {
    amdf_memory_native_profile_t profile = {};
    ASSERT_EQ(
        amdf_gpu_dxg_query_memory_profile(&memory_capabilities_, 1, &profile),
        AMDF_STATUS_OK);
    const amdf_memory_native_create_info_t create_info = {
        .device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL |
                          AMDF_MEMORY_FLAG_DEVICE_ADDRESS |
                          AMDF_MEMORY_FLAG_MAPPING_SOURCE,
        .byte_length = 65536,
        .minimum_alignment = 65536,
    };
    amdf_gpu_umd_memory_result_t result = {};
    ASSERT_EQ(amdf_gpu_dxg_memory_prepare(&device_, &profile, &create_info,
                                          &source_, &result),
              AMDF_STATUS_OK);
    ASSERT_EQ(result.device_address, 0x100000u);
    state_.operations.clear();
    state_.reservations.clear();
    state_.mappings.clear();
    state_.wait_targets.clear();
    state_.releases.clear();
    state_.map_count = 0;
    state_.next_paging_fence = 1;
  }

  amdf_gpu_dxg_memory_reservation_t* CreateReservation(
      uint64_t minimum_alignment = 0, uint64_t common_address = 0) {
    amdf_gpu_dxg_memory_reservation_t* reservation = nullptr;
    EXPECT_EQ(
        amdf_gpu_dxg_memory_reservation_create(
            device_.host_allocator, 1, DescribeDevice, &reservation_device_,
            65536, minimum_alignment, common_address, &reservation),
        AMDF_STATUS_OK);
    return reservation;
  }

  FakeState state_;
  volatile uint64_t paging_progress_ = 0;
  amdf_dxg_bridge_api_t bridge_api_ = {};
  amdf_gpu_dxg_adapter_t bridge_adapter_ = {};
  amdf_gpu_dxg_memory_native_api_t native_api_ = {};
  amdf_gpu_dxg_memory_device_t device_ = {};
  amdf_gpu_dxg_memory_capabilities_t memory_capabilities_ = {};
  amdf_gpu_dxg_memory_reservation_device_t reservation_device_ = {};
  amdf_gpu_dxg_memory_t* source_ = nullptr;
};

TEST_F(DxgMemoryReservationTest,
       DerivesCapabilitiesAndRejectsDifferentMappingDomains) {
  amdf_memory_reservation_capabilities_t capabilities = {};
  ASSERT_EQ(amdf_gpu_dxg_query_memory_reservation_capabilities(
                &reservation_device_, 0, &capabilities),
            AMDF_STATUS_OK);
  EXPECT_EQ(capabilities.supported_access, AMDF_MEMORY_ACCESS_READ |
                                               AMDF_MEMORY_ACCESS_WRITE |
                                               AMDF_MEMORY_ACCESS_EXECUTE);
  EXPECT_EQ(capabilities.required_access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(capabilities.reservation_granularity, 65536u);
  EXPECT_EQ(capabilities.mapping_granularity, 4096u);
  EXPECT_EQ(capabilities.minimum_address, 65536u);
  EXPECT_EQ(capabilities.maximum_address, (UINT64_C(1) << 48) - 1);

  auto unsupported_capabilities = capabilities;
  EXPECT_EQ(amdf_gpu_dxg_query_memory_reservation_capabilities(
                &reservation_device_, 1, &unsupported_capabilities),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(unsupported_capabilities.maximum_address,
            capabilities.maximum_address);

  amdf_gpu_dxg_memory_reservation_device_t devices[] = {reservation_device_,
                                                        reservation_device_};
  devices[1].physical_adapter_index = 4;
  auto* sentinel = reinterpret_cast<amdf_gpu_dxg_memory_reservation_t*>(1);
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_create(device_.host_allocator, 2,
                                                   DescribeDevice, devices,
                                                   65536, 0, 0, &sentinel),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
  EXPECT_EQ(sentinel, reinterpret_cast<void*>(1));
  EXPECT_TRUE(state_.operations.empty());
}

TEST_F(DxgMemoryReservationTest,
       AlignsReservationAndMapsAcrossPhysicalAllocationSpans) {
  PrepareSource();
  amdf_gpu_dxg_memory_reservation_t* reservation =
      CreateReservation(UINT64_C(2) * 1024 * 1024);
  ASSERT_NE(reservation, nullptr);

  uint64_t address = 0;
  uint32_t mapping_access_ordinal = UINT32_MAX;
  uint64_t mapping_granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_dxg_memory_reservation_query_access(
      reservation, 0, &address, &mapping_access_ordinal, &mapping_granularity,
      &required_access);
  EXPECT_EQ(address, 0x600000u);
  EXPECT_EQ(mapping_access_ordinal, 0u);
  EXPECT_EQ(mapping_granularity, 4096u);
  EXPECT_EQ(required_access, AMDF_MEMORY_ACCESS_READ);
  ASSERT_EQ(state_.reservations.size(), 1u);
  EXPECT_EQ(state_.reservations[0].base_address, 0u);
  EXPECT_EQ(state_.reservations[0].minimum_address, 65536u);
  EXPECT_EQ(state_.reservations[0].byte_length, UINT64_C(2) * 1024 * 1024);

  const amdf_gpu_umd_mapping_update_t update = {
      .target_byte_offset = 8192,
      .byte_length = 8192,
      .memory = reinterpret_cast<amdf_gpu_umd_memory_t*>(source_),
      .source_byte_offset = 28672,
      .access = AMDF_MEMORY_ACCESS_READ,
  };
  bool mutation_started = false;
  ASSERT_EQ(amdf_gpu_dxg_memory_reservation_update(reservation, 0, 1, &update,
                                                   &mutation_started),
            AMDF_STATUS_OK);
  EXPECT_TRUE(mutation_started);
  ASSERT_EQ(state_.mappings.size(), 2u);
  EXPECT_EQ(state_.mappings[0].address, 0x602000u);
  EXPECT_EQ(state_.mappings[0].allocation_handle, 0x20u);
  EXPECT_EQ(state_.mappings[0].source_byte_offset, 28672u);
  EXPECT_EQ(state_.mappings[0].byte_length, 4096u);
  EXPECT_EQ(state_.mappings[0].access, AMDF_MEMORY_ACCESS_READ);
  EXPECT_EQ(state_.mappings[1].address, 0x603000u);
  EXPECT_EQ(state_.mappings[1].allocation_handle, 0x21u);
  EXPECT_EQ(state_.mappings[1].source_byte_offset, 0u);
  EXPECT_EQ(state_.mappings[1].byte_length, 4096u);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{2}));

  ASSERT_EQ(amdf_gpu_dxg_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  ASSERT_EQ(state_.mappings.size(), 3u);
  EXPECT_EQ(state_.mappings[2].address, 0x600000u);
  EXPECT_EQ(state_.mappings[2].allocation_handle, 0u);
  EXPECT_EQ(state_.mappings[2].byte_length, 65536u);
  EXPECT_EQ(state_.mappings[2].access, 0u);
  ASSERT_EQ(state_.releases.size(), 1u);
  EXPECT_EQ(state_.releases[0].address, 0x410000u);
  EXPECT_EQ(state_.releases[0].byte_length, UINT64_C(2) * 1024 * 1024);
}

TEST_F(DxgMemoryReservationTest,
       ExpandsQueueOperationsAcrossPhysicalAllocationSpans) {
  PrepareSource();
  amdf_gpu_dxg_memory_reservation_t* reservation = CreateReservation();
  ASSERT_NE(reservation, nullptr);
  const amdf_gpu_umd_mapping_update_t updates[] = {
      {
          .target_byte_offset = 8192,
          .byte_length = 8192,
          .memory = reinterpret_cast<amdf_gpu_umd_memory_t*>(source_),
          .source_byte_offset = 28672,
          .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
      },
      {
          .target_byte_offset = 32768,
          .byte_length = 4096,
          .memory = nullptr,
          .source_byte_offset = 0,
          .access = 0,
      },
  };
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_mapping_operation_count(reservation,
                                                                    2, updates),
            3u);
  std::vector<amdf_gpu_dxg_mapping_operation_t> operations(3);
  amdf_gpu_dxg_memory_reservation_visit_mapping_operations(
      reservation, 2, updates,
      [](void* context, uint32_t ordinal,
         const amdf_gpu_dxg_mapping_operation_t* operation) {
        auto* operations =
            static_cast<std::vector<amdf_gpu_dxg_mapping_operation_t>*>(
                context);
        (*operations)[ordinal] = *operation;
      },
      &operations);
  EXPECT_EQ(operations[0].target_address, 0x412000u);
  EXPECT_EQ(operations[0].byte_length, 4096u);
  EXPECT_EQ(operations[0].allocation_handle, 0x20u);
  EXPECT_EQ(operations[0].source_byte_offset, 28672u);
  EXPECT_EQ(operations[0].access,
            AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE);
  EXPECT_EQ(operations[1].target_address, 0x413000u);
  EXPECT_EQ(operations[1].allocation_handle, 0x21u);
  EXPECT_EQ(operations[1].source_byte_offset, 0u);
  EXPECT_EQ(operations[2].target_address, 0x418000u);
  EXPECT_EQ(operations[2].byte_length, 4096u);
  EXPECT_EQ(operations[2].allocation_handle, 0u);
  EXPECT_EQ(operations[2].access, 0u);
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
}

TEST_F(DxgMemoryReservationTest,
       FailedMappingReportsMutationAndSkipsThePagingWait) {
  PrepareSource();
  amdf_gpu_dxg_memory_reservation_t* reservation = CreateReservation();
  ASSERT_NE(reservation, nullptr);
  state_.failing_map_ordinal = 1;
  const amdf_gpu_umd_mapping_update_t update = {
      .target_byte_offset = 0,
      .byte_length = 8192,
      .memory = reinterpret_cast<amdf_gpu_umd_memory_t*>(source_),
      .source_byte_offset = 28672,
      .access = AMDF_MEMORY_ACCESS_READ,
  };
  bool mutation_started = false;
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_update(reservation, 0, 1, &update,
                                                   &mutation_started),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  EXPECT_TRUE(mutation_started);
  EXPECT_TRUE(state_.wait_targets.empty());

  state_.failing_map_ordinal = UINT32_MAX;
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  EXPECT_EQ(state_.wait_targets, (std::vector<uint64_t>{2}));
}

TEST_F(DxgMemoryReservationTest, PreservesAnExactBorrowedCommonAddress) {
  constexpr uint64_t common_address = 0x800000;
  amdf_gpu_dxg_memory_reservation_t* reservation =
      CreateReservation(UINT64_C(2) * 1024 * 1024, common_address);
  ASSERT_NE(reservation, nullptr);
  uint64_t address = 0;
  uint32_t mapping_access_ordinal = 0;
  uint64_t mapping_granularity = 0;
  amdf_memory_access_t required_access = 0;
  amdf_gpu_dxg_memory_reservation_query_access(
      reservation, 0, &address, &mapping_access_ordinal, &mapping_granularity,
      &required_access);
  EXPECT_EQ(address, common_address);
  ASSERT_EQ(state_.reservations.size(), 1u);
  EXPECT_EQ(state_.reservations[0].base_address, common_address);
  EXPECT_EQ(state_.reservations[0].byte_length, 65536u);
  EXPECT_EQ(amdf_gpu_dxg_memory_reservation_destroy(reservation),
            AMDF_STATUS_OK);
  ASSERT_EQ(state_.releases.size(), 1u);
  EXPECT_EQ(state_.releases[0].address, common_address);
  EXPECT_EQ(state_.releases[0].byte_length, 65536u);
}

}  // namespace
