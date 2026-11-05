// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/wsl/allocation.h"

#include <sys/ioctl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/private_records.h"
#include "libamdf/src/gpu/umd/dxg/wsl/adapter_state.h"
#include "libamdf/src/platform/linux/dxg/kmt_types.h"

namespace {

constexpr uint64_t kMaximumAllocationByteLength = 2ull * 1024 * 1024 * 1024;
constexpr NTSTATUS kNativeFailure = static_cast<NTSTATUS>(0xC0000017u);

struct ExpectedAllocation {
  uint64_t byte_length;
  amdf_dxg_private_allocation_domain_t domain;
  amdf_dxg_private_allocation_flags_t flags;
  const void* host_pointer;
};

class WslDxgAllocationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    adapter_.host_allocator = amdf_allocator_system();
    adapter_.native_transport = {
        .context = this,
        .invoke = Invoke,
    };
    adapter_.private_info.gfx_ip_major = 11;
    adapter_.private_info.is_discrete = 1;
    adapter_.private_info.invisible_local_heap_byte_length = 64 * 1024;
  }

  static uint32_t Invoke(void* context, uint64_t request, void* arguments) {
    return static_cast<WslDxgAllocationTest*>(context)->Invoke(request,
                                                               arguments);
  }

  uint32_t Invoke(uint64_t request, void* arguments) {
    ++invoke_count_;
    EXPECT_EQ(_IOC_TYPE(request), static_cast<unsigned long>('G'));
    EXPECT_EQ(_IOC_NR(request), 0x06u);
    EXPECT_EQ(_IOC_SIZE(request), sizeof(D3DKMT_CREATEALLOCATION));
    auto* create = static_cast<D3DKMT_CREATEALLOCATION*>(arguments);
    EXPECT_EQ(create->hDevice, 0x42u);
    EXPECT_EQ(create->PrivateDriverDataSize,
              AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH);
    EXPECT_EQ(create->NumAllocations, expected_allocations_.size());
    EXPECT_NE(create->pPrivateDriverData, nullptr);
    EXPECT_NE(create->pAllocationInfo2, nullptr);

    std::array<uint8_t, AMDF_DXG_PRIVATE_ALLOCATION_DRIVER_RECORD_BYTE_LENGTH>
        expected_driver_record;
    amdf_dxg_private_initialize_allocation_driver_record(
        expected_driver_record.data());
    EXPECT_EQ(
        std::memcmp(create->pPrivateDriverData, expected_driver_record.data(),
                    expected_driver_record.size()),
        0);

    for (uint32_t i = 0; i < create->NumAllocations; ++i) {
      const ExpectedAllocation& expected = expected_allocations_[i];
      const D3DDDI_ALLOCATIONINFO2& native_info = create->pAllocationInfo2[i];
      EXPECT_EQ(native_info.pSystemMem, expected.host_pointer);
      EXPECT_EQ(native_info.PrivateDriverDataSize,
                AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH);
      EXPECT_EQ(native_info.VidPnSourceId, D3DDDI_ID_UNINITIALIZED);

      std::array<uint8_t, AMDF_DXG_PRIVATE_ALLOCATION_RECORD_BYTE_LENGTH>
          expected_record;
      const amdf_dxg_private_allocation_info_t private_info = {
          .byte_length = expected.byte_length,
          .domain = expected.domain,
          .flags = expected.flags,
      };
      amdf_dxg_private_initialize_wsl_allocation_record(
          &private_info, &adapter_.private_info, expected_record.data());
      EXPECT_EQ(std::memcmp(native_info.pPrivateDriverData,
                            expected_record.data(), expected_record.size()),
                0);
    }

    if (native_status_ == STATUS_SUCCESS) {
      create->hResource = 0x100;
      for (uint32_t i = 0; i < create->NumAllocations; ++i) {
        create->pAllocationInfo2[i].hAllocation = 0x200 + i;
      }
    }
    return static_cast<uint32_t>(native_status_);
  }

  amdf_dxg_provider_result_t Create(
      const amdf_dxg_provider_gpu_allocation_create_info_t& create_info,
      uint32_t capacity, uint32_t* handles, uint32_t* out_resource_handle,
      uint32_t* out_allocation_count, uint32_t* out_native_status) {
    return amdf::wsl_provider::GpuAllocationCreate(
        &adapter_, &create_info, capacity, handles, out_resource_handle,
        out_allocation_count, out_native_status);
  }

  amdf_dxg_provider_gpu_adapter_t adapter_;
  std::vector<ExpectedAllocation> expected_allocations_;
  NTSTATUS native_status_ = STATUS_SUCCESS;
  uint32_t invoke_count_ = 0;
};

TEST_F(WslDxgAllocationTest, EncodesSystemPoliciesAndPublishesNativeHandles) {
  alignas(4096) std::array<uint8_t, 4096> host_storage = {};
  expected_allocations_ = {{
      host_storage.size(),
      AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_SYSTEM,
      AMDF_DXG_PRIVATE_ALLOCATION_FLAG_FINE_GRAIN |
          AMDF_DXG_PRIVATE_ALLOCATION_FLAG_DEVICE_UNCACHED,
      host_storage.data(),
  }};
  const amdf_dxg_provider_gpu_allocation_create_info_t create_info = {
      .device_handle = 0x42,
      .domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_SYSTEM,
      .flags = AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_FINE_GRAIN |
               AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_QUEUE_STORAGE |
               AMDF_DXG_PROVIDER_GPU_ALLOCATION_FLAG_DEVICE_UNCACHED,
      .byte_length = host_storage.size(),
      .placement_device_address = 0,
      .host_pointer = host_storage.data(),
  };

  uint32_t handles[1] = {};
  uint32_t resource_handle = 0;
  uint32_t allocation_count = 0;
  uint32_t native_status = 99;
  EXPECT_EQ(Create(create_info, 1, handles, &resource_handle, &allocation_count,
                   &native_status),
            AMDF_DXG_PROVIDER_RESULT_SUCCESS);
  EXPECT_EQ(handles[0], 0x200u);
  EXPECT_EQ(resource_handle, 0x100u);
  EXPECT_EQ(allocation_count, 1u);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(invoke_count_, 1u);
}

TEST_F(WslDxgAllocationTest, SplitsLargeLocalAllocationsIntoFixedRecords) {
  expected_allocations_ = {
      {kMaximumAllocationByteLength, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL,
       0, nullptr},
      {4096, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL, 0, nullptr},
  };
  const amdf_dxg_provider_gpu_allocation_create_info_t create_info = {
      .device_handle = 0x42,
      .domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL,
      .flags = 0,
      .byte_length = kMaximumAllocationByteLength + 4096,
      .placement_device_address = 0x1000,
      .host_pointer = nullptr,
  };

  uint32_t handles[2] = {};
  uint32_t resource_handle = 0;
  uint32_t allocation_count = 0;
  uint32_t native_status = 99;
  EXPECT_EQ(Create(create_info, 2, handles, &resource_handle, &allocation_count,
                   &native_status),
            AMDF_DXG_PROVIDER_RESULT_SUCCESS);
  EXPECT_EQ(handles[0], 0x200u);
  EXPECT_EQ(handles[1], 0x201u);
  EXPECT_EQ(allocation_count, 2u);
}

TEST_F(WslDxgAllocationTest, PreservesOutputsBeforeAndAfterNativeInvocation) {
  const amdf_dxg_provider_gpu_allocation_create_info_t create_info = {
      .device_handle = 0x42,
      .domain = AMDF_DXG_PROVIDER_GPU_ALLOCATION_DOMAIN_LOCAL,
      .flags = 0,
      .byte_length = kMaximumAllocationByteLength + 4096,
      .placement_device_address = 0x1000,
      .host_pointer = nullptr,
  };
  uint32_t handles[2] = {0xA0, 0xA1};
  uint32_t resource_handle = 0xB0;
  uint32_t allocation_count = 0xB1;
  uint32_t native_status = 0xB2;

  EXPECT_EQ(Create(create_info, 1, handles, &resource_handle, &allocation_count,
                   &native_status),
            AMDF_DXG_PROVIDER_RESULT_BUFFER_TOO_SMALL);
  EXPECT_EQ(handles[0], 0xA0u);
  EXPECT_EQ(resource_handle, 0xB0u);
  EXPECT_EQ(allocation_count, 2u);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(invoke_count_, 0u);

  expected_allocations_ = {
      {kMaximumAllocationByteLength, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL,
       0, nullptr},
      {4096, AMDF_DXG_PRIVATE_ALLOCATION_DOMAIN_LOCAL, 0, nullptr},
  };
  native_status_ = kNativeFailure;
  allocation_count = 0xB1;
  EXPECT_EQ(Create(create_info, 2, handles, &resource_handle, &allocation_count,
                   &native_status),
            AMDF_DXG_PROVIDER_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(handles[0], 0xA0u);
  EXPECT_EQ(handles[1], 0xA1u);
  EXPECT_EQ(resource_handle, 0xB0u);
  EXPECT_EQ(allocation_count, 0xB1u);
  EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
  EXPECT_EQ(invoke_count_, 1u);
}

}  // namespace
