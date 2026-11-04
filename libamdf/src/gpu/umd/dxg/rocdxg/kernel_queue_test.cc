// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/dxg/rocdxg/kernel_queue.h"

#include <sys/ioctl.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/adapter_state.h"
#include "shared/include/d3dkmt_types.h"

namespace {

constexpr NTSTATUS kNativeFailure = static_cast<NTSTATUS>(0xC0000001u);
constexpr NTSTATUS kSecondNativeFailure = static_cast<NTSTATUS>(0xC000000Du);

struct TrackingAllocator {
  amdf_allocator_t system = amdf_allocator_system();
  uint32_t allocation_count = 0;
  uint32_t free_count = 0;

  static void* Allocate(void* user_data, size_t byte_length, size_t alignment) {
    auto* self = static_cast<TrackingAllocator*>(user_data);
    ++self->allocation_count;
    return self->system.allocate(self->system.user_data, byte_length,
                                 alignment);
  }

  static void Free(void* user_data, void* pointer) {
    auto* self = static_cast<TrackingAllocator*>(user_data);
    ++self->free_count;
    self->system.free(self->system.user_data, pointer);
  }

  amdf_allocator_t api() {
    return {
        .user_data = this,
        .allocate = Allocate,
        .free = Free,
    };
  }
};

class RocdxgKernelQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    adapter_.host_allocator = allocator_.api();
    adapter_.native_transport = {
        .context = this,
        .invoke = Invoke,
    };
    adapter_.state_shadowing_by_cpfw = true;
    adapter_.pm4_queue_profile = {
        .scheduler_id = 7,
        .node_ordinal = 3,
        .is_supported = true,
        .uses_hardware_scheduling = true,
    };
    adapter_.sdma_queue_profile = {
        .scheduler_id = 9,
        .node_ordinal = 4,
        .is_supported = true,
        .uses_hardware_scheduling = false,
    };
  }

  static uint32_t Invoke(void* context, uint64_t request, void* arguments) {
    return static_cast<RocdxgKernelQueueTest*>(context)->Invoke(request,
                                                                arguments);
  }

  uint32_t Invoke(uint64_t request, void* arguments) {
    EXPECT_EQ(_IOC_TYPE(request), static_cast<unsigned long>('G'));
    const uint32_t operation = _IOC_NR(request);
    operations_.push_back(operation);
    const NTSTATUS status = operation_status_[operation];
    switch (operation) {
      case 0x04: {
        auto* create = static_cast<D3DKMT_CREATECONTEXTVIRTUAL*>(arguments);
        EXPECT_EQ(create->hDevice, 0x42u);
        EXPECT_EQ(create->NodeOrdinal, expected_node_ordinal_);
        EXPECT_EQ(create->EngineAffinity, 1u);
        EXPECT_EQ(create->Flags.HwQueueSupported,
                  expected_hardware_scheduling_);
        EXPECT_EQ(create->PrivateDriverDataSize, 424u);
        EXPECT_NE(create->pPrivateDriverData, nullptr);
        EXPECT_EQ(create->ClientHint, D3DKMT_CLIENTHINT_OPENCL);
        if (status == STATUS_SUCCESS) {
          create->hContext = 0x101;
        }
        break;
      }
      case 0x05: {
        const auto* destroy =
            static_cast<const D3DKMT_DESTROYCONTEXT*>(arguments);
        EXPECT_EQ(destroy->hContext, 0x101u);
        break;
      }
      case 0x0F: {
        const auto* submit =
            static_cast<const D3DKMT_SUBMITCOMMAND*>(arguments);
        EXPECT_EQ(submit->Commands, expected_command_address_);
        EXPECT_EQ(submit->CommandLength, expected_command_length_);
        EXPECT_EQ(submit->BroadcastContextCount, 1u);
        EXPECT_EQ(submit->BroadcastContext[0], 0x101u);
        ObserveSubmissionPrivateData(submit->pPrivateDriverData,
                                     submit->PrivateDriverDataSize);
        break;
      }
      case 0x10: {
        auto* create =
            static_cast<D3DKMT_CREATESYNCHRONIZATIONOBJECT2*>(arguments);
        EXPECT_EQ(create->hDevice, 0x42u);
        EXPECT_EQ(create->Info.Type, D3DDDI_MONITORED_FENCE);
        EXPECT_EQ(create->Info.MonitoredFence.EngineAffinity, 1u);
        if (status == STATUS_SUCCESS) {
          create->hSyncObject = 0x301;
          create->Info.MonitoredFence.FenceValueCPUVirtualAddress =
              const_cast<uint64_t*>(&fence_value_);
          create->Info.MonitoredFence.FenceValueGPUVirtualAddress = 0x9000;
        }
        break;
      }
      case 0x18: {
        auto* create = static_cast<D3DKMT_CREATEHWQUEUE*>(arguments);
        EXPECT_EQ(create->hHwContext, 0x101u);
        EXPECT_EQ(create->Flags.DisableGpuTimeout, 0u);
        EXPECT_EQ(create->PrivateDriverDataSize, 64u);
        EXPECT_NE(create->pPrivateDriverData, nullptr);
        if (status == STATUS_SUCCESS) {
          create->hHwQueue = 0x201;
          create->hHwQueueProgressFence = 0x301;
          create->HwQueueProgressFenceCPUVirtualAddress =
              const_cast<uint64_t*>(&fence_value_);
          create->HwQueueProgressFenceGPUVirtualAddress = 0x9000;
        }
        break;
      }
      case 0x1B: {
        const auto* destroy =
            static_cast<const D3DKMT_DESTROYHWQUEUE*>(arguments);
        EXPECT_EQ(destroy->hHwQueue, 0x201u);
        break;
      }
      case 0x1D: {
        const auto* destroy =
            static_cast<const D3DKMT_DESTROYSYNCHRONIZATIONOBJECT*>(arguments);
        EXPECT_EQ(destroy->hSyncObject, 0x301u);
        break;
      }
      case 0x32: {
        const auto* signal =
            static_cast<const D3DKMT_SIGNALSYNCHRONIZATIONOBJECTFROMGPU*>(
                arguments);
        EXPECT_EQ(signal->hContext, 0x101u);
        EXPECT_EQ(signal->ObjectCount, 1u);
        EXPECT_NE(signal->ObjectHandleArray, nullptr);
        EXPECT_NE(signal->MonitoredFenceValueArray, nullptr);
        if (signal->ObjectHandleArray != nullptr &&
            signal->MonitoredFenceValueArray != nullptr) {
          EXPECT_EQ(signal->ObjectHandleArray[0], 0x301u);
          EXPECT_EQ(signal->MonitoredFenceValueArray[0],
                    expected_progress_value_);
        }
        break;
      }
      case 0x34: {
        const auto* submit =
            static_cast<const D3DKMT_SUBMITCOMMANDTOHWQUEUE*>(arguments);
        EXPECT_EQ(submit->hHwQueue, 0x201u);
        EXPECT_EQ(submit->HwQueueProgressFenceId, expected_progress_value_);
        EXPECT_EQ(submit->CommandBuffer, expected_command_address_);
        EXPECT_EQ(submit->CommandLength, expected_command_length_);
        ObserveSubmissionPrivateData(submit->pPrivateDriverData,
                                     submit->PrivateDriverDataSize);
        break;
      }
      default:
        ADD_FAILURE() << "unexpected DXG operation " << operation;
        break;
    }
    return static_cast<uint32_t>(status);
  }

  void ObserveSubmissionPrivateData(void* pointer, uint32_t byte_length) {
    EXPECT_EQ(byte_length, 72u);
    EXPECT_NE(pointer, nullptr);
    if (submission_private_data_ == nullptr) {
      submission_private_data_ = pointer;
    } else {
      EXPECT_EQ(pointer, submission_private_data_);
    }
  }

  amdf_dxg_bridge_result_t CreateQueue(
      amdf_dxg_bridge_gpu_queue_command_type_t command_type,
      amdf_dxg_bridge_gpu_kernel_queue_t** out_queue,
      amdf_dxg_bridge_gpu_kernel_queue_info_t* out_info,
      uint32_t* out_native_status) {
    expected_node_ordinal_ =
        command_type == AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4 ? 3 : 4;
    expected_hardware_scheduling_ =
        command_type == AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4;
    const amdf_dxg_bridge_gpu_kernel_queue_create_info_t create_info = {
        .structure_size =
            sizeof(amdf_dxg_bridge_gpu_kernel_queue_create_info_t),
        .device_handle = 0x42,
        .command_type = command_type,
    };
    return amdf::rocdxg_bridge::GpuKernelQueueCreate(
        &adapter_, &create_info, out_queue, out_info, out_native_status);
  }

  amdf_dxg_bridge_result_t Submit(amdf_dxg_bridge_gpu_kernel_queue_t* queue,
                                  uint64_t address, uint64_t byte_length,
                                  uint64_t progress_value,
                                  uint32_t* out_accepted,
                                  uint32_t* out_native_status) {
    expected_command_address_ = address;
    expected_command_length_ = static_cast<uint32_t>(byte_length);
    expected_progress_value_ = progress_value;
    return amdf::rocdxg_bridge::GpuKernelQueueSubmit(
        queue, address, byte_length, progress_value, out_accepted,
        out_native_status);
  }

  TrackingAllocator allocator_;
  amdf_dxg_bridge_gpu_adapter_t adapter_;
  std::array<NTSTATUS, 256> operation_status_ = {};
  std::vector<uint32_t> operations_;
  uint64_t fence_value_ = 0;
  uint32_t expected_node_ordinal_ = 0;
  uint32_t expected_hardware_scheduling_ = 0;
  uint64_t expected_command_address_ = 0;
  uint32_t expected_command_length_ = 0;
  uint64_t expected_progress_value_ = 0;
  void* submission_private_data_ = nullptr;
};

TEST_F(RocdxgKernelQueueTest,
       HardwareQueueOwnsFixedSubmissionStorageAndNativeLifetime) {
  amdf_dxg_bridge_gpu_kernel_queue_t* queue = nullptr;
  amdf_dxg_bridge_gpu_kernel_queue_info_t info = {};
  uint32_t native_status = 0;
  ASSERT_EQ(CreateQueue(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &queue,
                        &info, &native_status),
            AMDF_DXG_BRIDGE_RESULT_SUCCESS);
  EXPECT_EQ(native_status, 0u);
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x04, 0x18}));
  EXPECT_EQ(info.structure_size, sizeof(info));
  EXPECT_EQ(info.command_buffer_alignment, 4u);
  EXPECT_EQ(info.maximum_command_buffer_byte_length, 0xFFFFFFFCu);
  EXPECT_EQ(info.progress_fence_handle, 0x301u);
  EXPECT_EQ(info.progress_fence_pointer, &fence_value_);
  EXPECT_EQ(info.progress_fence_device_address, 0x9000u);
  EXPECT_EQ(info.context_handle, 0x101u);
  EXPECT_EQ(info.queue_handle, 0x201u);
  EXPECT_EQ(adapter_.live_queue_count.load(), 1u);

  uint32_t close_status = 0xFFFFFFFFu;
  EXPECT_EQ(
      amdf::rocdxg_bridge::PrepareGpuAdapterClose(&adapter_, &close_status),
      AMDF_DXG_BRIDGE_RESULT_BUSY);
  EXPECT_EQ(close_status, 0u);

  operations_.clear();
  const uint32_t allocation_count = allocator_.allocation_count;
  uint32_t accepted = 0;
  ASSERT_EQ(Submit(queue, 0x1000, 64, 1, &accepted, &native_status),
            AMDF_DXG_BRIDGE_RESULT_SUCCESS);
  EXPECT_EQ(accepted, 1u);
  ASSERT_EQ(Submit(queue, 0x2000, 128, 2, &accepted, &native_status),
            AMDF_DXG_BRIDGE_RESULT_SUCCESS);
  EXPECT_EQ(accepted, 1u);
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x34, 0x34}));
  EXPECT_EQ(allocator_.allocation_count, allocation_count);

  operations_.clear();
  operation_status_[0x1B] = kNativeFailure;
  operation_status_[0x05] = kSecondNativeFailure;
  EXPECT_EQ(amdf::rocdxg_bridge::GpuKernelQueueDestroy(queue, &native_status),
            AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x1B, 0x05}));
  EXPECT_EQ(adapter_.live_queue_count.load(), 0u);
  EXPECT_EQ(allocator_.allocation_count, allocator_.free_count);
}

TEST_F(RocdxgKernelQueueTest,
       SoftwareQueuePreservesAcceptanceAcrossSignalFailure) {
  amdf_dxg_bridge_gpu_kernel_queue_t* queue = nullptr;
  amdf_dxg_bridge_gpu_kernel_queue_info_t info = {};
  uint32_t native_status = 0;
  ASSERT_EQ(CreateQueue(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA, &queue,
                        &info, &native_status),
            AMDF_DXG_BRIDGE_RESULT_SUCCESS);
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x04, 0x10}));
  EXPECT_EQ(info.queue_handle, 0u);
  EXPECT_EQ(info.progress_fence_handle, 0x301u);

  operations_.clear();
  operation_status_[0x32] = kNativeFailure;
  const uint32_t allocation_count = allocator_.allocation_count;
  uint32_t accepted = 0;
  EXPECT_EQ(Submit(queue, 0x3000, 64, 7, &accepted, &native_status),
            AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(accepted, 1u);
  EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x0F, 0x32}));
  EXPECT_EQ(allocator_.allocation_count, allocation_count);

  operations_.clear();
  operation_status_[0x32] = STATUS_SUCCESS;
  operation_status_[0x05] = kNativeFailure;
  operation_status_[0x1D] = kSecondNativeFailure;
  EXPECT_EQ(amdf::rocdxg_bridge::GpuKernelQueueDestroy(queue, &native_status),
            AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE);
  EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
  EXPECT_EQ(operations_, (std::vector<uint32_t>{0x05, 0x1D}));
  EXPECT_EQ(allocator_.allocation_count, allocator_.free_count);
}

TEST_F(RocdxgKernelQueueTest,
       CreationFailuresRollBackAndPreserveCallerOutputs) {
  for (const auto [command_type, failed_operation] :
       std::array<std::pair<amdf_dxg_bridge_gpu_queue_command_type_t, uint32_t>,
                  2>{{
           {AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, 0x18},
           {AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_SDMA, 0x10},
       }}) {
    operations_.clear();
    operation_status_.fill(STATUS_SUCCESS);
    operation_status_[failed_operation] = kNativeFailure;
    amdf_dxg_bridge_gpu_kernel_queue_t* queue =
        reinterpret_cast<amdf_dxg_bridge_gpu_kernel_queue_t*>(0x1234);
    amdf_dxg_bridge_gpu_kernel_queue_info_t info;
    std::memset(&info, 0xA5, sizeof(info));
    const amdf_dxg_bridge_gpu_kernel_queue_info_t original_info = info;
    uint32_t native_status = 0;
    EXPECT_EQ(CreateQueue(command_type, &queue, &info, &native_status),
              AMDF_DXG_BRIDGE_RESULT_NATIVE_FAILURE);
    EXPECT_EQ(queue,
              reinterpret_cast<amdf_dxg_bridge_gpu_kernel_queue_t*>(0x1234));
    EXPECT_EQ(std::memcmp(&info, &original_info, sizeof(info)), 0);
    EXPECT_EQ(native_status, static_cast<uint32_t>(kNativeFailure));
    EXPECT_EQ(operations_,
              (std::vector<uint32_t>{0x04, failed_operation, 0x05}));
    EXPECT_EQ(adapter_.live_queue_count.load(), 0u);
    EXPECT_EQ(allocator_.allocation_count, allocator_.free_count);
  }
}

TEST_F(RocdxgKernelQueueTest, UnsupportedProfileDoesNotStartConstruction) {
  adapter_.pm4_queue_profile.is_supported = false;
  amdf_dxg_bridge_gpu_kernel_queue_t* queue =
      reinterpret_cast<amdf_dxg_bridge_gpu_kernel_queue_t*>(0x1234);
  amdf_dxg_bridge_gpu_kernel_queue_info_t info = {.structure_size = 0xA5A5};
  uint32_t native_status = 0xFFFFFFFFu;
  EXPECT_EQ(CreateQueue(AMDF_DXG_BRIDGE_GPU_QUEUE_COMMAND_TYPE_PM4, &queue,
                        &info, &native_status),
            AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED);
  EXPECT_EQ(queue,
            reinterpret_cast<amdf_dxg_bridge_gpu_kernel_queue_t*>(0x1234));
  EXPECT_EQ(info.structure_size, 0xA5A5u);
  EXPECT_EQ(native_status, 0u);
  EXPECT_TRUE(operations_.empty());
  EXPECT_EQ(allocator_.allocation_count, 0u);
}

}  // namespace
