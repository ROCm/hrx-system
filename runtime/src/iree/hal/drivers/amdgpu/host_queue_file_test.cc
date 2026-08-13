// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using iree::hal::cts::SemaphoreList;

constexpr iree_hal_queue_affinity_t kQueueAffinity0 =
    ((iree_hal_queue_affinity_t)1ull) << 0;

std::vector<uint8_t> MakePatternData(size_t size) {
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>((i * 131 + (i >> 7) * 17 + 0x5A) & 0xFF);
  }
  return data;
}

class TestLogicalDevice {
 public:
  ~TestLogicalDevice() {
    iree_hal_device_release(device_);
    iree_hal_device_group_release(device_group_);
  }

  iree_status_t Initialize(const iree_hal_amdgpu_libhsa_t* libhsa,
                           const iree_hal_amdgpu_topology_t* topology,
                           iree_allocator_t host_allocator) {
    IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
    iree_hal_amdgpu_logical_device_options_t options;
    iree_hal_amdgpu_logical_device_options_initialize(&options);
    options.preallocate_pools = 0;
    IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
        IREE_SV("amdgpu"), &options, libhsa, topology, create_context_.params(),
        host_allocator, &device_));
    return iree_hal_device_group_create_from_device(
        device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_);
  }

  iree_hal_device_t* device() const { return device_; }

 private:
  // Provides the proactor pool and causal frontier tracker for the device.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the owning device group.
  iree_hal_device_t* device_ = NULL;

  // Owns the topology assigned to the logical device.
  iree_hal_device_group_t* device_group_ = NULL;
};

class HostQueueFileTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (!iree_status_is_ok(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(iree_hal_amdgpu_topology_initialize_with_defaults(
        &libhsa_, &topology_));
    if (topology_.gpu_agent_count == 0) {
      GTEST_SKIP() << "no GPU devices available, skipping tests";
    }
  }

  static void TearDownTestSuite() {
    iree_hal_amdgpu_topology_deinitialize(&topology_);
    iree_hal_amdgpu_libhsa_deinitialize(&libhsa_);
  }

  iree_status_t ImportHostAllocationFile(iree_hal_device_t* device,
                                         std::vector<uint8_t>* data,
                                         iree_hal_memory_access_t access,
                                         iree_hal_file_t** out_file) {
    iree_io_file_handle_t* handle = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(data->data(), data->size()),
        iree_io_file_handle_release_callback_null(), host_allocator_, &handle));
    iree_status_t status = iree_hal_file_import(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  iree_status_t CreatePatternedDeviceBuffer(iree_hal_device_t* device,
                                            iree_device_size_t buffer_size,
                                            uint8_t pattern,
                                            iree_hal_buffer_t** out_buffer) {
    *out_buffer = NULL;
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_DISPATCH_STORAGE;

    iree_hal_buffer_t* buffer = NULL;
    iree_status_t status = iree_hal_allocator_allocate_buffer(
        iree_hal_device_allocator(device), params, buffer_size, &buffer);
    SemaphoreList signal_list(device, {0}, {1});
    if (iree_status_is_ok(status)) {
      status = iree_hal_device_queue_fill(
          device, kQueueAffinity0, iree_hal_semaphore_list_empty(), signal_list,
          buffer, /*target_offset=*/0, buffer_size, &pattern, sizeof(pattern),
          IREE_HAL_FILL_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_semaphore_list_wait(
          signal_list, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE);
    }
    if (iree_status_is_ok(status)) {
      *out_buffer = buffer;
    } else {
      iree_hal_buffer_release(buffer);
    }
    return status;
  }

  // Allocator shared by the suite-owned runtime objects.
  static iree_allocator_t host_allocator_;

  // Dynamically loaded HSA API table shared by the test suite.
  static iree_hal_amdgpu_libhsa_t libhsa_;

  // Host and device agents discovered for the test system.
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueFileTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueFileTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueFileTest::topology_;

TEST_F(HostQueueFileTest, MemoryFileReadWritePreservesQueueOrderingAndOffsets) {
  constexpr iree_device_size_t kDeviceAllocationSize = 640;
  constexpr iree_device_size_t kReadSubspanOffset = 73;
  constexpr iree_device_size_t kWriteSubspanOffset = 91;
  constexpr iree_device_size_t kDeviceSubspanLength = 400;
  constexpr iree_device_size_t kReadBufferOffset = 31;
  constexpr iree_device_size_t kWriteBufferOffset = 47;
  constexpr size_t kSourceFileOffset = 19;
  constexpr size_t kTargetFileOffset = 29;
  constexpr iree_device_size_t kTransferLength = 257;
  constexpr size_t kHostAllocationSize = 512;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  std::vector<uint8_t> source_data(kHostAllocationSize, 0x11);
  std::vector<uint8_t> target_data(kHostAllocationSize, 0xC7);
  Ref<iree_hal_file_t> source_file;
  IREE_ASSERT_OK(ImportHostAllocationFile(test_device.device(), &source_data,
                                          IREE_HAL_MEMORY_ACCESS_READ,
                                          source_file.out()));
  Ref<iree_hal_file_t> target_file;
  IREE_ASSERT_OK(ImportHostAllocationFile(test_device.device(), &target_data,
                                          IREE_HAL_MEMORY_ACCESS_WRITE,
                                          target_file.out()));

  Ref<iree_hal_buffer_t> read_device_allocation;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(test_device.device(),
                                             kDeviceAllocationSize, 0x5E,
                                             read_device_allocation.out()));
  Ref<iree_hal_buffer_t> read_device_subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      read_device_allocation, kReadSubspanOffset, kDeviceSubspanLength,
      host_allocator_, read_device_subspan.out()));
  Ref<iree_hal_buffer_t> write_device_allocation;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(test_device.device(),
                                             kDeviceAllocationSize, 0xA6,
                                             write_device_allocation.out()));
  Ref<iree_hal_buffer_t> write_device_subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      write_device_allocation, kWriteSubspanOffset, kDeviceSubspanLength,
      host_allocator_, write_device_subspan.out()));

  SemaphoreList gate_semaphore(test_device.device(), {0}, {1});
  SemaphoreList read_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_read(
      test_device.device(), kQueueAffinity0, gate_semaphore, read_semaphore,
      source_file, kSourceFileOffset, read_device_subspan, kReadBufferOffset,
      kTransferLength, IREE_HAL_READ_FLAG_NONE));

  SemaphoreList copy_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_copy(
      test_device.device(), kQueueAffinity0, read_semaphore, copy_semaphore,
      read_device_allocation, kReadSubspanOffset + kReadBufferOffset,
      write_device_allocation, kWriteSubspanOffset + kWriteBufferOffset,
      kTransferLength, IREE_HAL_COPY_FLAG_NONE));

  SemaphoreList write_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_device_queue_write(
      test_device.device(), kQueueAffinity0, copy_semaphore, write_semaphore,
      write_device_subspan, kWriteBufferOffset, target_file, kTargetFileOffset,
      kTransferLength, IREE_HAL_WRITE_FLAG_NONE));

  uint64_t completed_value = 0;
  IREE_ASSERT_OK(iree_hal_semaphore_query(write_semaphore.semaphores[0],
                                          &completed_value));
  EXPECT_EQ(0u, completed_value);

  const std::vector<uint8_t> transfer_data =
      MakePatternData(static_cast<size_t>(kTransferLength));
  std::memcpy(source_data.data() + kSourceFileOffset, transfer_data.data(),
              transfer_data.size());
  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_semaphore.semaphores[0], 1,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      write_semaphore, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  for (size_t i = 0; i < target_data.size(); ++i) {
    const bool is_transferred =
        i >= kTargetFileOffset &&
        i < kTargetFileOffset + static_cast<size_t>(kTransferLength);
    const uint8_t expected =
        is_transferred ? transfer_data[i - kTargetFileOffset] : 0xC7;
    ASSERT_EQ(expected, target_data[i]) << "byte mismatch at offset " << i;
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
