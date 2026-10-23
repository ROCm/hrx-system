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
#include "iree/hal/drivers/amdgpu/allocator.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;
using iree::hal::cts::SemaphoreList;

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
    iree_hal_pool_release(host_import_pool_);
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
        IREE_SV("legacy-amdgpu"), &options, libhsa, topology,
        create_context_.params(), host_allocator, &device_));
    IREE_RETURN_IF_ERROR(iree_hal_device_group_create_from_device(
        device_, create_context_.frontier_tracker(), host_allocator,
        &device_group_));
    const iree_hal_pool_family_access_t family_access = {
        .family = iree_hal_queue_family(queue()),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family_access,
    };
    const iree_hal_pool_import_class_t import_class = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
    };
    iree_hal_slab_pool_options_t pool_options;
    iree_hal_slab_pool_options_initialize(&pool_options);
    pool_options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
    pool_options.imports.class_count = 1;
    pool_options.imports.classes = &import_class;
    pool_options.trace_name = IREE_SV("amdgpu.test.host.import");
    return iree_hal_slab_pool_create(device_group_, scope, &pool_options,
                                     host_allocator, &host_import_pool_);
  }

  iree_hal_device_t* device() const { return device_; }

  iree_hal_queue_t* queue() const {
    return iree_hal_device_queue(device_, /*family_ordinal=*/0,
                                 /*queue_ordinal=*/0);
  }

  iree_hal_pool_t* host_import_pool() const { return host_import_pool_; }

 private:
  // Provides the proactor pool and causal frontier tracker for the device.
  iree::hal::cts::DeviceCreateContext create_context_;

  // Test-owned device reference released before the owning device group.
  iree_hal_device_t* device_ = NULL;

  // Owns the topology assigned to the logical device.
  iree_hal_device_group_t* device_group_ = NULL;

  // Explicit import route for caller-owned host allocations.
  iree_hal_pool_t* host_import_pool_ = NULL;
};

class HostQueueHostMemoryTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    host_allocator_ = iree_allocator_system();
    iree_status_t status = iree_hal_amdgpu_libhsa_initialize(
        IREE_HAL_AMDGPU_LIBHSA_FLAG_NONE, iree_string_view_list_empty(),
        host_allocator_, &libhsa_);
    if (iree_status_is_unavailable(status)) {
      iree_status_fprint(stderr, status);
      iree_status_free(status);
      GTEST_SKIP() << "HSA not available, skipping tests";
    }
    IREE_ASSERT_OK(status);
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

  iree_status_t ImportHostAllocationBuffer(iree_hal_pool_t* pool,
                                           std::vector<uint8_t>* data,
                                           iree_hal_buffer_t** out_buffer) {
    iree_hal_external_storage_t external_buffer = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        .byte_length = data->size(),
        .host_cacheability = IREE_HAL_HOST_CACHEABILITY_WRITE_BACK,
        .maintenance_range = iree_hal_make_buffer_range(0, data->size()),
    };
    external_buffer.payload.host_pointer = data->data();
    return iree_hal_pool_import_buffer(
        pool, &external_buffer,
        iree_hal_make_buffer_range(0, IREE_HAL_WHOLE_BUFFER), out_buffer);
  }

  iree_status_t CreatePatternedDeviceBuffer(iree_hal_device_t* device,
                                            iree_hal_queue_t* queue,
                                            iree_device_size_t buffer_size,
                                            uint8_t pattern,
                                            iree_hal_buffer_t** out_buffer) {
    *out_buffer = NULL;
    const iree_hal_amdgpu_buffer_spec_t spec = {
        .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
        .access = IREE_HAL_MEMORY_ACCESS_ALL,
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
        .queue_family_affinity = IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
    };

    iree_hal_buffer_t* buffer = NULL;
    iree_status_t status = iree_hal_amdgpu_allocator_allocate_direct_buffer(
        iree_hal_device_allocator(device), spec, buffer_size, &buffer);
    SemaphoreList signal_list(device, {0}, {1});
    if (iree_status_is_ok(status)) {
      status = iree_hal_queue_fill(
          queue, iree_hal_semaphore_list_empty(), signal_list, buffer,
          /*target_offset=*/0, buffer_size, &pattern, sizeof(pattern),
          /*barriers=*/NULL, IREE_HAL_FILL_FLAG_NONE);
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

iree_allocator_t HostQueueHostMemoryTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueHostMemoryTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueHostMemoryTest::topology_;

TEST_F(HostQueueHostMemoryTest, ImportedHostCopyPublishesWritesAndOffsets) {
  constexpr iree_device_size_t kDeviceAllocationSize = 640;
  constexpr iree_device_size_t kReadSubspanOffset = 73;
  constexpr iree_device_size_t kWriteSubspanOffset = 91;
  constexpr iree_device_size_t kDeviceSubspanLength = 400;
  constexpr iree_device_size_t kReadBufferOffset = 31;
  constexpr iree_device_size_t kWriteBufferOffset = 47;
  constexpr size_t kSourceHostOffset = 19;
  constexpr size_t kTargetHostOffset = 29;
  constexpr iree_device_size_t kTransferLength = 257;
  constexpr size_t kHostAllocationSize = 512;

  // Caller storage remains live through native attachment retirement.
  std::vector<uint8_t> source_data(kHostAllocationSize, 0x11);
  std::vector<uint8_t> target_data(kHostAllocationSize, 0xC7);
  TestLogicalDevice test_device;
  IREE_ASSERT_OK(test_device.Initialize(&libhsa_, &topology_, host_allocator_));

  Ref<iree_hal_buffer_t> source_host_buffer;
  IREE_ASSERT_OK(ImportHostAllocationBuffer(
      test_device.host_import_pool(), &source_data, source_host_buffer.out()));
  Ref<iree_hal_buffer_t> target_host_buffer;
  IREE_ASSERT_OK(ImportHostAllocationBuffer(
      test_device.host_import_pool(), &target_data, target_host_buffer.out()));

  Ref<iree_hal_buffer_t> read_device_allocation;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.device(), test_device.queue(), kDeviceAllocationSize, 0x5E,
      read_device_allocation.out()));
  Ref<iree_hal_buffer_t> read_device_subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      read_device_allocation, kReadSubspanOffset, kDeviceSubspanLength,
      host_allocator_, read_device_subspan.out()));
  Ref<iree_hal_buffer_t> write_device_allocation;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.device(), test_device.queue(), kDeviceAllocationSize, 0xA6,
      write_device_allocation.out()));
  Ref<iree_hal_buffer_t> write_device_subspan;
  IREE_ASSERT_OK(iree_hal_buffer_subspan(
      write_device_allocation, kWriteSubspanOffset, kDeviceSubspanLength,
      host_allocator_, write_device_subspan.out()));

  const std::vector<uint8_t> transfer_data =
      MakePatternData(static_cast<size_t>(kTransferLength));
  std::memcpy(source_data.data() + kSourceHostOffset, transfer_data.data(),
              transfer_data.size());

  SemaphoreList read_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_copy(
      test_device.queue(), iree_hal_semaphore_list_empty(), read_semaphore,
      source_host_buffer, kSourceHostOffset, read_device_subspan,
      kReadBufferOffset, kTransferLength, /*barriers=*/NULL,
      IREE_HAL_COPY_FLAG_NONE));

  SemaphoreList copy_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_copy(
      test_device.queue(), read_semaphore, copy_semaphore,
      read_device_allocation, kReadSubspanOffset + kReadBufferOffset,
      write_device_allocation, kWriteSubspanOffset + kWriteBufferOffset,
      kTransferLength, /*barriers=*/NULL, IREE_HAL_COPY_FLAG_NONE));

  SemaphoreList write_semaphore(test_device.device(), {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_copy(
      test_device.queue(), copy_semaphore, write_semaphore,
      write_device_subspan, kWriteBufferOffset, target_host_buffer,
      kTargetHostOffset, kTransferLength, /*barriers=*/NULL,
      IREE_HAL_COPY_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      write_semaphore, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  for (size_t i = 0; i < target_data.size(); ++i) {
    const bool is_transferred =
        i >= kTargetHostOffset &&
        i < kTargetHostOffset + static_cast<size_t>(kTransferLength);
    const uint8_t expected =
        is_transferred ? transfer_data[i - kTargetHostOffset] : 0xC7;
    ASSERT_EQ(expected, target_data[i]) << "byte mismatch at offset " << i;
  }
}

}  // namespace
}  // namespace iree::hal::amdgpu
