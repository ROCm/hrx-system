// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/drivers/amdgpu/host_queue_staging.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "iree/hal/api.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/hal/drivers/amdgpu/host_queue.h"
#include "iree/hal/drivers/amdgpu/logical_device.h"
#include "iree/hal/drivers/amdgpu/physical_device.h"
#include "iree/hal/drivers/amdgpu/queue_affinity.h"
#include "iree/hal/drivers/amdgpu/system.h"
#include "iree/hal/drivers/amdgpu/util/aql_emitter.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::amdgpu {
namespace {

using iree::hal::cts::Ref;

constexpr iree_hal_queue_affinity_t kQueueAffinity0 =
    ((iree_hal_queue_affinity_t)1ull) << 0;

constexpr iree_device_size_t kStagingSlotSize =
    IREE_HAL_AMDGPU_STAGING_SLOT_ALIGNMENT;
constexpr iree_device_size_t kMultiSlotTransferSize =
    kStagingSlotSize * 2 + 4096;

std::vector<uint8_t> MakePatternData(size_t size) {
  std::vector<uint8_t> data(size);
  for (size_t i = 0; i < size; ++i) {
    data[i] = static_cast<uint8_t>((i * 131 + (i >> 7) * 17 + 0x5A) & 0xFF);
  }
  return data;
}

static iree_status_t CreateSemaphore(iree_hal_device_t* device,
                                     iree_hal_semaphore_t** out_semaphore) {
  return iree_hal_semaphore_create(
      device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      /*initial_value=*/0, IREE_HAL_SEMAPHORE_FLAG_DEFAULT, out_semaphore);
}

static iree_hal_semaphore_list_t MakeSemaphoreList(
    iree_hal_semaphore_t** semaphore, uint64_t* payload_value) {
  return iree_hal_semaphore_list_t{
      /*count=*/1,
      /*semaphores=*/semaphore,
      /*payload_values=*/payload_value,
  };
}

static bool HostQueueHasPendingOperation(iree_hal_amdgpu_host_queue_t* queue) {
  iree_slim_mutex_lock(&queue->locks.submission_mutex);
  const bool has_operation = queue->pending_head != NULL;
  iree_slim_mutex_unlock(&queue->locks.submission_mutex);
  return has_operation;
}

static iree_status_t EnqueueRawBlockingBarrier(
    iree_hal_amdgpu_host_queue_t* queue, hsa_signal_t blocker_signal) {
  const uint64_t packet_id =
      iree_hal_amdgpu_aql_ring_reserve(&queue->aql_ring, /*count=*/1);
  iree_hal_amdgpu_aql_packet_t* packet =
      iree_hal_amdgpu_aql_ring_packet(&queue->aql_ring, packet_id);
  const hsa_signal_t dep_signals[1] = {blocker_signal};
  const uint16_t header = iree_hal_amdgpu_aql_emit_barrier_and(
      &packet->barrier_and, dep_signals, IREE_ARRAYSIZE(dep_signals),
      iree_hal_amdgpu_aql_packet_control_barrier_system(),
      iree_hsa_signal_null());
  iree_hal_amdgpu_aql_ring_commit(packet, header, /*setup=*/0);
  iree_hal_amdgpu_aql_ring_doorbell(&queue->aql_ring, packet_id);
  return iree_ok_status();
}

static void ExpectByteRangeRepeated(const std::vector<uint8_t>& data,
                                    uint8_t pattern) {
  for (size_t i = 0; i < data.size(); ++i) {
    if (data[i] != pattern) {
      ADD_FAILURE() << "byte mismatch at offset " << i << ": expected 0x"
                    << std::hex << static_cast<int>(pattern) << ", got 0x"
                    << static_cast<int>(data[i]);
      return;
    }
  }
}

static void ExpectByteRangeMatches(const std::vector<uint8_t>& actual,
                                   const std::vector<uint8_t>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  if (!actual.empty()) {
    EXPECT_EQ(std::memcmp(actual.data(), expected.data(), actual.size()), 0);
  }
}

class HostQueueStagingTest : public ::testing::Test {
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

  class TestLogicalDevice {
   public:
    ~TestLogicalDevice() {
      iree_hal_device_release(base_device_);
      iree_hal_device_group_release(device_group_);
    }

    iree_status_t Initialize(
        const iree_hal_amdgpu_logical_device_options_t* options,
        const iree_hal_amdgpu_libhsa_t* libhsa,
        const iree_hal_amdgpu_topology_t* topology,
        iree_allocator_t host_allocator) {
      IREE_RETURN_IF_ERROR(create_context_.Initialize(host_allocator));
      IREE_RETURN_IF_ERROR(iree_hal_amdgpu_logical_device_create(
          IREE_SV("amdgpu"), options, libhsa, topology,
          create_context_.params(), host_allocator, &base_device_));
      return iree_hal_device_group_create_from_device(
          base_device_, create_context_.frontier_tracker(), host_allocator,
          &device_group_);
    }

    iree_status_t ReinitializeFileStagingPool(
        const iree_hal_amdgpu_staging_pool_options_t* options,
        iree_allocator_t host_allocator) {
      iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
      iree_hal_amdgpu_physical_device_t* physical_device =
          this->first_physical_device();
      if (!physical_device) {
        return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                "test device has no physical devices");
      }

      iree_hal_amdgpu_staging_pool_deinitialize(
          &physical_device->file_staging_pool);
      return iree_hal_amdgpu_staging_pool_initialize(
          base_device_, &logical_device->system->libhsa,
          &logical_device->system->topology,
          &physical_device->host_memory_pools,
          iree_hal_make_queue_family_affinity(
              (iree_hal_queue_family_ordinal_t)physical_device->device_ordinal),
          options, host_allocator, &physical_device->file_staging_pool);
    }

    iree_hal_device_t* base_device() const { return base_device_; }

    iree_hal_allocator_t* allocator() const {
      return iree_hal_device_allocator(base_device_);
    }

    iree_hal_amdgpu_logical_device_t* logical_device() const {
      return (iree_hal_amdgpu_logical_device_t*)base_device_;
    }

    iree_hal_amdgpu_physical_device_t* first_physical_device() const {
      iree_hal_amdgpu_logical_device_t* logical_device = this->logical_device();
      if (logical_device->physical_device_count == 0) return NULL;
      return logical_device->physical_devices[0];
    }

    iree_hal_amdgpu_host_queue_t* first_host_queue() const {
      iree_hal_amdgpu_physical_device_t* physical_device =
          this->first_physical_device();
      if (!physical_device || physical_device->host_queue_count == 0) {
        return NULL;
      }
      return &physical_device->host_queues[0];
    }

   private:
    // Creation context supplying the proactor pool and frontier tracker.
    iree::hal::cts::DeviceCreateContext create_context_;

    // Test-owned device reference released before the topology-owning group.
    iree_hal_device_t* base_device_ = NULL;

    // Device group that owns the topology assigned to |base_device_|.
    iree_hal_device_group_t* device_group_ = NULL;
  };

  iree_hal_amdgpu_staging_pool_options_t OneSlotStagingOptions() {
    iree_hal_amdgpu_staging_pool_options_t options;
    iree_hal_amdgpu_staging_pool_options_initialize(&options);
    options.slot_size = kStagingSlotSize;
    options.slot_count = 1;
    return options;
  }

  iree_status_t CreateTestDevice(
      const iree_hal_amdgpu_logical_device_options_t* options,
      TestLogicalDevice* out_device) {
    IREE_RETURN_IF_ERROR(
        out_device->Initialize(options, &libhsa_, &topology_, host_allocator_));
    iree_hal_amdgpu_staging_pool_options_t staging_options =
        OneSlotStagingOptions();
    return out_device->ReinitializeFileStagingPool(&staging_options,
                                                   host_allocator_);
  }

  iree_status_t CreateTempFileWithContents(
      const std::vector<uint8_t>& data, iree::testing::TempFilePath* out_path) {
#if IREE_FILE_IO_ENABLE
    *out_path =
        iree::testing::TempFilePath("iree_hal_amdgpu_host_queue_staging");
    return iree_io_file_contents_write(
        out_path->path_view(),
        iree_make_const_byte_span(data.data(), data.size()),
        iree_allocator_system());
#else
    (void)data;
    out_path->Reset();
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
#endif  // IREE_FILE_IO_ENABLE
  }

  iree_status_t CreatePatternedTempFile(size_t size, uint8_t pattern,
                                        iree::testing::TempFilePath* out_path) {
    std::vector<uint8_t> data(size, pattern);
    return CreateTempFileWithContents(data, out_path);
  }

  iree_status_t TruncateTempFile(const iree::testing::TempFilePath& path) {
#if IREE_FILE_IO_ENABLE
    // The HAL retains its imported handle while this handle truncates the file.
    // Both handles must admit the other's access on Windows.
    iree_io_file_handle_t* handle = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_create(
        IREE_IO_FILE_MODE_WRITE | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE,
        path.path_view(), /*initial_size=*/0, iree_allocator_system(),
        &handle));
    iree_io_file_handle_release(handle);
    return iree_ok_status();
#else
    (void)path;
    return iree_make_status(IREE_STATUS_UNAVAILABLE, "file I/O is disabled");
#endif  // IREE_FILE_IO_ENABLE
  }

  iree_status_t ReadTempFileContents(const iree::testing::TempFilePath& path,
                                     std::vector<uint8_t>* out_data) {
    out_data->clear();
    iree_io_file_contents_t* contents = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_contents_read(
        path.path_view(), iree_allocator_system(), &contents));
    out_data->assign(
        contents->const_buffer.data,
        contents->const_buffer.data + contents->const_buffer.data_length);
    iree_io_file_contents_free(contents);
    return iree_ok_status();
  }

  iree_status_t ImportNativeFile(iree_hal_device_t* device,
                                 const iree::testing::TempFilePath& path,
                                 iree_hal_memory_access_t access,
                                 iree_hal_file_t** out_file) {
    // Tests inspect or truncate the backing path while the HAL retains this
    // imported handle.
    iree_io_file_mode_t mode =
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_ASYNC |
        IREE_IO_FILE_MODE_SHARE_READ | IREE_IO_FILE_MODE_SHARE_WRITE;
    if (iree_all_bits_set(access, IREE_HAL_MEMORY_ACCESS_WRITE)) {
      mode |= IREE_IO_FILE_MODE_WRITE;
    }
    iree_io_file_handle_t* handle = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        mode, path.path_view(), iree_allocator_system(), &handle));
    iree_status_t status = iree_hal_file_import(
        device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access, handle,
        IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  iree_status_t CreatePatternedDeviceBuffer(iree_hal_allocator_t* allocator,
                                            iree_hal_device_t* device,
                                            iree_device_size_t buffer_size,
                                            uint8_t pattern,
                                            iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage =
        IREE_HAL_BUFFER_USAGE_TRANSFER | IREE_HAL_BUFFER_USAGE_STORAGE;
    IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
        allocator, params, buffer_size, out_buffer));
    return FillDeviceBufferRange(device, *out_buffer, /*offset=*/0, buffer_size,
                                 pattern);
  }

  iree_status_t QueueAllocaHostVisibleTransientBuffer(
      iree_hal_device_t* device, iree_device_size_t buffer_size,
      iree_hal_semaphore_list_t signal_list, iree_hal_buffer_t** out_buffer) {
    iree_hal_buffer_params_t params = {0};
    params.type =
        IREE_HAL_MEMORY_TYPE_HOST_VISIBLE | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    params.access = IREE_HAL_MEMORY_ACCESS_ALL;
    params.usage = IREE_HAL_BUFFER_USAGE_TRANSFER |
                   IREE_HAL_BUFFER_USAGE_STORAGE |
                   IREE_HAL_BUFFER_USAGE_MAPPING_SCOPED;
    params.queue_family_affinity = iree_hal_make_queue_family_affinity(0);
    params.min_alignment = 16;
    return iree_hal_device_queue_alloca(
        device, kQueueAffinity0, iree_hal_semaphore_list_empty(), signal_list,
        /*pool=*/NULL, params, buffer_size, IREE_HAL_ALLOCA_FLAG_NONE,
        out_buffer);
  }

  iree_status_t ImportHostAllocationFile(iree_hal_device_t* device,
                                         std::vector<uint8_t>* data,
                                         iree_hal_file_t** out_file) {
    iree_io_file_handle_t* handle = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(data->data(), data->size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    iree_status_t status =
        iree_hal_file_import(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                             IREE_HAL_MEMORY_ACCESS_WRITE, handle,
                             IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file);
    iree_io_file_handle_release(handle);
    return status;
  }

  iree_status_t FillDeviceBufferRange(iree_hal_device_t* device,
                                      iree_hal_buffer_t* buffer,
                                      iree_device_size_t offset,
                                      iree_device_size_t length,
                                      uint8_t pattern) {
    if (length == 0) return iree_ok_status();
    Ref<iree_hal_semaphore_t> signal_semaphore;
    IREE_RETURN_IF_ERROR(CreateSemaphore(device, signal_semaphore.out()));
    uint64_t signal_value = 1;
    iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
    iree_hal_semaphore_list_t signal_list =
        MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_fill(
        device, IREE_HAL_QUEUE_AFFINITY_ANY, iree_hal_semaphore_list_empty(),
        signal_list, buffer, offset, length, &pattern, sizeof(pattern),
        IREE_HAL_FILL_FLAG_NONE));
    return iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t QueueReadAndWait(iree_hal_device_t* device,
                                 iree_hal_file_t* source_file,
                                 uint64_t source_offset,
                                 iree_hal_buffer_t* target_buffer,
                                 iree_device_size_t target_offset,
                                 iree_device_size_t length) {
    Ref<iree_hal_semaphore_t> signal_semaphore;
    IREE_RETURN_IF_ERROR(CreateSemaphore(device, signal_semaphore.out()));
    uint64_t signal_value = 1;
    iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
    iree_hal_semaphore_list_t signal_list =
        MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);
    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
    if (!queue) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "test device has no provisioned queue");
    }
    IREE_RETURN_IF_ERROR(
        iree_hal_queue_read(queue, iree_hal_semaphore_list_empty(), signal_list,
                            source_file, source_offset, target_buffer,
                            target_offset, length, IREE_HAL_READ_FLAG_NONE));
    return iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t QueueWriteAndWait(iree_hal_device_t* device,
                                  iree_hal_buffer_t* source_buffer,
                                  iree_device_size_t source_offset,
                                  iree_hal_file_t* target_file,
                                  uint64_t target_offset,
                                  iree_device_size_t length) {
    Ref<iree_hal_semaphore_t> signal_semaphore;
    IREE_RETURN_IF_ERROR(CreateSemaphore(device, signal_semaphore.out()));
    uint64_t signal_value = 1;
    iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
    iree_hal_semaphore_list_t signal_list =
        MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);
    iree_hal_queue_t* queue = iree_hal_device_queue(
        device, /*family_ordinal=*/0, /*queue_ordinal=*/0);
    if (!queue) {
      return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                              "test device has no provisioned queue");
    }
    IREE_RETURN_IF_ERROR(iree_hal_queue_write(
        queue, iree_hal_semaphore_list_empty(), signal_list, source_buffer,
        source_offset, target_file, target_offset, length,
        IREE_HAL_WRITE_FLAG_NONE));
    return iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                   iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE);
  }

  iree_status_t ReadBufferContents(iree_hal_device_t* device,
                                   iree_hal_buffer_t* buffer,
                                   iree_device_size_t offset,
                                   iree_device_size_t length,
                                   std::vector<uint8_t>* out_data) {
    out_data->assign((size_t)length, 0);
    iree_io_file_handle_t* handle = NULL;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(out_data->data(), out_data->size()),
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    Ref<iree_hal_file_t> file;
    iree_status_t status =
        iree_hal_file_import(device, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
                             IREE_HAL_MEMORY_ACCESS_WRITE, handle,
                             IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out());
    iree_io_file_handle_release(handle);
    if (iree_status_is_ok(status)) {
      status = QueueWriteAndWait(device, buffer, offset, file,
                                 /*target_offset=*/0, length);
    }
    return status;
  }

  static iree_allocator_t host_allocator_;
  static iree_hal_amdgpu_libhsa_t libhsa_;
  static iree_hal_amdgpu_topology_t topology_;
};

iree_allocator_t HostQueueStagingTest::host_allocator_;
iree_hal_amdgpu_libhsa_t HostQueueStagingTest::libhsa_;
iree_hal_amdgpu_topology_t HostQueueStagingTest::topology_;

TEST_F(HostQueueStagingTest, LogicalDeviceOptionsSetFileStagingPool) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;
  options.file_staging.slot_size = kStagingSlotSize;
  options.file_staging.slot_count = 2;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(
      test_device.Initialize(&options, &libhsa_, &topology_, host_allocator_));
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.first_physical_device();
  ASSERT_NE(physical_device, nullptr);
  EXPECT_EQ(physical_device->file_staging_pool.slot_size, kStagingSlotSize);
  EXPECT_EQ(physical_device->file_staging_pool.slot_count, 2u);
}

#if IREE_FILE_IO_ENABLE

TEST_F(HostQueueStagingTest, OneSlotLargeReadCompletesThroughSlotWaiter) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&options, &test_device));
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.first_physical_device();
  ASSERT_NE(physical_device, nullptr);
  ASSERT_EQ(physical_device->file_staging_pool.slot_count, 1u);

  std::vector<uint8_t> file_data = MakePatternData(kMultiSlotTransferSize);
  iree::testing::TempFilePath path;
  IREE_ASSERT_OK(CreateTempFileWithContents(file_data, &path));

  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(ImportNativeFile(test_device.base_device(), path,
                                  IREE_HAL_MEMORY_ACCESS_READ, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.allocator(), test_device.base_device(),
      kMultiSlotTransferSize, 0x00, buffer.out()));

  IREE_ASSERT_OK(QueueReadAndWait(test_device.base_device(), file,
                                  /*source_offset=*/0, buffer,
                                  /*target_offset=*/0, kMultiSlotTransferSize));

  std::vector<uint8_t> contents;
  IREE_ASSERT_OK(ReadBufferContents(test_device.base_device(), buffer,
                                    /*offset=*/0, kMultiSlotTransferSize,
                                    &contents));
  ExpectByteRangeMatches(contents, file_data);
}

TEST_F(HostQueueStagingTest,
       HostVisibleTransientReadWriteCompletesBeforeQueuedDealloca) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&options, &test_device));

  std::vector<uint8_t> file_data = MakePatternData(kMultiSlotTransferSize);
  iree::testing::TempFilePath path;
  IREE_ASSERT_OK(CreateTempFileWithContents(file_data, &path));

  Ref<iree_hal_file_t> source_file;
  IREE_ASSERT_OK(ImportNativeFile(test_device.base_device(), path,
                                  IREE_HAL_MEMORY_ACCESS_READ,
                                  source_file.out()));
  std::vector<uint8_t> output_data(kMultiSlotTransferSize, 0);
  Ref<iree_hal_file_t> output_file;
  IREE_ASSERT_OK(ImportHostAllocationFile(test_device.base_device(),
                                          &output_data, output_file.out()));

  Ref<iree_hal_semaphore_t> alloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), alloca_signal.out()));
  uint64_t alloca_signal_value = 1;
  iree_hal_semaphore_t* alloca_signal_ptr = alloca_signal.get();
  iree_hal_semaphore_list_t alloca_signal_list =
      MakeSemaphoreList(&alloca_signal_ptr, &alloca_signal_value);

  iree_hal_buffer_t* transient_raw = NULL;
  IREE_ASSERT_OK(QueueAllocaHostVisibleTransientBuffer(
      test_device.base_device(), kMultiSlotTransferSize, alloca_signal_list,
      &transient_raw));
  Ref<iree_hal_buffer_t> transient_buffer(transient_raw);

  Ref<iree_hal_semaphore_t> gate_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), gate_signal.out()));
  uint64_t gate_signal_value = 1;
  iree_hal_semaphore_t* read_wait_semaphores[] = {alloca_signal.get(),
                                                  gate_signal.get()};
  uint64_t read_wait_values[] = {alloca_signal_value, gate_signal_value};
  iree_hal_semaphore_list_t read_wait_list = {
      IREE_ARRAYSIZE(read_wait_semaphores),
      read_wait_semaphores,
      read_wait_values,
  };

  Ref<iree_hal_semaphore_t> read_signal;
  IREE_ASSERT_OK(CreateSemaphore(test_device.base_device(), read_signal.out()));
  uint64_t read_signal_value = 1;
  iree_hal_semaphore_t* read_signal_ptr = read_signal.get();
  iree_hal_semaphore_list_t read_signal_list =
      MakeSemaphoreList(&read_signal_ptr, &read_signal_value);
  IREE_ASSERT_OK(iree_hal_queue_read(
      &test_device.first_host_queue()->base, read_wait_list, read_signal_list,
      source_file, /*source_offset=*/0, transient_buffer, /*target_offset=*/0,
      kMultiSlotTransferSize, IREE_HAL_READ_FLAG_NONE));

  Ref<iree_hal_semaphore_t> write_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), write_signal.out()));
  uint64_t write_signal_value = 1;
  iree_hal_semaphore_t* write_signal_ptr = write_signal.get();
  iree_hal_semaphore_list_t write_wait_list =
      MakeSemaphoreList(&read_signal_ptr, &read_signal_value);
  iree_hal_semaphore_list_t write_signal_list =
      MakeSemaphoreList(&write_signal_ptr, &write_signal_value);
  IREE_ASSERT_OK(iree_hal_queue_write(
      &test_device.first_host_queue()->base, write_wait_list, write_signal_list,
      transient_buffer, /*source_offset=*/0, output_file, /*target_offset=*/0,
      kMultiSlotTransferSize, IREE_HAL_WRITE_FLAG_NONE));

  Ref<iree_hal_semaphore_t> dealloca_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), dealloca_signal.out()));
  uint64_t dealloca_signal_value = 1;
  iree_hal_semaphore_t* dealloca_signal_ptr = dealloca_signal.get();
  iree_hal_semaphore_list_t dealloca_wait_list =
      MakeSemaphoreList(&write_signal_ptr, &write_signal_value);
  iree_hal_semaphore_list_t dealloca_signal_list =
      MakeSemaphoreList(&dealloca_signal_ptr, &dealloca_signal_value);
  IREE_ASSERT_OK(iree_hal_device_queue_dealloca(
      test_device.base_device(), kQueueAffinity0, dealloca_wait_list,
      dealloca_signal_list, transient_buffer, IREE_HAL_DEALLOCA_FLAG_NONE));

  IREE_ASSERT_OK(iree_hal_semaphore_signal(gate_signal, gate_signal_value,
                                           /*frontier=*/NULL));
  IREE_ASSERT_OK(iree_hal_semaphore_wait(dealloca_signal, dealloca_signal_value,
                                         iree_infinite_timeout(),
                                         IREE_ASYNC_WAIT_FLAG_NONE));

  ExpectByteRangeMatches(output_data, file_data);
}

TEST_F(HostQueueStagingTest, OneSlotLargeWriteCompletesThroughSlotWaiter) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&options, &test_device));
  iree_hal_amdgpu_physical_device_t* physical_device =
      test_device.first_physical_device();
  ASSERT_NE(physical_device, nullptr);
  ASSERT_EQ(physical_device->file_staging_pool.slot_count, 1u);

  iree::testing::TempFilePath path;
  IREE_ASSERT_OK(CreatePatternedTempFile(kMultiSlotTransferSize, 0x00, &path));

  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(ImportNativeFile(
      test_device.base_device(), path,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.allocator(), test_device.base_device(),
      kMultiSlotTransferSize, 0xA7, buffer.out()));

  IREE_ASSERT_OK(QueueWriteAndWait(test_device.base_device(), buffer,
                                   /*source_offset=*/0, file,
                                   /*target_offset=*/0,
                                   kMultiSlotTransferSize));

  std::vector<uint8_t> contents;
  IREE_ASSERT_OK(ReadTempFileContents(path, &contents));
  ASSERT_EQ(contents.size(), kMultiSlotTransferSize);
  ExpectByteRangeRepeated(contents, 0xA7);
}

TEST_F(HostQueueStagingTest, CapacityParkedStagedWriteRetriesAfterPostDrain) {
  static constexpr uint32_t kAqlCapacity = 64;
  static constexpr uint32_t kNotificationCapacity = 1;
  static constexpr uint32_t kKernargCapacity = 2 * kAqlCapacity;

  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.host_queues.aql_capacity = kAqlCapacity;
  options.host_queues.notification_capacity = kNotificationCapacity;
  options.host_queues.kernarg_capacity = kKernargCapacity;
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&options, &test_device));
  iree_hal_amdgpu_host_queue_t* queue = test_device.first_host_queue();
  ASSERT_NE(queue, nullptr);

  iree::testing::TempFilePath path;
  IREE_ASSERT_OK(CreatePatternedTempFile(kMultiSlotTransferSize, 0x00, &path));
  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(ImportNativeFile(
      test_device.base_device(), path,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, file.out()));

  Ref<iree_hal_buffer_t> source_buffer;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.allocator(), test_device.base_device(),
      kMultiSlotTransferSize, 0x3C, source_buffer.out()));
  Ref<iree_hal_buffer_t> pressure_buffer;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.allocator(), test_device.base_device(), sizeof(uint32_t),
      0x00, pressure_buffer.out()));

  hsa_signal_t blocker_signal = iree_hsa_signal_null();
  IREE_ASSERT_OK(iree_hsa_amd_signal_create(
      IREE_LIBHSA(&libhsa_), /*initial_value=*/1, /*num_consumers=*/0,
      /*consumers=*/NULL, /*attributes=*/0, &blocker_signal));
  IREE_ASSERT_OK(EnqueueRawBlockingBarrier(queue, blocker_signal));

  Ref<iree_hal_semaphore_t> pressure_signal;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), pressure_signal.out()));
  uint64_t pressure_signal_value = 1;
  iree_hal_semaphore_t* pressure_signal_ptr = pressure_signal.get();
  iree_hal_semaphore_list_t pressure_signal_list =
      MakeSemaphoreList(&pressure_signal_ptr, &pressure_signal_value);
  const uint32_t pressure_pattern = 0xABCD1234u;
  iree_status_t status = iree_hal_device_queue_fill(
      test_device.base_device(), kQueueAffinity0,
      iree_hal_semaphore_list_empty(), pressure_signal_list, pressure_buffer,
      /*target_offset=*/0, sizeof(pressure_pattern), &pressure_pattern,
      sizeof(pressure_pattern), IREE_HAL_FILL_FLAG_NONE);

  Ref<iree_hal_semaphore_t> write_signal;
  if (iree_status_is_ok(status)) {
    status = CreateSemaphore(test_device.base_device(), write_signal.out());
  }
  uint64_t write_signal_value = 1;
  iree_hal_semaphore_t* write_signal_ptr = write_signal.get();
  iree_hal_semaphore_list_t write_signal_list =
      MakeSemaphoreList(&write_signal_ptr, &write_signal_value);
  if (iree_status_is_ok(status)) {
    status = iree_hal_queue_write(
        &queue->base, iree_hal_semaphore_list_empty(), write_signal_list,
        source_buffer, /*source_offset=*/0, file, /*target_offset=*/0,
        kMultiSlotTransferSize, IREE_HAL_WRITE_FLAG_NONE);
  }
  const bool write_parked =
      iree_status_is_ok(status) && HostQueueHasPendingOperation(queue);

  iree_hsa_signal_store_screlease(IREE_LIBHSA(&libhsa_), blocker_signal, 0);

  if (iree_status_is_ok(status)) {
    status = iree_hal_semaphore_wait(write_signal, write_signal_value,
                                     iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE);
  }
  IREE_EXPECT_OK(
      iree_hsa_signal_destroy(IREE_LIBHSA(&libhsa_), blocker_signal));

  IREE_ASSERT_OK(status);
  EXPECT_TRUE(write_parked);

  std::vector<uint8_t> contents;
  IREE_ASSERT_OK(ReadTempFileContents(path, &contents));
  ASSERT_EQ(contents.size(), kMultiSlotTransferSize);
  ExpectByteRangeRepeated(contents, 0x3C);
}

TEST_F(HostQueueStagingTest, ShortReadFailsTerminalSignal) {
  iree_hal_amdgpu_logical_device_options_t options;
  iree_hal_amdgpu_logical_device_options_initialize(&options);
  options.preallocate_pools = 0;

  TestLogicalDevice test_device;
  IREE_ASSERT_OK(CreateTestDevice(&options, &test_device));

  std::vector<uint8_t> file_data = MakePatternData(kStagingSlotSize);
  iree::testing::TempFilePath path;
  IREE_ASSERT_OK(CreateTempFileWithContents(file_data, &path));

  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(ImportNativeFile(test_device.base_device(), path,
                                  IREE_HAL_MEMORY_ACCESS_READ, file.out()));
  Ref<iree_hal_buffer_t> buffer;
  IREE_ASSERT_OK(CreatePatternedDeviceBuffer(
      test_device.allocator(), test_device.base_device(), kStagingSlotSize,
      0x00, buffer.out()));

  Ref<iree_hal_semaphore_t> wait_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), wait_semaphore.out()));
  uint64_t wait_value = 1;
  iree_hal_semaphore_t* wait_semaphore_ptr = wait_semaphore.get();
  iree_hal_semaphore_list_t wait_list =
      MakeSemaphoreList(&wait_semaphore_ptr, &wait_value);

  Ref<iree_hal_semaphore_t> signal_semaphore;
  IREE_ASSERT_OK(
      CreateSemaphore(test_device.base_device(), signal_semaphore.out()));
  uint64_t signal_value = 1;
  iree_hal_semaphore_t* signal_semaphore_ptr = signal_semaphore.get();
  iree_hal_semaphore_list_t signal_list =
      MakeSemaphoreList(&signal_semaphore_ptr, &signal_value);

  IREE_ASSERT_OK(iree_hal_queue_read(
      &test_device.first_host_queue()->base, wait_list, signal_list, file,
      /*source_offset=*/0, buffer, /*target_offset=*/0, kStagingSlotSize,
      IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(TruncateTempFile(path));
  IREE_ASSERT_OK(
      iree_hal_semaphore_signal(wait_semaphore, wait_value, /*frontier=*/NULL));

  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        iree_hal_semaphore_wait(signal_semaphore, signal_value,
                                                iree_infinite_timeout(),
                                                IREE_ASYNC_WAIT_FLAG_NONE));
}

#else

TEST_F(HostQueueStagingTest, FileIoDisabled) {
  GTEST_SKIP() << "file I/O is disabled";
}

#endif  // IREE_FILE_IO_ENABLE

}  // namespace
}  // namespace iree::hal::amdgpu
