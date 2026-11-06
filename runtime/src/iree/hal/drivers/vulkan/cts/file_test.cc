// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <utility>
#include <vector>

#include "iree/hal/cts/util/pool_test_util.h"
#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::cts {

namespace {

struct FileHandleDeleter {
  void operator()(iree_io_file_handle_t* handle) const {
    iree_io_file_handle_release(handle);
  }
};

using FileHandlePtr = std::unique_ptr<iree_io_file_handle_t, FileHandleDeleter>;

struct ResourceRefCountObservation {
  // Borrowed resource kept live by the test until observation completes.
  iree_hal_resource_t* resource = nullptr;
  // Result published by the terminal semaphore callback.
  std::promise<uint32_t> result;
};

static void ObserveResourceRefCount(void* user_data,
                                    iree_async_semaphore_timepoint_t* timepoint,
                                    iree_status_t status) {
  (void)timepoint;
  auto* observation = static_cast<ResourceRefCountObservation*>(user_data);
  const uint32_t ref_count =
      iree_atomic_ref_count_load(&observation->resource->ref_count);
  iree_status_free(status);
  observation->result.set_value(ref_count);
}

}  // namespace

class VulkanFileTest : public CtsTestBase<> {
 protected:
  iree_status_t ImportNativeFile(const std::vector<uint8_t>& file_contents,
                                 const char* temp_file_stem,
                                 iree_io_file_mode_t extra_mode,
                                 Ref<iree_hal_file_t>* out_file,
                                 FileHandlePtr* out_handle) {
    iree::testing::TempFilePath path(temp_file_stem);
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        path.path_view(),
        iree_make_const_byte_span(file_contents.data(), file_contents.size()),
        iree_allocator_system()));

    iree_io_file_handle_t* raw_handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC |
            extra_mode,
        path.path_view(), iree_allocator_system(), &raw_handle));
    FileHandlePtr handle(raw_handle);

    Ref<iree_hal_file_t> file;
    IREE_RETURN_IF_ERROR(iree_hal_file_import(
        device_, /*storage_pool=*/nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
        handle.get(), IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out()));
    *out_file = std::move(file);
    if (out_handle) {
      *out_handle = std::move(handle);
    }
    return iree_ok_status();
  }

  void NativeFileRoundTrip(iree_device_size_t file_length,
                           const char* temp_file_stem) {
    const std::vector<uint8_t> file_contents(file_length, 0);
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(file_length);

    Ref<iree_hal_file_t> file;
    IREE_ASSERT_OK(
        ImportNativeFile(file_contents, temp_file_stem, 0, &file, nullptr));

    Ref<iree_hal_buffer_t> source_buffer;
    IREE_ASSERT_OK(CreateDeviceBufferWithData(
        source_contents.data(), source_contents.size(), source_buffer.out()));
    Ref<iree_hal_buffer_t> target_buffer;
    IREE_ASSERT_OK(CreateZeroedDeviceBuffer(file_length, target_buffer.out()));

    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_write(
        transfer_queue_, empty_wait, write_signal, source_buffer.get(),
        /*source_offset=*/0, file.get(),
        /*target_offset=*/0, file_length, /*barriers=*/NULL,
        IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_read(
        transfer_queue_, write_signal, read_signal, file.get(),
        /*source_offset=*/0, target_buffer.get(),
        /*target_offset=*/0, file_length, /*barriers=*/NULL,
        IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

    const std::vector<uint8_t> actual_contents =
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length);
    ASSERT_EQ(actual_contents.size(), source_contents.size());
    EXPECT_EQ(0, std::memcmp(actual_contents.data(), source_contents.data(),
                             source_contents.size()));
  }
};

TEST_P(VulkanFileTest, NativeFileLargeRangeRoundTrip) {
  const iree_device_size_t file_length = 256 * 1024 + 31;
  NativeFileRoundTrip(file_length, "iree_vulkan_hal_cts_native_file");
}

TEST_P(VulkanFileTest, NativeFileRoundTripExceedsStagingRing) {
  const iree_device_size_t file_length = 20 * 1024 * 1024 + 31;
  NativeFileRoundTrip(file_length, "iree_vulkan_hal_cts_native_file_staging");
}

TEST_P(VulkanFileTest, NativeFileShortReadExceedsStagingRingFails) {
  const iree_device_size_t import_length = 20 * 1024 * 1024 + 31;
  const iree_device_size_t actual_length = 4 * 1024 * 1024 + 17;
  const std::vector<uint8_t> import_contents(import_length, 0);

  iree::testing::TempFilePath path("iree_vulkan_hal_cts_native_file_short");
  IREE_ASSERT_OK(iree_io_file_contents_write(
      path.path_view(),
      iree_make_const_byte_span(import_contents.data(), import_contents.size()),
      iree_allocator_system()));

  iree_io_file_handle_t* raw_handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
          IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
          IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC,
      path.path_view(), iree_allocator_system(), &raw_handle));
  FileHandlePtr handle(raw_handle);

  Ref<iree_hal_file_t> file;
  IREE_ASSERT_OK(iree_hal_file_import(
      device_, /*storage_pool=*/nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, handle.get(),
      IREE_HAL_EXTERNAL_FILE_FLAG_NONE, file.out()));

  IREE_ASSERT_OK(iree_io_file_handle_resize(handle.get(), actual_length));

  Ref<iree_hal_buffer_t> target_buffer;
  IREE_ASSERT_OK(CreateZeroedDeviceBuffer(import_length, target_buffer.out()));

  SemaphoreList empty_wait;
  SemaphoreList read_signal(device_, {0}, {1});
  ResourceRefCountObservation observation = {&target_buffer.get()->resource};
  std::future<uint32_t> observed_ref_count = observation.result.get_future();
  iree_async_semaphore_timepoint_t timepoint = {
      .callback = ObserveResourceRefCount,
      .user_data = &observation,
  };
  IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
      reinterpret_cast<iree_async_semaphore_t*>(read_signal.semaphores[0]), 1,
      &timepoint));
  IREE_ASSERT_OK(iree_hal_queue_read(
      transfer_queue_, empty_wait, read_signal, file.get(),
      /*source_offset=*/0, target_buffer.get(), /*target_offset=*/0,
      import_length, /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));
  EXPECT_EQ(observed_ref_count.get(), 1u);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_list_wait(read_signal, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
}

class VulkanImportedFileTest : public VulkanFileTest {
 protected:
  void SetUp() override {
    VulkanFileTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    const iree_hal_pool_family_access_t family = {
        .family = iree_hal_queue_family(transfer_queue_),
        .usage = IREE_HAL_BUFFER_USAGE_TRANSFER,
    };
    const iree_hal_pool_scope_t scope = {
        .family_count = 1,
        .families = &family,
    };
    const iree_hal_pool_import_class_t import_class = {
        .storage_class = iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
    };
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.owned.mode = IREE_HAL_SLAB_POOL_IMPORT_ONLY;
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
    iree_status_t status = iree_hal_slab_pool_create(
        device_group_, scope, &options, iree_allocator_system(), pool_.out());
    if (iree_status_is_unavailable(status)) {
      Status unavailable(std::move(status));
      GTEST_SKIP() << unavailable.ToString();
    }
    IREE_ASSERT_OK(status);
    ASSERT_TRUE(iree_hal_pool_query_import_requirements(
        pool_, import_class.storage_class, &requirements_));
  }

  void ImportBacking(
      iree_device_size_t length, iree_device_size_t file_alignment,
      Ref<iree_hal_buffer_t>* out_buffer,
      std::future<ImportedHostAllocationReleaseResult>* out_release) {
    const auto alignment =
        std::max(requirements_.base_alignment, file_alignment);
    const auto registered_length =
        iree_device_align(std::max(length, requirements_.min_byte_length),
                          requirements_.byte_length_granularity);
    const auto storage_length = alignment + registered_length;
    ASSERT_LE(storage_length, IREE_HOST_SIZE_MAX);
    auto allocation = std::make_unique<ImportedHostAllocation>();
    allocation->host_allocator = iree_allocator_system();
    allocation->storage_class = iree_hal_make_external_storage_class(
        IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER);
    *out_release = allocation->release_result.get_future();
    IREE_ASSERT_OK(iree_allocator_malloc_aligned(
        allocation->host_allocator,
        static_cast<iree_host_size_t>(storage_length),
        static_cast<iree_host_size_t>(alignment), 0, &allocation->pointer));
    iree_hal_external_storage_t storage = {
        .storage_class = allocation->storage_class,
        .byte_length = storage_length,
    };
    storage.payload.host_pointer = allocation->pointer;
    storage.host_properties = (IREE_HAL_HOST_MEMORY_PROPERTY_READ_CACHED |
                               IREE_HAL_HOST_MEMORY_PROPERTY_WRITE_BACK);
    storage.maintenance_range =
        iree_hal_make_buffer_range(alignment, registered_length);
    storage.release = ReleaseImportedHostAllocation;
    storage.release_user_data = allocation.get();
    IREE_ASSERT_OK(iree_hal_pool_import_buffer(
        pool_, &storage, iree_hal_make_buffer_range(alignment, length),
        out_buffer->out()));
    allocation.release();
  }

  enum RoundTripFlagBits : uint32_t {
    kRoundTripNone = 0,
    kRoundTripDirect = 1u << 0,
    kRoundTripQueuedAllocation = 1u << 1,
    kRoundTripEmptyPayloadBarriers = 1u << 2,
  };
  using RoundTripFlags = uint32_t;

  void QueuedRoundTrip(RoundTripFlags flags) {
    const bool direct = iree_any_bit_set(flags, kRoundTripDirect);
    const auto expected =
        MakeDeterministicBytes(20 * 1024 * 1024 + (direct ? 0 : 31));
    std::vector<uint8_t> expected_file(expected.size(), 0);
    Ref<iree_hal_file_t> input;
    Ref<iree_hal_file_t> output;
    const auto mode = direct ? IREE_IO_FILE_MODE_DIRECT : 0;
    iree::Status import_status = ImportNativeFile(
        expected, "iree_vulkan_imported_input", mode, &input, nullptr);
    if (direct && (import_status.code() == iree::StatusCode::kUnavailable ||
                   import_status.code() == iree::StatusCode::kUnimplemented)) {
      GTEST_SKIP() << import_status.ToString();
    }
    IREE_ASSERT_OK(import_status);
    IREE_ASSERT_OK(ImportNativeFile(
        expected_file, "iree_vulkan_imported_output", mode, &output, nullptr));
    const auto& read = iree_hal_file_requirements(input)->read;
    const auto& write = iree_hal_file_requirements(output)->write;
    const uint64_t alignment =
        std::max<uint64_t>({read.buffer_alignment, read.offset_alignment,
                            read.length_alignment, write.buffer_alignment,
                            write.offset_alignment, write.length_alignment});
    const auto offset = iree_device_align(13, alignment);
    const auto length = (expected.size() / 4) & ~(alignment - 1);
    ASSERT_GT(length, 1024 * 1024);
    const iree_device_size_t backing_length = length + 4 * offset;
    Ref<iree_hal_buffer_t> storage;
    std::future<ImportedHostAllocationReleaseResult> released;
    ASSERT_NO_FATAL_FAILURE(ImportBacking(
        iree_device_align(backing_length, requirements_.base_alignment),
        alignment, &storage, &released));
    SemaphoreList start(device_, {0}, {1});
    SemaphoreList allocated = start;
    Ref<iree_hal_pool_t> arena;
    Ref<iree_hal_buffer_t> backing;
    if (iree_any_bit_set(flags, kRoundTripQueuedAllocation)) {
      // Capacity and native attachment exist before queue execution. The
      // queued allocation only claims a view of this retained backing.
      const iree_hal_fixed_block_pool_options_t options = {
          .block_size = iree_hal_buffer_byte_length(storage),
          .frontier_capacity = 2,
      };
      IREE_ASSERT_OK(iree_hal_fixed_block_pool_create_from_buffer(
          storage, 0, IREE_HAL_WHOLE_BUFFER, &options, iree_allocator_system(),
          arena.out()));
      allocated = SemaphoreList(device_, {0}, {1});
      const auto request = iree_hal_make_pool_reservation_request(
          backing_length, /*min_alignment=*/1);
      IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, start, allocated,
                                           arena, 1, &request, backing.out()));
    } else {
      IREE_ASSERT_OK(iree_hal_buffer_subspan(
          storage, 0, backing_length, iree_allocator_system(), backing.out()));
    }
    Ref<iree_hal_buffer_t> outer;
    Ref<iree_hal_buffer_t> view;
    IREE_ASSERT_OK(iree_hal_buffer_subspan(backing, offset, length + 3 * offset,
                                           iree_allocator_system(),
                                           outer.out()));
    IREE_ASSERT_OK(iree_hal_buffer_subspan(outer, offset, length + 2 * offset,
                                           iree_allocator_system(),
                                           view.out()));
    outer.reset();

    std::array<Ref<iree_hal_buffer_t>, 3> consumed;
    for (auto& buffer : consumed) {
      IREE_ASSERT_OK(CreateUninitializedDeviceBuffer(length, buffer.out()));
    }
    SemaphoreList initialized(device_, {0}, {1});
    SemaphoreList loaded(device_, {0}, {1});
    SemaphoreList copied(device_, {0}, {1});
    SemaphoreList modified(device_, {0}, {1});
    SemaphoreList written(device_, {0}, {1});
    const iree_hal_barrier_list_t no_barriers = {};
    const iree_hal_queue_barriers_t no_payload_barriers = {
        .before = &no_barriers,
        .after = &no_barriers,
    };
    const auto* barriers =
        iree_any_bit_set(flags, kRoundTripEmptyPayloadBarriers)
            ? &no_payload_barriers
            : nullptr;
    const uint8_t guard = 0xCD;
    IREE_ASSERT_OK(
        iree_hal_queue_fill(transfer_queue_, allocated, initialized, backing, 0,
                            iree_hal_buffer_byte_length(backing), &guard,
                            sizeof(guard), nullptr, IREE_HAL_FILL_FLAG_NONE));
    for (size_t i = 0; i < consumed.size(); ++i) {
      const auto source_offset = i * length + offset;
      SemaphoreList wait = i == 0 ? initialized : written;
      loaded.payload_values[0] = copied.payload_values[0] =
          modified.payload_values[0] = written.payload_values[0] = i + 1;
      IREE_ASSERT_OK(iree_hal_queue_read(transfer_queue_, wait, loaded, input,
                                         source_offset, view, offset, length,
                                         barriers, IREE_HAL_READ_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_queue_copy(transfer_queue_, loaded, copied, view,
                                         offset, consumed[i], 0, length,
                                         barriers, IREE_HAL_COPY_FLAG_NONE));
      // File writes must observe a device producer, not merely read back the
      // preceding native file input that is already resident in host caches.
      const uint8_t pattern = static_cast<uint8_t>(0x40 + i);
      IREE_ASSERT_OK(iree_hal_queue_fill(
          transfer_queue_, copied, modified, view, offset, length, &pattern,
          sizeof(pattern), barriers, IREE_HAL_FILL_FLAG_NONE));
      IREE_ASSERT_OK(iree_hal_queue_write(
          transfer_queue_, modified, written, view, offset, output,
          source_offset, length, barriers, IREE_HAL_WRITE_FLAG_NONE));
      std::memset(expected_file.data() + source_offset, pattern, length);
    }
    ResourceRefCountObservation observation = {&view.get()->resource};
    auto observed_refs = observation.result.get_future();
    iree_async_semaphore_timepoint_t timepoint = {
        .callback = ObserveResourceRefCount,
        .user_data = &observation,
    };
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(written.semaphores[0]),
        written.payload_values[0], &timepoint));
    // Every read, consumer, write and reuse is already queued. No application
    // wait bridges native I/O readiness to device execution.
    IREE_ASSERT_OK(iree_hal_semaphore_list_signal(start, nullptr));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        written, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(observed_refs.get(), 1u);
    const auto guards = ReadBufferBytes(backing, 0, 3 * offset);
    EXPECT_EQ(guards, std::vector<uint8_t>(3 * offset, guard));
    EXPECT_EQ(ReadBufferBytes(backing, 3 * offset + length, offset),
              std::vector<uint8_t>(offset, guard));
    view.reset();
    if (iree_any_bit_set(flags, kRoundTripQueuedAllocation)) {
      SemaphoreList deallocated(device_, {0}, {1});
      auto* raw_buffer = backing.get();
      IREE_ASSERT_OK(iree_hal_queue_dealloca(transfer_queue_, written,
                                             deallocated, 1, &raw_buffer));
      IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
          deallocated, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    }
    backing.reset();
    arena.reset();
    storage.reset();
    const auto release = released.get();
    EXPECT_TRUE(release.storage_class_matches);
    EXPECT_TRUE(release.payload_matches);
    for (size_t i = 0; i < consumed.size(); ++i) {
      const auto actual = ReadBufferBytes(consumed[i], 0, length);
      EXPECT_EQ(std::memcmp(actual.data(),
                            expected.data() + i * length + offset, length),
                0);
    }
    Ref<iree_hal_buffer_t> readback;
    IREE_ASSERT_OK(
        CreateUninitializedDeviceBuffer(expected_file.size(), readback.out()));
    SemaphoreList readback_done(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_read(
        transfer_queue_, written, readback_done, output, 0, readback, 0,
        expected_file.size(), nullptr, IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        readback_done, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_EQ(ReadBufferBytes(readback, 0, expected_file.size()),
              expected_file);
  }

  void FailsAfterReleasingCaptures(bool fail_dependency) {
    const size_t length = 5 * 1024 * 1024 + 31;
    Ref<iree_hal_file_t> file;
    FileHandlePtr handle;
    IREE_ASSERT_OK(ImportNativeFile(std::vector<uint8_t>(length, 0xAB),
                                    "iree_vulkan_imported_failure", 0, &file,
                                    &handle));
    if (!fail_dependency) {
      // The imported extent is still valid, but native I/O must report EOF
      // after successful partial requests and retire the other accepted lanes.
      IREE_ASSERT_OK(
          iree_io_file_handle_resize(handle.get(), 2 * 1024 * 1024 + 17));
    }
    Ref<iree_hal_buffer_t> buffer;
    std::future<ImportedHostAllocationReleaseResult> released;
    ASSERT_NO_FATAL_FAILURE(ImportBacking(length, 1, &buffer, &released));
    SemaphoreList start(device_, {0}, {1});
    SemaphoreList done(device_, {0}, {1});
    ResourceRefCountObservation observation = {&buffer.get()->resource};
    auto observed_refs = observation.result.get_future();
    iree_async_semaphore_timepoint_t timepoint = {
        .callback = ObserveResourceRefCount,
        .user_data = &observation,
    };
    IREE_ASSERT_OK(iree_async_semaphore_acquire_timepoint(
        reinterpret_cast<iree_async_semaphore_t*>(done.semaphores[0]), 1,
        &timepoint));
    IREE_ASSERT_OK(iree_hal_queue_read(transfer_queue_, start, done, file, 0,
                                       buffer, 0, length, nullptr,
                                       IREE_HAL_READ_FLAG_NONE));
    if (fail_dependency) {
      iree_hal_semaphore_list_fail(start,
                                   iree_status_from_code(IREE_STATUS_ABORTED));
    } else {
      IREE_ASSERT_OK(iree_hal_semaphore_list_signal(start, nullptr));
    }
    EXPECT_EQ(observed_refs.get(), 1u);
    IREE_EXPECT_STATUS_IS(
        fail_dependency ? IREE_STATUS_ABORTED : IREE_STATUS_OUT_OF_RANGE,
        iree_hal_semaphore_list_wait(done, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    buffer.reset();
    const auto release = released.get();
    EXPECT_TRUE(release.storage_class_matches);
    EXPECT_TRUE(release.payload_matches);
  }

  // Import route and native attachment geometry selected once per test.
  Ref<iree_hal_pool_t> pool_;
  // Exact constraints for ordinary host-pointer imports.
  iree_hal_pool_import_requirements_t requirements_ = {};
};

TEST_P(VulkanImportedFileTest, QueuesNativeReadsConsumersAndReuse) {
  QueuedRoundTrip(kRoundTripNone);
}

TEST_P(VulkanImportedFileTest, QueuesDirectReadsConsumersAndReuse) {
  QueuedRoundTrip(kRoundTripDirect);
}

TEST_P(VulkanImportedFileTest, QueuedAllocationPreservesImportedFileBacking) {
  QueuedRoundTrip(kRoundTripQueuedAllocation);
}

TEST_P(VulkanImportedFileTest,
       DirectQueuedAllocationPreservesImportedFileBacking) {
  QueuedRoundTrip(kRoundTripDirect | kRoundTripQueuedAllocation);
}

TEST_P(VulkanImportedFileTest,
       NativeFileVisibilitySurvivesEmptyPayloadBarriers) {
  QueuedRoundTrip(kRoundTripEmptyPayloadBarriers);
}

TEST_P(VulkanImportedFileTest,
       DirectFileVisibilitySurvivesEmptyPayloadBarriers) {
  QueuedRoundTrip(kRoundTripDirect | kRoundTripEmptyPayloadBarriers);
}

TEST_P(VulkanImportedFileTest, NativeEofRetiresEveryAcceptedRequest) {
  FailsAfterReleasingCaptures(/*fail_dependency=*/false);
}

TEST_P(VulkanImportedFileTest, FailedDependencyReleasesCaptures) {
  FailsAfterReleasingCaptures(/*fail_dependency=*/true);
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(VulkanFileTest, {"file_io"}, {});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(VulkanImportedFileTest, {"file_io"}, {});

}  // namespace iree::hal::cts
