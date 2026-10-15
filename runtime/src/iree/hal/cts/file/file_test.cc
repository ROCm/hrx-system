// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include "iree/hal/cts/util/test_base.h"
#include "iree/io/file_contents.h"
#include "iree/io/file_handle.h"
#include "iree/testing/temp_file.h"

namespace iree::hal::cts {

using ::testing::ContainerEq;

namespace {

constexpr iree_device_size_t kFileAlignment = 128;

enum class FileProviderKind {
  kMemory,
  kNative,
};

struct FileProvider {
  // Human-readable provider name used in scoped test traces.
  const char* name;
  // Backing implementation shape exercised by the provider.
  FileProviderKind kind;
};

const FileProvider kFileProviders[] = {
    {
        "memory_file",
        FileProviderKind::kMemory,
    },
    {
        "native_file",
        FileProviderKind::kNative,
    },
};

void ReleaseAlignedAllocation(void* user_data,
                              iree_io_file_handle_primitive_t primitive) {
  (void)primitive;
  iree_allocator_free_aligned(iree_allocator_system(), user_data);
}

struct AlignedAllocationDeleter {
  void operator()(uint8_t* ptr) const {
    iree_allocator_free_aligned(iree_allocator_system(), ptr);
  }
};

using AlignedAllocation = std::unique_ptr<uint8_t, AlignedAllocationDeleter>;

}  // namespace

class FileTest : public CtsTestBase<> {
 protected:
  void TearDown() override {
    iree_hal_pool_release(file_storage_pool_);
    file_storage_pool_ = nullptr;
    iree_hal_pool_release(transfer_source_pool_);
    transfer_source_pool_ = nullptr;
    iree_hal_pool_release(transfer_target_pool_);
    transfer_target_pool_ = nullptr;
    CtsTestBase::TearDown();
  }

  class TestFile {
   public:
    TestFile() = default;
    ~TestFile() { Reset(); }

    TestFile(TestFile&& other) noexcept { MoveFrom(std::move(other)); }

    TestFile& operator=(TestFile&& other) noexcept {
      if (this != &other) {
        Reset();
        MoveFrom(std::move(other));
      }
      return *this;
    }

    TestFile(const TestFile&) = delete;
    TestFile& operator=(const TestFile&) = delete;

    iree_hal_file_t* get() const { return file_; }
    iree_hal_file_t** out_file() { return &file_; }
    iree_string_view_t native_path() const { return native_path_.path_view(); }

    // Drops caller-owned handles while preserving the path for inspection.
    void ReleaseHandles() {
      iree_hal_file_release(file_);
      file_ = nullptr;
      iree_io_file_handle_release(native_handle_);
      native_handle_ = nullptr;
    }

    void SetMemoryContents(uint8_t* memory_contents) {
      memory_contents_ = memory_contents;
    }

    void SetNativeFile(iree::testing::TempFilePath path,
                       iree_io_file_handle_t* handle) {
      native_path_ = std::move(path);
      native_handle_ = handle;
    }

    std::vector<uint8_t> ReadContents(iree_device_size_t offset,
                                      iree_device_size_t length) {
      std::vector<uint8_t> contents(static_cast<size_t>(length));
      if (memory_contents_) {
        std::memcpy(contents.data(), memory_contents_ + offset,
                    static_cast<size_t>(length));
        return contents;
      }

      if (native_path_) {
        ReleaseHandles();

        iree_io_file_contents_t* file_contents = nullptr;
        IREE_EXPECT_OK(iree_io_file_contents_read(
            native_path_.path_view(), iree_allocator_system(), &file_contents));
        if (file_contents) {
          const uint64_t read_end =
              static_cast<uint64_t>(offset) + static_cast<uint64_t>(length);
          EXPECT_LE(read_end,
                    static_cast<uint64_t>(file_contents->buffer.data_length));
          if (read_end <=
              static_cast<uint64_t>(file_contents->buffer.data_length)) {
            std::memcpy(
                contents.data(),
                file_contents->buffer.data + static_cast<size_t>(offset),
                static_cast<size_t>(length));
          }
        }
        iree_io_file_contents_free(file_contents);
        return contents;
      }

      ADD_FAILURE() << "test file has no inspectable backing storage";
      return contents;
    }

    void Reset() {
      ReleaseHandles();
      memory_contents_ = nullptr;
      if (native_path_) {
        EXPECT_TRUE(native_path_.Remove());
      }
      native_path_.Reset();
    }

   private:
    void MoveFrom(TestFile&& other) {
      file_ = other.file_;
      other.file_ = nullptr;
      memory_contents_ = other.memory_contents_;
      other.memory_contents_ = nullptr;
      native_handle_ = other.native_handle_;
      other.native_handle_ = nullptr;
      native_path_ = std::move(other.native_path_);
      other.native_path_.Reset();
    }

    // Imported HAL file under test.
    iree_hal_file_t* file_ = nullptr;
    // Host allocation backing memory_file providers, unowned.
    uint8_t* memory_contents_ = nullptr;
    // Native platform file handle opened for async HAL import.
    iree_io_file_handle_t* native_handle_ = nullptr;
    // Path to the native platform file for independent host inspection.
    iree::testing::TempFilePath native_path_;
  };

  template <typename Body>
  void ForEachProvider(Body body) {
    int provider_count = 0;
    for (const auto& provider : kFileProviders) {
      SCOPED_TRACE(provider.name);
      ++provider_count;
      body(provider);
    }
    EXPECT_GT(provider_count, 0);
  }

  bool TryCreateTestFile(const FileProvider& provider,
                         iree_hal_memory_access_t access,
                         const std::vector<uint8_t>& initial_contents,
                         TestFile* out_file) {
    iree_status_t status =
        CreateTestFileStatus(provider, access, initial_contents, out_file);
    if (ProviderSkipped(provider, status)) {
      return false;
    }
    if (!iree_status_is_ok(status)) {
      IREE_EXPECT_OK(status);
      return false;
    }
    return true;
  }

  Ref<iree_hal_buffer_t> CreateBufferWithData(
      const std::vector<uint8_t>& contents) {
    Ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(CreateDeviceBufferWithData(
        contents.data(), static_cast<iree_device_size_t>(contents.size()),
        buffer.out()));
    return buffer;
  }

  Ref<iree_hal_buffer_t> CreateZeroedBuffer(iree_device_size_t length) {
    Ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(CreateZeroedDeviceBuffer(length, buffer.out()));
    return buffer;
  }

  Ref<iree_hal_buffer_t> CreateBufferWithAccess(iree_hal_memory_access_t access,
                                                iree_device_size_t length) {
    iree_hal_pool_t** pool_slot = access == IREE_HAL_MEMORY_ACCESS_READ
                                      ? &transfer_source_pool_
                                      : &transfer_target_pool_;
    if (!*pool_slot) {
      const iree_hal_buffer_usage_t usage =
          access == IREE_HAL_MEMORY_ACCESS_READ
              ? IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE
              : IREE_HAL_BUFFER_USAGE_TRANSFER_TARGET;
      IREE_CHECK_OK(CreateDevicePool(usage, pool_slot));
    }
    Ref<iree_hal_buffer_t> buffer;
    IREE_CHECK_OK(iree_hal_pool_allocate_buffer(
        *pool_slot,
        iree_hal_make_pool_reservation_request(length, /*min_alignment=*/1),
        iree_infinite_timeout(), buffer.out()));
    return buffer;
  }

  iree_status_t CreateHostFileStoragePool(
      iree_hal_pool_t** out_pool, iree_hal_slab_pool_owned_mode_t owned_mode =
                                      IREE_HAL_SLAB_POOL_IMPORT_ONLY) {
    *out_pool = nullptr;
    const iree_hal_pool_family_access_t family_access = {
        .family = iree_hal_queue_family(transfer_queue_),
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
    iree_hal_slab_pool_options_t options;
    iree_hal_slab_pool_options_initialize(&options);
    options.owned.mode = owned_mode;
    options.imports.class_count = 1;
    options.imports.classes = &import_class;
    options.trace_name = IREE_SV("cts.file.storage");
    return iree_hal_slab_pool_create(device_group_, scope, &options,
                                     iree_allocator_system(), out_pool);
  }

  // Returns the fixture-owned host import pool when the backend exposes one.
  // A null result is a supported outcome for devices that stage host-only
  // memory files through another native I/O path.
  iree_status_t GetHostFileStoragePool(iree_hal_pool_t** out_pool) {
    *out_pool = nullptr;
    if (!file_storage_pool_queried_) {
      iree_status_t status = CreateHostFileStoragePool(&file_storage_pool_);
      if (iree_status_is_unavailable(status) ||
          iree_status_is_unimplemented(status)) {
        iree_status_free(status);
      } else {
        IREE_RETURN_IF_ERROR(status);
      }
      file_storage_pool_queried_ = true;
    }
    *out_pool = file_storage_pool_;
    return iree_ok_status();
  }

  static bool HostStorageMeetsImportRequirements(
      iree_const_byte_span_t storage,
      const iree_hal_pool_import_requirements_t& requirements) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(storage.data);
    return storage.data_length >= requirements.min_byte_length &&
           (!requirements.max_byte_length ||
            storage.data_length <= requirements.max_byte_length) &&
           requirements.base_alignment != 0 &&
           address % requirements.base_alignment == 0 &&
           requirements.byte_length_granularity != 0 &&
           storage.data_length % requirements.byte_length_granularity == 0;
  }

  iree_status_t SelectHostFileStoragePool(iree_const_byte_span_t storage,
                                          iree_hal_pool_t** out_pool) {
    IREE_RETURN_IF_ERROR(GetHostFileStoragePool(out_pool));
    if (!*out_pool) {
      return iree_ok_status();
    }
    iree_hal_pool_import_requirements_t requirements;
    if (!iree_hal_pool_query_import_requirements(
            *out_pool,
            iree_hal_make_external_storage_class(
                IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
            &requirements) ||
        !HostStorageMeetsImportRequirements(storage, requirements)) {
      *out_pool = nullptr;
    }
    return iree_ok_status();
  }

  iree_status_t AllocateBorrowedHostFileStorage(
      iree_device_size_t length, AlignedAllocation* out_allocation) {
    out_allocation->reset();
    iree_host_size_t alignment = kFileAlignment;
    iree_hal_pool_t* storage_pool = nullptr;
    IREE_RETURN_IF_ERROR(GetHostFileStoragePool(&storage_pool));
    if (storage_pool) {
      iree_hal_pool_import_requirements_t requirements;
      if (iree_hal_pool_query_import_requirements(
              storage_pool,
              iree_hal_make_external_storage_class(
                  IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
              &requirements)) {
        if (requirements.base_alignment > IREE_HOST_SIZE_MAX) {
          return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                  "host import alignment exceeds host size");
        }
        alignment = std::max(alignment, static_cast<iree_host_size_t>(
                                            requirements.base_alignment));
      }
    }
    void* ptr = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        iree_allocator_system(), static_cast<iree_host_size_t>(length),
        alignment, 0, &ptr));
    out_allocation->reset(static_cast<uint8_t*>(ptr));
    return iree_ok_status();
  }

  iree_status_t CreateBorrowedMemoryTestFile(iree_hal_memory_access_t access,
                                             iree_byte_span_t contents,
                                             TestFile* out_file) {
    out_file->Reset();
    iree_hal_pool_t* storage_pool = nullptr;
    IREE_RETURN_IF_ERROR(SelectHostFileStoragePool(
        iree_make_const_byte_span(contents.data, contents.data_length),
        &storage_pool));
    iree_io_file_handle_t* handle = nullptr;
    IREE_RETURN_IF_ERROR(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE, contents,
        iree_io_file_handle_release_callback_null(), iree_allocator_system(),
        &handle));
    iree_status_t status = iree_hal_file_import(
        device_, storage_pool, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access,
        handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file->out_file());
    iree_io_file_handle_release(handle);
    if (iree_status_is_ok(status)) {
      out_file->SetMemoryContents(contents.data);
    }
    return status;
  }

  void QueueReadAndWait(iree_hal_file_t* source_file,
                        iree_device_size_t source_offset,
                        iree_hal_buffer_t* target_buffer,
                        iree_device_size_t target_offset,
                        iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_read(
        transfer_queue_, empty_wait, read_signal, source_file, source_offset,
        target_buffer, target_offset, length, /*barriers=*/NULL,
        IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void QueueWriteAndWait(iree_hal_buffer_t* source_buffer,
                         iree_device_size_t source_offset,
                         iree_hal_file_t* target_file,
                         iree_device_size_t target_offset,
                         iree_device_size_t length) {
    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_write(
        transfer_queue_, empty_wait, write_signal, source_buffer, source_offset,
        target_file, target_offset, length, /*barriers=*/NULL,
        IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

  void QueueWriteThenRead(iree_hal_buffer_t* source_buffer,
                          iree_device_size_t source_offset,
                          iree_hal_file_t* file, iree_device_size_t file_offset,
                          iree_hal_buffer_t* target_buffer,
                          iree_device_size_t target_offset,
                          iree_device_size_t length) {
    iree_hal_barrier_t action = {
        .flags = IREE_HAL_BARRIER_FLAG_ACQUIRE_SYSTEM_SCOPE |
                 IREE_HAL_BARRIER_FLAG_RELEASE_SYSTEM_SCOPE,
    };
    const iree_hal_barrier_list_t list = {1, &action};
    const iree_hal_queue_barriers_t barriers = {&list, &list};
    SemaphoreList empty_wait;
    SemaphoreList write_signal(device_, {0}, {1});
    SemaphoreList read_signal(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_write(
        transfer_queue_, empty_wait, write_signal, source_buffer, source_offset,
        file, file_offset, length, &barriers, IREE_HAL_WRITE_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_queue_read(transfer_queue_, write_signal,
                                       read_signal, file, file_offset,
                                       target_buffer, target_offset, length,
                                       &barriers, IREE_HAL_READ_FLAG_NONE));
    IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
        read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  }

 private:
  // Owned source-only pool used by negative transfer-access coverage.
  iree_hal_pool_t* transfer_source_pool_ = nullptr;
  // Owned target-only pool used by negative transfer-access coverage.
  iree_hal_pool_t* transfer_target_pool_ = nullptr;
  // Owned import-only pool used when memory file storage satisfies its exact
  // native geometry requirements.
  iree_hal_pool_t* file_storage_pool_ = nullptr;
  // True after the backend's host import route has been queried once.
  bool file_storage_pool_queried_ = false;

  iree_status_t CreateTestFileStatus(
      const FileProvider& provider, iree_hal_memory_access_t access,
      const std::vector<uint8_t>& initial_contents, TestFile* out_file) {
    out_file->Reset();
    switch (provider.kind) {
      case FileProviderKind::kMemory:
        return CreateMemoryTestFile(access, initial_contents, out_file);
      case FileProviderKind::kNative:
        return CreateNativeTestFile(access, initial_contents, out_file);
    }
    return iree_status_from_code(IREE_STATUS_UNIMPLEMENTED);
  }

  bool ProviderSkipped(const FileProvider& provider, iree_status_t status) {
    if (iree_status_is_ok(status)) {
      return false;
    }
    const iree_status_code_t code = iree_status_code(status);
    if (provider.kind == FileProviderKind::kNative &&
        (code == IREE_STATUS_UNAVAILABLE ||
         code == IREE_STATUS_UNIMPLEMENTED)) {
      iree_status_free(status);
      return true;
    }
    return false;
  }

  iree_status_t CreateMemoryTestFile(
      iree_hal_memory_access_t access,
      const std::vector<uint8_t>& initial_contents, TestFile* out_file) {
    iree_hal_pool_t* storage_pool = nullptr;
    IREE_RETURN_IF_ERROR(GetHostFileStoragePool(&storage_pool));
    iree_host_size_t allocation_alignment = kFileAlignment;
    if (storage_pool) {
      iree_hal_pool_import_requirements_t requirements;
      if (!iree_hal_pool_query_import_requirements(
              storage_pool,
              iree_hal_make_external_storage_class(
                  IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
              &requirements) ||
          initial_contents.size() < requirements.min_byte_length ||
          (requirements.max_byte_length &&
           initial_contents.size() > requirements.max_byte_length) ||
          requirements.byte_length_granularity == 0 ||
          initial_contents.size() % requirements.byte_length_granularity != 0) {
        storage_pool = nullptr;
      } else if (requirements.base_alignment > IREE_HOST_SIZE_MAX) {
        return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "host import alignment exceeds host size");
      } else {
        allocation_alignment = std::max(
            allocation_alignment,
            static_cast<iree_host_size_t>(requirements.base_alignment));
      }
    }

    void* file_contents = nullptr;
    IREE_RETURN_IF_ERROR(iree_allocator_malloc_aligned(
        iree_allocator_system(), initial_contents.size(), allocation_alignment,
        0, &file_contents));
    std::memcpy(file_contents, initial_contents.data(),
                initial_contents.size());

    iree_io_file_handle_release_callback_t release_callback = {
        ReleaseAlignedAllocation,
        file_contents,
    };
    iree_io_file_handle_t* handle = nullptr;
    iree_status_t status = iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_contents, initial_contents.size()),
        release_callback, iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      EXPECT_FALSE(iree_io_file_handle_uses_async_io(handle));
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_file_import(
          device_, storage_pool, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access,
          handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE, out_file->out_file());
    }
    if (iree_status_is_ok(status)) {
      out_file->SetMemoryContents(static_cast<uint8_t*>(file_contents));
    } else if (!handle) {
      iree_io_file_handle_primitive_t primitive = {};
      release_callback.fn(release_callback.user_data, primitive);
    }
    iree_io_file_handle_release(handle);
    return status;
  }

 protected:
  iree_status_t CreateNativeTestFile(
      iree_hal_memory_access_t access,
      const std::vector<uint8_t>& initial_contents, TestFile* out_file,
      iree_io_file_mode_t additional_mode = 0) {
    iree::testing::TempFilePath path("iree_hal_cts_file");
    IREE_RETURN_IF_ERROR(iree_io_file_contents_write(
        path.path_view(),
        iree_make_const_byte_span(initial_contents.data(),
                                  initial_contents.size()),
        iree_allocator_system()));

    iree_io_file_handle_t* handle = nullptr;
    iree_status_t status = iree_io_file_handle_open(
        IREE_IO_FILE_MODE_READ | IREE_IO_FILE_MODE_WRITE |
            IREE_IO_FILE_MODE_RANDOM_ACCESS | IREE_IO_FILE_MODE_SHARE_READ |
            IREE_IO_FILE_MODE_SHARE_WRITE | IREE_IO_FILE_MODE_ASYNC |
            additional_mode,
        path.path_view(), iree_allocator_system(), &handle);
    if (iree_status_is_ok(status)) {
      EXPECT_TRUE(iree_io_file_handle_uses_async_io(handle));
    }
    if (iree_status_is_ok(status)) {
      status = iree_hal_file_import(device_, /*storage_pool=*/nullptr,
                                    IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY, access,
                                    handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
                                    out_file->out_file());
    }

    if (iree_status_is_ok(status)) {
      out_file->SetNativeFile(std::move(path), handle);
    } else {
      iree_io_file_handle_release(handle);
      EXPECT_TRUE(path.Remove());
    }
    return status;
  }
};

class AsyncFileTest : public FileTest {};

TEST_P(AsyncFileTest, DirectFileRoundTripUsesReportedRequirements) {
  // A range large enough to cross ordinary staging slots, with untouched file
  // and buffer guards. The native query supplies every required alignment.
  constexpr size_t kFileLength = 20 * 1024 * 1024;
  std::vector<uint8_t> expected(kFileLength, 0xCD);
  TestFile file;
  iree::Status status = CreateNativeTestFile(
      IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE, expected,
      &file, IREE_IO_FILE_MODE_DIRECT);
  if (status.code() == iree::StatusCode::kUnavailable ||
      status.code() == iree::StatusCode::kUnimplemented) {
    GTEST_SKIP() << status.ToString();
  }
  IREE_ASSERT_OK(status);
  const auto& requirements = *iree_hal_file_requirements(file.get());
  const auto alignment = std::max(requirements.read.buffer_alignment,
                                  requirements.write.buffer_alignment);
  const uint64_t stride = std::max(
      {requirements.read.offset_alignment, requirements.write.offset_alignment,
       uint64_t(requirements.read.length_alignment),
       uint64_t(requirements.write.length_alignment)});
  ASSERT_LE(alignment, 1024 * 1024u);
  ASSERT_LE(stride, 1024 * 1024u);
  const size_t length = iree_host_align(16 * 1024 * 1024 + 17, stride);
  const size_t buffer_offset = alignment;
  const uint64_t file_offset = stride;
  const size_t buffer_length = length + 2 * alignment;
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(GetDevicePool(&pool));
  Ref<iree_hal_buffer_t> source, target;
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(buffer_length, alignment),
      iree_infinite_timeout(), source.out()));
  IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
      pool, iree_hal_make_pool_reservation_request(buffer_length, alignment),
      iree_infinite_timeout(), target.out()));
  std::vector<uint8_t> source_contents = MakeDeterministicBytes(buffer_length);
  const std::vector<uint8_t> target_contents(buffer_length, 0xA5);
  SemaphoreList empty_wait;
  SemaphoreList initialized(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_upload(transfer_queue_, empty_wait, initialized,
                                       source_contents.data(), source.get(), 0,
                                       buffer_length, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      initialized, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  SemaphoreList target_initialized(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_upload(
      transfer_queue_, empty_wait, target_initialized, target_contents.data(),
      target.get(), 0, buffer_length, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      target_initialized, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
  ASSERT_NO_FATAL_FAILURE(
      QueueWriteThenRead(source.get(), buffer_offset, file.get(), file_offset,
                         target.get(), buffer_offset, length));
  std::vector<uint8_t> expected_buffer = target_contents;
  std::memcpy(expected_buffer.data() + buffer_offset,
              source_contents.data() + buffer_offset, length);
  std::memcpy(expected.data() + file_offset,
              source_contents.data() + buffer_offset, length);
  EXPECT_THAT(ReadBufferBytes(target.get(), 0, buffer_length),
              ContainerEq(expected_buffer));
  EXPECT_THAT(file.ReadContents(0, kFileLength), ContainerEq(expected));
}

TEST_P(FileTest, ReadEntireFile) {
  ForEachProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 128;
    const std::vector<uint8_t> file_contents(file_length, 0xDEu);
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ, file_contents,
                           &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(file_length);
    QueueReadAndWait(file.get(), /*source_offset=*/0, target_buffer.get(),
                     /*target_offset=*/0, file_length);

    EXPECT_THAT(ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length),
                ContainerEq(file_contents));
  });
}

TEST_P(FileTest, ImportRejectsUnknownFlags) {
  std::vector<uint8_t> contents(16, 0);
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(contents.data(), contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_file_import(
          device_, /*storage_pool=*/nullptr, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
          IREE_HAL_MEMORY_ACCESS_READ, handle,
          static_cast<iree_hal_external_file_flags_t>(1u), &file));
  EXPECT_EQ(file, nullptr);
  iree_io_file_handle_release(handle);
}

TEST_P(FileTest, MemoryFileUsesExplicitStoragePool) {
  for (auto owned_mode :
       {IREE_HAL_SLAB_POOL_IMPORT_ONLY, IREE_HAL_SLAB_POOL_OWNED_ENABLED}) {
    SCOPED_TRACE(owned_mode);
    Ref<iree_hal_pool_t> storage_pool;
    iree_status_t pool_status =
        CreateHostFileStoragePool(storage_pool.out(), owned_mode);
    if (iree_status_is_unavailable(pool_status) ||
        iree_status_is_unimplemented(pool_status)) {
      iree_status_free(pool_status);
      GTEST_SKIP() << "device does not support host-pointer pool imports";
    }
    IREE_ASSERT_OK(pool_status);

    iree_hal_pool_import_requirements_t requirements;
    ASSERT_TRUE(iree_hal_pool_query_import_requirements(
        storage_pool,
        iree_hal_make_external_storage_class(
            IREE_HAL_EXTERNAL_STORAGE_TYPE_HOST_POINTER),
        &requirements));
    const iree_device_size_t file_length =
        ((std::max<iree_device_size_t>(128, requirements.min_byte_length) - 1) /
             requirements.byte_length_granularity +
         1) *
        requirements.byte_length_granularity;
    ASSERT_TRUE(requirements.max_byte_length == 0 ||
                file_length <= requirements.max_byte_length);
    ASSERT_LE(file_length, static_cast<iree_device_size_t>(IREE_HOST_SIZE_MAX));
    const iree_device_size_t device_alignment = std::max<iree_device_size_t>(
        kFileAlignment, requirements.base_alignment);
    ASSERT_LE(device_alignment,
              static_cast<iree_device_size_t>(IREE_HOST_SIZE_MAX));
    const iree_host_size_t file_alignment =
        static_cast<iree_host_size_t>(device_alignment);

    void* file_contents = nullptr;
    IREE_ASSERT_OK(iree_allocator_malloc_aligned(
        iree_allocator_system(), static_cast<iree_host_size_t>(file_length),
        file_alignment, 0, &file_contents));
    AlignedAllocation allocation(static_cast<uint8_t*>(file_contents));
    const std::vector<uint8_t> expected = MakeDeterministicBytes(file_length);
    std::memcpy(file_contents, expected.data(), expected.size());
    const iree_io_file_handle_release_callback_t release_callback = {
        ReleaseAlignedAllocation,
        file_contents,
    };
    iree_io_file_handle_t* handle = nullptr;
    IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
        IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
        iree_make_byte_span(file_contents,
                            static_cast<iree_host_size_t>(file_length)),
        release_callback, iree_allocator_system(), &handle));
    allocation.release();

    TestFile file;
    iree::Status import_status = iree_hal_file_import(
        device_, storage_pool, IREE_HAL_QUEUE_FAMILY_AFFINITY_ANY,
        IREE_HAL_MEMORY_ACCESS_READ, handle, IREE_HAL_EXTERNAL_FILE_FLAG_NONE,
        file.out_file());
    iree_io_file_handle_release(handle);
    IREE_ASSERT_OK(import_status);

    iree_hal_buffer_t* storage_buffer =
        iree_hal_file_storage_buffer(file.get());
    ASSERT_NE(storage_buffer, nullptr);
    EXPECT_EQ(iree_hal_buffer_byte_length(storage_buffer), file_length);
    EXPECT_TRUE(iree_all_bits_set(iree_hal_buffer_memory_type(storage_buffer),
                                  IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
    EXPECT_EQ(iree_hal_buffer_allowed_access(storage_buffer),
              IREE_HAL_MEMORY_ACCESS_READ);
    EXPECT_EQ(iree_hal_buffer_allowed_usage(storage_buffer),
              IREE_HAL_BUFFER_USAGE_TRANSFER_SOURCE);
    Ref<iree_hal_buffer_t> target;
    if (owned_mode == IREE_HAL_SLAB_POOL_OWNED_ENABLED) {
      IREE_ASSERT_OK(iree_hal_pool_allocate_buffer(
          storage_pool, iree_hal_make_pool_reservation_request(file_length, 0),
          iree_infinite_timeout(), target.out()));
    } else {
      target = CreateZeroedBuffer(file_length);
    }
    ASSERT_TRUE(target);
    ASSERT_NO_FATAL_FAILURE(
        QueueReadAndWait(file.get(), 0, target.get(), 0, file_length));
    EXPECT_THAT(ReadBufferBytes(target.get(), 0, file_length),
                ContainerEq(expected));
  }
}

TEST_P(FileTest, ImportRejectsEmptyQueueFamilyAffinity) {
  std::vector<uint8_t> contents(16, 0);
  iree_io_file_handle_t* handle = nullptr;
  IREE_ASSERT_OK(iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ | IREE_IO_FILE_ACCESS_WRITE,
      iree_make_byte_span(contents.data(), contents.size()),
      iree_io_file_handle_release_callback_null(), iree_allocator_system(),
      &handle));

  iree_hal_file_t* file = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_file_import(device_, /*storage_pool=*/nullptr,
                           /*queue_family_affinity=*/0,
                           IREE_HAL_MEMORY_ACCESS_READ, handle,
                           IREE_HAL_EXTERNAL_FILE_FLAG_NONE, &file));
  EXPECT_EQ(file, nullptr);
  iree_io_file_handle_release(handle);
}

TEST_P(FileTest, ReadRejectsFileRangeOverflow) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_read(
          transfer_queue_, iree_hal_semaphore_list_empty(), signal, file.get(),
          UINT64_MAX - 3, target_buffer.get(), /*target_offset=*/0,
          /*length=*/8, /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));
}

TEST_P(FileTest, ReadRejectsRangePastBufferEnd) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_read(
          transfer_queue_, iree_hal_semaphore_list_empty(), signal, file.get(),
          /*source_offset=*/0, target_buffer.get(), /*target_offset=*/15,
          /*length=*/2, /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));
}

TEST_P(FileTest, WriteRejectsRangePastFileEnd) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> source_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_write(transfer_queue_, iree_hal_semaphore_list_empty(),
                           signal, source_buffer.get(), /*source_offset=*/0,
                           file.get(),
                           /*target_offset=*/15, /*length=*/2,
                           /*barriers=*/NULL, IREE_HAL_WRITE_FLAG_NONE));
}

TEST_P(FileTest, WriteRejectsRangePastBufferEnd) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> source_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_queue_write(transfer_queue_, iree_hal_semaphore_list_empty(),
                           signal, source_buffer.get(), /*source_offset=*/15,
                           file.get(),
                           /*target_offset=*/0, /*length=*/2, /*barriers=*/NULL,
                           IREE_HAL_WRITE_FLAG_NONE));
}

TEST_P(FileTest, ReadRejectsIncompatibleBufferAccess) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));

  Ref<iree_hal_buffer_t> read_only_buffer =
      CreateBufferWithAccess(IREE_HAL_MEMORY_ACCESS_READ, 16);
  SemaphoreList access_signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_read(transfer_queue_, iree_hal_semaphore_list_empty(),
                          access_signal, file.get(), /*source_offset=*/0,
                          read_only_buffer.get(), /*target_offset=*/0,
                          /*length=*/1, /*barriers=*/NULL,
                          IREE_HAL_READ_FLAG_NONE));
}

TEST_P(FileTest, WriteRejectsIncompatibleBufferAccess) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));

  Ref<iree_hal_buffer_t> write_only_buffer =
      CreateBufferWithAccess(IREE_HAL_MEMORY_ACCESS_WRITE, 16);
  SemaphoreList access_signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_write(transfer_queue_, iree_hal_semaphore_list_empty(),
                           access_signal, write_only_buffer.get(),
                           /*source_offset=*/0, file.get(),
                           /*target_offset=*/0, /*length=*/1, /*barriers=*/NULL,
                           IREE_HAL_WRITE_FLAG_NONE));
}

TEST_P(FileTest, ReadRejectsWriteOnlyFile) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_read(
          transfer_queue_, iree_hal_semaphore_list_empty(), signal, file.get(),
          /*source_offset=*/0, target_buffer.get(), /*target_offset=*/0,
          /*length=*/1, /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));
}

TEST_P(FileTest, WriteRejectsReadOnlyFile) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> source_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_queue_write(transfer_queue_, iree_hal_semaphore_list_empty(),
                           signal, source_buffer.get(), /*source_offset=*/0,
                           file.get(),
                           /*target_offset=*/0, /*length=*/1, /*barriers=*/NULL,
                           IREE_HAL_WRITE_FLAG_NONE));
}

TEST_P(FileTest, ReadRejectsUnknownFlags) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_read(
          transfer_queue_, iree_hal_semaphore_list_empty(), signal, file.get(),
          /*source_offset=*/0, target_buffer.get(), /*target_offset=*/0,
          /*length=*/1, /*barriers=*/NULL, (iree_hal_read_flags_t)1u));
}

TEST_P(FileTest, WriteRejectsUnknownFlags) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> source_buffer = CreateZeroedBuffer(16);

  SemaphoreList signal(device_, {0}, {1});
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_queue_write(transfer_queue_, iree_hal_semaphore_list_empty(),
                           signal, source_buffer.get(), /*source_offset=*/0,
                           file.get(),
                           /*target_offset=*/0, /*length=*/1, /*barriers=*/NULL,
                           (iree_hal_write_flags_t)1u));
}

TEST_P(FileTest, SynchronousReadRejectsRangePastFileEnd) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  uint8_t target[2] = {};

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_file_read(file.get(), /*file_offset=*/15,
                         iree_make_byte_span(target, sizeof(target))));
}

TEST_P(FileTest, SynchronousWriteRejectsReadOnlyFile) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  const uint8_t source = 0;

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_file_write(file.get(), /*file_offset=*/0,
                          iree_make_const_byte_span(&source, sizeof(source))));
}

TEST_P(FileTest, SynchronousHostSpans) {
  ForEachProvider([&](const FileProvider& provider) {
    std::vector<uint8_t> file_contents(256);
    for (size_t i = 0; i < file_contents.size(); ++i) {
      file_contents[i] = static_cast<uint8_t>(i);
    }
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_ALL, file_contents,
                           &file)) {
      return;
    }
    ASSERT_TRUE(iree_hal_file_supports_synchronous_io(file.get()));

    // Unaligned host spans need no HAL allocation, import or public mapping.
    std::vector<uint8_t> target(80, 0xA5);
    IREE_ASSERT_OK(iree_hal_file_read(
        file.get(), 19, iree_make_byte_span(target.data() + 3, 67)));
    std::vector<uint8_t> expected_target(80, 0xA5);
    std::copy_n(file_contents.data() + 19, 67, expected_target.data() + 3);
    EXPECT_THAT(target, ContainerEq(expected_target));

    const std::vector<uint8_t> source = target;
    IREE_ASSERT_OK(iree_hal_file_write(
        file.get(), 103, iree_make_const_byte_span(source.data() + 7, 59)));
    std::copy_n(source.data() + 7, 59, file_contents.data() + 103);

    // Empty spans permit a null pointer and preserve the logical file bound.
    IREE_ASSERT_OK(iree_hal_file_read(file.get(), file_contents.size(),
                                      iree_byte_span_empty()));
    IREE_ASSERT_OK(iree_hal_file_write(file.get(), file_contents.size(),
                                       iree_const_byte_span_empty()));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_file_read(file.get(), file_contents.size() + 1,
                           iree_byte_span_empty()));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_file_write(file.get(), file_contents.size() + 1,
                            iree_const_byte_span_empty()));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_OUT_OF_RANGE,
        iree_hal_file_write(file.get(), file_contents.size() - 1,
                            iree_make_const_byte_span(source.data(), 2)));
    EXPECT_THAT(file.ReadContents(0, file_contents.size()),
                ContainerEq(file_contents));
  });
}

TEST_P(FileTest, SynchronousSpansRequireStorage) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_ALL,
                                std::vector<uint8_t>(16, 0), &file));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_file_read(file.get(), 0, iree_make_byte_span(nullptr, 1)));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_hal_file_write(file.get(), 0,
                          iree_make_const_byte_span(nullptr, 1)));
}

TEST_P(AsyncFileTest, SynchronousAndQueuedAccessShareFile) {
  ForEachProvider([&](const FileProvider& provider) {
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_ALL,
                           std::vector<uint8_t>(256, 0), &file)) {
      return;
    }
    Ref<iree_hal_buffer_t> buffer = CreateZeroedBuffer(128);
    const std::vector<uint8_t> contents = MakeDeterministicBytes(81);

    // Native files have already been imported into the device's proactor.
    // Synchronous operations must consume only their own completion; queued
    // operations continue through the proactor on the same underlying file.
    IREE_ASSERT_OK(iree_hal_file_write(
        file.get(), 11,
        iree_make_const_byte_span(contents.data(), contents.size())));
    std::vector<uint8_t> readback(contents.size(), 0);
    IREE_ASSERT_OK(iree_hal_file_read(
        file.get(), 11, iree_make_byte_span(readback.data(), readback.size())));
    EXPECT_THAT(readback, ContainerEq(contents));

    QueueReadAndWait(file.get(), 11, buffer.get(), 7, contents.size());
    EXPECT_THAT(ReadBufferBytes(buffer.get(), 7, contents.size()),
                ContainerEq(contents));
    QueueWriteAndWait(buffer.get(), 7, file.get(), 139, contents.size());
    std::fill(readback.begin(), readback.end(), 0);
    IREE_ASSERT_OK(iree_hal_file_read(
        file.get(), 139,
        iree_make_byte_span(readback.data(), readback.size())));
    EXPECT_THAT(readback, ContainerEq(contents));
  });
}

TEST_P(FileTest, SynchronousReadRejectsWriteOnlyFile) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  uint8_t target = 0;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_PERMISSION_DENIED,
      iree_hal_file_read(file.get(), 0,
                         iree_make_byte_span(&target, sizeof(target))));
}

TEST_P(AsyncFileTest, ZeroLengthReadForwardsDependencies) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(16);

  SemaphoreList wait_signal(device_, {0}, {1});
  SemaphoreList read_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_read(
      transfer_queue_, wait_signal, read_signal, file.get(),
      /*source_offset=*/16, target_buffer.get(), /*target_offset=*/16,
      /*length=*/0, /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));

  uint64_t read_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(read_signal.semaphores[0], &read_value));
  EXPECT_EQ(read_value, 0u);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_signal.semaphores[0], 1,
                                           /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      read_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(AsyncFileTest, ZeroLengthWriteForwardsDependencies) {
  const FileProvider provider = {"memory_file", FileProviderKind::kMemory};
  TestFile file;
  ASSERT_TRUE(TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE,
                                std::vector<uint8_t>(16, 0), &file));
  Ref<iree_hal_buffer_t> source_buffer = CreateZeroedBuffer(16);

  SemaphoreList wait_signal(device_, {0}, {1});
  SemaphoreList write_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_write(
      transfer_queue_, wait_signal, write_signal, source_buffer.get(),
      /*source_offset=*/16, file.get(),
      /*target_offset=*/16, /*length=*/0, /*barriers=*/NULL,
      IREE_HAL_WRITE_FLAG_NONE));

  uint64_t write_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(write_signal.semaphores[0], &write_value));
  EXPECT_EQ(write_value, 0u);

  IREE_ASSERT_OK(iree_hal_semaphore_signal(wait_signal.semaphores[0], 1,
                                           /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(AsyncFileTest, MemoryFileChainRetainsDeferredResources) {
  const iree_device_size_t allocation_length = 640;
  const iree_device_size_t subspan_offset = 73;
  const iree_device_size_t subspan_length = 400;
  const iree_device_size_t buffer_offset = 31;
  const iree_device_size_t transfer_length = 257;
  const iree_device_size_t source_offset = 19;
  const iree_device_size_t target_offset = 29;
  const iree_device_size_t file_length = 512;

  AlignedAllocation source_contents;
  IREE_ASSERT_OK(
      AllocateBorrowedHostFileStorage(file_length, &source_contents));
  std::memset(source_contents.get(), 0x11, static_cast<size_t>(file_length));
  AlignedAllocation target_contents;
  IREE_ASSERT_OK(
      AllocateBorrowedHostFileStorage(file_length, &target_contents));
  std::memset(target_contents.get(), 0xC7, static_cast<size_t>(file_length));
  TestFile source_file;
  IREE_ASSERT_OK(CreateBorrowedMemoryTestFile(
      IREE_HAL_MEMORY_ACCESS_READ,
      iree_make_byte_span(source_contents.get(),
                          static_cast<iree_host_size_t>(file_length)),
      &source_file));
  TestFile target_file;
  IREE_ASSERT_OK(CreateBorrowedMemoryTestFile(
      IREE_HAL_MEMORY_ACCESS_WRITE,
      iree_make_byte_span(target_contents.get(),
                          static_cast<iree_host_size_t>(file_length)),
      &target_file));

  Ref<iree_hal_buffer_t> allocation = CreateZeroedBuffer(allocation_length);
  Ref<iree_hal_buffer_t> subspan;
  IREE_ASSERT_OK(
      iree_hal_buffer_subspan(allocation.get(), subspan_offset, subspan_length,
                              iree_allocator_system(), subspan.out()));

  SemaphoreList host_producer_signal(device_, {0}, {1});
  SemaphoreList read_signal(device_, {0}, {1});
  SemaphoreList write_signal(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_read(
      transfer_queue_, host_producer_signal, read_signal, source_file.get(),
      source_offset, subspan.get(), buffer_offset, transfer_length,
      /*barriers=*/NULL, IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_write(
      transfer_queue_, read_signal, write_signal, subspan.get(), buffer_offset,
      target_file.get(), target_offset, transfer_length, /*barriers=*/NULL,
      IREE_HAL_WRITE_FLAG_NONE));

  uint64_t write_value = 0;
  IREE_ASSERT_OK(
      iree_hal_semaphore_query(write_signal.semaphores[0], &write_value));
  EXPECT_EQ(write_value, 0u);

  const std::vector<uint8_t> produced_contents =
      MakeDeterministicBytes(transfer_length);
  std::memcpy(source_contents.get() + source_offset, produced_contents.data(),
              produced_contents.size());

  // Accepted operations retain every resource they need until terminal
  // completion. Keep only the externally owned host allocations and fences.
  source_file.Reset();
  target_file.Reset();
  subspan.reset();
  allocation.reset();

  IREE_ASSERT_OK(iree_hal_semaphore_signal(host_producer_signal.semaphores[0],
                                           1, /*frontier=*/nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      write_signal, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));

  // Completion releases the final owned view, so the caller may destroy the
  // borrowed source pool immediately without waiting for another callback.
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(GetDevicePool(&pool));
  iree_hal_pool_stats_t stats;
  iree_hal_pool_query_stats(pool, &stats);
  EXPECT_EQ(stats.reservation_count, 0u);

  std::vector<uint8_t> expected(file_length, 0xC7);
  std::copy(produced_contents.begin(), produced_contents.end(),
            expected.begin() + target_offset);
  const std::vector<uint8_t> actual(target_contents.get(),
                                    target_contents.get() + file_length);
  EXPECT_THAT(actual, ContainerEq(expected));
}

TEST_P(AsyncFileTest, NativeFileChainRetainsDeferredResources) {
  const iree_device_size_t length = 2 * 1024 * 1024 + 257;
  const iree_device_size_t file_offset = 19;
  const iree_device_size_t buffer_offset = 31;
  const iree_device_size_t file_length = length + 64;
  TestFile source_file, target_file;
  iree::Status file_status = CreateNativeTestFile(
      IREE_HAL_MEMORY_ACCESS_ALL, std::vector<uint8_t>(file_length, 0x11),
      &source_file);
  if (file_status.code() == iree::StatusCode::kUnavailable ||
      file_status.code() == iree::StatusCode::kUnimplemented) {
    GTEST_SKIP() << file_status.ToString();
  }
  IREE_ASSERT_OK(file_status);
  IREE_ASSERT_OK(CreateNativeTestFile(IREE_HAL_MEMORY_ACCESS_WRITE,
                                      std::vector<uint8_t>(file_length, 0xC7),
                                      &target_file));
  iree_hal_pool_t* pool = nullptr;
  IREE_ASSERT_OK(GetDevicePool(&pool));
  Ref<iree_hal_buffer_t> buffer;
  const auto contents = MakeDeterministicBytes(length);
  SemaphoreList producer(device_, {0}, {1});
  SemaphoreList allocated(device_, {0}, {1});
  SemaphoreList read_done(device_, {0}, {1});
  SemaphoreList write_done(device_, {0}, {1});
  SemaphoreList released(device_, {0}, {1});
  const auto request = iree_hal_make_pool_reservation_request(length + 64, 1);
  IREE_ASSERT_OK(iree_hal_queue_alloca(transfer_queue_, producer, allocated,
                                       pool, 1, &request, buffer.out()));
  iree_status_t status = iree_hal_queue_read(
      transfer_queue_, allocated, read_done, source_file.get(), file_offset,
      buffer.get(), buffer_offset, length, nullptr, IREE_HAL_READ_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(producer.semaphores[0], iree_status_clone(status));
  }
  IREE_ASSERT_OK(status);
  status =
      iree_hal_queue_write(transfer_queue_, read_done, write_done, buffer.get(),
                           buffer_offset, target_file.get(), file_offset,
                           length, nullptr, IREE_HAL_WRITE_FLAG_NONE);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(producer.semaphores[0], iree_status_clone(status));
  }
  IREE_ASSERT_OK(status);
  iree_hal_buffer_t* allocation = buffer.get();
  status = iree_hal_queue_dealloca(transfer_queue_, write_done, released, 1,
                                   &allocation);
  if (!iree_status_is_ok(status)) {
    iree_hal_semaphore_fail(producer.semaphores[0], iree_status_clone(status));
  }
  IREE_ASSERT_OK(status);

  // Populate bytes only after admission. The dependency prevents file I/O
  // from racing this producer or resolving the uncommitted allocation, and
  // native staging must span multiple chunks.
  IREE_EXPECT_OK(iree_hal_file_write(
      source_file.get(), file_offset,
      iree_make_const_byte_span(contents.data(), contents.size())));
  source_file.ReleaseHandles();
  target_file.ReleaseHandles();
  buffer.reset();
  IREE_ASSERT_OK(iree_hal_semaphore_signal(producer.semaphores[0], 1, nullptr));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(released, iree_infinite_timeout(),
                                              IREE_ASYNC_WAIT_FLAG_NONE));
  std::vector<uint8_t> expected(file_length, 0xC7);
  std::copy(contents.begin(), contents.end(), expected.begin() + file_offset);
  EXPECT_THAT(target_file.ReadContents(0, file_length), ContainerEq(expected));
}

TEST_P(AsyncFileTest, TruncatedFileFailsItsConsumer) {
  const auto contents = MakeDeterministicBytes(128);
  TestFile file;
  iree::Status file_status =
      CreateNativeTestFile(IREE_HAL_MEMORY_ACCESS_READ, contents, &file);
  if (file_status.code() == iree::StatusCode::kUnavailable ||
      file_status.code() == iree::StatusCode::kUnimplemented) {
    GTEST_SKIP() << file_status.ToString();
  }
  IREE_ASSERT_OK(file_status);
  Ref<iree_hal_buffer_t> buffer = CreateZeroedBuffer(contents.size());
  // Import captured the logical length. An external truncation must produce
  // an I/O failure instead of silently completing only a prefix.
  IREE_ASSERT_OK(iree_io_file_contents_write(
      file.native_path(), iree_make_const_byte_span(contents.data(), 7),
      iree_allocator_system()));
  SemaphoreList empty;
  SemaphoreList read_done(device_, {0}, {1});
  SemaphoreList consumer(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_read(
      transfer_queue_, empty, read_done, file.get(), 0, buffer.get(), 0,
      contents.size(), nullptr, IREE_HAL_READ_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_queue_barrier(transfer_queue_, read_done, consumer,
                                        nullptr,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  file.ReleaseHandles();
  buffer.reset();
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_OUT_OF_RANGE,
      iree_hal_semaphore_list_wait(consumer, iree_infinite_timeout(),
                                   IREE_ASYNC_WAIT_FLAG_NONE));
  SemaphoreList independent(device_, {0}, {1});
  IREE_ASSERT_OK(iree_hal_queue_barrier(transfer_queue_, empty, independent,
                                        nullptr,
                                        IREE_HAL_QUEUE_BARRIER_FLAG_NONE));
  IREE_ASSERT_OK(iree_hal_semaphore_list_wait(
      independent, iree_infinite_timeout(), IREE_ASYNC_WAIT_FLAG_NONE));
}

TEST_P(AsyncFileTest, FailedDependencyPreventsReadAndWrite) {
  ForEachProvider([&](const FileProvider& provider) {
    const auto source = MakeDeterministicBytes(256);
    const std::vector<uint8_t> unchanged(256, 0xA5);
    TestFile source_file, target_file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ, source,
                           &source_file) ||
        !TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_WRITE, unchanged,
                           &target_file)) {
      return;
    }
    Ref<iree_hal_buffer_t> source_buffer = CreateBufferWithData(source);
    Ref<iree_hal_buffer_t> target_buffer = CreateBufferWithData(unchanged);
    SemaphoreList producer(device_, {0}, {1});
    SemaphoreList read_done(device_, {0}, {1});
    SemaphoreList write_done(device_, {0}, {1});
    IREE_ASSERT_OK(iree_hal_queue_read(transfer_queue_, producer, read_done,
                                       source_file.get(), 0,
                                       target_buffer.get(), 0, source.size(),
                                       nullptr, IREE_HAL_READ_FLAG_NONE));
    iree_status_t status = iree_hal_queue_write(
        transfer_queue_, producer, write_done, source_buffer.get(), 0,
        target_file.get(), 0, source.size(), nullptr, IREE_HAL_WRITE_FLAG_NONE);
    iree_hal_semaphore_fail(
        producer.semaphores[0],
        iree_make_status(IREE_STATUS_ABORTED, "file producer failed"));
    IREE_ASSERT_OK(status);
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_list_wait(read_done, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_ABORTED,
        iree_hal_semaphore_list_wait(write_done, iree_infinite_timeout(),
                                     IREE_ASYNC_WAIT_FLAG_NONE));
    EXPECT_THAT(ReadBufferBytes(target_buffer.get(), 0, unchanged.size()),
                ContainerEq(unchanged));
    EXPECT_THAT(target_file.ReadContents(0, unchanged.size()),
                ContainerEq(unchanged));
  });
}

TEST_P(FileTest, ReadSubrangeWithOffsets) {
  ForEachProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 512;
    const iree_device_size_t target_length = 384;
    const iree_device_size_t source_offset = 97;
    const iree_device_size_t target_offset = 13;
    const iree_device_size_t read_length = 211;
    const std::vector<uint8_t> file_contents =
        MakeDeterministicBytes(file_length);
    TestFile file;
    if (!TryCreateTestFile(provider, IREE_HAL_MEMORY_ACCESS_READ, file_contents,
                           &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(target_length);
    QueueReadAndWait(file.get(), source_offset, target_buffer.get(),
                     target_offset, read_length);

    std::vector<uint8_t> expected(static_cast<size_t>(target_length), 0);
    std::copy(file_contents.begin() + source_offset,
              file_contents.begin() + source_offset + read_length,
              expected.begin() + target_offset);
    EXPECT_THAT(
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, target_length),
        ContainerEq(expected));
  });
}

TEST_P(FileTest, WriteSubrangeWithOffsets) {
  ForEachProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 512;
    const iree_device_size_t source_length = 384;
    const iree_device_size_t source_offset = 23;
    const iree_device_size_t target_offset = 41;
    const iree_device_size_t write_length = 307;
    std::vector<uint8_t> file_contents = MakeDeterministicBytes(file_length);
    std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(source_length);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    QueueWriteAndWait(source_buffer.get(), source_offset, file.get(),
                      target_offset, write_length);

    std::copy(source_contents.begin() + source_offset,
              source_contents.begin() + source_offset + write_length,
              file_contents.begin() + target_offset);
    EXPECT_THAT(file.ReadContents(/*offset=*/0, file_length),
                ContainerEq(file_contents));
  });
}

TEST_P(FileTest, ChainedWriteThenReadWithSemaphores) {
  ForEachProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 768;
    const iree_device_size_t source_length = 512;
    const iree_device_size_t target_length = 512;
    const iree_device_size_t source_offset = 19;
    const iree_device_size_t file_offset = 123;
    const iree_device_size_t target_offset = 37;
    const iree_device_size_t transfer_length = 333;
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(source_length);
    const std::vector<uint8_t> file_contents(file_length, 0);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(target_length);
    QueueWriteThenRead(source_buffer.get(), source_offset, file.get(),
                       file_offset, target_buffer.get(), target_offset,
                       transfer_length);

    std::vector<uint8_t> expected(static_cast<size_t>(target_length), 0);
    std::copy(source_contents.begin() + source_offset,
              source_contents.begin() + source_offset + transfer_length,
              expected.begin() + target_offset);
    EXPECT_THAT(
        ReadBufferBytes(target_buffer.get(), /*offset=*/0, target_length),
        ContainerEq(expected));
  });
}

TEST_P(FileTest, LargeRangeRoundTrip) {
  ForEachProvider([&](const FileProvider& provider) {
    const iree_device_size_t file_length = 3 * 1024 * 1024 + 31;
    const std::vector<uint8_t> source_contents =
        MakeDeterministicBytes(file_length);
    const std::vector<uint8_t> file_contents(file_length, 0);
    TestFile file;
    if (!TryCreateTestFile(
            provider,
            IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
            file_contents, &file)) {
      return;
    }

    Ref<iree_hal_buffer_t> source_buffer =
        CreateBufferWithData(source_contents);
    Ref<iree_hal_buffer_t> target_buffer = CreateZeroedBuffer(file_length);
    QueueWriteThenRead(source_buffer.get(), /*source_offset=*/0, file.get(),
                       /*file_offset=*/0, target_buffer.get(),
                       /*target_offset=*/0, file_length);

    EXPECT_THAT(ReadBufferBytes(target_buffer.get(), /*offset=*/0, file_length),
                ContainerEq(source_contents));
  });
}

CTS_REGISTER_TEST_SUITE_WITH_TAGS(FileTest, {"file_io"}, {});
CTS_REGISTER_TEST_SUITE_WITH_TAGS(
    AsyncFileTest, (std::vector<std::string>{"async_queue", "file_io"}), {});

}  // namespace iree::hal::cts
