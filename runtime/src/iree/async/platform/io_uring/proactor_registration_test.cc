// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

#include "iree/async/buffer_pool.h"
#include "iree/async/file.h"
#include "iree/async/operations/file.h"
#include "iree/async/operations/net.h"
#include "iree/async/platform/io_uring/api.h"
#include "iree/async/slab.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

class SlabRegistrationTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(getrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
    page_size_ = static_cast<iree_host_size_t>(sysconf(_SC_PAGESIZE));
    iree_status_t status = iree_async_proactor_create_io_uring(
        iree_async_proactor_options_default(), iree_allocator_system(),
        &proactor_);
    if (iree_status_is_unavailable(status)) {
      iree_status_free(status);
      GTEST_SKIP() << "io_uring is unavailable";
    }
    IREE_ASSERT_OK(status);
  }

  void TearDown() override {
    if (limit_changed_) {
      EXPECT_EQ(setrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
    }
    ResetSlab();
    iree_async_proactor_release(proactor_);
  }

  void ResetSlab() {
    iree_async_region_release(region_);
    region_ = nullptr;
    iree_async_slab_release(slab_);
    slab_ = nullptr;
    if (mapping_ != MAP_FAILED) {
      EXPECT_EQ(munmap(mapping_, mapping_length_), 0);
      mapping_ = MAP_FAILED;
    }
  }

  void CreateSlab(iree_host_size_t buffer_size, iree_host_size_t buffer_count) {
    ResetSlab();
    mapping_length_ = buffer_size * buffer_count;
    mapping_ = mmap(nullptr, mapping_length_, PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    ASSERT_NE(mapping_, MAP_FAILED) << strerror(errno);
    ASSERT_EQ(madvise(mapping_, mapping_length_, MADV_NOHUGEPAGE), 0);
    IREE_ASSERT_OK(iree_async_slab_wrap(mapping_, buffer_size, buffer_count,
                                        iree_allocator_system(), &slab_));
  }

  void RegisterAtFirstSlot() {
    IREE_ASSERT_OK(iree_async_proactor_register_slab(
        proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_));
    ASSERT_NE(region_, nullptr);
    EXPECT_EQ(region_->handles.iouring.base_buffer_index, 0);
  }

  // Proactor owning the kernel and userspace fixed-buffer tables.
  iree_async_proactor_t* proactor_ = nullptr;

  // Slab borrowing the mapped pages.
  iree_async_slab_t* slab_ = nullptr;

  // Region retained across assertions until fixture teardown.
  iree_async_region_t* region_ = nullptr;

  // Mapping owned until after all registrations and slab references expire.
  void* mapping_ = MAP_FAILED;

  // Byte length of the mapped slab storage.
  iree_host_size_t mapping_length_ = 0;

  // Native page size used to control the registration geometry.
  iree_host_size_t page_size_ = 0;

  // Original process pinning limit restored before teardown.
  struct rlimit original_limit_ = {};

  // True after the test lowers the process pinning limit.
  bool limit_changed_ = false;
};

class SlabRegistrationFaultTest : public SlabRegistrationTest,
                                  public ::testing::WithParamInterface<int> {};

TEST_P(SlabRegistrationFaultTest, ReportsFaultAndReleasesEveryReservedSlot) {
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 4));
  auto* fault_page = static_cast<uint8_t*>(mapping_) + GetParam() * page_size_;
  ASSERT_EQ(mprotect(fault_page, page_size_, PROT_NONE), 0);

  // Later faults follow a successfully registered prefix. They must preserve
  // the mapping error instead of reporting an unexpected short registration.
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_register_slab(
          proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_));
  EXPECT_EQ(region_, nullptr);
  iree_async_region_release(region_);
  region_ = nullptr;

  ASSERT_EQ(mprotect(fault_page, page_size_, PROT_READ | PROT_WRITE), 0);
  ASSERT_NO_FATAL_FAILURE(RegisterAtFirstSlot());
}

INSTANTIATE_TEST_SUITE_P(FaultPosition, SlabRegistrationFaultTest,
                         ::testing::Values(0, 1, 3));

TEST_P(SlabRegistrationFaultTest, HostBufferFailurePreservesRegistrationState) {
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 4));
  auto* fault_page = static_cast<uint8_t*>(mapping_) + GetParam() * page_size_;
  ASSERT_EQ(mprotect(fault_page, page_size_, PROT_NONE), 0);
  iree_async_buffer_registration_state_t state = {};
  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      iree_async_proactor_register_buffer(
          proactor_, &state, iree_make_byte_span(mapping_, mapping_length_),
          IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &entry));
  EXPECT_EQ(entry, nullptr);
  iree_async_buffer_registration_state_deinitialize(&state);

  ASSERT_EQ(mprotect(fault_page, page_size_, PROT_READ | PROT_WRITE), 0);
  ASSERT_NO_FATAL_FAILURE(RegisterAtFirstSlot());
}

TEST_F(SlabRegistrationTest, HostBufferAccessDoesNotSelectReceiveBuffers) {
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 3));
  const iree_async_buffer_access_flags_t accesses[] = {
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
      IREE_ASYNC_BUFFER_ACCESS_FLAG_READ | IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
  };
  for (auto access : accesses) {
    iree_async_buffer_registration_state_t state = {};
    iree_async_buffer_registration_entry_t* entry = nullptr;
    IREE_ASSERT_OK(iree_async_proactor_register_buffer(
        proactor_, &state, iree_make_byte_span(mapping_, mapping_length_),
        access, &entry));
    EXPECT_EQ(entry->region->access_flags, access);
    EXPECT_EQ(entry->region->buffer_size, mapping_length_);
    EXPECT_EQ(entry->region->buffer_count, 1u);
    EXPECT_EQ(entry->region->handles.iouring.base_buffer_index, 0);
    EXPECT_EQ(entry->region->handles.iouring.buffer_group_id, -1);
    iree_async_buffer_registration_state_deinitialize(&state);
  }
}

TEST_F(SlabRegistrationTest, HostBufferPinLimitRejectsUnpinnedAdmission) {
  constexpr rlim_t kMaxPinLimit = 8 * 1024 * 1024;
  struct rlimit limit = original_limit_;
  if (limit.rlim_cur > kMaxPinLimit) {
    limit.rlim_cur = kMaxPinLimit;
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateSlab(page_size_, limit.rlim_cur / page_size_ + 1));
  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &limit), 0);
  limit_changed_ = true;

  iree_async_buffer_registration_state_t state = {};
  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_RESOURCE_EXHAUSTED,
      iree_async_proactor_register_buffer(
          proactor_, &state, iree_make_byte_span(mapping_, mapping_length_),
          IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &entry));
  EXPECT_EQ(entry, nullptr);
  iree_async_buffer_registration_state_deinitialize(&state);

  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
  limit_changed_ = false;
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 4));
  ASSERT_NO_FATAL_FAILURE(RegisterAtFirstSlot());
}

TEST_F(SlabRegistrationTest, PinLimitReleasesPartialRegistration) {
  constexpr rlim_t kMaxPinLimit = 8 * 1024 * 1024;
  struct rlimit limit = original_limit_;
  if (limit.rlim_cur > kMaxPinLimit) {
    limit.rlim_cur = kMaxPinLimit;
  }
  const iree_host_size_t buffer_count = limit.rlim_cur / page_size_ + 1;
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, buffer_count));
  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &limit), 0);
  limit_changed_ = true;

  // The slab exceeds the entire pin budget by one page. Registration may
  // complete a prefix or fail immediately if other rings already consume
  // that budget. Both cases use the existing unpinned region representation.
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_READ, &region_));
  ASSERT_NE(region_, nullptr);
  EXPECT_EQ(region_->handles.iouring.base_buffer_index, -1);
  iree_async_region_release(region_);
  region_ = nullptr;

  ASSERT_EQ(setrlimit(RLIMIT_MEMLOCK, &original_limit_), 0);
  limit_changed_ = false;
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 4));
  ASSERT_NO_FATAL_FAILURE(RegisterAtFirstSlot());
}

class FileRegistrationTest : public SlabRegistrationTest,
                             public ::testing::WithParamInterface<bool> {
 protected:
  void SetUp() override {
    SlabRegistrationTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }

    // Shared backing preserves the registered pages when CPU translations are
    // discarded. Anonymous MADV_DONTNEED would replace the visible contents.
    backing_fd_ = memfd_create("iree_fixed_file_buffer", MFD_CLOEXEC);
    ASSERT_GE(backing_fd_, 0);
    mapping_length_ = 4 * page_size_;
    ASSERT_EQ(ftruncate(backing_fd_, mapping_length_), 0);
    mapping_ = mmap(nullptr, mapping_length_, PROT_READ | PROT_WRITE,
                    MAP_SHARED, backing_fd_, 0);
    ASSERT_NE(mapping_, MAP_FAILED);
    ASSERT_EQ(madvise(mapping_, mapping_length_, MADV_NOHUGEPAGE), 0);
    if (GetParam()) {
      iree_async_buffer_registration_entry_t* entry = nullptr;
      IREE_ASSERT_OK(iree_async_proactor_register_buffer(
          proactor_, &registration_state_,
          iree_make_byte_span(mapping_, mapping_length_),
          IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
              IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
          &entry));
      region_ = entry->region;
      iree_async_region_retain(region_);
      EXPECT_EQ(region_->handles.iouring.buffer_group_id, -1);
    } else {
      IREE_ASSERT_OK(iree_async_slab_wrap(mapping_, 2 * page_size_, 2,
                                          iree_allocator_system(), &slab_));
      IREE_ASSERT_OK(iree_async_proactor_register_slab(
          proactor_, slab_,
          IREE_ASYNC_BUFFER_ACCESS_FLAG_READ |
              IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE,
          &region_));
    }
    ASSERT_GE(region_->handles.iouring.base_buffer_index, 0);

    raw_file_fd_ = memfd_create("iree_fixed_file_contents", MFD_CLOEXEC);
    ASSERT_GE(raw_file_fd_, 0);
    IREE_ASSERT_OK(iree_async_file_import(
        proactor_, iree_async_primitive_from_fd(raw_file_fd_),
        IREE_ASYNC_FILE_FLAG_NONE, &file_));
    raw_file_fd_ = -1;

    // Pagemap exposes presence without reading the buffer and faulting it in.
    // PFN disclosure is privileged; the presence bit for our own VMA is not.
    pagemap_fd_ = open("/proc/self/pagemap", O_RDONLY | O_CLOEXEC);
    ASSERT_GE(pagemap_fd_, 0);
  }

  void TearDown() override {
    iree_async_file_release(file_);
    if (raw_file_fd_ >= 0) {
      EXPECT_EQ(close(raw_file_fd_), 0);
    }
    if (pagemap_fd_ >= 0) {
      EXPECT_EQ(close(pagemap_fd_), 0);
    }
    iree_async_buffer_registration_state_deinitialize(&registration_state_);
    SlabRegistrationTest::TearDown();
    if (backing_fd_ >= 0) {
      EXPECT_EQ(close(backing_fd_), 0);
    }
  }

  void DiscardCpuTranslations() {
    ASSERT_EQ(madvise(mapping_, mapping_length_, MADV_DONTNEED), 0);
    ASSERT_NO_FATAL_FAILURE(ExpectNoCpuTranslations());
  }

  void ExpectNoCpuTranslations() {
    for (iree_host_size_t offset = 0; offset < mapping_length_;
         offset += page_size_) {
      uint64_t entry = 0;
      uint64_t page_index =
          (reinterpret_cast<uintptr_t>(mapping_) + offset) / page_size_;
      ASSERT_EQ(
          pread(pagemap_fd_, &entry, sizeof(entry), page_index * sizeof(entry)),
          sizeof(entry));
      EXPECT_EQ(entry & (uint64_t{1} << 63), 0u) << "byte offset " << offset;
    }
  }

  void Run(iree_async_operation_t* operation) {
    struct Completion {
      // Published by the terminal callback on this polling thread.
      bool done = false;
      // Terminal result consumed after polling completes.
      iree::Status status;
    } completion;
    operation->completion_fn =
        +[](void* user_data, iree_async_operation_t*, iree_status_t status,
            iree_async_completion_flags_t flags) {
          auto& completion = *static_cast<Completion*>(user_data);
          EXPECT_EQ(flags & IREE_ASYNC_COMPLETION_FLAG_MORE, 0u);
          completion.status = std::move(status);
          completion.done = true;
        };
    operation->user_data = &completion;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, operation));
    while (!completion.done) {
      IREE_ASSERT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), /*out_completed_count=*/nullptr));
    }
    IREE_ASSERT_OK(completion.status.release());
  }

  // Registration owner whose references may retire before accepted I/O.
  iree_async_buffer_registration_state_t registration_state_ = {};
  // Shared physical backing retained until all registrations are released.
  int backing_fd_ = -1;
  // Own-process translation metadata, independent of payload access.
  int pagemap_fd_ = -1;
  // Owned file descriptor until import transfers it to |file_|.
  int raw_file_fd_ = -1;
  // Source and destination file owned until all operations complete.
  iree_async_file_t* file_ = nullptr;
};

TEST_P(FileRegistrationTest, ReadWriteWithoutCpuTranslations) {
  const iree_async_span_t spans[] = {
      iree_async_span_make(region_, 0, 2 * page_size_),
      iree_async_span_make(region_, 2 * page_size_, 2 * page_size_),
      iree_async_span_make(region_, 2 * page_size_ + 17, page_size_ - 33),
  };
  std::vector<uint8_t> expected(mapping_length_);
  std::vector<uint8_t> actual(mapping_length_);
  for (iree_async_span_t span : spans) {
    SCOPED_TRACE(span.offset);
    for (iree_host_size_t i = 0; i < expected.size(); ++i) {
      expected[i] = static_cast<uint8_t>(i * 13 + 7);
    }
    memcpy(mapping_, expected.data(), expected.size());
    ASSERT_NO_FATAL_FAILURE(DiscardCpuTranslations());

    iree_async_file_write_operation_t write = {};
    write.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
    write.file = file_;
    write.offset = 29;
    write.buffer = span;
    ASSERT_NO_FATAL_FAILURE(Run(&write.base));
    ASSERT_EQ(write.bytes_written, span.length);
    ASSERT_NO_FATAL_FAILURE(ExpectNoCpuTranslations());
    ASSERT_EQ(pread(file_->primitive.value.fd, actual.data(), span.length, 29),
              span.length);
    EXPECT_EQ(memcmp(actual.data(), expected.data() + span.offset, span.length),
              0);

    // Overwrite only the requested subrange. Prefix/suffix sentinels check that
    // a fixed slot is not confused with the complete registered region.
    memset(actual.data(), 0xA5, span.length);
    ASSERT_EQ(pwrite(file_->primitive.value.fd, actual.data(), span.length, 29),
              span.length);
    memset(expected.data() + span.offset, 0xA5, span.length);
    iree_async_file_read_operation_t read = {};
    read.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
    read.file = file_;
    read.offset = 29;
    read.buffer = span;
    ASSERT_NO_FATAL_FAILURE(Run(&read.base));
    ASSERT_EQ(read.bytes_read, span.length);
    ASSERT_NO_FATAL_FAILURE(ExpectNoCpuTranslations());
    EXPECT_EQ(memcmp(mapping_, expected.data(), expected.size()), 0);
  }
}

TEST_P(FileRegistrationTest, RawAndCrossSlotSpansPreserveByteRanges) {
  // The first span crosses indexed slab slots but fits within a host buffer's
  // fixed slot. Both representations preserve exact bytes, as do raw pointers
  // and empty ranges at the end of the region.
  const iree_async_span_t spans[] = {
      iree_async_span_make(region_, page_size_ + 5, 2 * page_size_),
      iree_async_span_from_ptr(mapping_, mapping_length_),
      iree_async_span_make(region_, mapping_length_, 0),
  };
  std::vector<uint8_t> expected(mapping_length_);
  for (iree_host_size_t i = 0; i < expected.size(); ++i) {
    expected[i] = static_cast<uint8_t>(i * 17 + 11);
  }
  for (iree_async_span_t span : spans) {
    memcpy(mapping_, expected.data(), expected.size());
    iree_async_file_write_operation_t write = {};
    write.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
    write.file = file_;
    write.buffer = span;
    ASSERT_NO_FATAL_FAILURE(Run(&write.base));
    ASSERT_EQ(write.bytes_written, span.length);

    memset(iree_async_span_ptr(span), 0, span.length);
    iree_async_file_read_operation_t read = {};
    read.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_READ;
    read.file = file_;
    read.buffer = span;
    ASSERT_NO_FATAL_FAILURE(Run(&read.base));
    ASSERT_EQ(read.bytes_read, span.length);
    EXPECT_EQ(memcmp(mapping_, expected.data(), expected.size()), 0);
  }
}

TEST_P(FileRegistrationTest, AcceptedFileWriteRetainsRegistration) {
  std::vector<uint8_t> expected(page_size_, 0xB7);
  memcpy(mapping_, expected.data(), expected.size());
  ASSERT_NO_FATAL_FAILURE(DiscardCpuTranslations());
  struct Completion {
    // Published by the terminal callback on this polling thread.
    bool done = false;
    // Terminal result consumed after polling completes.
    iree::Status status;
  } completion;
  iree_async_file_write_operation_t write = {};
  write.base.type = IREE_ASYNC_OPERATION_TYPE_FILE_WRITE;
  write.base.completion_fn =
      +[](void* user_data, iree_async_operation_t*, iree_status_t status,
          iree_async_completion_flags_t) {
        auto& completion = *static_cast<Completion*>(user_data);
        completion.status = std::move(status);
        completion.done = true;
      };
  write.base.user_data = &completion;
  write.file = file_;
  write.buffer = iree_async_span_make(region_, 0, page_size_);
  IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &write.base));

  // Only the accepted operation retains the registration from this point.
  // The backing still belongs to this fixture, independently of its region.
  iree_async_buffer_registration_state_deinitialize(&registration_state_);
  iree_async_region_release(region_);
  region_ = nullptr;
  while (!completion.done) {
    IREE_ASSERT_OK(iree_async_proactor_poll(proactor_, iree_infinite_timeout(),
                                            /*out_completed_count=*/nullptr));
  }
  IREE_ASSERT_OK(completion.status.release());
  EXPECT_EQ(write.bytes_written, expected.size());
  ASSERT_NO_FATAL_FAILURE(ExpectNoCpuTranslations());
  std::vector<uint8_t> actual(expected.size());
  ASSERT_EQ(pread(file_->primitive.value.fd, actual.data(), actual.size(), 0),
            actual.size());
  EXPECT_EQ(actual, expected);

  // Terminal retirement returns the exact fixed-table range for reuse.
  iree_async_buffer_registration_entry_t* entry = nullptr;
  IREE_ASSERT_OK(iree_async_proactor_register_buffer(
      proactor_, &registration_state_,
      iree_make_byte_span(mapping_, mapping_length_),
      IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &entry));
  EXPECT_EQ(entry->region->handles.iouring.base_buffer_index, 0);
}

INSTANTIATE_TEST_SUITE_P(RegistrationKind, FileRegistrationTest,
                         ::testing::Bool(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "HostBuffer" : "IndexedSlab";
                         });

class ReceiveSlabRegistrationTest : public SlabRegistrationTest {
 protected:
  void TearDown() override {
    iree_async_socket_release(socket_);
    for (int fd : socket_fds_) {
      if (fd >= 0) {
        EXPECT_EQ(close(fd), 0);
      }
    }
    iree_async_buffer_pool_release(pool_);
    SlabRegistrationTest::TearDown();
  }

  // Owned descriptors not yet transferred to the proactor.
  int socket_fds_[2] = {-1, -1};
  // Receive socket owned until all one-shot receives complete.
  iree_async_socket_t* socket_ = nullptr;
  // Pool retaining the receive region through every returned lease.
  iree_async_buffer_pool_t* pool_ = nullptr;
};

TEST_F(ReceiveSlabRegistrationTest, OneShotPoolSurvivesDisabledMultishot) {
  if (!iree_any_bit_set(iree_async_proactor_query_capabilities(proactor_),
                        IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT)) {
    GTEST_SKIP() << "kernel predates provided-buffer rings";
  }
  iree_async_proactor_release(proactor_);
  proactor_ = nullptr;
  auto options = iree_async_proactor_options_default();
  options.allowed_capabilities &= ~IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT;
  IREE_ASSERT_OK(iree_async_proactor_create_io_uring(
      options, iree_allocator_system(), &proactor_));
  EXPECT_FALSE(
      iree_any_bit_set(iree_async_proactor_query_capabilities(proactor_),
                       IREE_ASYNC_PROACTOR_CAPABILITY_MULTISHOT));
  ASSERT_NO_FATAL_FAILURE(CreateSlab(page_size_, 2));
  IREE_ASSERT_OK(iree_async_proactor_register_slab(
      proactor_, slab_, IREE_ASYNC_BUFFER_ACCESS_FLAG_WRITE, &region_));
  IREE_ASSERT_OK(
      iree_async_buffer_pool_create(region_, iree_allocator_system(), &pool_));
  ASSERT_EQ(socketpair(AF_UNIX, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0,
                       socket_fds_),
            0);
  IREE_ASSERT_OK(iree_async_socket_import(
      proactor_, iree_async_primitive_from_fd(socket_fds_[0]),
      IREE_ASYNC_SOCKET_TYPE_UNIX_STREAM, IREE_ASYNC_SOCKET_FLAG_NONE,
      &socket_));
  socket_fds_[0] = -1;

  // More receives than pool slots require returned leases to replenish the
  // kernel ring. Multishot stays disabled throughout these real byte transfers.
  for (uint8_t expected = 1; expected <= 9; ++expected) {
    ASSERT_EQ(send(socket_fds_[1], &expected, sizeof(expected), MSG_NOSIGNAL),
              1);
    struct Completion {
      // Set by the poll-owner callback when operation storage is reusable.
      bool done = false;
      // Terminal receive result consumed after the callback.
      iree::Status status;
    } completion;
    iree_async_socket_recv_pool_operation_t operation = {};
    iree_async_operation_initialize(
        &operation.base, IREE_ASYNC_OPERATION_TYPE_SOCKET_RECV_POOL,
        IREE_ASYNC_OPERATION_FLAG_NONE,
        +[](void* user_data, iree_async_operation_t*, iree_status_t status,
            iree_async_completion_flags_t flags) {
          auto& completion = *static_cast<Completion*>(user_data);
          EXPECT_EQ(flags & IREE_ASYNC_COMPLETION_FLAG_MORE, 0u);
          completion.status = std::move(status);
          completion.done = true;
        },
        &completion);
    operation.socket = socket_;
    operation.pool = pool_;
    IREE_ASSERT_OK(iree_async_proactor_submit_one(proactor_, &operation.base));
    while (!completion.done) {
      IREE_EXPECT_OK(iree_async_proactor_poll(
          proactor_, iree_infinite_timeout(), /*out_completed_count=*/nullptr));
    }
    IREE_EXPECT_OK(completion.status.release());
    EXPECT_EQ(operation.bytes_received, sizeof(expected));
    if (operation.bytes_received == sizeof(expected)) {
      EXPECT_EQ(*iree_async_span_ptr(operation.lease.span), expected);
    }
    iree_async_buffer_lease_release(&operation.lease);
  }
}

}  // namespace
