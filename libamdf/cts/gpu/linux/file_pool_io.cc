// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/linux/file_pool_io.h"

#include <fcntl.h>
#include <linux/fs.h>
#include <linux/magic.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "gtest/gtest.h"

namespace {

uint32_t Load(const uint8_t* memory, uint32_t offset) {
  return __atomic_load_n(reinterpret_cast<const uint32_t*>(memory + offset),
                         __ATOMIC_ACQUIRE);
}

void Store(uint8_t* memory, uint32_t offset, uint32_t value) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(memory + offset), value,
                   __ATOMIC_RELEASE);
}

void TransferFile(int file, bool write, void* pointer, size_t byte_length) {
  auto* bytes = static_cast<uint8_t*>(pointer);
  size_t offset = 0;
  while (offset < byte_length) {
    const ssize_t result =
        write ? pwrite(file, bytes + offset, byte_length - offset, offset)
              : pread(file, bytes + offset, byte_length - offset, offset);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    ASSERT_GT(result, 0) << std::strerror(errno);
    offset += static_cast<size_t>(result);
  }
}

}  // namespace

void FilePoolIo::CreateFile(const std::vector<uint32_t>& seed, bool direct,
                            size_t buffer_alignment) {
  const char* temporary_directory = std::getenv("TEST_TMPDIR");
  std::string path =
      std::string(temporary_directory ? temporary_directory : "/tmp") +
      "/amdf-file-pool-XXXXXX";
  file_ = mkstemp(path.data());
  ASSERT_GE(file_, 0) << std::strerror(errno);
  ASSERT_EQ(unlink(path.c_str()), 0);
  struct statfs filesystem = {};
  ASSERT_EQ(fstatfs(file_, &filesystem), 0);
  ::testing::Test::RecordProperty("io_filesystem_type",
                                  std::to_string(filesystem.f_type));
  ::testing::Test::RecordProperty("io_mode", direct ? "direct" : "buffered");
  if (direct && filesystem.f_type == TMPFS_MAGIC) {
    GTEST_SKIP() << "direct-storage witness requires a disk-backed file";
  }
  if (direct && filesystem.f_type == BTRFS_SUPER_MAGIC) {
    // Before allocating extents, NOCOW also establishes NODATASUM. Checksummed
    // writes may otherwise bounce through a buffered path on modern Btrfs.
    unsigned long flags = 0;
    ASSERT_EQ(ioctl(file_, FS_IOC_GETFLAGS, &flags), 0);
    flags |= FS_NOCOW_FL;
    ASSERT_EQ(ioctl(file_, FS_IOC_SETFLAGS, &flags), 0);
  }
  ASSERT_NO_FATAL_FAILURE(TransferFile(file_, true,
                                       const_cast<uint32_t*>(seed.data()),
                                       seed.size() * sizeof(uint32_t)));
  if (!direct) {
    return;
  }
  struct statx geometry = {};
  ASSERT_EQ(
      syscall(__NR_statx, file_, "", AT_EMPTY_PATH, STATX_DIOALIGN, &geometry),
      0)
      << std::strerror(errno);
  uint32_t memory_alignment = geometry.stx_dio_mem_align;
  uint32_t offset_alignment = geometry.stx_dio_offset_align;
  if ((geometry.stx_mask & STATX_DIOALIGN) == 0 &&
      filesystem.f_type == BTRFS_SUPER_MAGIC) {
    // Btrfs exports its sector size as f_bsize. Older kernels omit DIOALIGN.
    memory_alignment = offset_alignment = filesystem.f_bsize;
  }
  if (memory_alignment == 0 || offset_alignment == 0) {
    GTEST_SKIP() << "filesystem does not report its direct-I/O alignment";
  }
  ASSERT_EQ(buffer_alignment % memory_alignment, 0u);
  ASSERT_EQ(buffer_alignment % offset_alignment, 0u);
  ::testing::Test::RecordProperty("io_direct_memory_alignment",
                                  memory_alignment);
  ::testing::Test::RecordProperty("io_direct_offset_alignment",
                                  offset_alignment);
  ASSERT_EQ(fdatasync(file_), 0) << std::strerror(errno);
  const int flags = fcntl(file_, F_GETFL);
  ASSERT_GE(flags, 0);
  ASSERT_EQ(fcntl(file_, F_SETFL, flags | O_DIRECT), 0);
}

void FilePoolIo::Register(void* pointer, size_t byte_length) {
  ring_ = syscall(__NR_io_uring_setup, 8, &parameters_);
  if (ring_ < 0 && (errno == EPERM || errno == ENOSYS)) {
    GTEST_SKIP() << "native io_uring policy: " << std::strerror(errno);
  }
  ASSERT_GE(ring_, 0) << std::strerror(errno);
  submission_length_ =
      parameters_.sq_off.array + parameters_.sq_entries * sizeof(uint32_t);
  completion_length_ =
      parameters_.cq_off.cqes + parameters_.cq_entries * sizeof(io_uring_cqe);
  const bool shared = (parameters_.features & IORING_FEAT_SINGLE_MMAP) != 0;
  if (shared) {
    submission_length_ = completion_length_ =
        std::max(submission_length_, completion_length_);
  }
  void* mapping = mmap(nullptr, submission_length_, PROT_READ | PROT_WRITE,
                       MAP_SHARED, ring_, IORING_OFF_SQ_RING);
  ASSERT_NE(mapping, MAP_FAILED) << std::strerror(errno);
  submission_ = static_cast<uint8_t*>(mapping);
  if (shared) {
    completion_ = submission_;
  } else {
    mapping = mmap(nullptr, completion_length_, PROT_READ | PROT_WRITE,
                   MAP_SHARED, ring_, IORING_OFF_CQ_RING);
    ASSERT_NE(mapping, MAP_FAILED) << std::strerror(errno);
    completion_ = static_cast<uint8_t*>(mapping);
  }
  mapping = mmap(nullptr, parameters_.sq_entries * sizeof(io_uring_sqe),
                 PROT_READ | PROT_WRITE, MAP_SHARED, ring_, IORING_OFF_SQES);
  ASSERT_NE(mapping, MAP_FAILED) << std::strerror(errno);
  entries_ = static_cast<io_uring_sqe*>(mapping);
  const iovec region = {pointer, byte_length};
  ASSERT_EQ(syscall(__NR_io_uring_register, ring_, IORING_REGISTER_BUFFERS,
                    &region, 1),
            0)
      << std::strerror(errno);
  registered_buffer_ = true;
  ASSERT_EQ(
      syscall(__NR_io_uring_register, ring_, IORING_REGISTER_FILES, &file_, 1),
      0)
      << std::strerror(errno);
  registered_file_ = true;
}

void FilePoolIo::Submit(const io_uring_sqe& request) {
  const uint32_t tail = Load(submission_, parameters_.sq_off.tail);
  ASSERT_LT(tail - Load(submission_, parameters_.sq_off.head),
            parameters_.sq_entries);
  const uint32_t index = tail & Load(submission_, parameters_.sq_off.ring_mask);
  entries_[index] = request;
  reinterpret_cast<uint32_t*>(submission_ + parameters_.sq_off.array)[index] =
      index;
  Store(submission_, parameters_.sq_off.tail, tail + 1);
  // The published SQE is now owned even if the entry syscall is interrupted.
  ++pending_;
  int result;
  do {
    result = syscall(__NR_io_uring_enter, ring_, 1, 0, 0, nullptr, 0);
  } while (result < 0 && errno == EINTR);
  ASSERT_EQ(result, 1) << std::strerror(errno);
}

void FilePoolIo::Begin(bool write, void* pointer, uint32_t byte_length,
                       uint64_t file_offset, uint64_t identity) {
  io_uring_sqe request = {};
  request.opcode = write ? IORING_OP_WRITE_FIXED : IORING_OP_READ_FIXED;
  request.flags = IOSQE_FIXED_FILE;
  request.fd = 0;
  request.off = file_offset;
  request.addr = reinterpret_cast<uintptr_t>(pointer);
  request.len = byte_length;
  request.buf_index = 0;
  request.user_data = identity;
  ASSERT_NO_FATAL_FAILURE(Submit(request));
}

void FilePoolIo::Next(io_uring_cqe* out_completion) {
  ASSERT_GT(pending_, 0u);
  const uint32_t head = Load(completion_, parameters_.cq_off.head);
  while (head == Load(completion_, parameters_.cq_off.tail)) {
    // Sleep on explicit native completion; the outer CTS harness owns timeout.
    // Submitting the current SQ tail also retires an interrupted submission.
    const uint32_t queued = Load(submission_, parameters_.sq_off.tail) -
                            Load(submission_, parameters_.sq_off.head);
    const int result = syscall(__NR_io_uring_enter, ring_, queued, 1,
                               IORING_ENTER_GETEVENTS, nullptr, 0);
    ASSERT_TRUE(result >= 0 || errno == EINTR) << std::strerror(errno);
  }
  const auto* records = reinterpret_cast<const io_uring_cqe*>(
      completion_ + parameters_.cq_off.cqes);
  *out_completion =
      records[head & Load(completion_, parameters_.cq_off.ring_mask)];
  Store(completion_, parameters_.cq_off.head, head + 1);
  --pending_;
}

void FilePoolIo::Complete(uint64_t identity, uint32_t byte_length) {
  io_uring_cqe completion = {};
  ASSERT_NO_FATAL_FAILURE(Next(&completion));
  ASSERT_EQ(completion.user_data, identity);
  ASSERT_EQ(completion.res, static_cast<int32_t>(byte_length))
      << (completion.res < 0 ? std::strerror(-completion.res) : "short I/O");
}

void FilePoolIo::CancelAndRetire(void* pointer, uint32_t byte_length,
                                 uint64_t file_offset) {
  ASSERT_NO_FATAL_FAILURE(
      Begin(false, pointer, byte_length, file_offset, 1001));
  const io_uring_sqe cancel = {
      .opcode = IORING_OP_ASYNC_CANCEL,
      .addr = 1001,
      .user_data = 1002,
  };
  ASSERT_NO_FATAL_FAILURE(Submit(cancel));
  std::array<io_uring_cqe, 2> completions;
  ASSERT_NO_FATAL_FAILURE(Next(&completions[0]));
  ASSERT_NO_FATAL_FAILURE(Next(&completions[1]));
  if (completions[0].user_data == 1002) {
    std::swap(completions[0], completions[1]);
  }
  ASSERT_EQ(completions[0].user_data, 1001u);
  ASSERT_EQ(completions[1].user_data, 1002u);
  EXPECT_TRUE(completions[0].res == static_cast<int32_t>(byte_length) ||
              completions[0].res == -ECANCELED);
  EXPECT_TRUE(completions[1].res == 0 || completions[1].res == -ENOENT ||
              completions[1].res == -EALREADY);
  ::testing::Test::RecordProperty("cancelled_io_result", completions[0].res);
  ::testing::Test::RecordProperty("cancellation_result", completions[1].res);
}

void FilePoolIo::VerifyFile(const std::vector<uint32_t>& expected) {
  ASSERT_EQ(pending_, 0u);
  const int flags = fcntl(file_, F_GETFL);
  ASSERT_GE(flags, 0);
  ASSERT_EQ(fcntl(file_, F_SETFL, flags & ~O_DIRECT), 0);
  std::vector<uint32_t> observed(expected.size());
  ASSERT_NO_FATAL_FAILURE(TransferFile(file_, false, observed.data(),
                                       observed.size() * sizeof(uint32_t)));
  EXPECT_EQ(observed, expected);
}

void FilePoolIo::Release() {
  while (pending_ != 0) {
    io_uring_cqe completion = {};
    ASSERT_NO_FATAL_FAILURE(Next(&completion));
  }
  if (submission_ != nullptr) {
    EXPECT_EQ(Load(submission_, parameters_.sq_off.dropped), 0u);
  }
  if (completion_ != nullptr) {
    EXPECT_EQ(Load(completion_, parameters_.cq_off.overflow), 0u);
  }
  if (registered_buffer_) {
    ASSERT_EQ(syscall(__NR_io_uring_register, ring_, IORING_UNREGISTER_BUFFERS,
                      nullptr, 0),
              0)
        << std::strerror(errno);
    registered_buffer_ = false;
  }
  if (registered_file_) {
    ASSERT_EQ(syscall(__NR_io_uring_register, ring_, IORING_UNREGISTER_FILES,
                      nullptr, 0),
              0)
        << std::strerror(errno);
    registered_file_ = false;
  }
  if (entries_ != nullptr) {
    ASSERT_EQ(munmap(entries_, parameters_.sq_entries * sizeof(io_uring_sqe)),
              0);
    entries_ = nullptr;
  }
  if (completion_ != nullptr && completion_ != submission_) {
    ASSERT_EQ(munmap(completion_, completion_length_), 0);
  }
  completion_ = nullptr;
  if (submission_ != nullptr) {
    ASSERT_EQ(munmap(submission_, submission_length_), 0);
    submission_ = nullptr;
  }
  if (ring_ >= 0) {
    ASSERT_EQ(close(std::exchange(ring_, -1)), 0);
  }
  if (file_ >= 0) {
    ASSERT_EQ(close(std::exchange(file_, -1)), 0);
  }
}
