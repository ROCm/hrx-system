// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/linux/io_uring/file_io_resources.h"

#include <fcntl.h>
#include <linux/magic.h>
#include <linux/stat.h>
#include <poll.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statfs.h>
#include <sys/syscall.h>
#include <sys/uio.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <utility>

#include "libamdf/cts/gpu/gpu_device_fixture.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace {

// Every transport uses caller-owned storage without a submission-index array.
constexpr uint32_t kCallerRingSetupFlags =
    IORING_SETUP_NO_MMAP | IORING_SETUP_NO_SQARRAY | IORING_SETUP_R_DISABLED;

}  // namespace

const char* FileIoPathName(FileIoPath path) {
  switch (path) {
    case FileIoPath::kDevice:
      return "device";
    case FileIoPath::kDeviceWait:
      return "device_wait";
    case FileIoPath::kHostRelay:
      return "host_relay";
    case FileIoPath::kHostWait:
      return "host_wait";
    case FileIoPath::kHostPoll:
      return "host_poll";
  }
  std::abort();
}

void GpuFileIoResources::InitializeFileIo(const amdf_api_t* api,
                                          amdf_memory_scope_t* scope,
                                          amdf_device_t* device,
                                          amdf_gpu_device_features_t features) {
  gpu_ = {api, scope, device};
  if (!(features & AMDF_GPU_DEVICE_FEATURE_HOST_REGISTRATION)) {
    GTEST_SKIP() << "selected native lifetime cannot register caller pages";
  }
  const long page_size = sysconf(_SC_PAGESIZE);
  ASSERT_GE(page_size, 4096);
  page_byte_length_ = static_cast<size_t>(page_size);
  struct utsname identity = {};
  ASSERT_EQ(uname(&identity), 0);
  ::testing::Test::RecordProperty("io_kernel_release", identity.release);

  // Probe a fixed, valid caller-owned ring before creating workload resources.
  // An unsupported flag yields EINVAL here; workload setup errors still fail.
  // The disabled probe submits no work and creates no SQPOLL thread.
  const size_t probe_byte_length = 3 * page_byte_length_;
  void* probe_pages = mmap(nullptr, probe_byte_length, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(probe_pages, MAP_FAILED) << std::strerror(errno);
  caller_pages_.push_back({probe_pages, probe_byte_length});
  io_uring_params probe = {};
  probe.flags = kCallerRingSetupFlags;
  probe.sq_off.user_addr = reinterpret_cast<uintptr_t>(probe_pages);
  probe.cq_off.user_addr = probe.sq_off.user_addr + page_byte_length_;
  const int probe_file =
      static_cast<int>(syscall(__NR_io_uring_setup, 8, &probe));
  if (probe_file < 0) {
    const int setup_error = errno;
    if (setup_error == ENOSYS || setup_error == EINVAL) {
      GTEST_SKIP() << "kernel does not support caller-owned io_uring rings: "
                   << std::strerror(setup_error);
    }
    if (setup_error == EPERM || setup_error == EACCES) {
      GTEST_SKIP() << "execution policy denies io_uring setup: "
                   << std::strerror(setup_error);
    }
    FAIL() << "io_uring admission: " << std::strerror(setup_error);
  }
  ASSERT_EQ(close(probe_file), 0) << std::strerror(errno);
}

void GpuFileIoResources::ReleaseFileIo() {
  for (auto& memory : memories_) {
    ASSERT_TRUE(memory.Release(gpu_.api));
  }
  for (auto& ring : rings_) {
    if (ring->file >= 0) {
      ASSERT_EQ(close(std::exchange(ring->file, -1)), 0);
    }
    if (ring->notification >= 0) {
      ASSERT_EQ(close(std::exchange(ring->notification, -1)), 0);
    }
  }
  if (data_file_ >= 0) {
    ASSERT_EQ(close(std::exchange(data_file_, -1)), 0);
  }
  for (auto& pages : caller_pages_) {
    ASSERT_EQ(munmap(pages.pointer, pages.byte_length), 0);
    pages.pointer = nullptr;
  }
}

void GpuFileIoResources::CreateRegisteredPages(size_t byte_length,
                                               uint32_t initial_word,
                                               GpuMemory** out_memory) {
  void* pointer = mmap(nullptr, byte_length, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  ASSERT_NE(pointer, MAP_FAILED) << std::strerror(errno);
  caller_pages_.push_back({pointer, byte_length});
  std::fill_n(static_cast<uint32_t*>(pointer), byte_length / sizeof(uint32_t),
              initial_word);
  const amdf_memory_device_access_t access = {
      gpu_.device,
      {.access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
       .flags =
           AMDF_MEMORY_FLAG_HOST_COHERENT | AMDF_MEMORY_FLAG_DEVICE_ADDRESS},
  };
  amdf_memory_create_info_t creation = {};
  creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
  creation.structure_size = sizeof(creation);
  creation.access_count = 1;
  creation.accesses = &access;
  creation.required_flags = AMDF_MEMORY_FLAG_HOST_VISIBLE;
  creation.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
      gpu_.api, gpu_.scope, gpu_.device,
      AMDF_MEMORY_PROFILE_ROLE_REGISTER | AMDF_MEMORY_PROFILE_ROLE_HOST_MAP,
      creation.required_flags, access.requirements);
  ASSERT_NE(creation.memory_profile_ordinal,
            AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
  creation.byte_length = byte_length;
  creation.minimum_alignment = page_byte_length_;
  creation.registered_host_pointer = pointer;
  creation.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
  auto& memory = memories_.emplace_back();
  ASSERT_NO_FATAL_FAILURE(memory.Initialize(gpu_.api, gpu_.scope, creation));
  *out_memory = &memory;
  ASSERT_EQ((*out_memory)->host.pointer, pointer);
}

void GpuFileIoResources::CreateFile(const std::vector<uint32_t>& words,
                                    FileMode mode) {
  const char* temporary_directory = std::getenv("TEST_TMPDIR");
  const std::string directory =
      temporary_directory ? temporary_directory : "/tmp";
  std::string path = directory + "/amdf-device-io-XXXXXX";
  data_file_ = mkstemp(path.data());
  ASSERT_GE(data_file_, 0) << std::strerror(errno);
  ASSERT_EQ(unlink(path.c_str()), 0) << std::strerror(errno);
  size_t written = 0;
  const auto* bytes = reinterpret_cast<const uint8_t*>(words.data());
  const size_t byte_length = words.size() * sizeof(uint32_t);
  while (written < byte_length) {
    const ssize_t result =
        pwrite(data_file_, bytes + written, byte_length - written, written);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    ASSERT_GT(result, 0) << std::strerror(errno);
    written += static_cast<size_t>(result);
  }
  struct statfs filesystem = {};
  ASSERT_EQ(fstatfs(data_file_, &filesystem), 0);
  ::testing::Test::RecordProperty("io_filesystem_type",
                                  std::to_string(filesystem.f_type));
  ::testing::Test::RecordProperty(
      "io_mode", mode == FileMode::kDirect ? "direct" : "buffered");
  if (mode == FileMode::kDirect) {
    if (filesystem.f_type == TMPFS_MAGIC) {
      GTEST_SKIP() << "direct-storage witness requires a disk-backed file";
    }
    struct statx alignment = {};
    const int alignment_result = static_cast<int>(syscall(
        __NR_statx, data_file_, "", AT_EMPTY_PATH, STATX_DIOALIGN, &alignment));
    if (alignment_result < 0 && (errno == ENOSYS || errno == EOPNOTSUPP)) {
      GTEST_SKIP() << "statx direct-I/O alignment query is unavailable: "
                   << std::strerror(errno);
    }
    ASSERT_EQ(alignment_result, 0) << std::strerror(errno);
    if (!(alignment.stx_mask & STATX_DIOALIGN) ||
        alignment.stx_dio_mem_align == 0 ||
        alignment.stx_dio_offset_align == 0) {
      GTEST_SKIP() << "filesystem does not report a direct-I/O contract";
    }
    ASSERT_EQ(page_byte_length_ % alignment.stx_dio_mem_align, 0u);
    ASSERT_EQ(page_byte_length_ % alignment.stx_dio_offset_align, 0u);
    ::testing::Test::RecordProperty("io_direct_memory_alignment",
                                    alignment.stx_dio_mem_align);
    ::testing::Test::RecordProperty("io_direct_offset_alignment",
                                    alignment.stx_dio_offset_align);
    ASSERT_EQ(fdatasync(data_file_), 0) << std::strerror(errno);
    const int flags = fcntl(data_file_, F_GETFL);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(fcntl(data_file_, F_SETFL, flags | O_DIRECT), 0)
        << std::strerror(errno);
  }
}

void GpuFileIoResources::CreateRing(GpuMemory* payload, FileIoPath path,
                                    uint32_t idle_milliseconds,
                                    uint32_t submission_entries) {
  rings_.push_back(std::make_unique<Ring>());
  ring_ = rings_.back().get();
  ring_->path = path;
  ring_->control_offset =
      (submission_entries * sizeof(io_uring_sqe) + page_byte_length_ - 1) &
      ~(page_byte_length_ - 1);
  // Reserve one page for native control words and page-rounded CQEs. Returned
  // offsets, not a copied kernel-private header, locate shared control words.
  // NO_SQARRAY omits the extra submission-index array.
  const size_t control_length =
      page_byte_length_ +
      ((2 * submission_entries * sizeof(io_uring_cqe) + page_byte_length_ - 1) &
       ~(page_byte_length_ - 1));
  const size_t ring_length = ring_->control_offset + control_length;
  ASSERT_NO_FATAL_FAILURE(
      CreateRegisteredPages(ring_length, 0, &ring_->memory));
  ring_->parameters.flags = kCallerRingSetupFlags;
  if (path == FileIoPath::kDevice || path == FileIoPath::kHostRelay) {
    ring_->parameters.flags |= IORING_SETUP_SQPOLL;
    ring_->parameters.sq_thread_idle = idle_milliseconds;
  } else if (path == FileIoPath::kDeviceWait) {
    ring_->parameters.flags |= IORING_SETUP_SINGLE_ISSUER |
                               IORING_SETUP_DEFER_TASKRUN |
                               IORING_SETUP_TASKRUN_FLAG;
  }
  ring_->parameters.sq_off.user_addr =
      reinterpret_cast<uintptr_t>(ring_->memory->host.pointer);
  ring_->parameters.cq_off.user_addr =
      ring_->parameters.sq_off.user_addr + ring_->control_offset;
  ring_->file = static_cast<int>(
      syscall(__NR_io_uring_setup, submission_entries, &ring_->parameters));
  if (ring_->file < 0) {
    const int setup_error = errno;
    // SQPOLL has a separate native policy check. Admission of the base ring
    // does not establish permission for this workload's submission strategy.
    if (setup_error == EPERM || setup_error == EACCES) {
      GTEST_SKIP() << "execution policy denies io_uring path "
                   << FileIoPathName(path) << " (setup flags "
                   << ring_->parameters.flags
                   << "): " << std::strerror(setup_error);
    }
    FAIL() << "io_uring_setup: " << std::strerror(setup_error);
  }
  ASSERT_EQ(ring_->parameters.sq_entries, submission_entries);
  ASSERT_GE(ring_->parameters.cq_entries, ring_->parameters.sq_entries);
  ASSERT_LE(ring_->parameters.cq_off.cqes +
                ring_->parameters.cq_entries * sizeof(io_uring_cqe),
            control_length);
  ASSERT_LE(ring_->parameters.sq_entries * sizeof(io_uring_sqe),
            ring_->control_offset);

  const struct iovec region = {payload->host.pointer,
                               static_cast<size_t>(payload->host.byte_length)};
  ASSERT_EQ(syscall(__NR_io_uring_register, ring_->file,
                    IORING_REGISTER_BUFFERS, &region, 1),
            0)
      << "register buffers: " << std::strerror(errno);
  ASSERT_EQ(syscall(__NR_io_uring_register, ring_->file, IORING_REGISTER_FILES,
                    &data_file_, 1),
            0)
      << "register file: " << std::strerror(errno);

  if (path == FileIoPath::kDeviceWait) {
    ring_->notification = eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK);
    ASSERT_GE(ring_->notification, 0) << std::strerror(errno);
    ASSERT_EQ(syscall(__NR_io_uring_register, ring_->file,
                      IORING_REGISTER_EVENTFD, &ring_->notification, 1),
              0)
        << "register notification: " << std::strerror(errno);
  }

  const std::array<io_uring_restriction, 5> restrictions = {{
      {.opcode = IORING_RESTRICTION_SQE_OP, .sqe_op = IORING_OP_READ_FIXED},
      {.opcode = IORING_RESTRICTION_SQE_OP, .sqe_op = IORING_OP_WRITE_FIXED},
      {.opcode = IORING_RESTRICTION_SQE_FLAGS_ALLOWED,
       .sqe_flags = IOSQE_FIXED_FILE},
      {.opcode = IORING_RESTRICTION_SQE_FLAGS_REQUIRED,
       .sqe_flags = IOSQE_FIXED_FILE},
      {.opcode = IORING_RESTRICTION_REGISTER_OP,
       .register_op = IORING_REGISTER_ENABLE_RINGS},
  }};
  ASSERT_EQ(
      syscall(__NR_io_uring_register, ring_->file, IORING_REGISTER_RESTRICTIONS,
              restrictions.data(), restrictions.size()),
      0)
      << "restrict ring: " << std::strerror(errno);
  ASSERT_EQ(syscall(__NR_io_uring_register, ring_->file,
                    IORING_REGISTER_ENABLE_RINGS, nullptr, 0),
            0)
      << "enable ring: " << std::strerror(errno);
  ::testing::Test::RecordProperty("io_setup_flags", ring_->parameters.flags);
  ::testing::Test::RecordProperty("io_setup_features",
                                  ring_->parameters.features);
  ::testing::Test::RecordProperty("io_submission_entries",
                                  ring_->parameters.sq_entries);
  ::testing::Test::RecordProperty("io_completion_entries",
                                  ring_->parameters.cq_entries);
  device_ring_memory_ = ring_->memory;
  if (path != FileIoPath::kDevice && path != FileIoPath::kDeviceWait) {
    ASSERT_NO_FATAL_FAILURE(
        CreateRegisteredPages(ring_length, 0, &ring_->relay_memory));
    device_ring_memory_ = ring_->relay_memory;
  }
  ::testing::Test::RecordProperty("io_path", FileIoPathName(path));
}

uintptr_t GpuFileIoResources::RingWord(uint32_t offset) const {
  return reinterpret_cast<uintptr_t>(ring_->memory->host.pointer) +
         ring_->control_offset + offset;
}

uint64_t GpuFileIoResources::RingAddress(uint32_t offset) const {
  return device_ring_memory_->device_address + ring_->control_offset + offset;
}

void GpuFileIoResources::RelayFileIo() {
  if (device_ring_memory_ == ring_->memory) {
    return;
  }
  const uintptr_t device_base =
      reinterpret_cast<uintptr_t>(device_ring_memory_->host.pointer);
  const uintptr_t device_control = device_base + ring_->control_offset;
  auto* native_submissions =
      static_cast<io_uring_sqe*>(ring_->memory->host.pointer);
  const auto* device_submissions =
      reinterpret_cast<const io_uring_sqe*>(device_base);
  const auto* native_completions = reinterpret_cast<const io_uring_cqe*>(
      RingWord(ring_->parameters.cq_off.cqes));
  auto* device_completions = reinterpret_cast<io_uring_cqe*>(
      device_control + ring_->parameters.cq_off.cqes);

  // Completion publication also carries the kernel's payload writes to the
  // GPU. Returning native CQ space does not release an application payload.
  // These finite owners bound outstanding work below both ring capacities;
  // no producer can lap its consumer even if the host drains CQs first.
  uint32_t consumed =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.head));
  const uint32_t completed =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail));
  if (consumed != completed) {
    for (; consumed != completed; ++consumed) {
      const uint32_t index = consumed & (ring_->parameters.cq_entries - 1);
      device_completions[index] = native_completions[index];
    }
    GpuStoreRelease<uint32_t>(RingWord(ring_->parameters.cq_off.head),
                              completed);
    GpuStoreRelease<uint32_t>(device_control + ring_->parameters.cq_off.tail,
                              completed);
  }

  uint32_t submitted =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
  const uint32_t available =
      GpuLoadAcquire<uint32_t>(device_control + ring_->parameters.sq_off.tail);
  if (submitted != available) {
    for (; submitted != available; ++submitted) {
      const uint32_t index = submitted & (ring_->parameters.sq_entries - 1);
      native_submissions[index] = device_submissions[index];
    }
    GpuStoreRelease<uint32_t>(RingWord(ring_->parameters.sq_off.tail),
                              available);
    GpuStoreRelease<uint32_t>(device_control + ring_->parameters.sq_off.head,
                              available);
  }
}

void GpuFileIoResources::ServiceHostIo(uint64_t* enter_calls) {
  const uint32_t tail =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
  const uint32_t head =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head));
  const uint32_t consumed =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.head));
  // Each accepted request has exactly one CQE. The host exclusively advances
  // this native CQ head; GPU consumption happens in the separate relay ring.
  // With no published work, waiting would prevent servicing GPU admission.
  if (tail == consumed) {
    return;
  }
  ++*enter_calls;
  const uint32_t minimum = ring_->path == FileIoPath::kHostWait ? 1 : 0;
  const long result = syscall(__NR_io_uring_enter, ring_->file, tail - head,
                              minimum, IORING_ENTER_GETEVENTS, nullptr, 0);
  if (result < 0 && errno == EINTR) {
    return;
  }
  ASSERT_GE(result, 0) << "host submission: " << std::strerror(errno);
  // Publish the completion before applying the next host service delay.
  RelayFileIo();
}

void GpuFileIoResources::ServiceDeviceIo(uint64_t* enter_calls,
                                         uint64_t* wait_calls) {
  const uint32_t initial_completed =
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail));
  if (GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail)) ==
      initial_completed) {
    return;
  }
  for (;;) {
    // Drain notification before servicing work. Deferred task work signals
    // the eventfd even before a CQE exists; bounded task-work service can also
    // leave TASKRUN set without producing another notification edge.
    uint64_t notifications = 0;
    const ssize_t drained =
        read(ring_->notification, &notifications, sizeof(notifications));
    if (drained < 0 && errno == EINTR) {
      return;
    }
    ASSERT_TRUE(drained == sizeof(notifications) ||
                (drained < 0 && errno == EAGAIN))
        << "read notification: " << std::strerror(errno);
    const uint32_t tail =
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail));
    const uint32_t head =
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head));
    ++*enter_calls;
    const long result = syscall(__NR_io_uring_enter, ring_->file, tail - head,
                                0, IORING_ENTER_GETEVENTS, nullptr, 0);
    if (result < 0 && errno == EINTR) {
      return;
    }
    ASSERT_GE(result, 0) << "device submission: " << std::strerror(errno);
    const uint32_t completed =
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail));
    if (completed != initial_completed) {
      return;
    }
    if (GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
        IORING_SQ_TASKRUN) {
      continue;
    }
    // One CQE per accepted request makes this kernel-owned frontier immune
    // to concurrent GPU CQ consumption. No accepted I/O means no guaranteed
    // future eventfd wake; new GPU SQ publication is observed by the caller.
    if (GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head)) ==
        completed) {
      return;
    }
    pollfd notification = {ring_->notification, POLLIN, 0};
    ++*wait_calls;
    const int ready = poll(&notification, 1, -1);
    if (ready < 0 && errno == EINTR) {
      return;
    }
    ASSERT_EQ(ready, 1) << "wait notification: " << std::strerror(errno);
    ASSERT_EQ(notification.revents, POLLIN);
  }
}

void GpuFileIoResources::VerifyFile(const std::vector<uint32_t>& expected_file,
                                    FileMode mode) {
  struct stat file_info = {};
  ASSERT_EQ(fstat(data_file_, &file_info), 0) << std::strerror(errno);
  ASSERT_EQ(file_info.st_size,
            static_cast<off_t>(expected_file.size() * sizeof(uint32_t)));
  // Verification happens only after the GPU consumed and reloaded the
  // payload. Clearing O_DIRECT here cannot change the qualified device path.
  if (mode == FileMode::kDirect) {
    const int flags = fcntl(data_file_, F_GETFL);
    ASSERT_GE(flags, 0);
    ASSERT_EQ(fcntl(data_file_, F_SETFL, flags & ~O_DIRECT), 0);
  }
  std::vector<uint32_t> actual_file(expected_file.size());
  auto* bytes = reinterpret_cast<uint8_t*>(actual_file.data());
  const size_t byte_length = actual_file.size() * sizeof(uint32_t);
  size_t read = 0;
  while (read < byte_length) {
    const ssize_t result =
        pread(data_file_, bytes + read, byte_length - read, read);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    ASSERT_GT(result, 0) << std::strerror(errno);
    read += static_cast<size_t>(result);
  }
  EXPECT_EQ(actual_file, expected_file);
}
