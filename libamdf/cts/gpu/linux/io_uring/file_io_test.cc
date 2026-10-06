// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <tuple>
#include <vector>

#include "libamdf/cts/gpu/kernels/file_exchange.h"
#if defined(AMDF_FILE_IO_CXX_KERNELS)
#include "libamdf/cts/gpu/kernels/file_exchange_cxx_kernels.h"
#else
#include "libamdf/cts/gpu/kernels/file_exchange_kernels.h"
#endif
#include "libamdf/cts/gpu/linux/io_uring/file_io_fixture.h"

namespace {

namespace protocol = kernels::file_exchange;
#if defined(AMDF_FILE_IO_CXX_KERNELS)
namespace products = kernels::file_exchange_cxx;
#else
namespace products = kernels::file_exchange;
#endif

// The shader writes the native UAPI, not a host-translated command record.
static_assert(sizeof(io_uring_sqe) == 64);
static_assert(offsetof(io_uring_sqe, opcode) == 0);
static_assert(offsetof(io_uring_sqe, flags) == 1);
static_assert(offsetof(io_uring_sqe, fd) == 4);
static_assert(offsetof(io_uring_sqe, off) == 8);
static_assert(offsetof(io_uring_sqe, addr) == 16);
static_assert(offsetof(io_uring_sqe, len) == 24);
static_assert(offsetof(io_uring_sqe, user_data) == 32);
static_assert(offsetof(io_uring_sqe, buf_index) == 40);
static_assert(sizeof(io_uring_cqe) == 16);
static_assert(offsetof(io_uring_cqe, res) == 8);
static_assert(IOSQE_FIXED_FILE == 1);
static_assert(IORING_OP_READ_FIXED == protocol::kReadFixed);
static_assert(IORING_OP_WRITE_FIXED == protocol::kWriteFixed);
static_assert(-ENODATA == protocol::kIncompleteRead);

constexpr uint32_t kFileBlockCount = 16;
constexpr uint32_t kGuard = 0x9d372be5u;
constexpr uint32_t kRecordGuardWords = 16;

enum class Workload { kRoundTrip, kShortInput, kInvalidFile };

uint32_t InputWord(uint32_t block, uint32_t word) {
  return 0x01020304u + block * 0x01010101u + word * 0x00010003u;
}

class GpuFileIoTest : public GpuFileIoFixture {
 protected:
  void Run(FileMode mode, Workload workload,
           FileIoPath path = FileIoPath::kDevice,
           uint32_t submission_entries = 8) {
    const auto* product = products::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(product, nullptr) << "missing compiled file-exchange kernel";
    const auto& kernel = *product;
    ASSERT_EQ(kernel.private_segment_byte_length, 0u);
    const uint32_t word_count = page_byte_length_ / sizeof(uint32_t);
    ASSERT_LE(word_count, 16384u);
    const uint32_t round_count = workload == Workload::kRoundTrip ? 33 : 1;
    const uint32_t seed = workload == Workload::kShortInput ? 0 : 0x80000001u;
    const uint32_t stride = 2 * page_byte_length_;
    const uint32_t file_index = workload == Workload::kInvalidFile ? 1 : 0;
    std::vector<uint32_t> expected_file(2 * kFileBlockCount * word_count,
                                        kGuard);
    for (uint32_t block = 0; block < kFileBlockCount; ++block) {
      for (uint32_t word = 0; word < word_count; ++word) {
        expected_file[block * word_count + word] = InputWord(block, word);
      }
    }
    if (workload == Workload::kShortInput) {
      expected_file.resize(word_count / 2);
    }
    ASSERT_NO_FATAL_FAILURE(CreateFile(expected_file, mode));
    if (IsSkipped()) {
      return;
    }

    GpuMemory* payload = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        CreateRegisteredPages(7 * page_byte_length_, kGuard, &payload));
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, path, 1, submission_entries));
    const size_t record_word_count =
        2 * kRecordGuardWords + protocol::kSummaryWordCount +
        round_count * (protocol::kRecordHeaderWordCount + word_count);
    std::vector<uint32_t> expected_records(record_word_count, kGuard);
    GpuMemory* records = nullptr;
    GpuMemory* arguments = nullptr;
    GpuMemory* completion = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     record_word_count * sizeof(uint32_t), &records));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
    std::memcpy(records->host.pointer, expected_records.data(),
                records->host.byte_length);
    std::memset(completion->host.pointer, 0, completion->host.byte_length);

    const protocol::Arguments device_arguments = {
        .submission_entries = device_ring_memory_->device_address,
        .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
        .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
        .completion_head = RingAddress(ring_->parameters.cq_off.head),
        .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
        .payload = payload->device_address + page_byte_length_,
        .records =
            records->device_address + kRecordGuardWords * sizeof(uint32_t),
        .host_payload = reinterpret_cast<uintptr_t>(payload->host.pointer) +
                        page_byte_length_,
        .submission_mask = ring_->parameters.sq_entries - 1,
        .completion_mask = ring_->parameters.cq_entries - 1,
        .round_count = round_count,
        .word_count = word_count,
        .file_block_mask = kFileBlockCount - 1,
        .seed = seed,
        .payload_stride = stride,
        .file_index = file_index,
    };
    ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
    ASSERT_LE(kernel.arguments.byte_length, sizeof(device_arguments));
    std::memset(arguments->host.pointer, 0, arguments->host.byte_length);
    std::memcpy(arguments->host.pointer, &device_arguments,
                kernel.arguments.byte_length);
    ASSERT_NO_FATAL_FAILURE(
        Execute(kernel, arguments, completion, "file_exchange"));

    uint32_t cause = seed;
    uint32_t request_count = 0;
    int32_t terminal = 0;
    uint32_t finished = 0;
    std::vector<uint32_t> expected_payload(7 * word_count, kGuard);
    if (workload == Workload::kRoundTrip) {
      for (uint32_t round = 0; round < round_count; ++round) {
        const uint32_t block = cause % kFileBlockCount;
        const uint32_t write_block =
            kFileBlockCount + (block * 5 + 1) % kFileBlockCount;
        const size_t record_offset =
            kRecordGuardWords + protocol::kSummaryWordCount +
            round * (protocol::kRecordHeaderWordCount + word_count);
        expected_records[record_offset] = block;
        expected_records[record_offset + 1] = cause;
        expected_records[record_offset + 2] = write_block;
        for (uint32_t word = 0; word < word_count; ++word) {
          const uint32_t input = InputWord(block, word);
          const uint32_t output =
              static_cast<uint32_t>(uint64_t{input} * 3 + cause + round);
          expected_payload[word_count + word] = input;
          expected_payload[3 * word_count + word] = output;
          expected_payload[5 * word_count + word] = output;
          expected_file[write_block * word_count + word] = output;
          expected_records[record_offset + protocol::kRecordHeaderWordCount +
                           word] = output;
        }
        cause = expected_payload[5 * word_count];
      }
      request_count = round_count * 3;
      finished = round_count;
    } else if (workload == Workload::kShortInput) {
      std::copy(expected_file.begin(), expected_file.end(),
                expected_payload.begin() + word_count);
      request_count = 2;
      terminal = protocol::kIncompleteRead;
    } else {
      request_count = 1;
      terminal = -EBADF;
    }
    expected_records[kRecordGuardWords] = finished;
    expected_records[kRecordGuardWords + 1] = static_cast<uint32_t>(terminal);
    expected_records[kRecordGuardWords + 2] = request_count;
    expected_records[kRecordGuardWords + 3] = cause;
    const auto* actual_records =
        static_cast<const uint32_t*>(records->host.pointer);
    const auto* actual_payload =
        static_cast<const uint32_t*>(payload->host.pointer);
    for (size_t word = 0; word < expected_records.size(); ++word) {
      EXPECT_EQ(actual_records[word], expected_records[word])
          << "record word=" << word;
    }
    for (size_t word = 0; word < expected_payload.size(); ++word) {
      EXPECT_EQ(actual_payload[word], expected_payload[word])
          << "payload or guard word=" << word;
    }
    EXPECT_EQ(std::memcmp(arguments->host.pointer, &device_arguments,
                          kernel.arguments.byte_length),
              0);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head)),
              request_count);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail)),
              request_count);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.head)),
              request_count);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail)),
              request_count);
    EXPECT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)),
        0u);
    EXPECT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
        0u);
    const auto* completions = reinterpret_cast<const io_uring_cqe*>(
        RingWord(ring_->parameters.cq_off.cqes));
    for (uint32_t recent = 0;
         recent < std::min(request_count, ring_->parameters.cq_entries);
         ++recent) {
      const uint32_t ticket = request_count - recent - 1;
      const auto& entry =
          completions[ticket & (ring_->parameters.cq_entries - 1)];
      EXPECT_EQ(entry.user_data, ticket);
      EXPECT_EQ(entry.flags, 0u);
    }
    RecordProperty("io_completed_requests", request_count);
    RecordProperty("io_completed_rounds", finished);
    RecordProperty("io_terminal_result", terminal);

    ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, mode));
  }
};

TEST_F(GpuFileIoTest, BufferedCausalReadWriteReload) {
  Run(FileMode::kBuffered, Workload::kRoundTrip);
}

TEST_F(GpuFileIoTest, DirectCausalReadWriteReload) {
  Run(FileMode::kDirect, Workload::kRoundTrip);
}

TEST_F(GpuFileIoTest,
       PartialReadThenEofRetiresWithoutConsumingIncompletePayload) {
  Run(FileMode::kBuffered, Workload::kShortInput);
}

TEST_F(GpuFileIoTest, InvalidFixedFileRetiresWithTheNativeError) {
  Run(FileMode::kBuffered, Workload::kInvalidFile);
}

class GpuFileProgressIoTest : public GpuFileIoTest,
                              public ::testing::WithParamInterface<FileIoPath> {
};

TEST_P(GpuFileProgressIoTest, BufferedCausalReadWriteReload) {
  Run(FileMode::kBuffered, Workload::kRoundTrip, GetParam());
}

TEST_P(GpuFileProgressIoTest, DirectCausalReadWriteReload) {
  Run(FileMode::kDirect, Workload::kRoundTrip, GetParam());
}

TEST_P(GpuFileProgressIoTest, PartialReadThenEof) {
  Run(FileMode::kBuffered, Workload::kShortInput, GetParam());
}

TEST_P(GpuFileProgressIoTest, InvalidFixedFile) {
  Run(FileMode::kBuffered, Workload::kInvalidFile, GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ProgressPaths, GpuFileProgressIoTest,
    ::testing::Values(FileIoPath::kDeviceWait, FileIoPath::kHostRelay,
                      FileIoPath::kHostWait, FileIoPath::kHostPoll),
    [](const auto& info) { return FileIoPathName(info.param); });

class GpuFileRingGeometryTest
    : public GpuFileIoTest,
      public ::testing::WithParamInterface<std::tuple<FileIoPath, uint32_t>> {};

TEST_P(GpuFileRingGeometryTest, DirectRoundTrip) {
  Run(FileMode::kDirect, Workload::kRoundTrip, std::get<0>(GetParam()),
      std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    NativeLayouts, GpuFileRingGeometryTest,
    ::testing::Combine(::testing::Values(FileIoPath::kDevice,
                                         FileIoPath::kDeviceWait,
                                         FileIoPath::kHostRelay,
                                         FileIoPath::kHostWait,
                                         FileIoPath::kHostPoll),
                       ::testing::Values(8u, 64u, 256u, 512u)));

}  // namespace
