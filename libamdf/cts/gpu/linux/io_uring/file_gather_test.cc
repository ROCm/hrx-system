// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/file_gather.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <tuple>
#include <vector>

#if defined(AMDF_FILE_IO_CXX_KERNELS)
#include "libamdf/cts/gpu/kernels/file_gather_cxx_kernels.h"
#else
#include "libamdf/cts/gpu/kernels/file_gather_kernels.h"
#endif
#include "libamdf/cts/gpu/linux/io_uring/file_io_fixture.h"

namespace {

namespace protocol = kernels::file_gather;
#if defined(AMDF_FILE_IO_CXX_KERNELS)
namespace products = kernels::file_gather_cxx;
#else
namespace products = kernels::file_gather;
#endif

// The fixture writes native fields directly, including both user_data words.
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
static_assert(offsetof(io_uring_cqe, user_data) == 0);
static_assert(offsetof(io_uring_cqe, res) == 8);
static_assert(offsetof(io_uring_cqe, flags) == 12);
static_assert(IORING_OP_READ_FIXED == 4);
static_assert(IORING_OP_WRITE_FIXED == 5);
static_assert(IOSQE_FIXED_FILE == 1);

constexpr uint32_t kGuard = 0x9d372be5u;
constexpr uint32_t kGuardWords = 16;
constexpr uint32_t kPeerRounds = 33;
constexpr uint32_t kConsumerCount = 3 + 2 * kPeerRounds;
constexpr uint32_t kSeed = 0x80000001u;
constexpr uint32_t kAbsent = std::numeric_limits<uint32_t>::max();

enum class Fault : uint32_t {
  kNone = 0,
  kInvalidFile = 1,
  kShortInput = 2,
  kInvalidWrite = 3,
  kInvalidReload = 4,
};

uint32_t InputWord(uint32_t block, uint32_t word) {
  return 0x01020304u + block * 0x01010101u + word * 0x00010003u;
}

uint32_t RecordNumber(uint32_t slot, uint32_t consumer, uint32_t held_slot) {
  if (slot == held_slot) {
    return consumer;
  }
  return 3 + (slot < held_slot ? slot : slot - 1) * kPeerRounds + consumer;
}

uint32_t WriteWindow(uint32_t slot) { return 5 - slot; }
uint32_t ReloadWindow(uint32_t slot) { return 6 + (slot + 1) % 3; }

struct Consumer {
  // Independent oracle cause, never derived from GPU output.
  uint32_t cause;
  // Immutable input key shared by the two held-source readers.
  uint32_t key;
  // Source generation; held consumers zero and one share generation zero.
  uint32_t generation;
  // Disjoint scattered file destination for this consumer.
  uint32_t output_block;
  // Complete expected arithmetic result, including every payload word.
  std::vector<uint32_t> words;
};

std::vector<Consumer> MakeConsumers(uint32_t word_count, uint32_t held_slot) {
  std::vector<Consumer> consumers(kConsumerCount);
  for (uint32_t slot = 0; slot < protocol::kSlotCount; ++slot) {
    uint32_t cause = kSeed + slot * 17;
    const uint32_t count = slot == held_slot ? 3 : kPeerRounds;
    for (uint32_t local = 0; local < count; ++local) {
      const uint32_t record = RecordNumber(slot, local, held_slot);
      auto& consumer = consumers[record];
      consumer.cause = cause;
      consumer.key = (cause % 4) * 4 + slot;
      consumer.generation = slot == held_slot ? (local == 2 ? 1 : 0) : local;
      consumer.output_block = protocol::kInputBlockCount +
                              (record * 5 + 1) % protocol::kOutputBlockCount;
      consumer.words.resize(word_count);
      for (uint32_t word = 0; word < word_count; ++word) {
        consumer.words[word] = static_cast<uint32_t>(
            uint64_t{InputWord(consumer.key, word)} * 3 + cause + record * 257);
      }
      if (slot != held_slot || local != 0) {
        cause = consumer.words[0];
      }
    }
  }
  return consumers;
}

void CheckWords(std::span<const uint32_t> actual,
                std::span<const uint32_t> expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t word = 0; word < expected.size(); ++word) {
    ASSERT_EQ(actual[word], expected[word]) << "word=" << word;
  }
}

// A per-source semantic oracle follows read/write/reload completion, not the
// shader's scheduling loop or native CQ order. It also accounts for partial
// operations that finish after useful processing has stopped.
struct SourceOracle {
  // Next operation allowed to borrow this source credit.
  uint32_t phase = 0;
  // Consumer that owns the next operation.
  uint32_t consumer = 0;
  // Completed byte count for a positive short operation.
  uint32_t progress = 0;
  // Previous submission ticket used to check the CQ dependency edge.
  uint32_t previous_ticket = kAbsent;
};

class GpuFileGatherTest : public GpuFileIoFixture {
 protected:
  void CreateGuardedMemory(const std::vector<uint32_t>& initial,
                           GpuMemory** out_memory) {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE,
                     initial.size() * sizeof(uint32_t), out_memory));
    std::memcpy((*out_memory)->host.pointer, initial.data(),
                initial.size() * sizeof(uint32_t));
  }

  void Run(FileMode mode, uint32_t held_slot, Fault fault = Fault::kNone,
           uint32_t request_capacity = protocol::kRequestCapacity,
           FileIoPath path = FileIoPath::kDevice) {
    const auto* product = products::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(product, nullptr) << "missing compiled file-gather kernel";
    const auto& kernel = *product;
    ASSERT_EQ(kernel.private_segment_byte_length, 0u);
    const uint32_t word_count = page_byte_length_ / sizeof(uint32_t);
    ASSERT_LE(word_count, 16384u);
    const uint32_t block_bytes = word_count * sizeof(uint32_t);
    const uint32_t stride_words = 2 * word_count;
    const uint32_t fault_slot = (held_slot + 1) % protocol::kSlotCount;
    const uint32_t file_blocks =
        protocol::kInputBlockCount + protocol::kOutputBlockCount;
    std::vector<uint32_t> initial_file(file_blocks * word_count, kGuard);
    for (uint32_t block = 0; block < protocol::kInputBlockCount; ++block) {
      for (uint32_t word = 0; word < word_count; ++word) {
        initial_file[block * word_count + word] = InputWord(block, word);
      }
    }
    if (fault == Fault::kShortInput) {
      for (uint32_t word = 0; word < word_count / 2; ++word) {
        initial_file.push_back(InputWord(file_blocks, word));
      }
    }
    ASSERT_NO_FATAL_FAILURE(CreateFile(initial_file, mode));
    if (IsSkipped()) {
      return;
    }
    auto expected_file = initial_file;
    const auto consumers = MakeConsumers(word_count, held_slot);
    std::vector<uint32_t> expected_payload(
        (2 * protocol::kWindowCount + 1) * word_count, kGuard);
    const size_t record_words = protocol::kRecordHeaderWords + word_count;
    std::vector<uint32_t> expected_records(
        2 * kGuardWords + kConsumerCount * record_words, kGuard);
    std::vector<uint32_t> expected_state(
        2 * kGuardWords + sizeof(protocol::State) / sizeof(uint32_t), kGuard);
    std::vector<uint32_t> expected_requests(
        2 * kGuardWords +
            request_capacity * sizeof(protocol::Request) / sizeof(uint32_t),
        kGuard);
    GpuMemory* payload = nullptr;
    GpuMemory* state = nullptr;
    GpuMemory* records = nullptr;
    GpuMemory* requests = nullptr;
    GpuMemory* arguments = nullptr;
    GpuMemory* completion = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateRegisteredPages(
        expected_payload.size() * sizeof(uint32_t), kGuard, &payload));
    ASSERT_NO_FATAL_FAILURE(CreateRing(payload, path));
    ASSERT_NO_FATAL_FAILURE(CreateGuardedMemory(expected_state, &state));
    ASSERT_NO_FATAL_FAILURE(CreateGuardedMemory(expected_records, &records));
    ASSERT_NO_FATAL_FAILURE(CreateGuardedMemory(expected_requests, &requests));
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
    std::memset(completion->host.pointer, 0, completion->host.byte_length);
    const protocol::Arguments device_arguments = {
        .submission_entries = device_ring_memory_->device_address,
        .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
        .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
        .completion_head = RingAddress(ring_->parameters.cq_off.head),
        .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
        .payload = payload->device_address + page_byte_length_,
        .state = state->device_address + kGuardWords * sizeof(uint32_t),
        .records = records->device_address + kGuardWords * sizeof(uint32_t),
        .requests = requests->device_address + kGuardWords * sizeof(uint32_t),
        .host_payload = reinterpret_cast<uintptr_t>(payload->host.pointer) +
                        page_byte_length_,
        .submission_mask = ring_->parameters.sq_entries - 1,
        .completion_mask = ring_->parameters.cq_entries - 1,
        .peer_round_count = kPeerRounds,
        .word_count = word_count,
        .seed = kSeed,
        .payload_stride = 2 * block_bytes,
        .held_slot = held_slot,
        .fault = static_cast<uint32_t>(fault),
        .request_capacity = request_capacity,
    };
    ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);
    ASSERT_LE(kernel.arguments.byte_length, sizeof(device_arguments));
    std::vector<uint8_t> expected_arguments(arguments->host.byte_length, 0);
    std::memcpy(expected_arguments.data(), &device_arguments,
                kernel.arguments.byte_length);
    std::memcpy(arguments->host.pointer, expected_arguments.data(),
                expected_arguments.size());
    ASSERT_NO_FATAL_FAILURE(
        Execute(kernel, arguments, completion, "file_gather"));

    const auto& actual = *reinterpret_cast<const protocol::State*>(
        static_cast<const uint32_t*>(state->host.pointer) + kGuardWords);
    const auto& summary = actual.summary;
    const auto* journal = reinterpret_cast<const protocol::Request*>(
        static_cast<const uint32_t*>(requests->host.pointer) + kGuardWords);
    ASSERT_GE(summary.submitted, 3u);
    ASSERT_LE(summary.submitted, request_capacity);
    ASSERT_EQ(summary.completed, summary.submitted);
    ASSERT_EQ(summary.peak_outstanding, 3u);
    ASSERT_EQ(summary.deduplicated, 1u);
    ASSERT_EQ(summary.duplicate_slot, held_slot);
    const bool journal_full = request_capacity < protocol::kRequestCapacity;
    const int32_t expected_status = journal_full                  ? -EOVERFLOW
                                    : fault == Fault::kShortInput ? -ENODATA
                                    : fault != Fault::kNone       ? -EBADF
                                                                  : 0;
    ASSERT_EQ(summary.status, expected_status);
    const uint32_t processing_end =
        expected_status ? summary.failure_completion : summary.completed;
    ASSERT_LE(processing_end, summary.completed);
    if (expected_status) {
      ASSERT_EQ(summary.failure_tail, summary.submitted);
    } else {
      EXPECT_EQ(summary.failure_tail, 0u);
      EXPECT_EQ(summary.failure_completion, 0u);
      EXPECT_GE(summary.submitted, 6 * kPeerRounds + 8);
    }
    if (journal_full) {
      EXPECT_EQ(summary.submitted, request_capacity);
      // With four journal entries, the first completion is replenished and
      // the second exposes exhaustion with two accepted requests to drain.
      ASSERT_EQ(request_capacity, 4u);
      EXPECT_EQ(processing_end, 2u);
    }
    std::vector<uint32_t> completion_tickets(summary.completed, kAbsent);
    std::array<SourceOracle, protocol::kSlotCount> sources;
    protocol::State predicted = {};
    predicted.summary = summary;
    uint32_t unique_reads = 0;
    uint32_t peer_completed = 0;
    uint32_t held_peer_reloads = 0;
    uint32_t held_ready = 0;
    uint32_t reordered = 0;
    uint32_t previous_frontier = 0;
    uint32_t first_native_error = kAbsent;
    std::array<uint32_t, protocol::kSlotCount> last_reload = {kAbsent, kAbsent,
                                                              kAbsent};
    uint32_t held_first_reload = kAbsent;
    uint32_t held_second_write = kAbsent;

    for (uint32_t ticket = 0; ticket < summary.submitted; ++ticket) {
      SCOPED_TRACE(ticket);
      const auto& request = journal[ticket];
      ASSERT_LT(request.slot, protocol::kSlotCount);
      ASSERT_LT(request.completion, summary.completed);
      ASSERT_EQ(completion_tickets[request.completion], kAbsent);
      completion_tickets[request.completion] = ticket;
      EXPECT_EQ(request.flags, 0u);
      EXPECT_LE(request.completion_frontier, request.completion);
      EXPECT_GE(request.completion_frontier, previous_frontier);
      EXPECT_LE(request.completion_frontier, ticket);
      EXPECT_LE(ticket + 1 - request.completion_frontier, 3u);
      EXPECT_GT(request.submitted_at_completion, ticket);
      EXPECT_LE(request.submitted_at_completion, summary.submitted);
      if (expected_status) {
        EXPECT_LE(request.completion_frontier, processing_end);
        if (request.completion >= processing_end) {
          EXPECT_EQ(request.submitted_at_completion, summary.failure_tail);
        }
      }
      previous_frontier = request.completion_frontier;
      reordered += request.completion != ticket;
      auto& source = sources[request.slot];
      EXPECT_EQ(request.phase, source.phase);
      EXPECT_EQ(request.consumer, source.consumer);
      EXPECT_EQ(request.progress, source.progress);
      if (source.previous_ticket != kAbsent) {
        EXPECT_LT(journal[source.previous_ticket].completion,
                  request.completion_frontier);
      }
      source.previous_ticket = ticket;
      const uint32_t consumer_limit =
          request.slot == held_slot ? 3 : kPeerRounds;
      ASSERT_LT(request.consumer, consumer_limit);
      ASSERT_LE(request.progress, block_bytes);
      ASSERT_EQ(request.length, block_bytes - request.progress);
      const uint32_t record =
          RecordNumber(request.slot, request.consumer, held_slot);
      const auto& consumer = consumers[record];
      EXPECT_EQ(request.cause, consumer.cause);
      const bool reading = request.phase == 0;
      const bool writing = request.phase == 2;
      const bool reloading = request.phase == 4;
      ASSERT_TRUE(reading || writing || reloading);
      const bool fault_read =
          reading && request.slot == fault_slot && request.consumer == 0;
      const uint32_t key = fault_read && fault == Fault::kShortInput
                               ? file_blocks
                               : consumer.key;
      EXPECT_EQ(request.block, reading ? key : consumer.output_block);
      const bool shared = request.slot == held_slot && request.consumer == 0;
      const uint32_t readers = shared ? 2 : 1;
      const uint32_t expected_references = reading ? (ticket < 3 ? 1 : readers)
                                           : writing && request.progress == 0
                                               ? readers
                                               : readers - 1;
      EXPECT_EQ(request.references, expected_references);
      if (ticket < 3) {
        EXPECT_EQ(request.slot, ticket);
        EXPECT_EQ(request.phase, 0u);
        EXPECT_EQ(request.completion_frontier, 0u);
      }
      unique_reads += reading && request.progress == 0;
      const bool invalid_file = request.slot == fault_slot &&
                                request.consumer == 0 &&
                                ((reading && fault == Fault::kInvalidFile) ||
                                 (writing && fault == Fault::kInvalidWrite) ||
                                 (reloading && fault == Fault::kInvalidReload));
      if (invalid_file) {
        EXPECT_EQ(request.result, -EBADF);
      } else if (fault_read && fault == Fault::kShortInput) {
        ASSERT_LE(request.progress, block_bytes / 2);
        EXPECT_EQ(request.result, block_bytes / 2 - request.progress);
      } else {
        ASSERT_GT(request.result, 0);
      }
      if (request.result <= 0) {
        first_native_error = std::min(first_native_error, request.completion);
      }
      const uint32_t transferred = request.result > 0 ? request.result : 0;
      ASSERT_LE(transferred, request.length);
      const uint32_t window = reading   ? request.slot
                              : writing ? WriteWindow(request.slot)
                                        : ReloadWindow(request.slot);
      const size_t payload_word = word_count + window * stride_words;
      auto* destination =
          reinterpret_cast<uint8_t*>(expected_payload.data() + payload_word);
      if (writing && request.progress == 0) {
        std::memcpy(destination, consumer.words.data(), block_bytes);
      }
      if (reading) {
        const size_t file_byte = size_t{key} * block_bytes + request.progress;
        ASSERT_LE(file_byte + transferred,
                  initial_file.size() * sizeof(uint32_t));
        std::memcpy(
            destination + request.progress,
            reinterpret_cast<const uint8_t*>(initial_file.data()) + file_byte,
            transferred);
      } else if (writing) {
        std::memcpy(reinterpret_cast<uint8_t*>(expected_file.data()) +
                        size_t{consumer.output_block} * block_bytes +
                        request.progress,
                    destination + request.progress, transferred);
      } else {
        std::memcpy(destination + request.progress,
                    reinterpret_cast<const uint8_t*>(consumer.words.data()) +
                        request.progress,
                    transferred);
      }
      const bool complete =
          request.result > 0 && request.progress + transferred == block_bytes;
      const bool processed = request.completion < processing_end;
      auto& slot = predicted.slots[request.slot];
      slot = {.phase = 7,
              .consumer = request.consumer,
              .cause = consumer.cause,
              .key = key,
              .references = 0,
              .progress = request.progress,
              .ticket = ticket,
              .generation = consumer.generation};
      if (processed && request.result > 0) {
        slot.progress = complete ? 0 : request.progress + transferred;
      }
      if (complete && processed && reloading) {
        const size_t start = kGuardWords + record * record_words;
        expected_records[start] = consumer.key;
        expected_records[start + 1] = consumer.cause;
        expected_records[start + 2] = request.consumer;
        expected_records[start + 3] = consumer.generation;
        expected_records[start + 4] = request.slot;
        expected_records[start + 5] = WriteWindow(request.slot);
        expected_records[start + 6] = ReloadWindow(request.slot);
        const bool held_witness =
            request.slot != held_slot && request.consumer != 0;
        expected_records[start + 7] = held_witness ? 1 : 0;
        std::copy(
            consumer.words.begin(), consumer.words.end(),
            expected_records.begin() + start + protocol::kRecordHeaderWords);
        held_peer_reloads += held_witness;
        peer_completed += request.slot != held_slot;
        held_ready |= shared;
        slot.consumer += 1;
        if (!shared) {
          slot.cause = consumer.words[0];
          slot.generation += 1;
        }
        last_reload[request.slot] = ticket;
        if (shared) {
          held_first_reload = ticket;
        }
      }
      if (writing && request.slot == held_slot && request.consumer == 1) {
        held_second_write = ticket;
      }
      if (request.result > 0) {
        source.progress += transferred;
        if (complete) {
          source.progress = 0;
          if (reloading) {
            source.consumer += 1;
            source.phase = shared ? 2 : 0;
          } else {
            source.phase += 2;
          }
        }
      }
      // Every populated journal field has now been checked against a semantic
      // expectation or a native ordering/result constraint. Remaining storage
      // must retain its guard pattern.
      std::memcpy(expected_requests.data() + kGuardWords +
                      ticket * sizeof(protocol::Request) / sizeof(uint32_t),
                  &request, sizeof(request));
    }
    if (fault != Fault::kNone) {
      EXPECT_EQ(first_native_error, processing_end);
    } else {
      EXPECT_EQ(first_native_error, kAbsent);
    }
    EXPECT_EQ(summary.unique_reads, unique_reads);
    EXPECT_EQ(summary.peer_completed, peer_completed);
    EXPECT_EQ(summary.held_peer_reloads, held_peer_reloads);
    EXPECT_EQ(summary.held_ready, held_ready);
    if (!expected_status) {
      EXPECT_EQ(unique_reads, 2 * kPeerRounds + 2);
      EXPECT_EQ(peer_completed, 2 * kPeerRounds);
      EXPECT_EQ(held_peer_reloads, 2 * (kPeerRounds - 1));
      ASSERT_NE(held_first_reload, kAbsent);
      ASSERT_NE(held_second_write, kAbsent);
      for (uint32_t ticket = 0; ticket < summary.submitted; ++ticket) {
        const auto& request = journal[ticket];
        if (request.slot != held_slot && request.consumer != 0) {
          EXPECT_LT(journal[held_first_reload].completion,
                    request.completion_frontier);
        }
      }
      for (uint32_t slot = 0; slot < protocol::kSlotCount; ++slot) {
        if (slot == held_slot) {
          continue;
        }
        ASSERT_NE(last_reload[slot], kAbsent);
        EXPECT_LT(journal[last_reload[slot]].completion,
                  journal[held_second_write].completion_frontier);
      }
    }
    std::memcpy(expected_state.data() + kGuardWords, &predicted,
                sizeof(predicted));
    ASSERT_NO_FATAL_FAILURE(
        CheckWords({static_cast<const uint32_t*>(payload->host.pointer),
                    expected_payload.size()},
                   expected_payload));
    ASSERT_NO_FATAL_FAILURE(
        CheckWords({static_cast<const uint32_t*>(records->host.pointer),
                    expected_records.size()},
                   expected_records));
    ASSERT_NO_FATAL_FAILURE(
        CheckWords({static_cast<const uint32_t*>(state->host.pointer),
                    expected_state.size()},
                   expected_state));
    ASSERT_NO_FATAL_FAILURE(
        CheckWords({static_cast<const uint32_t*>(requests->host.pointer),
                    expected_requests.size()},
                   expected_requests));
    EXPECT_EQ(std::memcmp(arguments->host.pointer, expected_arguments.data(),
                          expected_arguments.size()),
              0);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head)),
              summary.submitted);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail)),
              summary.submitted);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.head)),
              summary.completed);
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.tail)),
              summary.completed);
    EXPECT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)),
        0u);
    EXPECT_EQ(
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
        0u);
    const auto* native = reinterpret_cast<const io_uring_cqe*>(
        RingWord(ring_->parameters.cq_off.cqes));
    for (uint32_t position =
             summary.completed > ring_->parameters.cq_entries
                 ? summary.completed - ring_->parameters.cq_entries
                 : 0;
         position < summary.completed; ++position) {
      const auto& entry = native[position & (ring_->parameters.cq_entries - 1)];
      const uint32_t ticket = completion_tickets[position];
      ASSERT_NE(ticket, kAbsent);
      const auto& request = journal[ticket];
      EXPECT_EQ(entry.user_data, (uint64_t{ticket} << 32) | request.slot);
      EXPECT_EQ(entry.res, request.result);
      EXPECT_EQ(entry.flags, request.flags);
    }
    RecordProperty("io_held_slot", held_slot);
    RecordProperty("io_completed_requests", summary.completed);
    RecordProperty("io_unique_reads", summary.unique_reads);
    RecordProperty("io_deduplicated", summary.deduplicated);
    RecordProperty("io_peak_outstanding", summary.peak_outstanding);
    RecordProperty("io_held_peer_reloads", summary.held_peer_reloads);
    RecordProperty("io_reordered_completions", reordered);
    RecordProperty("io_terminal_result", summary.status);
    RecordProperty("io_drained_after_failure",
                   expected_status ? summary.completed - processing_end -
                                         (journal_full ? 0 : 1)
                                   : 0);
    ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, mode));
  }
};

TEST_F(GpuFileGatherTest, BufferedHeldFirst) { Run(FileMode::kBuffered, 0); }
TEST_F(GpuFileGatherTest, BufferedHeldMiddle) { Run(FileMode::kBuffered, 1); }
TEST_F(GpuFileGatherTest, BufferedHeldLast) { Run(FileMode::kBuffered, 2); }
TEST_F(GpuFileGatherTest, DirectHeldFirst) { Run(FileMode::kDirect, 0); }
TEST_F(GpuFileGatherTest, DirectHeldMiddle) { Run(FileMode::kDirect, 1); }
TEST_F(GpuFileGatherTest, DirectHeldLast) { Run(FileMode::kDirect, 2); }

TEST_F(GpuFileGatherTest, InvalidFixedFileStopsIssuanceAndDrainsAcceptedIo) {
  Run(FileMode::kBuffered, 2, Fault::kInvalidFile);
}

TEST_F(GpuFileGatherTest, FailedWriteDrainsWithoutPublishingAConsumerResult) {
  Run(FileMode::kBuffered, 0, Fault::kInvalidWrite);
}

TEST_F(GpuFileGatherTest, FailedReloadDrainsWithoutPublishingAConsumerResult) {
  Run(FileMode::kBuffered, 1, Fault::kInvalidReload);
}

TEST_F(GpuFileGatherTest,
       PartialReadThenEofDrainsWithoutConsumingIncompleteInput) {
  Run(FileMode::kBuffered, 1, Fault::kShortInput);
}

TEST_F(GpuFileGatherTest,
       FullJournalStopsPublicationAndDrainsTwoAcceptedRequests) {
  Run(FileMode::kBuffered, 0, Fault::kNone, 4);
}

class GpuFileGatherProgressTest
    : public GpuFileGatherTest,
      public ::testing::WithParamInterface<std::tuple<uint32_t, FileIoPath>> {};

TEST_P(GpuFileGatherProgressTest, BufferedHeldReader) {
  Run(FileMode::kBuffered, std::get<0>(GetParam()), Fault::kNone,
      protocol::kRequestCapacity, std::get<1>(GetParam()));
}

TEST_P(GpuFileGatherProgressTest, DirectHeldReader) {
  Run(FileMode::kDirect, std::get<0>(GetParam()), Fault::kNone,
      protocol::kRequestCapacity, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    AllSlots, GpuFileGatherProgressTest,
    ::testing::Combine(::testing::Values(0u, 1u, 2u),
                       ::testing::Values(FileIoPath::kDeviceWait,
                                         FileIoPath::kHostRelay,
                                         FileIoPath::kHostWait,
                                         FileIoPath::kHostPoll)));

class GpuFileGatherProgressErrorTest
    : public GpuFileGatherTest,
      public ::testing::WithParamInterface<std::tuple<Fault, FileIoPath>> {};

TEST_P(GpuFileGatherProgressErrorTest, StopsPublicationAndDrains) {
  Run(FileMode::kBuffered, 1, std::get<0>(GetParam()),
      protocol::kRequestCapacity, std::get<1>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(
    AllPhases, GpuFileGatherProgressErrorTest,
    ::testing::Combine(
        ::testing::Values(Fault::kInvalidFile, Fault::kInvalidWrite,
                          Fault::kInvalidReload, Fault::kShortInput),
        ::testing::Values(FileIoPath::kDeviceWait, FileIoPath::kHostRelay,
                          FileIoPath::kHostWait, FileIoPath::kHostPoll)));

class GpuFileGatherProgressJournalTest
    : public GpuFileGatherTest,
      public ::testing::WithParamInterface<FileIoPath> {};

TEST_P(GpuFileGatherProgressJournalTest, FullJournalDrainsAcceptedRequests) {
  Run(FileMode::kBuffered, 0, Fault::kNone, 4, GetParam());
}

INSTANTIATE_TEST_SUITE_P(ProgressPaths, GpuFileGatherProgressJournalTest,
                         ::testing::Values(FileIoPath::kDeviceWait,
                                           FileIoPath::kHostRelay,
                                           FileIoPath::kHostWait,
                                           FileIoPath::kHostPoll));

}  // namespace
