// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/device_sdma_lookahead.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/device_sdma_reader_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_upload_kernels.h"
#include "libamdf/cts/gpu/recipes/device_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

namespace lookahead = kernels::device_sdma_lookahead;

struct alignas(64) ReaderStart {
  // Published by workitem zero; never used to permit reuse or shader launch.
  uint32_t value;
  // Untouched cache-line suffix, separate from native signal records.
  uint32_t guards[15];
};

// Matches the observation offsets consumed by device_sdma_upload.
struct JobSignals {
  // Upload dispatch completion admits every reader of this job.
  aql::Signal upload;
  // Independent reader completions, both joined before input-slot reuse.
  std::array<aql::Signal, 2> readers;
  // Independent reader start markers used only for interleaving observations.
  std::array<ReaderStart, 2> started;
};
static_assert(offsetof(JobSignals, readers) == 64);
static_assert(offsetof(JobSignals, started) == 192);
static_assert(sizeof(JobSignals) == 320);

struct LookaheadCase {
  // Exact required input placement; LOCAL never substitutes SYSTEM backing.
  amdf_memory_class_t memory_class;
  // Number of reusable input slots, controlling the maximum lookahead.
  uint32_t slot_count;
  // Number of independent readers whose completion permits slot reuse.
  uint32_t reader_count;
  // Four positive word counts selected by device computation.
  std::array<uint32_t, 4> word_counts;
  // Finite number of jobs published before the first upload starts.
  uint32_t job_count;
  // Stable placement, size, slot and reader parameter name.
  std::string name;
};

// Compare every word without expanding an entire allocation into a failure log.
void CheckWords(const void* observed, const std::vector<uint32_t>& expected) {
  const auto* words = static_cast<const uint32_t*>(observed);
  const auto mismatch = std::mismatch(expected.begin(), expected.end(), words);
  ASSERT_EQ(mismatch.first, expected.end())
      << "word " << (mismatch.first - expected.begin()) << ": expected "
      << *mismatch.first << ", observed " << *mismatch.second;
}

class DeviceSdmaLookaheadTest
    : public DeviceGeneratedSdmaTest,
      public ::testing::WithParamInterface<LookaheadCase> {};

TEST_P(DeviceSdmaLookaheadTest, ReuseJoinsEveryIndependentReader) {
  const auto& test_case = GetParam();
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
      (features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "LOCAL placement is not advertised by this device";
  }
  constexpr uint64_t kAqlRingByteLength = 65536;
  ASSERT_NE(family_.ring_byte_length_alignment, 0u);
  if (kAqlRingByteLength < family_.minimum_ring_byte_length ||
      kAqlRingByteLength > family_.maximum_ring_byte_length ||
      kAqlRingByteLength % family_.ring_byte_length_alignment != 0) {
    GTEST_SKIP() << "finite lookahead batch requires a 64-KiB AQL ring";
  }
  const auto* upload_kernel =
      kernels::device_sdma_upload::kKernels.Find(gpu_endpoint_info_);
  const auto* reader_kernel =
      kernels::device_sdma_reader::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(upload_kernel, nullptr);
  ASSERT_NE(reader_kernel, nullptr);
  ASSERT_EQ(upload_kernel->workgroup_size(), 1u);
  ASSERT_EQ(reader_kernel->workgroup_size(), 64u);
  ASSERT_LE(upload_kernel->arguments.byte_length,
            sizeof(lookahead::UploadArguments));
  ASSERT_LE(reader_kernel->arguments.byte_length,
            sizeof(lookahead::ReaderArguments));
  for (const auto* kernel : {upload_kernel, reader_kernel}) {
    ASSERT_EQ(kernel->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  }

  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  constexpr uint32_t kSourceGuard = 0x751e29c3u;
  constexpr uint32_t kInputGuard = 0x329b76a1u;
  constexpr uint32_t kOutputGuard = 0x8abd432fu;
  constexpr uint32_t kControlGuard = 0xc49a132bu;
  constexpr uint32_t kPrefixWords = 16;
  constexpr uint64_t kCompletionOffset = 64;
  constexpr uint64_t kFrontierOffset = 128;
  constexpr uint64_t kLengthsOffset = 192;
  constexpr uint64_t kArgumentStride = 256;
  constexpr uint64_t kReaderArgumentsOffset = 160;
  const auto align_page = [](uint64_t bytes) {
    return (bytes + 4095) & ~UINT64_C(4095);
  };
  const uint32_t maximum_words = *std::max_element(
      test_case.word_counts.begin(), test_case.word_counts.end());
  const uint64_t slot_bytes = align_page(uint64_t{maximum_words} * 4 + 128);
  const uint64_t slot_words = slot_bytes / 4;
  const uint64_t input_bytes = slot_bytes * test_case.slot_count;
  const uint64_t output_bytes =
      slot_bytes * test_case.job_count * test_case.reader_count;
  GpuMemory* source = nullptr;
  GpuMemory* seed = nullptr;
  GpuMemory* input = nullptr;
  GpuMemory* readback = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* requests = nullptr;
  GpuMemory* selections = nullptr;
  GpuMemory* signals = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 8 * slot_bytes, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, slot_bytes, &seed));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, input_bytes, &readback));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, output_bytes, &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ, align_page(test_case.job_count * 4), &requests));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite,
      align_page(test_case.job_count * sizeof(lookahead::Selection)),
      &selections));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite, align_page((test_case.job_count + 1) * sizeof(JobSignals)),
      &signals));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, 4096, &control));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ,
      align_page(test_case.job_count * kArgumentStride), &arguments));
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL) {
    ASSERT_NE(local_scope_, nullptr);
    const amdf_memory_device_access_t attachment = {
        device_,
        {.access = kReadWrite, .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    const uint32_t profile = FindGpuMemoryProfileOrdinal(
        api_, local_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL, attachment.requirements);
    ASSERT_NE(profile, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    const amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = profile,
        .access_count = 1,
        .required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL,
        .byte_length = input_bytes,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &input));
    ASSERT_EQ(input->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
    ASSERT_NE(input->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
    ASSERT_EQ(input->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    ASSERT_EQ(input->host.pointer, nullptr);
    ASSERT_EQ(input->mapping, nullptr);
  } else {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, input_bytes, &input));
  }
  uint32_t upload_flags = 0;
  uint32_t seed_flags = 0;
  uint32_t readback_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, input, output, &upload_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, input, selections, &seed_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryDownloadCacheFlags(input, readback, &readback_flags));

  GpuUserQueue* publisher = nullptr;
  GpuUserQueue* consumer = nullptr;
  GpuUserQueue* transfer = nullptr;
  for (auto** queue : {&publisher, &consumer}) {
    ASSERT_NO_FATAL_FAILURE(CreateQueue(
        family_, queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kAqlRingByteLength));
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  uint64_t publisher_index = 0;
  uint64_t consumer_index = 0;
  uint64_t upload_descriptor = 0;
  uint64_t reader_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*publisher, *upload_kernel,
                                        "lookahead_upload", &publisher_index,
                                        &upload_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*consumer, *reader_kernel,
                                        "lookahead_reader", &consumer_index,
                                        &reader_descriptor));

  std::vector<uint32_t> expected_source(source->info.byte_length / 4,
                                        kSourceGuard);
  for (uint32_t page = 0; page < 8; ++page) {
    for (uint32_t word = 0; word < maximum_words; ++word) {
      expected_source[page * slot_words + kPrefixWords + word] =
          0x31415927u + page * 0x243f6a89u + word * 0x1020305u;
    }
  }
  std::memcpy(source->host.pointer, expected_source.data(),
              source->info.byte_length);
  std::fill_n(static_cast<uint32_t*>(seed->host.pointer), slot_words,
              kInputGuard);
  std::fill_n(static_cast<uint32_t*>(readback->host.pointer), input_bytes / 4,
              kOutputGuard);
  std::fill_n(static_cast<uint32_t*>(output->host.pointer), output_bytes / 4,
              kOutputGuard);
  std::vector<uint32_t> expected_selections(selections->info.byte_length / 4,
                                            kControlGuard);
  std::memcpy(selections->host.pointer, expected_selections.data(),
              selections->info.byte_length);
  std::vector<uint32_t> expected_requests(requests->info.byte_length / 4,
                                          kControlGuard);
  uint32_t token = 0x91e10da5u;
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    token = token * 1664525u + 1013904223u;
    expected_requests[job] = token;
  }
  std::memcpy(requests->host.pointer, expected_requests.data(),
              requests->info.byte_length);

  std::vector<uint32_t> expected_signals(signals->info.byte_length / 4,
                                         kControlGuard);
  auto* job_signals = static_cast<JobSignals*>(signals->host.pointer);
  for (uint32_t job = 0; job <= test_case.job_count; ++job) {
    JobSignals state = {};
    state.upload.kind = 1;
    for (auto& signal : state.readers) {
      signal.kind = 1;
    }
    for (uint32_t reader = 0; reader < 2; ++reader) {
      state.started[reader].value =
          job < test_case.job_count && reader < test_case.reader_count ? 1 : 0;
      std::fill_n(state.started[reader].guards, 15, kControlGuard);
    }
    std::memcpy(expected_signals.data() + job * sizeof(JobSignals) / 4, &state,
                sizeof(state));
  }
  std::memcpy(signals->host.pointer, expected_signals.data(),
              signals->info.byte_length);
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    job_signals[job].upload.value = 1;
    for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
      job_signals[job].readers[reader].value = 1;
      job_signals[job].started[reader].value = 0;
    }
  }
  std::array<uint32_t, 1024> expected_control;
  expected_control.fill(kControlGuard);
  aql::Signal joined = {.kind = 1};
  std::memcpy(expected_control.data(), &joined, sizeof(joined));
  expected_control[kCompletionOffset / 4] = 0;
  std::memcpy(expected_control.data() + kLengthsOffset / 4,
              test_case.word_counts.data(), sizeof(test_case.word_counts));
  std::memcpy(control->host.pointer, expected_control.data(),
              control->info.byte_length);
  auto& terminal_signal = *static_cast<aql::Signal*>(control->host.pointer);
  terminal_signal.value = 1;
  const uint64_t completion_address =
      control->device_address + kCompletionOffset;
  const uintptr_t completion_host =
      reinterpret_cast<uintptr_t>(control->host.pointer) + kCompletionOffset;

  // Only cold initialization and final inspection use the host producer. Both
  // ownership transfers join SDMA command retirement as well as completion.
  uint64_t frontier = 0;
  auto copy_slots = [&](uint64_t source_address, uint64_t source_stride,
                        uint64_t target_address, uint32_t flags,
                        uint32_t generation) {
    std::array<uint32_t, 64> words = {};
    SdmaCommandWriter writer(words.data(), sdma_family_.format_features);
    if ((flags & 1) != 0) {
      writer.AcquireFromSystem();
    }
    for (uint32_t slot = 0; slot < test_case.slot_count; ++slot) {
      writer.CopyLinear(source_address + slot * source_stride,
                        target_address + slot * slot_bytes,
                        static_cast<uint32_t>(slot_bytes));
    }
    if ((flags & 2) != 0) {
      writer.ReleaseToSystem();
    }
    writer.Fence32(completion_address, generation);
    const uint64_t bytes = writer.word_count() * sizeof(uint32_t);
    const uint64_t offset = frontier % mapping.ring_byte_length;
    const uint64_t tail = mapping.ring_byte_length - offset;
    auto* ring = reinterpret_cast<uint8_t*>(transfer->host.ring_address);
    if (tail < bytes) {
      std::memset(ring + offset, 0, tail);
      frontier += tail;
    }
    std::memcpy(ring + frontier % mapping.ring_byte_length, words.data(),
                bytes);
    frontier += bytes;
    transfer->PublishStream(frontier);
    GpuWaitEqual<uint32_t>(completion_host, generation);
    ASSERT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  };
  ASSERT_NO_FATAL_FAILURE(copy_slots(seed->device_address, 0,
                                     input->device_address, seed_flags, 1));
  GpuStoreRelease<uint32_t>(completion_host, 0);
  std::memcpy(static_cast<uint8_t*>(control->host.pointer) + kFrontierOffset,
              &frontier, sizeof(frontier));

  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, input->device_address, 4);
  encoder.Fence32(completion_address, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());
  std::vector<uint8_t> expected_arguments(arguments->info.byte_length, 0);
  std::vector<uint32_t> expected_input(input_bytes / 4, kInputGuard);
  std::vector<uint32_t> expected_output(output_bytes / 4, kOutputGuard);
  std::vector<aql::Packet> uploads;
  std::vector<aql::Packet> readers;
  // Dependencies provide readiness; each dispatch owns its SYSTEM visibility.
  constexpr aql::FenceScopes kDependencyScopes = {aql::FenceScope::kNone,
                                                  aql::FenceScope::kNone};
  const auto job_address = [&](uint32_t job) {
    return signals->device_address + job * sizeof(JobSignals);
  };
  const auto reader_dependencies = [&](uint32_t job) {
    std::array<uint64_t, 5> dependencies = {};
    for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
      dependencies[reader] = job_address(job) + offsetof(JobSignals, readers) +
                             reader * sizeof(aql::Signal);
    }
    return dependencies;
  };
  uint32_t selected_pages = 0;
  uint32_t selected_lengths = 0;
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    const uint32_t slot = job % test_case.slot_count;
    const uint64_t argument_offset = job * kArgumentStride;
    const uint64_t selection_address =
        selections->device_address + job * sizeof(lookahead::Selection);
    const lookahead::UploadArguments upload = {
        .ring = mapping.ring_address,
        .read_index = mapping.read_index_address,
        .write_index = mapping.write_index_address,
        .notification = mapping.doorbell_address,
        .completion = completion_address,
        .state = control->device_address + kFrontierOffset,
        .selection = selection_address,
        .request = requests->device_address + job * 4,
        .previous_readers = job_address(job ? job - 1 : test_case.job_count),
        .lengths = control->device_address + kLengthsOffset,
        .source_address = source->device_address,
        .input_address = input->device_address + slot * slot_bytes,
        .completion_address = completion_address,
        .capacity = mapping.ring_byte_length,
        .slot_byte_length = slot_bytes,
        .slot = slot,
        .generation = job + 1,
        .copy_control = encoding[2],
        .fence_header = encoding[7],
        .cache_flags = upload_flags,
        .reserved = 0,
    };
    std::memcpy(expected_arguments.data() + argument_offset, &upload,
                upload_kernel->arguments.byte_length);
    if (job >= test_case.slot_count) {
      uploads.push_back(aql::Barrier(
          aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled, 0,
          reader_dependencies(job - test_case.slot_count), kDependencyScopes));
    }
    uploads.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
        upload_descriptor, arguments->device_address + argument_offset,
        job_address(job)));
    // Earlier readers need not finish to admit a different uploaded slot.
    readers.push_back(
        aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
                     {job_address(job), 0, 0, 0, 0}, kDependencyScopes));

    lookahead::Selection selection = {};
    std::fill_n(selection.guards, 8, kControlGuard);
    selection.hash = expected_requests[job] ^ (expected_requests[job] >> 13) ^
                     ((job + 1) * 0x9e3779b1u);
    selection.page = selection.hash & 7;
    selection.slot = slot;
    const uint32_t length_index = (selection.hash >> 8) & 3;
    selection.word_count = test_case.word_counts[length_index];
    selected_pages |= 1u << selection.page;
    selected_lengths |= 1u << length_index;
    const uint64_t chain_bytes =
        44 + ((upload_flags & 1) ? 20 : 0) + ((upload_flags & 2) ? 20 : 0);
    const uint64_t tail =
        mapping.ring_byte_length - frontier % mapping.ring_byte_length;
    if (tail < chain_bytes) {
      frontier += tail;
    }
    frontier += chain_bytes;
    selection.frontier = frontier;
    std::memcpy(expected_selections.data() + job * sizeof(selection) / 4,
                &selection, sizeof(selection));
    const auto* source_words =
        expected_source.data() + selection.page * slot_words + kPrefixWords;
    std::copy_n(source_words, selection.word_count,
                expected_input.data() + slot * slot_words + kPrefixWords);
    for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
      const uint64_t output_offset =
          (job * test_case.reader_count + reader) * slot_bytes +
          kPrefixWords * 4;
      // Alternate the slower reader so joining only the first or last packet
      // cannot masquerade as joining every input user.
      const uint32_t round_count = ((reader + job) & 1) ? 257 : 3;
      const lookahead::ReaderArguments read = {
          .input = upload.input_address + kPrefixWords * 4,
          .output = output->device_address + output_offset,
          .selection = selection_address,
          .started = job_address(job) + offsetof(JobSignals, started) +
                     reader * sizeof(ReaderStart),
          .round_count = round_count,
          .reader = reader,
      };
      const uint64_t reader_offset =
          argument_offset + kReaderArgumentsOffset +
          reader * sizeof(lookahead::ReaderArguments);
      std::memcpy(expected_arguments.data() + reader_offset, &read,
                  reader_kernel->arguments.byte_length);
      readers.push_back(aql::Dispatch(
          aql::HeaderBarrier::kDisabled,
          {1, {64, 1, 1}, {(maximum_words + 63) & ~63u, 1, 1}}, 0, 0,
          reader_descriptor, arguments->device_address + reader_offset,
          reader_dependencies(job)[reader]));
      for (uint32_t word = 0; word < selection.word_count; ++word) {
        uint32_t value = selection.hash ^ reader;
        for (uint32_t round = 0; round < round_count; ++round) {
          value += source_words[(word + round) % selection.word_count];
          value = (value ^ (value >> 13)) * 1664525u + 1013904223u;
        }
        expected_output[output_offset / 4 + word] = value;
      }
    }
  }
  EXPECT_EQ(selected_pages, 0xffu);
  EXPECT_EQ(selected_lengths, 0xfu);
  if (test_case.job_count == 129) {
    EXPECT_GT(frontier, mapping.ring_byte_length)
        << "row batches must reuse SDMA command storage across ring wrap";
  }
  // Header-barrier completion joins all earlier reader packets, including any
  // that remain live after the final job's own readers finish.
  readers.push_back(aql::Barrier(aql::BarrierType::kAnd,
                                 aql::HeaderBarrier::kEnabled,
                                 control->device_address));
  std::memcpy(arguments->host.pointer, expected_arguments.data(),
              arguments->info.byte_length);
  expected_control[kCompletionOffset / 4] = test_case.job_count;
  std::memcpy(expected_control.data() + kFrontierOffset / 4, &frontier,
              sizeof(frontier));
  ASSERT_NO_FATAL_FAILURE(
      aql::PrepareBatch(*publisher, publisher_index, uploads));
  ASSERT_NO_FATAL_FAILURE(
      aql::PrepareBatch(*consumer, consumer_index, readers));
  aql::PublishBatch(*consumer, consumer_index, readers);
  aql::PublishBatch(*publisher, publisher_index, uploads);

  // One terminal host join; no per-job refill, inspection or release.
  GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&terminal_signal.value), 0);
  EXPECT_NO_FATAL_FAILURE(CheckWords(output->host.pointer, expected_output));
  EXPECT_NO_FATAL_FAILURE(CheckWords(source->host.pointer, expected_source));
  EXPECT_NO_FATAL_FAILURE(
      CheckWords(requests->host.pointer, expected_requests));
  EXPECT_NO_FATAL_FAILURE(CheckWords(signals->host.pointer, expected_signals));
  EXPECT_EQ(std::memcmp(arguments->host.pointer, expected_arguments.data(),
                        arguments->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(control->host.pointer, expected_control.data(),
                        control->info.byte_length),
            0);
  const auto* observed =
      static_cast<const lookahead::Selection*>(selections->host.pointer);
  uint32_t interleaved_uploads = 0;
  uint32_t live_second_readers = 0;
  std::string reader_observations;
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    SCOPED_TRACE(job);
    const uint32_t reader_mask = (1u << test_case.reader_count) - 1;
    EXPECT_EQ(observed[job].started_before & ~reader_mask, 0u);
    EXPECT_EQ(observed[job].pending_after & ~reader_mask, 0u);
    if (job == 0 || test_case.slot_count == 1) {
      EXPECT_EQ(observed[job].pending_after, 0u);
    }
    const uint32_t live =
        observed[job].started_before & observed[job].pending_after;
    if (!reader_observations.empty()) {
      reader_observations += ",";
    }
    reader_observations += std::to_string(observed[job].started_before) + ":" +
                           std::to_string(observed[job].pending_after);
    interleaved_uploads += live != 0;
    live_second_readers += (live & 2) != 0;
    const uint64_t expected_offset = job * sizeof(lookahead::Selection) / 4;
    expected_selections[expected_offset + 6] = observed[job].started_before;
    expected_selections[expected_offset + 7] = observed[job].pending_after;
  }
  EXPECT_NO_FATAL_FAILURE(
      CheckWords(selections->host.pointer, expected_selections));
  EXPECT_EQ(GpuLoadAcquire<uint64_t>(transfer->host.write_index_address),
            frontier);
  ASSERT_NO_FATAL_FAILURE(
      publisher->WaitConsumed(api_, publisher_index + uploads.size()));
  ASSERT_NO_FATAL_FAILURE(
      consumer->WaitConsumed(api_, consumer_index + readers.size()));
  ASSERT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  const uint64_t upload_frontier = frontier;
  ASSERT_NO_FATAL_FAILURE(copy_slots(input->device_address, slot_bytes,
                                     readback->device_address, readback_flags,
                                     test_case.job_count + 1));
  EXPECT_NO_FATAL_FAILURE(CheckWords(readback->host.pointer, expected_input));
  RecordProperty("lookahead_jobs", test_case.job_count);
  RecordProperty("lookahead_slots", test_case.slot_count);
  RecordProperty("lookahead_readers", test_case.reader_count);
  RecordProperty("lookahead_memory_class", input->info.memory_class);
  RecordProperty("lookahead_upload_cache_flags", upload_flags);
  RecordProperty("lookahead_sdma_ring_bytes",
                 std::to_string(mapping.ring_byte_length));
  RecordProperty("lookahead_uploads_during_live_reader", interleaved_uploads);
  RecordProperty("lookahead_uploads_during_live_second_reader",
                 live_second_readers);
  RecordProperty("lookahead_reader_start_pending_observations",
                 reader_observations);
  RecordProperty("lookahead_sdma_ring_wraps",
                 std::to_string(upload_frontier / mapping.ring_byte_length));
}

std::vector<LookaheadCase> LookaheadCases() {
  std::vector<LookaheadCase> cases;
  for (const amdf_memory_class_t memory_class :
       {AMDF_MEMORY_CLASS_SYSTEM, AMDF_MEMORY_CLASS_LOCAL}) {
    const std::string placement =
        memory_class == AMDF_MEMORY_CLASS_LOCAL ? "Local" : "System";
    for (uint32_t slot_count : {1u, 2u, 4u}) {
      for (uint32_t reader_count : {1u, 2u}) {
        const std::string suffix = "Slots" + std::to_string(slot_count) +
                                   "Readers" + std::to_string(reader_count);
        cases.push_back({memory_class,
                         slot_count,
                         reader_count,
                         {17, 66, 72, 1584},
                         129,
                         placement + "Rows" + suffix});
        cases.push_back({memory_class,
                         slot_count,
                         reader_count,
                         {1024, 4352, 18432, 16384},
                         17,
                         placement + "Blocks" + suffix});
      }
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Placement, DeviceSdmaLookaheadTest, ::testing::ValuesIn(LookaheadCases()),
    [](const ::testing::TestParamInfo<LookaheadCase>& info) {
      return info.param.name;
    });

}  // namespace
