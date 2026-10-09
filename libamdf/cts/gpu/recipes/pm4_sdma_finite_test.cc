// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/recipes/pm4_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"

namespace {

constexpr uint32_t kMaximumCredits = 8;
constexpr uint32_t kGraphsPerStream = 64;
constexpr uint32_t kStreamsPerCreditCount = 5;
constexpr uint64_t kRingByteLength = 65536;
constexpr std::array<uint32_t, 4> kCreditCounts = {1, 2, 4, 8};
// A shared transfer queue must put each required upload before any download
// that waits for it. Grouping by available credits also preserves slot reuse.
enum class TransferQueues { kSeparate, kSharedPaired, kSharedCreditGrouped };

enum QueueIndex : size_t {
  kComputeQueue,
  kUploadQueue,
  kDownloadQueue,
  kQueueCount
};
// PM4 positions count DWORDs; SDMA positions count bytes.
constexpr std::array<uint64_t, kQueueCount> kIndexByteLengths = {
    sizeof(uint32_t), 1, 1};

class Pm4SdmaFiniteStreamTest : public Pm4SdmaTest {
 protected:
  Pm4SdmaFiniteStreamTest() : Pm4SdmaTest(AMDF_QUEUE_PUBLICATION_MODE_USER) {}
  static constexpr std::array<Edge, 15> kEdges = {{
      {"source_host_to_sdma", kSource, Site::kHost, Site::kSdma, kUpload},
      {"input_sdma_to_pm4", kInput, Site::kSdma, Site::kPm4, kUpload},
      {"output_pm4_to_sdma", kOutput, Site::kPm4, Site::kSdma, kDownload},
      {"readback_sdma_to_host", kReadback, Site::kSdma, Site::kHost, kDownload},
      {"input_seed_host_to_pm4", kInput, Site::kHost, Site::kPm4, kUpload},
      {"output_seed_host_to_sdma", kOutput, Site::kHost, Site::kSdma,
       kDownload},
      {"input_pm4_to_sdma_reuse", kInput, Site::kPm4, Site::kSdma, kUpload},
      {"output_sdma_to_pm4_reuse", kOutput, Site::kSdma, Site::kPm4, kDownload},
      {"input_sdma_to_host", kInput, Site::kSdma, Site::kHost, kUpload},
      {"output_pm4_to_host", kOutput, Site::kPm4, Site::kHost, kDownload},
      {"arguments_host_to_pm4", kArguments, Site::kHost, Site::kPm4, kUpload},
      {"code_host_to_pm4", kCode, Site::kHost, Site::kPm4, kUpload},
      {"control_host_to_sdma", kControl, Site::kHost, Site::kSdma, kUpload},
      {"control_pm4_to_sdma", kControl, Site::kPm4, Site::kSdma, kDownload},
      {"control_sdma_to_host", kControl, Site::kSdma, Site::kHost, kDownload},
  }};

  void RunCoherentStreams(TransferQueues transfer_queues) {
    for (const auto& family : {family_, sdma_family_}) {
      if (!(family.user_queue_capabilities &
            AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER)) {
        GTEST_SKIP() << "family does not admit host USER publication";
      }
      ASSERT_NE(family.ring_byte_length_alignment, 0u);
      if (kRingByteLength < family.minimum_ring_byte_length ||
          kRingByteLength > family.maximum_ring_byte_length ||
          kRingByteLength % family.ring_byte_length_alignment != 0) {
        GTEST_SKIP() << "family does not admit the 64-KiB command ring";
      }
    }
    const bool separate = transfer_queues == TransferQueues::kSeparate;
    const auto* kernel_product =
        kernels::transform::kKernels.Find(gpu_endpoint_info_);
    ASSERT_NE(kernel_product, nullptr)
        << "missing compiled transform kernel for endpoint";
    const auto& kernel = *kernel_product;
    RecordProperty("transform_kernel_target", kernel.target);

    constexpr amdf_memory_access_t kReadWrite =
        AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
    constexpr uint32_t kGridSize = 1024;
    constexpr uint32_t kPayloadWordCount = 2048;
    constexpr uint32_t kPageWordCount = 1024;
    constexpr uint32_t kPayloadWordOffset = 16;
    constexpr uint64_t kPayloadByteOffset =
        kPayloadWordOffset * sizeof(uint32_t);
    constexpr uint64_t kPayloadByteLength =
        kPayloadWordCount * sizeof(uint32_t);
    constexpr uint64_t kPageByteLength = kPageWordCount * sizeof(uint32_t);
    // Each slot has three independently written progress lines.
    constexpr uint64_t kUploadByteOffset = 0;
    constexpr uint64_t kComputeByteOffset = 64;
    constexpr uint64_t kDownloadByteOffset = 128;
    constexpr uint32_t kStreamCount =
        kCreditCounts.size() * kStreamsPerCreditCount;
    constexpr uint32_t kControlGuard = 0x68d329b7u;
    constexpr std::array<uint32_t, 4> kGuards = {0x759bf13du, 0x26a4e8c3u,
                                                 0x93b57fd1u, 0x4cd218a7u};
    constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
    constexpr std::array<uint32_t, 2> kAddends = {7, 0x80000023u};

    static_assert(sizeof(kernels::transform::Arguments) == 32);
    std::array<Backing, kBackingCount> backings = {{
        {"source", kGraphsPerStream * kPayloadByteLength,
         AMDF_MEMORY_ACCESS_READ},
        {"input", kMaximumCredits * kPayloadByteLength, kReadWrite},
        {"output", kMaximumCredits * kPayloadByteLength, kReadWrite},
        {"readback", kGraphsPerStream * kPayloadByteLength, kReadWrite},
        {"arguments", kMaximumCredits * kPageByteLength,
         AMDF_MEMORY_ACCESS_READ},
        {"control", kMaximumCredits * kPageByteLength, kReadWrite},
        {"code", kPageByteLength,
         AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE},
    }};
    Pm4ComputeProgram program = {};
    std::array<amdf_cache_operations_t, kTransferPhaseCount> sdma_operations =
        {};
    ASSERT_NO_FATAL_FAILURE(PrepareCoherentHandoff(kernel, PairQuery::kConcrete,
                                                   kEdges, backings, &program,
                                                   &sdma_operations));
    auto& source = *backings[kSource].memory;
    auto& input = *backings[kInput].memory;
    auto& output = *backings[kOutput].memory;
    auto& readback = *backings[kReadback].memory;
    auto& arguments = *backings[kArguments].memory;
    auto& control = *backings[kControl].memory;
    auto& code = *backings[kCode].memory;
    ASSERT_EQ(arguments.device_address % kernel.arguments.alignment, 0u);
    ASSERT_EQ(control.device_address % 64, 0u);
    std::array<uint32_t, kPageWordCount> expected_code = {};
    std::memcpy(expected_code.data(), kernel.executable.words,
                kernel.executable.byte_length);

    GpuUserQueue* pm4_queue = nullptr;
    GpuUserQueue* upload_queue = nullptr;
    GpuUserQueue* download_queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(
        CreateQueue(family_, &pm4_queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                    AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kRingByteLength));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(
        sdma_family_, &upload_queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kRingByteLength));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(
        sdma_family_, &download_queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
        AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kRingByteLength));
    for (auto* queue : {upload_queue, download_queue}) {
      ASSERT_TRUE(amdf_device_id_is_equal(&pm4_queue->info.device_id,
                                          &queue->info.device_id));
      ASSERT_NE(pm4_queue->queue, queue->queue);
    }
    ASSERT_NE(upload_queue->queue, download_queue->queue);
    constexpr uint64_t kPm4SlotWords = 128;
    constexpr uint64_t kSdmaSlotWords = 64;
    constexpr uint64_t kPm4SlotBytes = kPm4SlotWords * sizeof(uint32_t);
    constexpr uint64_t kSdmaSlotBytes = kSdmaSlotWords * sizeof(uint32_t);
    const uint64_t pm4_capacity = pm4_queue->host.ring_byte_length;
    const uint64_t upload_capacity = upload_queue->host.ring_byte_length;
    const uint64_t download_capacity = download_queue->host.ring_byte_length;
    ASSERT_GT(pm4_capacity, kPm4SlotBytes);
    ASSERT_GT(upload_capacity, kSdmaSlotBytes);
    ASSERT_GT(download_capacity, kSdmaSlotBytes);
    ASSERT_EQ(pm4_capacity % kPm4SlotBytes, 0u);
    ASSERT_EQ(upload_capacity % kSdmaSlotBytes, 0u);
    ASSERT_EQ(download_capacity % kSdmaSlotBytes, 0u);
    ASSERT_EQ(pm4_capacity, kRingByteLength);
    ASSERT_EQ(upload_capacity, kRingByteLength);
    ASSERT_EQ(download_capacity, kRingByteLength);
    ASSERT_LT(kGraphsPerStream * kPm4SlotBytes, pm4_capacity);
    ASSERT_LT(2 * kGraphsPerStream * kSdmaSlotBytes, upload_capacity);
    const std::array<GpuUserQueue*, kQueueCount> queues = {
        pm4_queue, upload_queue, download_queue};
    // Per-queue frontiers retain the inactive queue's consumed value.
    std::array<uint64_t, kQueueCount> frontiers = {};
    std::array<uint64_t, kQueueCount> credit_begin = {};
    std::array<std::vector<uint32_t>, kQueueCount> expected_rings;
    std::array<std::vector<uint32_t>, kQueueCount> observed_rings;
    for (size_t queue = 0; queue < queues.size(); ++queue) {
      expected_rings[queue].resize(kRingByteLength / sizeof(uint32_t));
      observed_rings[queue].resize(expected_rings[queue].size());
      std::memcpy(reinterpret_cast<void*>(queues[queue]->host.ring_address),
                  expected_rings[queue].data(), kRingByteLength);
    }
    RecordProperty("stream_graphs", kGraphsPerStream);
    RecordProperty("stream_credit_counts", "1,2,4,8");
    RecordProperty("stream_queue_count", kQueueCount);
    RecordProperty("stream_active_queue_count", separate ? 3 : 2);
    RecordProperty("stream_pm4_slot_dwords", kPm4SlotWords);
    RecordProperty("stream_sdma_slot_dwords", kSdmaSlotWords);
    RecordProperty("stream_pm4_capacity_bytes", std::to_string(pm4_capacity));
    RecordProperty("stream_upload_capacity_bytes",
                   std::to_string(upload_capacity));
    RecordProperty("stream_download_capacity_bytes",
                   std::to_string(download_capacity));
    RecordProperty("stream_publication_scope",
                   "fully_recorded_before_publication");

    std::array<std::vector<uint32_t>, 4> expected;
    std::array<std::vector<uint32_t>, 4> observed;
    for (size_t owner = 0; owner < expected.size(); ++owner) {
      expected[owner].resize(backings[owner].byte_length / sizeof(uint32_t));
      observed[owner].resize(expected[owner].size());
    }
    std::array<uint32_t, kMaximumCredits * kPageWordCount> expected_arguments;
    std::array<uint32_t, kMaximumCredits * kPageWordCount> observed_arguments;
    std::array<uint32_t, kMaximumCredits * kPageWordCount> expected_control;
    std::array<uint32_t, kMaximumCredits * kPageWordCount> observed_control;
    std::array<uint32_t, kPageWordCount> observed_code;
    uint64_t completed_graphs = 0;

    const auto check_words = [&](const char* name, const auto& actual,
                                 const auto& wanted) {
      const auto mismatch =
          std::mismatch(actual.begin(), actual.end(), wanted.begin());
      if (mismatch.first != actual.end()) {
        ADD_FAILURE() << name << " word=" << (mismatch.first - actual.begin())
                      << " actual=" << *mismatch.first
                      << " expected=" << *mismatch.second;
      }
    };
    for (uint32_t stream = 0; stream < kStreamCount; ++stream) {
      const uint32_t credits = kCreditCounts[stream / kStreamsPerCreditCount];
      // Each organization receives the same changing inputs, arguments and
      // guards, including slots that this credit count leaves inactive.
      const uint32_t workload = stream;
      SCOPED_TRACE(::testing::Message()
                   << "stream=" << stream << " credits=" << credits);
      if (stream % kStreamsPerCreditCount == 0) {
        credit_begin = frontiers;
      }
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        std::fill(expected[owner].begin(), expected[owner].end(),
                  kGuards[owner] ^ (workload * 0x1020304u));
      }
      expected_control.fill(kControlGuard);
      for (uint32_t slot = 0; slot < kMaximumCredits; ++slot) {
        const size_t page = slot * kPageWordCount;
        expected_control[page + kUploadByteOffset / sizeof(uint32_t)] = 0;
        expected_control[page + kComputeByteOffset / sizeof(uint32_t)] = 0;
        expected_control[page + kDownloadByteOffset / sizeof(uint32_t)] = 0;
        std::fill_n(expected_arguments.data() + page, kPageWordCount,
                    UINT32_C(0x713ace09) ^ (workload * kMaximumCredits + slot));
        if (slot >= credits) {
          continue;
        }
        const uint64_t payload_offset =
            slot * kPayloadWordCount * sizeof(uint32_t);
        const kernels::transform::Arguments payload = {
            input.device_address + payload_offset + kPayloadByteOffset,
            output.device_address + payload_offset + kPayloadByteOffset,
            kCounts[(workload + slot) % 2], kAddends[(workload + slot) % 2]};
        std::memcpy(expected_arguments.data() + page, &payload,
                    kernel.arguments.byte_length);
      }
      // The previous stream's complete native retirement precedes this reset.
      // No CPU write to these progress lines occurs during submitted work.
      std::memcpy(control.host.pointer, expected_control.data(),
                  sizeof(expected_control));
      std::memcpy(arguments.host.pointer, expected_arguments.data(),
                  sizeof(expected_arguments));

      std::array<std::array<uint32_t, kPm4SlotWords>, kGraphsPerStream>
          pm4_streams = {};
      // Zero DWORDs are ordinary SDMA NOPs, including each unused slot tail.
      std::array<std::array<uint32_t, kSdmaSlotWords>, kGraphsPerStream>
          upload_streams = {};
      std::array<std::array<uint32_t, kSdmaSlotWords>, kGraphsPerStream>
          download_streams = {};
      for (uint32_t graph = 0; graph < kGraphsPerStream; ++graph) {
        const uint32_t slot = graph % credits;
        const uint32_t generation = graph / credits + 1;
        const uint32_t tag = workload * kGraphsPerStream + graph + 1;
        const size_t record_base = graph * kPayloadWordCount;
        const size_t slot_base = slot * kPayloadWordCount;
        const size_t page = slot * kPageWordCount;
        const uint64_t record_offset = record_base * sizeof(uint32_t);
        const uint64_t slot_offset = slot_base * sizeof(uint32_t);
        const uint64_t progress =
            control.device_address + page * sizeof(uint32_t);
        Pm4CommandWriter pm4(pm4_streams[graph].data(), *pm4_profile_);
        SdmaCommandWriter upload(upload_streams[graph].data(),
                                 sdma_family_.format_features);
        SdmaCommandWriter download(download_streams[graph].data(),
                                   sdma_family_.format_features);

        // Compute owns the input until C; download owns the output until D.
        // Upload and compute wait for those previous-generation readers.
        upload.WaitMemory32(progress + kComputeByteOffset, generation - 1);
        if (sdma_operations[kUpload] &
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
          upload.AcquireFromSystem();
        }
        upload.CopyLinear(
            source.device_address + record_offset + kPayloadByteOffset,
            input.device_address + slot_offset + kPayloadByteOffset,
            kGridSize * sizeof(uint32_t));
        if (sdma_operations[kUpload] &
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
          upload.ReleaseToSystem();
        }
        upload.Fence32(progress + kUploadByteOffset, generation);

        pm4.WaitMemory32(progress + kUploadByteOffset, generation);
        pm4.WaitMemory32(progress + kDownloadByteOffset, generation - 1);
        pm4.SystemBarrier();
        pm4.BindCompute(program,
                        arguments.device_address + page * sizeof(uint32_t));
        pm4.Dispatch(program, kGridSize, 1, 1);
        pm4.ReleaseSystem32(progress + kComputeByteOffset, generation);
        ASSERT_LE(pm4.word_count(), kPm4SlotWords);
        while (pm4.word_count() < kPm4SlotWords) {
          pm4.PadToEightWords();
        }
        ASSERT_EQ(pm4.word_count(), kPm4SlotWords);

        download.WaitMemory32(progress + kComputeByteOffset, generation);
        if (sdma_operations[kDownload] &
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
          download.AcquireFromSystem();
        }
        download.CopyLinear(
            output.device_address + slot_offset + kPayloadByteOffset,
            readback.device_address + record_offset + kPayloadByteOffset,
            kGridSize * sizeof(uint32_t));
        if (sdma_operations[kDownload] &
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
          download.ReleaseToSystem();
        }
        download.Fence32(progress + kDownloadByteOffset, generation);
        ASSERT_LE(upload.word_count(), kSdmaSlotWords);
        ASSERT_LE(download.word_count(), kSdmaSlotWords);

        for (uint32_t word = 0; word < kGridSize; ++word) {
          const size_t record_word = record_base + kPayloadWordOffset + word;
          const size_t slot_word = slot_base + kPayloadWordOffset + word;
          const uint32_t value = static_cast<uint32_t>(
              UINT64_C(0xfffffff0) + uint64_t{word} * 0x01030507u +
              uint64_t{tag} * 0x11111111u);
          expected[kSource][record_word] = value;
          expected[kInput][slot_word] = value;
          if (word < kCounts[(workload + slot) % 2]) {
            expected[kOutput][slot_word] = static_cast<uint32_t>(
                uint64_t{value} * 3 + kAddends[(workload + slot) % 2]);
          }
          expected[kReadback][record_word] = expected[kOutput][slot_word];
        }
        expected_control[page + kUploadByteOffset / sizeof(uint32_t)] =
            generation;
        expected_control[page + kComputeByteOffset / sizeof(uint32_t)] =
            generation;
        expected_control[page + kDownloadByteOffset / sizeof(uint32_t)] =
            generation;
      }
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        std::memcpy(backings[owner].memory->host.pointer,
                    expected[owner].data(),
                    expected[owner].size() * sizeof(uint32_t));
      }
      for (uint32_t slot = 0; slot < credits; ++slot) {
        for (uint32_t word = 0; word < kGridSize; ++word) {
          const size_t offset =
              slot * kPayloadWordCount + kPayloadWordOffset + word;
          static_cast<uint32_t*>(input.host.pointer)[offset] =
              ~expected[kInput][offset];
          if (word < kCounts[(workload + slot) % 2]) {
            static_cast<uint32_t*>(output.host.pointer)[offset] =
                ~expected[kOutput][offset];
          }
        }
      }
      for (uint32_t graph = 0; graph < kGraphsPerStream; ++graph) {
        for (uint32_t word = 0; word < kGridSize; ++word) {
          const size_t offset =
              graph * kPayloadWordCount + kPayloadWordOffset + word;
          static_cast<uint32_t*>(readback.host.pointer)[offset] =
              ~expected[kReadback][offset];
        }
      }

      const auto append = [&](size_t queue, const auto& commands) {
        const uint64_t index_byte_length = kIndexByteLengths[queue];
        const uint64_t byte_offset =
            frontiers[queue] * index_byte_length % kRingByteLength;
        const uint64_t byte_length = commands.size() * sizeof(uint32_t);
        ASSERT_LE(byte_offset + byte_length, kRingByteLength);
        std::memcpy(reinterpret_cast<uint8_t*>(expected_rings[queue].data()) +
                        byte_offset,
                    commands.data(), byte_length);
        // Only this already-consumed span is overwritten in the live ring.
        std::memcpy(
            reinterpret_cast<uint8_t*>(queues[queue]->host.ring_address) +
                byte_offset,
            commands.data(), byte_length);
        frontiers[queue] += byte_length / index_byte_length;
      };
      for (uint32_t graph = 0; graph < kGraphsPerStream; ++graph) {
        ASSERT_NO_FATAL_FAILURE(append(kComputeQueue, pm4_streams[graph]));
      }
      if (separate) {
        for (uint32_t graph = 0; graph < kGraphsPerStream; ++graph) {
          ASSERT_NO_FATAL_FAILURE(append(kUploadQueue, upload_streams[graph]));
          ASSERT_NO_FATAL_FAILURE(
              append(kDownloadQueue, download_streams[graph]));
        }
      } else {
        const uint32_t group_size =
            transfer_queues == TransferQueues::kSharedPaired ? 1 : credits;
        for (uint32_t begin = 0; begin < kGraphsPerStream;
             begin += group_size) {
          for (uint32_t graph = begin; graph < begin + group_size; ++graph) {
            ASSERT_NO_FATAL_FAILURE(
                append(kUploadQueue, upload_streams[graph]));
          }
          for (uint32_t graph = begin; graph < begin + group_size; ++graph) {
            ASSERT_NO_FATAL_FAILURE(
                append(kUploadQueue, download_streams[graph]));
          }
        }
      }
      // Every command is resident before the first publication. The host
      // waits only for the final download; it never refills a live stream.
      ASSERT_NO_FATAL_FAILURE(
          queues[kComputeQueue]->PublishStream(frontiers[kComputeQueue]));
      if (separate) {
        ASSERT_NO_FATAL_FAILURE(
            queues[kDownloadQueue]->PublishStream(frontiers[kDownloadQueue]));
      }
      ASSERT_NO_FATAL_FAILURE(
          queues[kUploadQueue]->PublishStream(frontiers[kUploadQueue]));
      const auto last_download =
          reinterpret_cast<uintptr_t>(control.host.pointer) +
          ((kGraphsPerStream - 1) % credits) * kPageByteLength +
          kDownloadByteOffset;
      GpuWaitEqual<uint32_t>(last_download, kGraphsPerStream / credits);
      // Retain every graph's decisive output before independent owner joins.
      std::memcpy(observed[kReadback].data(), readback.host.pointer,
                  backings[kReadback].byte_length);
      for (uint32_t slot = 0; slot < credits; ++slot) {
        const uint64_t address =
            reinterpret_cast<uintptr_t>(control.host.pointer) +
            slot * kPageWordCount * sizeof(uint32_t);
        GpuWaitEqual<uint32_t>(address + kComputeByteOffset,
                               kGraphsPerStream / credits);
        GpuWaitEqual<uint32_t>(address + kUploadByteOffset,
                               kGraphsPerStream / credits);
      }
      for (size_t owner = kSource; owner <= kOutput; ++owner) {
        std::memcpy(observed[owner].data(),
                    backings[owner].memory->host.pointer,
                    backings[owner].byte_length);
      }
      std::memcpy(observed_arguments.data(), arguments.host.pointer,
                  sizeof(observed_arguments));
      std::memcpy(observed_control.data(), control.host.pointer,
                  sizeof(observed_control));
      std::memcpy(observed_code.data(), code.host.pointer,
                  sizeof(observed_code));
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        check_words(backings[owner].name, observed[owner], expected[owner]);
      }
      check_words("arguments", observed_arguments, expected_arguments);
      check_words("control", observed_control, expected_control);
      check_words("code", observed_code, expected_code);
      for (size_t queue = 0; queue < queues.size(); ++queue) {
        EXPECT_NO_FATAL_FAILURE(
            queues[queue]->WaitConsumed(api_, frontiers[queue]));
      }
      if (HasFailure()) {
        return;
      }
      for (size_t queue = 0; queue < queues.size(); ++queue) {
        SCOPED_TRACE(::testing::Message() << "queue=" << queue);
        std::memcpy(observed_rings[queue].data(),
                    reinterpret_cast<void*>(queues[queue]->host.ring_address),
                    kRingByteLength);
        check_words("ring", observed_rings[queue], expected_rings[queue]);
      }
      if (HasFailure()) {
        return;
      }
      completed_graphs += kGraphsPerStream;
      if ((stream + 1) % kStreamsPerCreditCount == 0) {
        const std::array<uint64_t, kQueueCount> advanced_bytes = {
            (frontiers[kComputeQueue] - credit_begin[kComputeQueue]) *
                sizeof(uint32_t),
            frontiers[kUploadQueue] - credit_begin[kUploadQueue],
            frontiers[kDownloadQueue] - credit_begin[kDownloadQueue]};
        EXPECT_EQ(advanced_bytes[kComputeQueue],
                  kStreamsPerCreditCount * kGraphsPerStream * kPm4SlotBytes);
        EXPECT_EQ(advanced_bytes[kUploadQueue],
                  kStreamsPerCreditCount * kGraphsPerStream * kSdmaSlotBytes *
                      (separate ? 1 : 2));
        EXPECT_EQ(advanced_bytes[kDownloadQueue],
                  separate ? kStreamsPerCreditCount * kGraphsPerStream *
                                 kSdmaSlotBytes
                           : 0);
        for (size_t queue = 0; queue < queues.size(); ++queue) {
          if (queue != kDownloadQueue || separate) {
            EXPECT_GT(advanced_bytes[queue], kRingByteLength);
          }
        }
      }
    }
    RecordProperty("completed_streams", kStreamCount);
    RecordProperty("completed_graphs", std::to_string(completed_graphs));
    for (size_t queue = 0; queue < queues.size(); ++queue) {
      amdf_user_queue_status_t status = {
          .type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS,
          .structure_size = sizeof(status)};
      ASSERT_EQ(api_->user_queue_query_status(queues[queue]->queue, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      EXPECT_EQ(status.producer_index, status.consumed_index);
      EXPECT_EQ(status.producer_index, frontiers[queue]);
      RecordProperty(queue == kComputeQueue  ? "stream_pm4_frontier"
                     : queue == kUploadQueue ? "stream_upload_frontier"
                                             : "stream_download_frontier",
                     std::to_string(status.producer_index));
    }
  }
};

TEST_F(Pm4SdmaFiniteStreamTest, SeparateTransfersRetireEveryGraph) {
  RunCoherentStreams(TransferQueues::kSeparate);
}

TEST_F(Pm4SdmaFiniteStreamTest, SharedPairedTransfersRetireEveryGraph) {
  RunCoherentStreams(TransferQueues::kSharedPaired);
}

TEST_F(Pm4SdmaFiniteStreamTest, SharedCreditGroupedTransfersRetireEveryGraph) {
  RunCoherentStreams(TransferQueues::kSharedCreditGrouped);
}

}  // namespace
