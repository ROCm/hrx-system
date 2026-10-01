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
#include <thread>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/recipes/pm4_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/user_queue.h"

namespace {

constexpr uint32_t kMaximumCredits = 8;
constexpr std::array<uint32_t, 4> kCreditCounts = {1, 2, 4, 8};

// Separate coherent control lines carry each slot's ownership transfers:
// H -> U permits source refill, U -> C permits input reuse, C -> D permits
// output reuse, and D -> R permits readback overwrite after its CPU snapshot.
// Every value is a per-slot generation, separate from native ring positions.
constexpr uint32_t kUploadCompleteByteOffset = 0;
constexpr uint32_t kComputeCompleteByteOffset = 64;
constexpr uint32_t kDownloadCompleteByteOffset = 128;
constexpr uint32_t kSourceReadyByteOffset = 192;
constexpr uint32_t kReadbackConsumedByteOffset = 256;
constexpr std::array<uint32_t, 5> kControlByteOffsets = {
    kUploadCompleteByteOffset, kComputeCompleteByteOffset,
    kDownloadCompleteByteOffset, kSourceReadyByteOffset,
    kReadbackConsumedByteOffset};

enum class AcquireMode { kFullBarrier, kOrderedData };

enum class ProducerClosure { kFullStream, kPendingIngress, kUnusedInput };

struct StreamCase {
  // Number of reusable payload slots in this stream.
  uint32_t credits;
  // Complete three-queue graph groups accepted before closing the producer.
  uint32_t graph_limit;
};

// Tokens are the low 32 bits of a logical generation. Reader acknowledgments
// keep each value stable until its consumers advance, including across zero.
uint32_t CreditToken(uint32_t initial_token, uint32_t generation) {
  return static_cast<uint32_t>(uint64_t{initial_token} + generation);
}

uint32_t SlotGeneration(uint32_t graph_count, uint32_t slot, uint32_t credits) {
  return graph_count / credits + (slot < graph_count % credits);
}

class Pm4SdmaStreamingTest : public Pm4SdmaTest {
 protected:
  Pm4SdmaStreamingTest() : Pm4SdmaTest(AMDF_QUEUE_PUBLICATION_MODE_USER) {}

  // Besides the four main payload edges, preserve the CPU seed and diagnostic
  // paths. Code publication additionally needs the qualified instruction-cache
  // operation in the first SystemBarrier below.
  static constexpr std::array<Edge, 15> kEdges = {{
      {"source_host_to_sdma", kSource, Site::kHost, Site::kSdma, kUpload},
      {"input_sdma_to_pm4", kInput, Site::kSdma, Site::kPm4, kUpload},
      {"output_pm4_to_sdma", kOutput, Site::kPm4, Site::kSdma, kDownload},
      {"readback_sdma_to_host", kReadback, Site::kSdma, Site::kHost, kDownload},
      {"input_seed_host_to_pm4", kInput, Site::kHost, Site::kPm4, kUpload},
      {"output_seed_host_to_sdma", kOutput, Site::kHost, Site::kSdma,
       kDownload},
      {"control_host_to_download", kControl, Site::kHost, Site::kSdma,
       kDownload},
      {"control_upload_to_host", kControl, Site::kSdma, Site::kHost, kUpload},
      {"input_sdma_to_host", kInput, Site::kSdma, Site::kHost, kUpload},
      {"output_pm4_to_host", kOutput, Site::kPm4, Site::kHost, kDownload},
      {"arguments_host_to_pm4", kArguments, Site::kHost, Site::kPm4, kUpload},
      {"code_host_to_pm4", kCode, Site::kHost, Site::kPm4, kUpload},
      {"control_host_to_sdma", kControl, Site::kHost, Site::kSdma, kUpload},
      {"control_pm4_to_sdma", kControl, Site::kPm4, Site::kSdma, kDownload},
      {"control_sdma_to_host", kControl, Site::kSdma, Site::kHost, kDownload},
  }};

  void RunStreaming(AcquireMode acquire_mode, uint32_t initial_token = 0,
                    ProducerClosure closure = ProducerClosure::kFullStream) {
    const bool ordered = acquire_mode == AcquireMode::kOrderedData;
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
    constexpr uint32_t kPayloadOffset = 16;
    constexpr uint32_t kControlGuard = 0x68d329b7u;
    constexpr std::array<uint32_t, 4> kGuards = {0x759bf13du, 0x26a4e8c3u,
                                                 0x93b57fd1u, 0x4cd218a7u};
    constexpr std::array<uint32_t, 2> kCounts = {1003, 997};
    constexpr std::array<uint32_t, 2> kAddends = {7, 0x80000023u};

    static_assert(sizeof(kernels::transform::Arguments) == 32);
    std::array<Backing, kBackingCount> backings = {{
        {"source", kMaximumCredits * 8192, AMDF_MEMORY_ACCESS_READ},
        {"input", kMaximumCredits * 8192, kReadWrite},
        {"output", kMaximumCredits * 8192, kReadWrite},
        {"readback", kMaximumCredits * 8192, kReadWrite},
        {"arguments", kMaximumCredits * 4096, AMDF_MEMORY_ACCESS_READ},
        {"control", kMaximumCredits * 4096, kReadWrite},
        {"code", 4096, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE},
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
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&pm4_queue));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &upload_queue));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &download_queue));
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
    const std::array<GpuUserQueue*, 3> queues = {pm4_queue, upload_queue,
                                                 download_queue};
    const std::array<uint64_t, 3> slot_bytes = {kPm4SlotBytes, kSdmaSlotBytes,
                                                kSdmaSlotBytes};
    const std::array<uint64_t, 3> slot_units = {kPm4SlotWords, kSdmaSlotBytes,
                                                kSdmaSlotBytes};
    const std::array<uint64_t, 3> capacity_units = {
        pm4_capacity / sizeof(uint32_t), upload_capacity, download_capacity};
    // Raw PM4 RPTR wraps in DWORD units. All command slots are aligned so
    // no packet straddles a native ring wrap. Keep the witness finite while
    // exercising every discovered ring, independently of provider defaults.
    ASSERT_EQ(capacity_units[0] & (capacity_units[0] - 1), 0u);
    uint64_t required_graph_count = 64;
    for (size_t queue = 0; queue < queues.size(); ++queue) {
      required_graph_count =
          std::max(required_graph_count,
                   3 * (capacity_units[queue] / slot_units[queue]));
    }
    ASSERT_LE(required_graph_count, UINT32_MAX - (kMaximumCredits - 1));
    const uint32_t graph_count =
        (static_cast<uint32_t>(required_graph_count) + kMaximumCredits - 1) &
        ~(kMaximumCredits - 1);
    RecordProperty("stream_graphs", graph_count);
    RecordProperty("stream_credit_counts", "1,2,4,8");
    RecordProperty("stream_initial_token", std::to_string(initial_token));
    RecordProperty("stream_queue_count", 3);
    RecordProperty("stream_publication_order", "pm4,download,upload");
    RecordProperty("stream_pm4_slot_dwords", kPm4SlotWords);
    RecordProperty("stream_sdma_slot_dwords", kSdmaSlotWords);
    RecordProperty("stream_pm4_capacity_bytes", std::to_string(pm4_capacity));
    RecordProperty("stream_upload_capacity_bytes",
                   std::to_string(upload_capacity));
    RecordProperty("stream_download_capacity_bytes",
                   std::to_string(download_capacity));

    std::vector<StreamCase> stream_cases;
    for (const uint32_t credits : kCreditCounts) {
      if (closure == ProducerClosure::kFullStream) {
        stream_cases.push_back({credits, graph_count});
      } else {
        std::array<uint32_t, 6> limits = {
            0, 1, credits - 1, credits, credits + 1, graph_count - 1};
        std::sort(limits.begin(), limits.end());
        const auto end = std::unique(limits.begin(), limits.end());
        for (auto limit = limits.begin(); limit != end; ++limit) {
          stream_cases.push_back({credits, *limit});
        }
      }
    }

    RecordProperty("host_staging_slots_allocated", kMaximumCredits);
    RecordProperty("host_ingress_bytes_per_graph",
                   kGridSize * sizeof(uint32_t));
    RecordProperty("host_egress_bytes_per_graph", kGridSize * sizeof(uint32_t));
    RecordProperty("host_source_ready_byte_offset", kSourceReadyByteOffset);
    RecordProperty("host_readback_released_byte_offset",
                   kReadbackConsumedByteOffset);
    // CPU-only transcripts are never attached or visible to the native queues.
    const size_t record_word_count = size_t{graph_count} * kGridSize;
    std::vector<uint32_t> source_records(record_word_count);
    std::vector<uint32_t> expected_records(record_word_count);
    std::vector<uint32_t> observed_records(record_word_count);
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
    uint64_t prepared_inputs = 0;
    bool program_published = false;

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
    for (uint32_t stream = 0; stream < stream_cases.size(); ++stream) {
      const uint32_t credits = stream_cases[stream].credits;
      const uint32_t graph_limit = stream_cases[stream].graph_limit;
      SCOPED_TRACE(::testing::Message()
                   << "stream=" << stream << " credits=" << credits
                   << " accepted=" << graph_limit);
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        std::fill(expected[owner].begin(), expected[owner].end(),
                  kGuards[owner] ^ (stream * 0x1020304u));
        std::memcpy(backings[owner].memory->host.pointer,
                    expected[owner].data(),
                    expected[owner].size() * sizeof(uint32_t));
      }
      std::fill(expected_records.begin(), expected_records.end(), 0x2badb007u);
      expected_control.fill(kControlGuard);
      std::fill(observed_records.begin(), observed_records.end(), 0x2badb007u);
      for (uint32_t slot = 0; slot < kMaximumCredits; ++slot) {
        const size_t page = slot * kPageWordCount;
        for (const uint32_t byte_offset : kControlByteOffsets) {
          expected_control[page + byte_offset / sizeof(uint32_t)] =
              initial_token;
        }
        std::fill_n(expected_arguments.data() + page, kPageWordCount,
                    UINT32_C(0x713ace09) ^ (stream * kMaximumCredits + slot));
        if (slot >= credits) {
          continue;
        }
        const uint64_t payload_offset =
            slot * kPayloadWordCount * sizeof(uint32_t);
        const kernels::transform::Arguments payload = {
            input.device_address + payload_offset + 64,
            output.device_address + payload_offset + 64,
            kCounts[(stream + slot) % 2], kAddends[(stream + slot) % 2]};
        std::memcpy(expected_arguments.data() + page, &payload,
                    kernel.arguments.byte_length);
      }
      // The previous stream's complete native retirement precedes this reset.
      // While live, the CPU writes only H/R; GPU queues own U/C/D.
      std::memcpy(control.host.pointer, expected_control.data(),
                  sizeof(expected_control));
      std::memcpy(arguments.host.pointer, expected_arguments.data(),
                  sizeof(expected_arguments));

      std::vector<std::array<uint32_t, kPm4SlotWords>> pm4_streams(graph_count);
      // Zero DWORDs are ordinary SDMA NOPs, including each unused slot tail.
      std::vector<std::array<uint32_t, kSdmaSlotWords>> upload_streams(
          graph_count);
      std::vector<std::array<uint32_t, kSdmaSlotWords>> download_streams(
          graph_count);
      for (uint32_t graph = 0; graph < graph_count; ++graph) {
        const uint32_t slot = graph % credits;
        const uint32_t generation = graph / credits + 1;
        const uint32_t token = CreditToken(initial_token, generation);
        const uint32_t previous_token =
            CreditToken(initial_token, generation - 1);
        const uint64_t tag = uint64_t{stream} * graph_count + graph + 1;
        const size_t record_base = size_t{graph} * kGridSize;
        const size_t slot_base = slot * kPayloadWordCount;
        const size_t page = slot * kPageWordCount;
        const uint64_t slot_offset = slot_base * sizeof(uint32_t);
        const uint64_t progress =
            control.device_address + page * sizeof(uint32_t);
        Pm4CommandWriter pm4(pm4_streams[graph].data(), *pm4_profile_);
        SdmaCommandWriter upload(upload_streams[graph].data(),
                                 sdma_family_.format_features);
        SdmaCommandWriter download(download_streams[graph].data(),
                                   sdma_family_.format_features);

        upload.WaitMemory32(progress + kSourceReadyByteOffset, token);
        upload.WaitMemory32(progress + kComputeCompleteByteOffset,
                            previous_token);
        if (sdma_operations[kUpload] &
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
          upload.AcquireFromSystem();
        }
        upload.CopyLinear(source.device_address + slot_offset + 64,
                          input.device_address + slot_offset + 64,
                          kGridSize * sizeof(uint32_t));
        if (sdma_operations[kUpload] &
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
          upload.ReleaseToSystem();
        }
        upload.Fence32(progress + kUploadCompleteByteOffset, token);

        pm4.WaitMemory32(progress + kUploadCompleteByteOffset, token);
        pm4.WaitMemory32(progress + kDownloadCompleteByteOffset,
                         previous_token);
        if (!ordered || (!program_published && graph == 0)) {
          pm4.SystemBarrier();
        } else {
          if (graph != 0) {
            const uint32_t previous = graph - 1;
            const uint64_t previous_address =
                control.device_address +
                (previous % credits) * kPageWordCount * sizeof(uint32_t) +
                kComputeCompleteByteOffset;
            pm4.WaitMemory32(
                previous_address,
                CreditToken(initial_token, previous / credits + 1));
          }
          // A prior stream is fully drained by the host. Within this stream,
          // the explicit previous-C wait joins shader users before rebinding.
          pm4.AcquireFromSystem();
        }
        pm4.BindCompute(program,
                        arguments.device_address + page * sizeof(uint32_t));
        pm4.DispatchWave32(kGridSize, 1, 1);
        pm4.ReleaseSystem32(progress + kComputeCompleteByteOffset, token);
        ASSERT_LE(pm4.word_count(), kPm4SlotWords);
        while (pm4.word_count() < kPm4SlotWords) {
          pm4.PadToEightWords();
        }
        ASSERT_EQ(pm4.word_count(), kPm4SlotWords);

        download.WaitMemory32(progress + kComputeCompleteByteOffset, token);
        download.WaitMemory32(progress + kReadbackConsumedByteOffset,
                              previous_token);
        if (sdma_operations[kDownload] &
            AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
          download.AcquireFromSystem();
        }
        download.CopyLinear(output.device_address + slot_offset + 64,
                            readback.device_address + slot_offset + 64,
                            kGridSize * sizeof(uint32_t));
        if (sdma_operations[kDownload] &
            AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
          download.ReleaseToSystem();
        }
        download.Fence32(progress + kDownloadCompleteByteOffset, token);
        ASSERT_LE(upload.word_count(), kSdmaSlotWords);
        ASSERT_LE(download.word_count(), kSdmaSlotWords);

        for (uint32_t word = 0; word < kGridSize; ++word) {
          const size_t record_word = record_base + word;
          const size_t slot_word = slot_base + kPayloadOffset + word;
          const uint32_t value = static_cast<uint32_t>(
              UINT64_C(0xfffffff0) + uint64_t{word} * 0x01030507u +
              uint64_t{tag} * 0x11111111u);
          source_records[record_word] = value;
          if (graph >= graph_limit) {
            continue;
          }
          expected[kInput][slot_word] = value;
          if (word < kCounts[(stream + slot) % 2]) {
            expected[kOutput][slot_word] = static_cast<uint32_t>(
                uint64_t{value} * 3 + kAddends[(stream + slot) % 2]);
          }
          expected[kReadback][slot_word] = expected[kOutput][slot_word];
          expected_records[record_word] = expected[kOutput][slot_word];
        }
      }

      uint32_t input_published = 0;
      uint32_t accepted_graphs = 0;
      uint32_t ingress_limit = graph_limit;
      if (closure == ProducerClosure::kPendingIngress && graph_limit != 0) {
        // The final accepted graph waits for input supplied after closure.
        --ingress_limit;
      } else if (closure == ProducerClosure::kUnusedInput) {
        // One prepared record has no submitted native consumer.
        ++ingress_limit;
      }
      uint32_t output_consumed = 0;
      const uint64_t host_control =
          reinterpret_cast<uintptr_t>(control.host.pointer);
      const auto service_host = [&] {
        bool progressed = false;
        if (output_consumed < accepted_graphs) {
          const uint32_t slot = output_consumed % credits;
          const uint32_t generation = output_consumed / credits + 1;
          const uint32_t token = CreditToken(initial_token, generation);
          const uint64_t address =
              host_control + slot * kPageWordCount * sizeof(uint32_t);
          if (GpuLoadAcquire<uint32_t>(address + kDownloadCompleteByteOffset) ==
              token) {
            // The next download cannot overwrite this slot before R. Snapshot
            // the entire payload, including the shader's untouched tail.
            std::memcpy(
                observed_records.data() + size_t{output_consumed} * kGridSize,
                static_cast<uint32_t*>(readback.host.pointer) +
                    slot * kPayloadWordCount + kPayloadOffset,
                kGridSize * sizeof(uint32_t));
            GpuStoreRelease(address + kReadbackConsumedByteOffset, token);
            ++output_consumed;
            progressed = true;
          }
        }
        if (input_published < ingress_limit) {
          const uint32_t slot = input_published % credits;
          const uint32_t generation = input_published / credits + 1;
          const uint32_t token = CreditToken(initial_token, generation);
          const uint32_t previous_token =
              CreditToken(initial_token, generation - 1);
          const uint64_t address =
              host_control + slot * kPageWordCount * sizeof(uint32_t);
          if (GpuLoadAcquire<uint32_t>(address + kUploadCompleteByteOffset) ==
              previous_token) {
            // U retires the previous SDMA source read. The next source read
            // waits on H, independently of the shader/input and output owners.
            std::memcpy(
                static_cast<uint32_t*>(source.host.pointer) +
                    slot * kPayloadWordCount + kPayloadOffset,
                source_records.data() + size_t{input_published} * kGridSize,
                kGridSize * sizeof(uint32_t));
            GpuStoreRelease(address + kSourceReadyByteOffset, token);
            ++input_published;
            progressed = true;
          }
        }
        return progressed;
      };
      const auto drain_outputs = [&](uint32_t count) {
        while (output_consumed < count) {
          if (!service_host()) {
            std::this_thread::yield();
          }
        }
      };

      const auto submit_stream = [&] {
        for (uint32_t graph = 0; graph < graph_limit; ++graph) {
          service_host();
          const uint64_t begin_graph = completed_graphs + graph;
          const uint64_t end_graph = begin_graph + 1;
          // Earlier graph publications include all three engines. Waiting for
          // command capacity never hides an earlier unpublished producer.
          for (size_t queue = 0; queue < queues.size(); ++queue) {
            const uint64_t published = begin_graph * slot_units[queue];
            const uint64_t end = end_graph * slot_units[queue];
            // Format-1 PM4 exposes ring-relative RPTR; SDMA exposes a
            // monotonic byte index. This single producer retains stable W
            // while expanding PM4's native value into its current window.
            const uint64_t mask =
                queue == 0 ? capacity_units[queue] - 1 : UINT64_MAX;
            while (true) {
              const uint64_t native_read = GpuLoadAcquire<uint64_t>(
                  queues[queue]->host.read_index_address);
              const uint64_t consumed =
                  published - ((published - native_read) & mask);
              if (end - consumed < capacity_units[queue]) {
                break;
              }
              if (!service_host()) {
                std::this_thread::yield();
              }
            }
          }
          const std::array<const uint32_t*, 3> commands = {
              pm4_streams[graph].data(), upload_streams[graph].data(),
              download_streams[graph].data()};
          for (size_t queue = 0; queue < queues.size(); ++queue) {
            auto* ring =
                reinterpret_cast<uint8_t*>(queues[queue]->host.ring_address);
            const uint64_t offset = begin_graph * slot_bytes[queue] %
                                    queues[queue]->host.ring_byte_length;
            std::memcpy(ring + offset, commands[queue], slot_bytes[queue]);
          }
          for (size_t queue : {0u, 2u, 1u}) {
            const uint64_t end = end_graph * slot_units[queue];
            ASSERT_NO_FATAL_FAILURE(queues[queue]->PublishStream(end));
          }
          accepted_graphs = graph + 1;
        }
      };
      ASSERT_NO_FATAL_FAILURE(submit_stream());
      if (closure == ProducerClosure::kUnusedInput) {
        while (input_published < ingress_limit) {
          if (!service_host()) {
            std::this_thread::yield();
          }
        }
      }
      const uint32_t staged_at_close = input_published;
      const uint32_t consumed_at_close = output_consumed;
      EXPECT_EQ(accepted_graphs, graph_limit);
      if (closure == ProducerClosure::kPendingIngress && accepted_graphs != 0) {
        EXPECT_LT(staged_at_close, accepted_graphs);
        EXPECT_LT(consumed_at_close, accepted_graphs);
      } else if (closure == ProducerClosure::kUnusedInput) {
        EXPECT_EQ(staged_at_close, accepted_graphs + 1);
      }
      // Closing acceptance preserves ingress and egress for accepted groups.
      // Extra preparation has no native consumer and needs no completion.
      ingress_limit = accepted_graphs;
      drain_outputs(accepted_graphs);
      EXPECT_EQ(input_published, std::max(staged_at_close, accepted_graphs));
      EXPECT_EQ(output_consumed, accepted_graphs);
      for (uint32_t graph = 0; graph < input_published; ++graph) {
        const uint32_t slot = graph % credits;
        std::copy_n(source_records.data() + size_t{graph} * kGridSize,
                    kGridSize,
                    expected[kSource].data() + slot * kPayloadWordCount +
                        kPayloadOffset);
      }
      // Each transcript copy preceded its R acknowledgement and all later
      // reuse. Compare before independent C/U joins or native retirement.
      check_words("retained_cpu_results", observed_records, expected_records);
      std::memcpy(observed[kReadback].data(), readback.host.pointer,
                  backings[kReadback].byte_length);
      for (uint32_t slot = 0; slot < credits; ++slot) {
        const uint32_t generation =
            SlotGeneration(accepted_graphs, slot, credits);
        const uint32_t token = CreditToken(initial_token, generation);
        const uint64_t address =
            reinterpret_cast<uintptr_t>(control.host.pointer) +
            slot * kPageWordCount * sizeof(uint32_t);
        GpuWaitEqual<uint32_t>(address + kComputeCompleteByteOffset, token);
        GpuWaitEqual<uint32_t>(address + kUploadCompleteByteOffset, token);
        for (const uint32_t byte_offset : kControlByteOffsets) {
          const uint32_t control_generation =
              byte_offset == kSourceReadyByteOffset
                  ? SlotGeneration(input_published, slot, credits)
                  : generation;
          expected_control[slot * kPageWordCount +
                           byte_offset / sizeof(uint32_t)] =
              CreditToken(initial_token, control_generation);
        }
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
        EXPECT_NO_FATAL_FAILURE(queues[queue]->WaitConsumed(
            api_, (completed_graphs + accepted_graphs) * slot_units[queue]));
      }
      if (HasFailure()) {
        return;
      }
      completed_graphs += accepted_graphs;
      prepared_inputs += input_published;
      program_published |= accepted_graphs != 0;
      const std::string prefix = "stream_" + std::to_string(stream);
      RecordProperty(prefix + "_credits", credits);
      RecordProperty(prefix + "_completed_graphs", accepted_graphs);
      RecordProperty(prefix + "_staged_at_close", staged_at_close);
      RecordProperty(prefix + "_consumed_at_close", consumed_at_close);
      RecordProperty(prefix + "_prepared_inputs", input_published);
      std::string generations;
      std::string tokens;
      for (uint32_t slot = 0; slot < credits; ++slot) {
        const uint32_t generation =
            SlotGeneration(accepted_graphs, slot, credits);
        if (slot != 0) {
          generations += ",";
          tokens += ",";
        }
        generations += std::to_string(generation);
        tokens += std::to_string(CreditToken(initial_token, generation));
      }
      RecordProperty(prefix + "_slot_generations", generations);
      RecordProperty(prefix + "_slot_tokens", tokens);
    }
    RecordProperty("completed_streams", stream_cases.size());
    RecordProperty("completed_graphs", std::to_string(completed_graphs));
    RecordProperty("host_source_publications", std::to_string(prepared_inputs));
    RecordProperty("host_readback_consumptions",
                   std::to_string(completed_graphs));
    for (size_t queue = 0; queue < queues.size(); ++queue) {
      amdf_user_queue_status_t status = {};
      status.type = AMDF_STRUCTURE_TYPE_USER_QUEUE_STATUS;
      status.structure_size = sizeof(status);
      ASSERT_EQ(api_->user_queue_query_status(queues[queue]->queue, &status),
                AMDF_STATUS_OK);
      EXPECT_EQ(status.state, AMDF_QUEUE_STATE_ACTIVE);
      EXPECT_EQ(status.terminal_status, AMDF_STATUS_OK);
      EXPECT_EQ(status.producer_index, status.consumed_index);
      EXPECT_EQ(status.producer_index, completed_graphs * slot_units[queue]);
      RecordProperty(queue == 0   ? "stream_pm4_frontier"
                     : queue == 1 ? "stream_upload_frontier"
                                  : "stream_download_frontier",
                     std::to_string(status.producer_index));
    }
    RecordProperty("stream_acquire_mode",
                   ordered ? "ordered_data" : "full_barrier");
  }
};

TEST_F(Pm4SdmaStreamingTest, CoherentHostStreaming) {
  RunStreaming(AcquireMode::kFullBarrier);
}

TEST_F(Pm4SdmaStreamingTest, OrderedDataAcquireHostStreaming) {
  RunStreaming(AcquireMode::kOrderedData);
}

TEST_F(Pm4SdmaStreamingTest, CreditTokensCrossHighBit) {
  RunStreaming(AcquireMode::kFullBarrier, UINT32_C(0x7ffffffe));
}

TEST_F(Pm4SdmaStreamingTest, OrderedDataAcquireCreditTokensCrossHighBit) {
  RunStreaming(AcquireMode::kOrderedData, UINT32_C(0x7ffffffe));
}

TEST_F(Pm4SdmaStreamingTest, CreditTokensWrap) {
  RunStreaming(AcquireMode::kFullBarrier, UINT32_C(0xfffffffe));
}

TEST_F(Pm4SdmaStreamingTest, OrderedDataAcquireCreditTokensWrap) {
  RunStreaming(AcquireMode::kOrderedData, UINT32_C(0xfffffffe));
}

TEST_F(Pm4SdmaStreamingTest, ProducerClosureWithPendingIngress) {
  RunStreaming(AcquireMode::kFullBarrier, 0, ProducerClosure::kPendingIngress);
}

TEST_F(Pm4SdmaStreamingTest,
       OrderedDataAcquireProducerClosureWithPendingIngress) {
  RunStreaming(AcquireMode::kOrderedData, 0, ProducerClosure::kPendingIngress);
}

TEST_F(Pm4SdmaStreamingTest, ProducerClosureWithUnusedInput) {
  RunStreaming(AcquireMode::kFullBarrier, 0, ProducerClosure::kUnusedInput);
}

TEST_F(Pm4SdmaStreamingTest, OrderedDataAcquireProducerClosureWithUnusedInput) {
  RunStreaming(AcquireMode::kOrderedData, 0, ProducerClosure::kUnusedInput);
}

}  // namespace
