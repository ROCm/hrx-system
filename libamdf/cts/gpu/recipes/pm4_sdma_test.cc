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

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/recipes/pm4_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

enum class AcquireMode { kFullBarrier, kOrderedData };
enum class TransferQueues { kShared, kSeparate };

class Pm4SdmaRecipeTest : public Pm4SdmaTest {
 protected:
  // Besides the four main payload edges, preserve the CPU seed and diagnostic
  // paths. Code publication additionally needs the qualified instruction-cache
  // operation in the first SystemBarrier below.
  static constexpr std::array<Edge, 13> kEdges = {{
      {"source_host_to_sdma", kSource, Site::kHost, Site::kSdma, kUpload},
      {"input_sdma_to_pm4", kInput, Site::kSdma, Site::kPm4, kUpload},
      {"output_pm4_to_sdma", kOutput, Site::kPm4, Site::kSdma, kDownload},
      {"readback_sdma_to_host", kReadback, Site::kSdma, Site::kHost, kDownload},
      {"input_seed_host_to_pm4", kInput, Site::kHost, Site::kPm4, kUpload},
      {"output_seed_host_to_sdma", kOutput, Site::kHost, Site::kSdma,
       kDownload},
      {"input_sdma_to_host", kInput, Site::kSdma, Site::kHost, kUpload},
      {"output_pm4_to_host", kOutput, Site::kPm4, Site::kHost, kDownload},
      {"arguments_host_to_pm4", kArguments, Site::kHost, Site::kPm4, kUpload},
      {"code_host_to_pm4", kCode, Site::kHost, Site::kPm4, kUpload},
      {"control_host_to_sdma", kControl, Site::kHost, Site::kSdma, kUpload},
      {"control_pm4_to_sdma", kControl, Site::kPm4, Site::kSdma, kDownload},
      {"control_sdma_to_host", kControl, Site::kSdma, Site::kHost, kDownload},
  }};

  template <uint32_t kGraphCount = 1>
  void RunCoherentHandoff(
      PairQuery query_kind,
      AcquireMode acquire_mode = AcquireMode::kFullBarrier,
      TransferQueues transfer_queues = TransferQueues::kShared,
      amdf_cache_operations_t required_sdma_operations = 0) {
    if (required_sdma_operations != 0 &&
        (sdma_family_.format_features &
         AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR) == 0) {
      GTEST_SKIP() << "explicit SDMA cache operations require USER_GCR";
    }
    ASSERT_EQ(sdma_family_.cache_operations & required_sdma_operations,
              required_sdma_operations);
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
    constexpr uint32_t kMaximumPm4WordsPerGraph = 64;
    constexpr uint32_t kMinimumSdmaWordsPerGraph = 28;
    constexpr uint32_t kMaximumSdmaWordsPerGraph =
        kMinimumSdmaWordsPerGraph + 20;
    constexpr uint32_t kEpochCount = 2;
    constexpr uint32_t kControlGuard = 0x68d329b7u;
    constexpr std::array<uint32_t, 4> kGuards = {0x759bf13du, 0x26a4e8c3u,
                                                 0x93b57fd1u, 0x4cd218a7u};
    constexpr std::array<uint32_t, kEpochCount> kCounts = {1003, 997};
    constexpr std::array<uint32_t, kEpochCount> kAddends = {7, 0x80000023u};

    static_assert(sizeof(kernels::transform::Arguments) == 32);
    std::array<Backing, kBackingCount> backings = {{
        {"source", kGraphCount * 8192, AMDF_MEMORY_ACCESS_READ},
        {"input", kGraphCount * 8192, kReadWrite},
        {"output", kGraphCount * 8192, kReadWrite},
        {"readback", kGraphCount * 8192, kReadWrite},
        {"arguments", kGraphCount * 4096, AMDF_MEMORY_ACCESS_READ},
        {"control", kGraphCount * 4096, kReadWrite},
        {"code", 4096, AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE},
    }};
    Pm4ComputeProgram program = {};
    // Explicit GCR cases retain their requested operations in both phases.
    std::array<amdf_cache_operations_t, kTransferPhaseCount> sdma_operations = {
        required_sdma_operations, required_sdma_operations};
    ASSERT_NO_FATAL_FAILURE(PrepareCoherentHandoff(
        kernel, query_kind, kEdges, backings, &program, &sdma_operations));
    auto& source = *backings[kSource].memory;
    auto& input = *backings[kInput].memory;
    auto& output = *backings[kOutput].memory;
    auto& readback = *backings[kReadback].memory;
    auto& arguments = *backings[kArguments].memory;
    auto& control = *backings[kControl].memory;
    auto& code = *backings[kCode].memory;
    ASSERT_EQ(arguments.device_address % kernel.arguments.alignment, 0u);
    ASSERT_EQ(control.device_address % 64, 0u);
    // Progress words use the selected coherent mapping and the ordinary
    // memory-wait, uncached SDMA FENCE32 and confirmed PM4 EOP protocols. The U
    // poll precedes the payload acquire, which cannot publish its predicate.
    // This separate packet/mapping contract is part of source admission; a
    // generic pair answer alone does not establish it. C's host observation
    // uses ReleaseSystem32's qualified completion contract. These are plain
    // DWORDs, with no HSA signal ABI and no host reset after this
    // initialization.
    std::array<uint32_t, kGraphCount * kPageWordCount> expected_control;
    expected_control.fill(kControlGuard);
    for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
      const size_t base = graph * kPageWordCount;
      expected_control[base] = expected_control[base + 16] =
          expected_control[base + 32] = 0;
    }
    std::memcpy(control.host.pointer, expected_control.data(),
                sizeof(expected_control));
    std::array<uint32_t, kPageWordCount> expected_code = {};
    std::memcpy(expected_code.data(), kernel.executable.words,
                kernel.executable.byte_length);

    GpuCommandQueue* pm4_queue = nullptr;
    GpuCommandQueue* upload_queue = nullptr;
    ASSERT_NO_FATAL_FAILURE(CreateQueue(&pm4_queue));
    ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &upload_queue));
    GpuCommandQueue* download_queue = upload_queue;
    if (transfer_queues == TransferQueues::kSeparate) {
      ASSERT_NO_FATAL_FAILURE(CreateQueue(sdma_family_, &download_queue));
      ASSERT_NE(upload_queue->native_handle(), download_queue->native_handle());
    }
    ASSERT_TRUE(amdf_device_id_is_equal(&pm4_queue->device_id(),
                                        &upload_queue->device_id()));
    ASSERT_TRUE(amdf_device_id_is_equal(&pm4_queue->device_id(),
                                        &download_queue->device_id()));
    ASSERT_NE(pm4_queue->native_handle(), upload_queue->native_handle());
    ASSERT_NE(pm4_queue->native_handle(), download_queue->native_handle());
    ASSERT_GT(pm4_queue->words().size(),
              kEpochCount * kGraphCount * kMaximumPm4WordsPerGraph);
    ASSERT_GT(upload_queue->words().size(),
              kEpochCount * kGraphCount * kMaximumSdmaWordsPerGraph);
    ASSERT_GT(download_queue->words().size(),
              kEpochCount * kGraphCount * kMaximumSdmaWordsPerGraph);
    Pm4CommandWriter pm4(pm4_queue->words().data(), *pm4_profile_);
    SdmaCommandWriter upload(upload_queue->words().data(),
                             sdma_family_.format_features);
    SdmaCommandWriter separate_download(download_queue->words().data(),
                                        sdma_family_.format_features);
    // Shared transfers append both phases through one writer and one owner.
    SdmaCommandWriter& download = transfer_queues == TransferQueues::kSeparate
                                      ? separate_download
                                      : upload;
    std::array<size_t, kEpochCount> pm4_frontiers;
    std::array<std::array<size_t, kTransferPhaseCount>, kEpochCount>
        transfer_frontiers;
    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t generation = epoch * kGraphCount + graph + 1;
        const uint64_t payload_offset = graph * 8192 + 64;
        const uint64_t page_offset = graph * 4096;
        const uint64_t progress_address = control.device_address + page_offset;
        pm4.WaitMemory32(progress_address, generation);
        if (acquire_mode == AcquireMode::kFullBarrier || generation == 1) {
          // Queried GLOBAL data acquire and cold code publication.
          pm4.SystemBarrier();
        } else {
          if (transfer_queues == TransferQueues::kSeparate) {
            // Upload can run ahead of compute. Its marker does not join the
            // previous shader, so wait for that shader's confirmed EOP before
            // acquiring data or changing compute registers.
            const uint32_t previous_graph =
                (graph + kGraphCount - 1) % kGraphCount;
            pm4.WaitMemory32(
                control.device_address + previous_graph * 4096 + 64,
                generation - 1);
          }
          // Shared SDMA reaches U_i only after observing C_(i-1). Separate
          // transfers use the explicit join above. Code remains immutable
          // after its initial publication in both layouts.
          pm4.AcquireFromSystem();
        }
        pm4.BindCompute(program, arguments.device_address + page_offset);
        pm4.DispatchWave32(kGridSize, 1, 1);
        pm4.ReleaseSystem32(progress_address + 64, generation);
        pm4.PadToEightWords();
        if ((sdma_operations[kUpload] &
             AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
          upload.AcquireFromSystem();
        }
        upload.CopyLinear(source.device_address + payload_offset,
                          input.device_address + payload_offset,
                          kGridSize * sizeof(uint32_t));
        if ((sdma_operations[kUpload] &
             AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
          upload.ReleaseToSystem();
        }
        upload.Fence32(progress_address, generation);
        download.WaitMemory32(progress_address + 64, generation);
        if ((sdma_operations[kDownload] &
             AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) != 0) {
          download.AcquireFromSystem();
        }
        download.CopyLinear(output.device_address + payload_offset,
                            readback.device_address + payload_offset,
                            kGridSize * sizeof(uint32_t));
        if ((sdma_operations[kDownload] &
             AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) != 0) {
          download.ReleaseToSystem();
        }
        download.Fence32(progress_address + 128, generation);
        ASSERT_LE(pm4.word_count(), generation * kMaximumPm4WordsPerGraph);
        const size_t transfer_word_count =
            upload.word_count() + (transfer_queues == TransferQueues::kSeparate
                                       ? download.word_count()
                                       : 0);
        ASSERT_GE(transfer_word_count, generation * kMinimumSdmaWordsPerGraph);
        ASSERT_LE(transfer_word_count, generation * kMaximumSdmaWordsPerGraph);
      }
      pm4_frontiers[epoch] = pm4.word_count();
      transfer_frontiers[epoch] = {upload.word_count(), download.word_count()};
    }
    // All streams are resident before publication. PM4's native index uses
    // DWORDs; SDMA's uses bytes. The command owners publish word frontiers.
    // No stream wraps or exhausts its storage.
    RecordProperty("pm4_sdma_pm4_family", family_.ordinal);
    RecordProperty("pm4_sdma_sdma_family", sdma_family_.ordinal);
    RecordProperty("pm4_sdma_sdma_format_features",
                   std::to_string(sdma_family_.format_features));
    RecordProperty("pm4_sdma_pm4_capacity_dwords",
                   std::to_string(pm4_queue->words().size()));
    RecordProperty("pm4_sdma_sdma_capacity_bytes",
                   std::to_string(upload_queue->words().size_bytes()));
    RecordProperty("pm4_sdma_download_capacity_bytes",
                   std::to_string(download_queue->words().size_bytes()));
    RecordProperty(
        "pm4_sdma_transfer_queues",
        transfer_queues == TransferQueues::kShared ? "shared" : "separate");
    RecordProperty("pm4_sdma_payload_byte_offset", 64);
    RecordProperty("pm4_sdma_copy_byte_length", kGridSize * sizeof(uint32_t));
    RecordProperty("pm4_sdma_grid_size", kGridSize);
    RecordProperty("pm4_sdma_workgroup_size", kernel.workgroup_size());
    RecordProperty("pm4_sdma_kernarg_byte_length",
                   kernel.arguments.byte_length);
    RecordProperty("pm4_sdma_host_cacheability",
                   AMDF_HOST_CACHEABILITY_WRITE_BACK);
    RecordProperty("pm4_sdma_control_offsets", "0,64,128");
    RecordProperty("pm4_sdma_publication_order",
                   transfer_queues == TransferQueues::kShared
                       ? "pm4,sdma"
                       : "pm4,download,upload");

    std::array<std::array<uint32_t, kGraphCount * kPayloadWordCount>, 4>
        expected;
    std::array<std::array<uint32_t, kGraphCount * kPayloadWordCount>, 4>
        observed;
    std::array<uint32_t, kGraphCount * kPageWordCount> expected_arguments;
    std::array<uint32_t, kGraphCount * kPageWordCount> observed_arguments;
    std::array<uint32_t, kGraphCount * kPageWordCount> observed_control;
    std::array<uint32_t, kPageWordCount> observed_code;
    for (uint32_t epoch = 0; epoch < kEpochCount; ++epoch) {
      SCOPED_TRACE(epoch);
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        const size_t payload_base = graph * kPayloadWordCount;
        const size_t page_base = graph * kPageWordCount;
        for (size_t owner = 0; owner < expected.size(); ++owner) {
          std::fill_n(expected[owner].data() + payload_base, kPayloadWordCount,
                      kGuards[owner] ^ (ordinal * 0x1020304u));
        }
        for (uint32_t i = 0; i < kGridSize; ++i) {
          const size_t word = payload_base + kPayloadOffset + i;
          const uint32_t value = static_cast<uint32_t>(
              UINT64_C(0xfffffff0) + uint64_t{i} * 0x01030507u +
              uint64_t{ordinal} * 0x11111111u);
          expected[kSource][word] = value;
          expected[kInput][word] = value;
          if (i < kCounts[ordinal % kCounts.size()]) {
            expected[kOutput][word] = static_cast<uint32_t>(
                uint64_t{value} * 3 + kAddends[ordinal % kAddends.size()]);
          }
          expected[kReadback][word] = expected[kOutput][word];
        }
        const kernels::transform::Arguments payload = {
            input.device_address + graph * 8192 + 64,
            output.device_address + graph * 8192 + 64,
            kCounts[ordinal % kCounts.size()],
            kAddends[ordinal % kAddends.size()]};
        std::fill_n(expected_arguments.data() + page_base, kPageWordCount,
                    UINT32_C(0x713ace09) ^ ordinal);
        std::memcpy(expected_arguments.data() + page_base, &payload,
                    kernel.arguments.byte_length);
        expected_control[page_base] = expected_control[page_base + 16] =
            expected_control[page_base + 32] = ordinal + 1;
      }
      for (size_t owner = 0; owner < expected.size(); ++owner) {
        std::memcpy(backings[owner].memory->host.pointer,
                    expected[owner].data(), sizeof(expected[owner]));
      }
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        for (uint32_t i = 0; i < kGridSize; ++i) {
          const size_t word = graph * kPayloadWordCount + kPayloadOffset + i;
          static_cast<uint32_t*>(input.host.pointer)[word] =
              ~expected[kInput][word];
          if (i < kCounts[ordinal % kCounts.size()]) {
            static_cast<uint32_t*>(output.host.pointer)[word] =
                ~expected[kOutput][word];
          }
          static_cast<uint32_t*>(readback.host.pointer)[word] =
              ~expected[kReadback][word];
        }
      }
      std::memcpy(arguments.host.pointer, expected_arguments.data(),
                  sizeof(expected_arguments));
      const uint64_t pm4_frontier = pm4_frontiers[epoch];
      ASSERT_NO_FATAL_FAILURE(pm4_queue->Publish(api_, gpu_api_, pm4_frontier));
      // Arm both consumers before starting the independent uploader.
      if (transfer_queues == TransferQueues::kSeparate) {
        ASSERT_NO_FATAL_FAILURE(download_queue->Publish(
            api_, gpu_api_, transfer_frontiers[epoch][kDownload]));
      }
      ASSERT_NO_FATAL_FAILURE(upload_queue->Publish(
          api_, gpu_api_, transfer_frontiers[epoch][kUpload]));
      const auto last_control =
          reinterpret_cast<uintptr_t>(control.host.pointer) +
          (kGraphCount - 1) * 4096;
      const uint32_t last_generation = (epoch + 1) * kGraphCount;
      GpuWaitEqual<uint32_t>(last_control + 128, last_generation);
      // One final download joins the complete batch. Snapshot every readback
      // before any independent shader join: this is the dependency observation.
      std::memcpy(observed[kReadback].data(), readback.host.pointer,
                  sizeof(observed[kReadback]));
      // Independently join the shader before reading its other owners, even if
      // the dependency under test allowed SDMA to publish D prematurely.
      GpuWaitEqual<uint32_t>(last_control + 64, last_generation);
      // Upload has its own owner in the separate layout. Independently join
      // its final write before observing input backing, even if a dependency
      // under test allowed compute or download to advance prematurely.
      GpuWaitEqual<uint32_t>(last_control, last_generation);
      for (size_t owner = kSource; owner <= kOutput; ++owner) {
        std::memcpy(observed[owner].data(),
                    backings[owner].memory->host.pointer,
                    sizeof(observed[owner]));
      }
      std::memcpy(observed_arguments.data(), arguments.host.pointer,
                  sizeof(observed_arguments));
      std::memcpy(observed_control.data(), control.host.pointer,
                  sizeof(observed_control));
      std::memcpy(observed_code.data(), code.host.pointer,
                  sizeof(observed_code));

      for (size_t owner = 0; owner < expected.size(); ++owner) {
        for (size_t word = 0; word < expected[owner].size(); ++word) {
          EXPECT_EQ(observed[owner][word], expected[owner][word])
              << backings[owner].name << " word=" << word;
        }
      }
      for (size_t word = 0; word < expected_arguments.size(); ++word) {
        EXPECT_EQ(observed_arguments[word], expected_arguments[word])
            << "arguments word=" << word;
        EXPECT_EQ(observed_control[word], expected_control[word])
            << "control word=" << word;
      }
      for (size_t word = 0; word < expected_code.size(); ++word) {
        EXPECT_EQ(observed_code[word], expected_code[word])
            << "code word=" << word;
      }
      // All retirement calls run after every nonfatal oracle, then any failure
      // stops the next CPU rewrite. Native teardown failure retains all owners.
      EXPECT_NO_FATAL_FAILURE(pm4_queue->WaitRetired(api_));
      EXPECT_NO_FATAL_FAILURE(upload_queue->WaitRetired(api_));
      if (transfer_queues == TransferQueues::kSeparate) {
        EXPECT_NO_FATAL_FAILURE(download_queue->WaitRetired(api_));
      }
      if (HasFailure()) {
        return;
      }
      const std::string prefix = "pm4_sdma_epoch_" + std::to_string(epoch);
      for (uint32_t graph = 0; graph < kGraphCount; ++graph) {
        const uint32_t ordinal = epoch * kGraphCount + graph;
        const std::string graph_prefix =
            prefix + "_graph_" + std::to_string(graph);
        RecordProperty(graph_prefix + "_count",
                       kCounts[ordinal % kCounts.size()]);
        RecordProperty(graph_prefix + "_addend",
                       std::to_string(kAddends[ordinal % kAddends.size()]));
        RecordProperty(graph_prefix + "_completion",
                       observed_control[graph * kPageWordCount + 32]);
      }
      RecordProperty(prefix + "_pm4_frontier", std::to_string(pm4_frontier));
      RecordProperty(prefix + "_upload_word_frontier",
                     std::to_string(transfer_frontiers[epoch][kUpload]));
      RecordProperty(prefix + "_download_word_frontier",
                     std::to_string(transfer_frontiers[epoch][kDownload]));
    }
    RecordProperty("pm4_sdma_completed_epochs", kEpochCount);
    RecordProperty("pm4_sdma_graphs_per_epoch", kGraphCount);
    RecordProperty("pm4_sdma_acquire_mode",
                   acquire_mode == AcquireMode::kFullBarrier ? "full_barrier"
                                                             : "ordered_data");
    RecordProperty("pm4_sdma_pm4_command_dwords",
                   std::to_string(pm4.word_count()));
    RecordProperty("pm4_sdma_sdma_command_dwords",
                   std::to_string(upload.word_count() +
                                  (transfer_queues == TransferQueues::kSeparate
                                       ? download.word_count()
                                       : 0)));
  }
};

TEST_F(Pm4SdmaRecipeTest, ConcreteCoherentUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kConcrete);
}

TEST_F(Pm4SdmaRecipeTest, ProfileCoherentUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kProfile);
}

TEST_F(Pm4SdmaRecipeTest, BatchedCoherentUploadDispatchDownload) {
  RunCoherentHandoff<4>(PairQuery::kConcrete);
}

TEST_F(Pm4SdmaRecipeTest, OrderedDataAcquireBatchedUploadDispatchDownload) {
  RunCoherentHandoff<4>(PairQuery::kConcrete, AcquireMode::kOrderedData);
}

TEST_F(Pm4SdmaRecipeTest, IndependentUploadDispatchDownloadQueues) {
  RunCoherentHandoff<4>(PairQuery::kConcrete, AcquireMode::kFullBarrier,
                        TransferQueues::kSeparate);
}

TEST_F(Pm4SdmaRecipeTest, OrderedDataAcquireIndependentTransferQueues) {
  RunCoherentHandoff<4>(PairQuery::kConcrete, AcquireMode::kOrderedData,
                        TransferQueues::kSeparate);
}

TEST_F(Pm4SdmaRecipeTest, UserGcrUploadDispatchDownload) {
  RunCoherentHandoff(PairQuery::kConcrete, AcquireMode::kFullBarrier,
                     TransferQueues::kShared,
                     AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM |
                         AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM);
}

}  // namespace
