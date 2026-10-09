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
#include "libamdf/cts/gpu/recipes/pm4_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

struct ReaderReuseCase {
  // Exact native publication path, selected before acquiring the device.
  amdf_queue_publication_modes_t publication_mode;
  // Required destination placement, without a SYSTEM substitute for LOCAL.
  amdf_memory_class_t memory_class;
  // Source slots released after their SDMA reads complete.
  uint32_t source_count;
  // Destination slots released after every independent reader completes.
  uint32_t destination_count;
  // Distinct PM4 reader queues consuming each copied input.
  uint32_t reader_count;
  // Complete positive payload extent in bytes.
  uint32_t byte_length;
};

struct alignas(64) ProgressWord {
  // Native completion value or sampled progress observation.
  uint32_t value;
  // Untouched suffix separates independently owned progress values.
  uint32_t guards[15];
};

struct JobProgress {
  // The source page contains this job's complete input.
  ProgressWord refill;
  // SDMA has finished reading the source and writing the destination.
  ProgressWord upload;
  // Each independent reader has released its output and destination use.
  std::array<ProgressWord, 2> readers;
  // Observed prior reader value after source overwrite, before refill release.
  ProgressWord prior_reader;
};
static_assert(sizeof(JobProgress) == 320);

struct CheckedMemory {
  // Borrowed from the fixture's queue-first native resource owner.
  GpuMemory* memory = nullptr;
  // Complete allocation oracle, including unmodified tails and guard words.
  std::vector<uint32_t> expected;
};

void CheckWords(const void* observed, const std::vector<uint32_t>& expected) {
  const auto* words = static_cast<const uint32_t*>(observed);
  const auto mismatch = std::mismatch(expected.begin(), expected.end(), words);
  ASSERT_EQ(mismatch.first, expected.end())
      << "word " << (mismatch.first - expected.begin()) << ": expected "
      << *mismatch.first << ", observed " << *mismatch.second;
}

class Pm4ReaderReuseTest
    : public Pm4SdmaTest,
      public ::testing::WithParamInterface<ReaderReuseCase> {
 protected:
  Pm4ReaderReuseTest() : Pm4SdmaTest(GetParam().publication_mode) {}

  void CreateChecked(amdf_memory_access_t access, uint64_t byte_length,
                     uint32_t guard, CheckedMemory* storage) {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(
        access, (byte_length + 4095) & ~UINT64_C(4095), &storage->memory));
    storage->expected.assign(storage->memory->info.byte_length / 4, guard);
    std::memcpy(storage->memory->host.pointer, storage->expected.data(),
                storage->memory->info.byte_length);
  }

  void QueryEdge(const char* name, const GpuMemory& memory, Site producer,
                 Site consumer, amdf_cache_operations_t* operations) {
    SCOPED_TRACE(name);
    const auto site = [&](Site actor) {
      return actor == Site::kHost
                 ? memory.HostSite()
                 : memory.DeviceSite(actor == Site::kPm4
                                         ? family_.ordinal
                                         : sdma_family_.ordinal);
    };
    const auto source = site(producer);
    const auto target = site(consumer);
    amdf_memory_pair_info_t pair = {};
    pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
    pair.structure_size = sizeof(pair);
    ASSERT_EQ(api_->memory_query_pair_info(&source, &target, &pair),
              AMDF_STATUS_OK);
    ASSERT_NE(pair.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE, 0u);
    ASSERT_NO_FATAL_FAILURE(
        ResolveTransition(pair.release, producer,
                          AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM, operations));
    ASSERT_NO_FATAL_FAILURE(ResolveTransition(
        pair.acquire, consumer, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        operations));
  }

  void Run();
};

void Pm4ReaderReuseTest::Run() {
  const auto& test_case = GetParam();
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
      (features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "LOCAL placement is not advertised by this device";
  }
  constexpr uint32_t kJobCount = 33;
  constexpr uint64_t kCommandByteLength = 65536;
  constexpr uint32_t kPrefixWords = 16;
  constexpr uint32_t kDatasetGuard = 0x318abd75u;
  constexpr uint32_t kSourceGuard = 0x79521aceu;
  constexpr uint32_t kInputGuard = 0x172ac935u;
  constexpr uint32_t kOutputGuard = 0x34581acbu;
  constexpr uint32_t kControlGuard = 0x712be853u;
  constexpr uint64_t kSeedOffset = kJobCount * sizeof(JobProgress);
  constexpr uint64_t kFinalOffset = kSeedOffset + sizeof(ProgressWord);
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const uint32_t word_count = test_case.byte_length / 4;
  const uint64_t slot_words = word_count + 2 * kPrefixWords;
  const uint64_t slot_bytes = slot_words * 4;
  const uint64_t input_bytes =
      (slot_bytes * test_case.destination_count + 4095) & ~UINT64_C(4095);
  const auto* kernel = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(kernel, nullptr);
  ASSERT_EQ(kernel->private_segment_byte_length, 0u);
  ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  ASSERT_EQ(kernel->workgroup_size(), 64u);
  ASSERT_LE(kernel->arguments.byte_length,
            sizeof(kernels::transform::Arguments));
  RecordProperty("pm4_reuse_kernel_target", kernel->target);

  CheckedMemory dataset;
  CheckedMemory source;
  CheckedMemory input;
  CheckedMemory output;
  CheckedMemory readback;
  CheckedMemory arguments;
  CheckedMemory control;
  ASSERT_NO_FATAL_FAILURE(CreateChecked(AMDF_MEMORY_ACCESS_READ,
                                        kJobCount * slot_bytes, kDatasetGuard,
                                        &dataset));
  ASSERT_NO_FATAL_FAILURE(CreateChecked(
      kReadWrite, test_case.source_count * slot_bytes, kSourceGuard, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateChecked(kReadWrite, kJobCount * test_case.reader_count * slot_bytes,
                    kOutputGuard, &output));
  // This seed becomes the final readback after every reader releases its slot.
  ASSERT_NO_FATAL_FAILURE(
      CreateChecked(kReadWrite, input_bytes, kInputGuard, &readback));
  ASSERT_NO_FATAL_FAILURE(
      CreateChecked(AMDF_MEMORY_ACCESS_READ,
                    kJobCount * test_case.reader_count *
                        sizeof(kernels::transform::Arguments),
                    0, &arguments));
  ASSERT_NO_FATAL_FAILURE(CreateChecked(kReadWrite,
                                        kFinalOffset + sizeof(ProgressWord),
                                        kControlGuard, &control));
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL) {
    ASSERT_NE(local_scope_, nullptr);
    const amdf_memory_device_access_t access = {
        device_,
        {.access = kReadWrite, .flags = AMDF_MEMORY_FLAG_DEVICE_ADDRESS}};
    amdf_memory_create_info_t creation = {};
    creation.type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO;
    creation.structure_size = sizeof(creation);
    creation.memory_profile_ordinal = FindGpuMemoryProfileOrdinal(
        api_, local_scope_, device_, AMDF_MEMORY_PROFILE_ROLE_CREATE,
        AMDF_MEMORY_FLAG_DEVICE_LOCAL, access.requirements);
    ASSERT_NE(creation.memory_profile_ordinal,
              AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    creation.required_flags = AMDF_MEMORY_FLAG_DEVICE_LOCAL;
    creation.access_count = 1;
    creation.accesses = &access;
    creation.byte_length = input_bytes;
    creation.minimum_alignment = 4096;
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(local_scope_, creation, &input.memory));
    ASSERT_EQ(input.memory->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
    ASSERT_NE(input.memory->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
    ASSERT_EQ(input.memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    ASSERT_EQ(input.memory->host.pointer, nullptr);
  } else {
    ASSERT_NO_FATAL_FAILURE(
        CreateMemory(kReadWrite, input_bytes, &input.memory));
    ASSERT_EQ(input.memory->info.memory_class, AMDF_MEMORY_CLASS_SYSTEM);
  }
  input.expected.assign(input_bytes / 4, kInputGuard);
  Pm4ComputeProgram program = {
      0,
      kernel->program.resource1,
      kernel->program.resource2,
      kernel->program.resource3,
      kernel->group_segment_byte_length,
      kernel->wavefront_size,
      {64, 1, 1},
  };
  GpuMemory* code = nullptr;
  ASSERT_NO_FATAL_FAILURE(PrepareProgram(kernel->executable,
                                         kernel->entry_byte_offset, &program,
                                         "pm4_reuse", &code));
  std::vector<uint32_t> expected_code(code->info.byte_length / 4, 0);
  std::memcpy(expected_code.data(), kernel->executable.words,
              kernel->executable.byte_length);

  amdf_cache_operations_t upload_operations = 0;
  amdf_cache_operations_t readback_operations = 0;
  for (const auto* memory :
       {dataset.memory, source.memory, output.memory, readback.memory,
        arguments.memory, control.memory, code}) {
    ASSERT_NO_FATAL_FAILURE(QueryEdge("host_seed", *memory, Site::kHost,
                                      Site::kPm4, &upload_operations));
  }
  ASSERT_NO_FATAL_FAILURE(QueryEdge("source_ready", *source.memory, Site::kPm4,
                                    Site::kSdma, &upload_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("source_reuse", *source.memory, Site::kSdma,
                                    Site::kPm4, &upload_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("input_ready", *input.memory, Site::kSdma,
                                    Site::kPm4, &upload_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("input_reuse", *input.memory, Site::kPm4,
                                    Site::kSdma, &upload_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("input_readback", *input.memory, Site::kPm4,
                                    Site::kSdma, &readback_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("readback", *readback.memory, Site::kSdma,
                                    Site::kHost, &readback_operations));
  ASSERT_NO_FATAL_FAILURE(QueryEdge("readback_reuse", *readback.memory,
                                    Site::kHost, Site::kSdma,
                                    &readback_operations));
  for (const auto* memory : {source.memory, output.memory, control.memory}) {
    ASSERT_NO_FATAL_FAILURE(QueryEdge("host_observation", *memory, Site::kPm4,
                                      Site::kHost, &upload_operations));
  }
  for (const auto& actors : {std::array{Site::kHost, Site::kSdma},
                             std::array{Site::kPm4, Site::kSdma},
                             std::array{Site::kSdma, Site::kPm4},
                             std::array{Site::kSdma, Site::kHost}}) {
    ASSERT_NO_FATAL_FAILURE(QueryEdge("control", *control.memory, actors[0],
                                      actors[1], &upload_operations));
  }

  GpuCommandQueue* refill_queue = nullptr;
  GpuCommandQueue* upload_queue = nullptr;
  std::array<GpuCommandQueue*, 2> reader_queues = {};
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&refill_queue, kCommandByteLength));
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &upload_queue, kCommandByteLength));
  std::vector<GpuCommandQueue*> queues = {refill_queue, upload_queue};
  for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
    ASSERT_NO_FATAL_FAILURE(
        CreateQueue(&reader_queues[reader], kCommandByteLength));
    queues.push_back(reader_queues[reader]);
  }
  for (size_t index = 0; index < queues.size(); ++index) {
    ASSERT_EQ(queues[index]->publication_mode(), test_case.publication_mode);
    ASSERT_GE(queues[index]->words().size_bytes(), kCommandByteLength);
    ASSERT_TRUE(amdf_device_id_is_equal(&queues[index]->device_id(),
                                        &refill_queue->device_id()));
    for (size_t previous = 0; previous < index; ++previous) {
      ASSERT_NE(queues[index]->native_handle(),
                queues[previous]->native_handle());
    }
    std::memset(queues[index]->words().data(), 0,
                queues[index]->words().size_bytes());
  }
  const auto address = [&](uint32_t job, uint64_t offset) {
    return control.memory->device_address + job * sizeof(JobProgress) + offset;
  };
  const auto reader_offset = [](uint32_t reader) {
    return offsetof(JobProgress, readers) + reader * sizeof(ProgressWord);
  };
  const auto held_reader = [&](uint32_t job) {
    return job % test_case.reader_count;
  };
  auto initial_control = control.expected;
  for (uint32_t job = 0; job < kJobCount; ++job) {
    const size_t base = job * sizeof(JobProgress) / 4;
    initial_control[base + offsetof(JobProgress, refill) / 4] = 0;
    initial_control[base + offsetof(JobProgress, upload) / 4] = 0;
    control.expected[base + offsetof(JobProgress, refill) / 4] = 1;
    control.expected[base + offsetof(JobProgress, upload) / 4] = 1;
    for (uint32_t reader = 0; reader < 2; ++reader) {
      initial_control[base + reader_offset(reader) / 4] = 0;
      control.expected[base + reader_offset(reader) / 4] =
          reader < test_case.reader_count ? 1 : 0;
    }
    if (job >= test_case.source_count) {
      control.expected[base + offsetof(JobProgress, prior_reader) / 4] = 0;
    }
  }
  initial_control[kSeedOffset / 4] = initial_control[kFinalOffset / 4] = 0;
  control.expected[kSeedOffset / 4] = control.expected[kFinalOffset / 4] = 1;
  std::memcpy(control.memory->host.pointer, initial_control.data(),
              control.memory->info.byte_length);

  Pm4CommandWriter refill(refill_queue->words().data(), *pm4_profile_);
  SdmaCommandWriter upload(upload_queue->words().data(),
                           sdma_family_.format_features);
  refill.SystemBarrier();
  refill.DmaCopy(readback.memory->device_address, input.memory->device_address,
                 static_cast<uint32_t>(input_bytes));
  refill.WaitDma();
  refill.ReleaseSystem32(control.memory->device_address + kSeedOffset, 1);
  refill.WaitMemory32(control.memory->device_address + kSeedOffset, 1);
  upload.WaitMemory32(control.memory->device_address + kSeedOffset, 1);
  for (uint32_t job = 0; job < kJobCount; ++job) {
    const uint64_t dataset_word = job * slot_words + kPrefixWords;
    const uint64_t source_word =
        (job % test_case.source_count) * slot_words + kPrefixWords;
    const uint64_t input_word =
        (job % test_case.destination_count) * slot_words + kPrefixWords;
    for (uint32_t word = 0; word < word_count; ++word) {
      const uint32_t value =
          0x31415927u + job * 0x243f6a89u + word * 0x01020305u;
      dataset.expected[dataset_word + word] = value;
      source.expected[source_word + word] = value;
      input.expected[input_word + word] = value;
      for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
        const uint64_t output_word =
            (job * test_case.reader_count + reader) * slot_words + kPrefixWords;
        output.expected[output_word + word] =
            value * 3 + job * 0x9e3779b1u + reader;
      }
    }
    if (job >= test_case.source_count) {
      refill.WaitMemory32(
          address(job - test_case.source_count, offsetof(JobProgress, upload)),
          1);
    }
    refill.AcquireFromSystem();
    refill.DmaCopy(dataset.memory->device_address + dataset_word * 4,
                   source.memory->device_address + source_word * 4,
                   test_case.byte_length);
    refill.WaitDma();
    if (job >= test_case.source_count) {
      const uint32_t previous = job - test_case.source_count;
      refill.CopyData32(address(previous, reader_offset(held_reader(previous))),
                        address(job, offsetof(JobProgress, prior_reader)));
    }
    refill.ReleaseSystem32(address(job, offsetof(JobProgress, refill)), 1);
    refill.WaitMemory32(address(job, offsetof(JobProgress, refill)), 1);

    upload.WaitMemory32(address(job, offsetof(JobProgress, refill)), 1);
    if (job >= test_case.destination_count) {
      for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
        upload.WaitMemory32(
            address(job - test_case.destination_count, reader_offset(reader)),
            1);
      }
    }
    if (upload_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
      upload.AcquireFromSystem();
    }
    upload.CopyLinear(source.memory->device_address + source_word * 4,
                      input.memory->device_address + input_word * 4,
                      test_case.byte_length);
    if (upload_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
      upload.ReleaseToSystem();
    }
    upload.Fence32(address(job, offsetof(JobProgress, upload)), 1);
  }
  std::vector<size_t> command_counts = {0, 0};
  for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
    Pm4CommandWriter consume(reader_queues[reader]->words().data(),
                             *pm4_profile_);
    consume.SystemBarrier();
    for (uint32_t job = 0; job < kJobCount; ++job) {
      consume.WaitMemory32(address(job, offsetof(JobProgress, upload)), 1);
      if (reader == held_reader(job) &&
          job + test_case.source_count < kJobCount) {
        consume.WaitMemory32(address(job + test_case.source_count,
                                     offsetof(JobProgress, refill)),
                             1);
      }
      consume.AcquireFromSystem();
      const uint64_t ordinal = job * test_case.reader_count + reader;
      const uint64_t argument_offset =
          ordinal * sizeof(kernels::transform::Arguments);
      const kernels::transform::Arguments payload = {
          input.memory->device_address +
              ((job % test_case.destination_count) * slot_words +
               kPrefixWords) *
                  4,
          output.memory->device_address +
              (ordinal * slot_words + kPrefixWords) * 4,
          word_count,
          job * 0x9e3779b1u + reader,
      };
      std::memcpy(arguments.expected.data() + argument_offset / 4, &payload,
                  kernel->arguments.byte_length);
      consume.BindCompute(program,
                          arguments.memory->device_address + argument_offset);
      consume.Dispatch(program, word_count, 1, 1);
      consume.ReleaseSystem32(address(job, reader_offset(reader)), 1);
      // Join this dispatch before the next data acquire. Other reader queues
      // remain independent; destination reuse still joins every reader.
      consume.WaitMemory32(address(job, reader_offset(reader)), 1);
    }
    consume.PadToEightWords();
    command_counts.push_back(consume.word_count());
    upload.WaitMemory32(address(kJobCount - 1, reader_offset(reader)), 1);
  }
  if (readback_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
    upload.AcquireFromSystem();
  }
  upload.CopyLinear(input.memory->device_address,
                    readback.memory->device_address,
                    static_cast<uint32_t>(input_bytes));
  if ((readback_operations | upload_operations) &
      AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
    upload.ReleaseToSystem();
  }
  upload.Fence32(control.memory->device_address + kFinalOffset, 1);
  refill.PadToEightWords();
  command_counts[0] = refill.word_count();
  command_counts[1] = upload.word_count();
  std::memcpy(dataset.memory->host.pointer, dataset.expected.data(),
              dataset.memory->info.byte_length);
  std::memcpy(arguments.memory->host.pointer, arguments.expected.data(),
              arguments.memory->info.byte_length);
  std::vector<std::vector<uint32_t>> expected_commands;
  for (size_t index = 0; index < queues.size(); ++index) {
    ASSERT_LT(command_counts[index], queues[index]->words().size());
    expected_commands.emplace_back(queues[index]->words().begin(),
                                   queues[index]->words().end());
  }
  // Every stream is resident before publication. Consumers can be submitted
  // first because their native waits do not occupy shader workgroups.
  for (size_t index = queues.size(); index != 0; --index) {
    ASSERT_NO_FATAL_FAILURE(
        queues[index - 1]->Publish(api_, gpu_api_, command_counts[index - 1]));
  }
  GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(control.memory->host.pointer) + kFinalOffset,
      1);
  // Capture payload visibility at the in-stream completion, before native
  // retirement can supply any additional driver-owned synchronization.
  const std::array<const CheckedMemory*, 5> checked = {
      &dataset, &source, &output, &arguments, &control};
  std::array<std::vector<uint32_t>, checked.size()> observed;
  for (size_t index = 0; index < checked.size(); ++index) {
    observed[index].resize(checked[index]->expected.size());
    std::memcpy(observed[index].data(), checked[index]->memory->host.pointer,
                checked[index]->memory->info.byte_length);
  }
  std::vector<uint32_t> observed_input(input.expected.size());
  std::memcpy(observed_input.data(), readback.memory->host.pointer,
              input_bytes);
  for (auto* queue : queues) {
    ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));
  }
  for (size_t index = 0; index < checked.size(); ++index) {
    ASSERT_NO_FATAL_FAILURE(
        CheckWords(observed[index].data(), checked[index]->expected));
  }
  ASSERT_NO_FATAL_FAILURE(CheckWords(observed_input.data(), input.expected));
  ASSERT_NO_FATAL_FAILURE(CheckWords(code->host.pointer, expected_code));
  for (size_t index = 0; index < queues.size(); ++index) {
    ASSERT_NO_FATAL_FAILURE(
        CheckWords(queues[index]->words().data(), expected_commands[index]));
  }
  RecordProperty("pm4_reuse_jobs", kJobCount);
  RecordProperty("pm4_reuse_readers", kJobCount * test_case.reader_count);
  RecordProperty(
      "pm4_reuse_largest_command_byte_length",
      std::to_string(
          *std::max_element(command_counts.begin(), command_counts.end()) * 4));
  RecordProperty("pm4_reuse_source_overwrites_with_reader_pending",
                 kJobCount - test_case.source_count);
  RecordProperty("pm4_reuse_upload_cache_operations",
                 std::to_string(upload_operations));
  RecordProperty("pm4_reuse_readback_cache_operations",
                 std::to_string(readback_operations));
}

TEST_P(Pm4ReaderReuseTest, SourceCopyAndEveryReaderOwnSeparateReuseBoundaries) {
  Run();
}

std::vector<ReaderReuseCase> ReaderReuseCases() {
  std::vector<ReaderReuseCase> cases;
  constexpr std::array<std::array<uint32_t, 3>, 4> kCredits = {{
      {1, 1, 1},
      {1, 2, 2},
      {2, 1, 2},
      {2, 4, 2},
  }};
  for (amdf_queue_publication_modes_t publication :
       {AMDF_QUEUE_PUBLICATION_MODE_USER, AMDF_QUEUE_PUBLICATION_MODE_KERNEL}) {
    for (amdf_memory_class_t memory_class :
         {AMDF_MEMORY_CLASS_SYSTEM, AMDF_MEMORY_CLASS_LOCAL}) {
      for (uint32_t bytes : {4096u, 65536u}) {
        for (const auto& credits : kCredits) {
          cases.push_back({publication, memory_class, credits[0], credits[1],
                           credits[2], bytes});
        }
      }
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    Publication, Pm4ReaderReuseTest, ::testing::ValuesIn(ReaderReuseCases()),
    [](const ::testing::TestParamInfo<ReaderReuseCase>& info) {
      const auto& item = info.param;
      std::string name =
          item.publication_mode == AMDF_QUEUE_PUBLICATION_MODE_USER ? "User"
                                                                    : "Kernel";
      name += item.memory_class == AMDF_MEMORY_CLASS_LOCAL ? "Local" : "System";
      name += std::to_string(item.byte_length / 1024) + "KiBS";
      name += std::to_string(item.source_count) + "D";
      name += std::to_string(item.destination_count) + "R";
      return name + std::to_string(item.reader_count);
    });

}  // namespace
