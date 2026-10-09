// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/kernels/file_staged.h"

#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/file_staged_read_kernels.h"
#include "libamdf/cts/gpu/kernels/file_staged_reader_kernels.h"
#include "libamdf/cts/gpu/kernels/file_staged_upload_kernels.h"
#include "libamdf/cts/gpu/linux/io_uring/file_io_resources.h"
#include "libamdf/cts/gpu/recipes/device_sdma_fixture.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

namespace {

namespace protocol = kernels::file_staged;

static_assert(sizeof(io_uring_sqe) == 64);
static_assert(offsetof(io_uring_sqe, opcode) == 0);
static_assert(offsetof(io_uring_sqe, flags) == 1);
static_assert(offsetof(io_uring_sqe, fd) == 4);
static_assert(offsetof(io_uring_sqe, off) == 8);
static_assert(offsetof(io_uring_sqe, addr) == 16);
static_assert(offsetof(io_uring_sqe, len) == 24);
static_assert(offsetof(io_uring_sqe, user_data) == 32);
static_assert(sizeof(io_uring_cqe) == 16);
static_assert(offsetof(io_uring_cqe, user_data) == 0);
static_assert(offsetof(io_uring_cqe, res) == 8);
static_assert(IORING_OP_READ_FIXED == 4 && IOSQE_FIXED_FILE == 1);
static_assert(ENODATA == 61 && EPROTO == 71);

struct JobSignals {
  // File completion admits the upload, including a terminal read error.
  aql::Signal file;
  // Copy completion permits source reuse and admits the independent readers.
  aql::Signal upload;
  // Every reader must finish before destination-slot reuse.
  std::array<aql::Signal, 2> readers;
};
static_assert(sizeof(JobSignals) == 256);

enum class FileFailure { kNone, kPartialEof, kInvalidFile };

struct StagedCase {
  // Native file contract, with no buffered substitute for direct I/O.
  FileMode mode;
  // Exact destination placement; LOCAL never substitutes SYSTEM.
  amdf_memory_class_t memory_class;
  // Independent registered source credits released by SDMA completion.
  uint32_t source_count;
  // Destination credits released by the last independent reader.
  uint32_t destination_count;
  // Independently completed dispatches consuming each copied input.
  uint32_t reader_count;
  // Positive page-multiple payload length, at most 64 KiB.
  uint32_t byte_length;
  // Native file error introduced after successful jobs, or a complete batch.
  FileFailure failure;
};

void CheckWords(const void* observed, const std::vector<uint32_t>& expected) {
  const auto* words = static_cast<const uint32_t*>(observed);
  const auto mismatch = std::mismatch(expected.begin(), expected.end(), words);
  ASSERT_EQ(mismatch.first, expected.end())
      << "word " << (mismatch.first - expected.begin()) << ": expected "
      << *mismatch.first << ", observed " << *mismatch.second;
}

uint32_t RequestHash(uint32_t request, uint32_t generation) {
  return request ^ (request >> 13) ^ (generation * 0x9e3779b1u);
}

class GpuFileStagedTest : public DeviceGeneratedSdmaTest,
                          protected GpuFileIoResources {
 protected:
  void SetUp() override {
    DeviceGeneratedSdmaTest::SetUp();
    if (HasFatalFailure() || IsSkipped()) {
      return;
    }
    InitializeFileIo(api_, system_scope_, device_, features_);
  }

  void TearDown() override {
    ASSERT_NO_FATAL_FAILURE(DeviceGeneratedSdmaTest::TearDown());
    ReleaseFileIo();
  }

  void Run(const StagedCase& test_case);
};

void GpuFileStagedTest::Run(const StagedCase& test_case) {
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
      (features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "LOCAL placement is not advertised by this device";
  }
  constexpr uint32_t kJobCount = 33;
  constexpr uint32_t kFailureJob = 7;
  constexpr uint32_t kFileBlocks = 8;
  constexpr uint32_t kSourceGuard = 0x713a9cb5u;
  constexpr uint32_t kInputGuard = 0x278ba935u;
  constexpr uint32_t kOutputGuard = 0x329ade71u;
  constexpr uint32_t kControlGuard = 0xc4319a75u;
  constexpr uint64_t kAqlRingBytes = 65536;
  constexpr uint64_t kCompletionOffset = 64;
  constexpr uint64_t kFrontierOffset = 128;
  constexpr uint64_t kFileStateOffset = 192;
  constexpr uint64_t kArgumentStride = 320;
  constexpr uint64_t kUploadOffset = 128;
  constexpr uint64_t kReadersOffset = 256;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  ASSERT_EQ(test_case.byte_length % page_byte_length_, 0u);
  ASSERT_NE(family_.ring_byte_length_alignment, 0u);
  if (kAqlRingBytes < family_.minimum_ring_byte_length ||
      kAqlRingBytes > family_.maximum_ring_byte_length ||
      kAqlRingBytes % family_.ring_byte_length_alignment != 0) {
    GTEST_SKIP() << "finite file batch requires a 64-KiB AQL ring";
  }
  const auto* read_kernel =
      kernels::file_staged_read::kKernels.Find(gpu_endpoint_info_);
  const auto* upload_kernel =
      kernels::file_staged_upload::kKernels.Find(gpu_endpoint_info_);
  const auto* reader_kernel =
      kernels::file_staged_reader::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(read_kernel, nullptr);
  ASSERT_NE(upload_kernel, nullptr);
  ASSERT_NE(reader_kernel, nullptr);
  ASSERT_EQ(read_kernel->workgroup_size(), 1u);
  ASSERT_EQ(upload_kernel->workgroup_size(), 1u);
  ASSERT_EQ(reader_kernel->workgroup_size(), 64u);
  ASSERT_LE(read_kernel->arguments.byte_length,
            sizeof(protocol::ReadArguments));
  ASSERT_LE(upload_kernel->arguments.byte_length,
            sizeof(protocol::UploadArguments));
  ASSERT_LE(reader_kernel->arguments.byte_length,
            sizeof(protocol::ReaderArguments));
  for (const auto* kernel : {read_kernel, upload_kernel, reader_kernel}) {
    ASSERT_EQ(kernel->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  }
  const auto align_page = [&](uint64_t bytes) {
    return (bytes + page_byte_length_ - 1) & ~(page_byte_length_ - 1);
  };
  const uint32_t word_count = test_case.byte_length / 4;
  const uint64_t slot_bytes = test_case.byte_length + 2 * page_byte_length_;
  const uint64_t slot_words = slot_bytes / 4;
  const uint64_t prefix_words = page_byte_length_ / 4;
  const uint64_t source_bytes = slot_bytes * test_case.source_count;
  const uint64_t input_bytes = slot_bytes * test_case.destination_count;
  const uint64_t output_bytes = slot_bytes * kJobCount * test_case.reader_count;
  GpuMemory* source = nullptr;
  GpuMemory* input = nullptr;
  GpuMemory* seed = nullptr;
  GpuMemory* readback = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* requests = nullptr;
  GpuMemory* records = nullptr;
  GpuMemory* signals = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateRegisteredPages(source_bytes, kSourceGuard, &source));

  // Registered WB pages have their own visibility contract. In particular they
  // cannot inherit an owned allocation's selected no-GCR backing policy.
  const auto host_site = source->HostSite();
  const auto sdma_site = source->DeviceSite(sdma_family_.ordinal);
  amdf_memory_pair_info_t source_pair = {};
  source_pair.type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO;
  source_pair.structure_size = sizeof(source_pair);
  const amdf_status_t source_status =
      api_->memory_query_pair_info(&host_site, &sdma_site, &source_pair);
  if (amdf_status_domain(source_status) == AMDF_STATUS_DOMAIN_API &&
      amdf_status_code(source_status) == AMDF_STATUS_CODE_UNSUPPORTED) {
    GTEST_SKIP() << "registered SYSTEM source has no SDMA visibility contract";
  }
  ASSERT_EQ(source_status, AMDF_STATUS_OK);

  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, slot_bytes, &seed));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, input_bytes, &readback));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, output_bytes, &output));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       align_page(kJobCount * 4), &requests));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite, align_page(kJobCount * sizeof(protocol::Result)), &records));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite, align_page((kJobCount + 1) * sizeof(JobSignals)), &signals));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(kReadWrite, page_byte_length_, &control));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(AMDF_MEMORY_ACCESS_READ,
                                       align_page(kJobCount * kArgumentStride),
                                       &arguments));
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
        .minimum_alignment = page_byte_length_,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &input));
    ASSERT_EQ(input->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
    ASSERT_NE(input->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
    ASSERT_EQ(input->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
    ASSERT_EQ(input->host.pointer, nullptr);
  } else {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, input_bytes, &input));
  }
  uint32_t upload_flags = 0;
  uint32_t seed_flags = 0;
  uint32_t readback_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, input, output, &upload_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, input, records, &seed_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryDownloadCacheFlags(input, readback, &readback_flags));

  std::vector<uint32_t> expected_file(kFileBlocks * word_count);
  for (uint32_t block = 0; block < kFileBlocks; ++block) {
    for (uint32_t word = 0; word < word_count; ++word) {
      expected_file[block * word_count + word] =
          0x31415927u + block * 0x243f6a89u + word * 0x01020305u;
    }
  }
  if (test_case.failure == FileFailure::kPartialEof) {
    expected_file.resize(expected_file.size() - word_count / 2);
  }
  ASSERT_NO_FATAL_FAILURE(CreateFile(expected_file, test_case.mode));
  if (IsSkipped()) {
    return;
  }
  ASSERT_NO_FATAL_FAILURE(CreateRing(source));
  if (IsSkipped()) {
    return;
  }

  GpuUserQueue* file_queue = nullptr;
  GpuUserQueue* upload_queue = nullptr;
  GpuUserQueue* reader_queue = nullptr;
  GpuUserQueue* transfer = nullptr;
  for (auto** queue : {&file_queue, &upload_queue, &reader_queue}) {
    ASSERT_NO_FATAL_FAILURE(
        CreateQueue(family_, queue, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                    AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kAqlRingBytes));
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  uint64_t file_packet_index = 0;
  uint64_t upload_index = 0;
  uint64_t reader_index = 0;
  uint64_t file_descriptor = 0;
  uint64_t upload_descriptor = 0;
  uint64_t reader_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*file_queue, *read_kernel,
                                        "staged_read", &file_packet_index,
                                        &file_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*upload_queue, *upload_kernel,
                                        "staged_upload", &upload_index,
                                        &upload_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*reader_queue, *reader_kernel,
                                        "staged_reader", &reader_index,
                                        &reader_descriptor));

  std::fill_n(static_cast<uint32_t*>(seed->host.pointer), slot_words,
              kInputGuard);
  std::fill_n(static_cast<uint32_t*>(readback->host.pointer), input_bytes / 4,
              kOutputGuard);
  std::fill_n(static_cast<uint32_t*>(output->host.pointer), output_bytes / 4,
              kOutputGuard);
  std::fill_n(static_cast<uint32_t*>(records->host.pointer),
              records->info.byte_length / 4, kControlGuard);
  std::vector<uint32_t> expected_requests(requests->info.byte_length / 4,
                                          kControlGuard);
  uint32_t token = 0x91e10da5u;
  for (uint32_t job = 0; job < kJobCount; ++job) {
    token = token * 1664525u + 1013904223u;
    uint32_t request = token;
    if (test_case.failure != FileFailure::kNone) {
      // Keep the partial final block out of earlier successful requests, then
      // select it exactly at the error boundary through the same GPU hash.
      const uint32_t block = RequestHash(request, job + 1) & 7;
      if (job == kFailureJob) {
        request ^= block ^ 7;
      } else if (job < kFailureJob && block == 7) {
        request ^= 1;
      }
    }
    // Every reused source receives different bytes. A reader accidentally
    // following the reused source address cannot pass by seeing the same block.
    if (job >= test_case.source_count &&
        !(test_case.failure != FileFailure::kNone && job == kFailureJob)) {
      const uint32_t previous =
          RequestHash(expected_requests[job - test_case.source_count],
                      job - test_case.source_count + 1) &
          7;
      if ((RequestHash(request, job + 1) & 7) == previous) {
        request ^= 2;
        if (test_case.failure != FileFailure::kNone && job < kFailureJob &&
            (RequestHash(request, job + 1) & 7) == 7) {
          request ^= 4;
        }
      }
    }
    expected_requests[job] = request;
  }
  std::memcpy(requests->host.pointer, expected_requests.data(),
              requests->info.byte_length);
  std::vector<uint32_t> expected_signals(signals->info.byte_length / 4,
                                         kControlGuard);
  auto* job_signals = static_cast<JobSignals*>(signals->host.pointer);
  for (uint32_t job = 0; job <= kJobCount; ++job) {
    JobSignals state = {};
    state.file.kind = state.upload.kind = 1;
    for (auto& reader : state.readers) {
      reader.kind = 1;
    }
    std::memcpy(expected_signals.data() + job * sizeof(JobSignals) / 4, &state,
                sizeof(state));
  }
  std::memcpy(signals->host.pointer, expected_signals.data(),
              signals->info.byte_length);
  for (uint32_t job = 0; job < kJobCount; ++job) {
    job_signals[job].file.value = job_signals[job].upload.value = 1;
    for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
      job_signals[job].readers[reader].value = 1;
    }
  }
  std::vector<uint32_t> expected_control(control->info.byte_length / 4,
                                         kControlGuard);
  aql::Signal terminal = {};
  terminal.kind = 1;
  std::memcpy(expected_control.data(), &terminal, sizeof(terminal));
  expected_control[kCompletionOffset / 4] = 0;
  expected_control[kFileStateOffset / 4] = 0;
  expected_control[kFileStateOffset / 4 + 1] = 0;
  std::memcpy(control->host.pointer, expected_control.data(),
              control->info.byte_length);
  auto& joined = *static_cast<aql::Signal*>(control->host.pointer);
  joined.value = 1;
  const uint64_t completion_address =
      control->device_address + kCompletionOffset;
  const uintptr_t completion_host =
      reinterpret_cast<uintptr_t>(control->host.pointer) + kCompletionOffset;
  uint64_t frontier = 0;
  // Host publication owns only cold initialization and final readback. Both
  // ownership transfers retire command storage as well as the copied bytes.
  auto copy_slots = [&](uint64_t source_address, uint64_t source_stride,
                        uint64_t destination_address, uint32_t flags,
                        uint32_t generation) {
    std::array<uint32_t, 64> words = {};
    SdmaCommandWriter writer(words.data(), sdma_family_.format_features);
    if ((flags & 1) != 0) {
      writer.AcquireFromSystem();
    }
    for (uint32_t slot = 0; slot < test_case.destination_count; ++slot) {
      writer.CopyLinear(source_address + slot * source_stride,
                        destination_address + slot * slot_bytes, slot_bytes);
    }
    if ((flags & 2) != 0) {
      writer.ReleaseToSystem();
    }
    writer.Fence32(completion_address, generation);
    const uint64_t bytes = writer.word_count() * 4;
    const uint64_t tail =
        mapping.ring_byte_length - frontier % mapping.ring_byte_length;
    auto* ring = reinterpret_cast<uint8_t*>(transfer->host.ring_address);
    if (tail < bytes) {
      std::memset(ring + frontier % mapping.ring_byte_length, 0, tail);
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
  const uint64_t initial_frontier = frontier;
  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, input->device_address, 4);
  encoder.Fence32(completion_address, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());

  std::vector<uint8_t> expected_arguments(arguments->info.byte_length, 0);
  std::vector<aql::Packet> files;
  std::vector<aql::Packet> uploads;
  std::vector<aql::Packet> readers;
  constexpr aql::FenceScopes kDependencyScopes = {aql::FenceScope::kNone,
                                                  aql::FenceScope::kNone};
  const auto job_address = [&](uint32_t job) {
    return signals->device_address + job * sizeof(JobSignals);
  };
  const auto reader_address = [&](uint32_t job, uint32_t reader) {
    return job_address(job) + offsetof(JobSignals, readers) +
           reader * sizeof(aql::Signal);
  };
  const auto held_reader = [&](uint32_t job) {
    return job % test_case.reader_count;
  };
  for (uint32_t job = 0; job < kJobCount; ++job) {
    const uint64_t argument_offset = job * kArgumentStride;
    const uint64_t source_offset =
        (job % test_case.source_count) * slot_bytes + page_byte_length_;
    const uint64_t input_offset =
        (job % test_case.destination_count) * slot_bytes + page_byte_length_;
    const uint64_t result_address =
        records->device_address + job * sizeof(protocol::Result);
    const uint32_t previous_job = job >= test_case.source_count
                                      ? job - test_case.source_count
                                      : kJobCount;
    const protocol::ReadArguments read = {
        .submission_entries = device_ring_memory_->device_address,
        .submission_tail = RingAddress(ring_->parameters.sq_off.tail),
        .completion_entries = RingAddress(ring_->parameters.cq_off.cqes),
        .completion_head = RingAddress(ring_->parameters.cq_off.head),
        .completion_tail = RingAddress(ring_->parameters.cq_off.tail),
        .state = control->device_address + kFileStateOffset,
        .result = result_address,
        .request = requests->device_address + job * 4,
        .previous_reader =
            reader_address(previous_job, held_reader(previous_job)) +
            offsetof(aql::Signal, value),
        .host_payload =
            reinterpret_cast<uintptr_t>(source->host.pointer) + source_offset,
        .submission_mask = ring_->parameters.sq_entries - 1,
        .completion_mask = ring_->parameters.cq_entries - 1,
        .word_count = word_count,
        .file_index =
            test_case.failure == FileFailure::kInvalidFile && job == kFailureJob
                ? 1u
                : 0u,
        .generation = job + 1,
        .file_block_mask = kFileBlocks - 1,
        .reserved = {},
    };
    std::memcpy(expected_arguments.data() + argument_offset, &read,
                read_kernel->arguments.byte_length);
    if (job >= test_case.source_count) {
      files.push_back(aql::Barrier(
          aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled, 0,
          {job_address(previous_job) + offsetof(JobSignals, upload), 0, 0, 0,
           0},
          kDependencyScopes));
    }
    files.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
        file_descriptor, arguments->device_address + argument_offset,
        job_address(job)));
    const protocol::UploadArguments upload = {
        .ring = mapping.ring_address,
        .read_index = mapping.read_index_address,
        .write_index = mapping.write_index_address,
        .notification = mapping.doorbell_address,
        .completion = completion_address,
        .state = control->device_address + kFrontierOffset,
        .result = result_address,
        .source = source->device_address + source_offset,
        .destination = input->device_address + input_offset,
        .completion_address = completion_address,
        .capacity = mapping.ring_byte_length,
        .generation = job + 1,
        .copy_control = encoding[2],
        .fence_header = encoding[7],
        .cache_flags = upload_flags,
        .reserved = {},
    };
    std::memcpy(expected_arguments.data() + argument_offset + kUploadOffset,
                &upload, upload_kernel->arguments.byte_length);
    std::array<uint64_t, 5> dependencies = {job_address(job), 0, 0, 0, 0};
    if (job >= test_case.destination_count) {
      for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
        dependencies[1 + reader] =
            reader_address(job - test_case.destination_count, reader);
      }
    }
    uploads.push_back(aql::Barrier(aql::BarrierType::kAnd,
                                   aql::HeaderBarrier::kEnabled, 0,
                                   dependencies, kDependencyScopes));
    uploads.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
        upload_descriptor,
        arguments->device_address + argument_offset + kUploadOffset,
        job_address(job) + offsetof(JobSignals, upload)));
    readers.push_back(aql::Barrier(
        aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
        {job_address(job) + offsetof(JobSignals, upload), 0, 0, 0, 0},
        kDependencyScopes));
    for (uint32_t ordinal = 0; ordinal < test_case.reader_count; ++ordinal) {
      const uint32_t reader =
          (held_reader(job) + 1 + ordinal) % test_case.reader_count;
      // This reader sees the retained copy after the source page is
      // overwritten. The future file read waits on this job's upload, never on
      // its readers.
      if (reader == held_reader(job) &&
          job + test_case.source_count < kJobCount) {
        readers.push_back(aql::Barrier(
            aql::BarrierType::kAnd, aql::HeaderBarrier::kDisabled, 0,
            {job_address(job + test_case.source_count), 0, 0, 0, 0},
            kDependencyScopes));
      }
      const uint64_t reader_offset = argument_offset + kReadersOffset +
                                     reader * sizeof(protocol::ReaderArguments);
      const protocol::ReaderArguments consume = {
          .input = input->device_address + input_offset,
          .output = output->device_address +
                    (job * test_case.reader_count + reader) * slot_bytes +
                    page_byte_length_,
          .result = result_address,
          .reader = reader,
          .reserved = 0,
      };
      std::memcpy(expected_arguments.data() + reader_offset, &consume,
                  reader_kernel->arguments.byte_length);
      readers.push_back(aql::Dispatch(
          aql::HeaderBarrier::kDisabled, {1, {64, 1, 1}, {word_count, 1, 1}}, 0,
          0, reader_descriptor, arguments->device_address + reader_offset,
          reader_address(job, reader)));
    }
  }
  readers.push_back(
      aql::Barrier(aql::BarrierType::kAnd, aql::HeaderBarrier::kEnabled,
                   control->device_address, {}, kDependencyScopes));
  std::memcpy(arguments->host.pointer, expected_arguments.data(),
              arguments->info.byte_length);
  ASSERT_NO_FATAL_FAILURE(
      aql::PrepareBatch(*file_queue, file_packet_index, files));
  ASSERT_NO_FATAL_FAILURE(
      aql::PrepareBatch(*upload_queue, upload_index, uploads));
  ASSERT_NO_FATAL_FAILURE(
      aql::PrepareBatch(*reader_queue, reader_index, readers));
  while (!(GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
           IORING_SQ_NEED_WAKEUP)) {
    std::this_thread::yield();
  }
  aql::PublishBatch(*reader_queue, reader_index, readers);
  aql::PublishBatch(*upload_queue, upload_index, uploads);
  aql::PublishBatch(*file_queue, file_packet_index, files);
  uint64_t wake_count = 0;
  while (GpuLoadAcquire<int64_t>(reinterpret_cast<uintptr_t>(&joined.value)) !=
         0) {
    // Idle wake only: the GPU owns every SQE, CQE and dependent dispatch.
    if ((GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.flags)) &
         IORING_SQ_NEED_WAKEUP) &&
        GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.head)) !=
            GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.tail))) {
      const long result = syscall(__NR_io_uring_enter, ring_->file, 0, 0,
                                  IORING_ENTER_SQ_WAKEUP, nullptr, 0);
      if (result < 0 && errno == EINTR) {
        continue;
      }
      ASSERT_EQ(result, 0) << "wake SQPOLL: " << std::strerror(errno);
      ++wake_count;
    }
    std::this_thread::yield();
  }
  ASSERT_NO_FATAL_FAILURE(
      file_queue->WaitConsumed(api_, file_packet_index + files.size()));
  ASSERT_NO_FATAL_FAILURE(
      upload_queue->WaitConsumed(api_, upload_index + uploads.size()));
  ASSERT_NO_FATAL_FAILURE(
      reader_queue->WaitConsumed(api_, reader_index + readers.size()));

  std::vector<uint32_t> expected_source(source_bytes / 4, kSourceGuard);
  std::vector<uint32_t> expected_input(input_bytes / 4, kInputGuard);
  std::vector<uint32_t> expected_output(output_bytes / 4, kOutputGuard);
  std::vector<uint32_t> expected_records(records->info.byte_length / 4,
                                         kControlGuard);
  const auto* results =
      static_cast<const protocol::Result*>(records->host.pointer);
  uint64_t expected_frontier = initial_frontier;
  uint32_t request_count = 0;
  uint32_t successful_jobs = 0;
  uint32_t overwritten_sources = 0;
  for (uint32_t job = 0; job < kJobCount; ++job) {
    SCOPED_TRACE(job);
    const bool complete =
        test_case.failure == FileFailure::kNone || job < kFailureJob;
    const bool partial =
        test_case.failure == FileFailure::kPartialEof && job == kFailureJob;
    protocol::Result expected = {};
    expected.block =
        RequestHash(expected_requests[job], job + 1) & (kFileBlocks - 1);
    expected.word_count = word_count;
    expected.hash = RequestHash(expected_requests[job], job + 1);
    expected.status = complete                                        ? 0
                      : test_case.failure == FileFailure::kPartialEof ? -ENODATA
                                                                      : -EBADF;
    expected.bytes_read = complete  ? test_case.byte_length
                          : partial ? test_case.byte_length / 2
                                    : 0;
    const uint32_t minimum_requests = complete             ? 1
                                      : partial            ? 2
                                      : job == kFailureJob ? 1
                                                           : 0;
    EXPECT_GE(results[job].request_count, minimum_requests);
    EXPECT_LE(results[job].request_count,
              minimum_requests ? expected.bytes_read + 1 : 0);
    expected.request_count = results[job].request_count;
    request_count += expected.request_count;
    expected.previous_reader_pending = job >= test_case.source_count ? 1 : 0;
    expected.guard = kControlGuard;
    std::fill_n(expected.guards, 6, kControlGuard);
    const uint64_t source_word =
        (job % test_case.source_count) * slot_words + prefix_words;
    const uint64_t input_word =
        (job % test_case.destination_count) * slot_words + prefix_words;
    for (uint32_t word = 0; word < expected.bytes_read / 4; ++word) {
      expected_source[source_word + word] =
          expected_file[expected.block * word_count + word];
    }
    if (complete) {
      ++successful_jobs;
      overwritten_sources += job >= test_case.source_count ? 1 : 0;
      const uint64_t chain_bytes =
          44 + ((upload_flags & 1) ? 20 : 0) + ((upload_flags & 2) ? 20 : 0);
      const uint64_t tail = mapping.ring_byte_length -
                            expected_frontier % mapping.ring_byte_length;
      if (tail < chain_bytes) {
        expected_frontier += tail;
      }
      expected_frontier += chain_bytes;
      for (uint32_t word = 0; word < word_count; ++word) {
        const uint32_t value =
            expected_file[expected.block * word_count + word];
        expected_input[input_word + word] = value;
        for (uint32_t reader = 0; reader < test_case.reader_count; ++reader) {
          const uint64_t output_word =
              (job * test_case.reader_count + reader) * slot_words +
              prefix_words + word;
          expected_output[output_word] = value * 3 + expected.hash + reader;
        }
      }
    }
    expected.frontier = expected_frontier;
    std::memcpy(expected_records.data() + job * sizeof(expected) / 4, &expected,
                sizeof(expected));
  }
  std::memcpy(
      &frontier,
      static_cast<const uint8_t*>(control->host.pointer) + kFrontierOffset,
      sizeof(frontier));
  ASSERT_EQ(frontier, expected_frontier);
  ASSERT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  std::memcpy(expected_control.data() + kFrontierOffset / 4, &frontier,
              sizeof(frontier));
  expected_control[kFileStateOffset / 4] = request_count;
  expected_control[kFileStateOffset / 4 + 1] =
      test_case.failure == FileFailure::kNone ? 0
      : test_case.failure == FileFailure::kPartialEof
          ? static_cast<uint32_t>(-ENODATA)
          : static_cast<uint32_t>(-EBADF);
  expected_control[kCompletionOffset / 4] = successful_jobs;
  ASSERT_NO_FATAL_FAILURE(CheckWords(control->host.pointer, expected_control));
  ASSERT_NO_FATAL_FAILURE(copy_slots(input->device_address, slot_bytes,
                                     readback->device_address, readback_flags,
                                     kJobCount + 1));
  ASSERT_NO_FATAL_FAILURE(CheckWords(source->host.pointer, expected_source));
  ASSERT_NO_FATAL_FAILURE(CheckWords(readback->host.pointer, expected_input));
  ASSERT_NO_FATAL_FAILURE(CheckWords(output->host.pointer, expected_output));
  ASSERT_NO_FATAL_FAILURE(CheckWords(records->host.pointer, expected_records));
  ASSERT_NO_FATAL_FAILURE(CheckWords(signals->host.pointer, expected_signals));
  ASSERT_NO_FATAL_FAILURE(
      CheckWords(requests->host.pointer, expected_requests));
  ASSERT_EQ(std::memcmp(arguments->host.pointer, expected_arguments.data(),
                        arguments->info.byte_length),
            0);
  for (uint32_t offset :
       {ring_->parameters.sq_off.head, ring_->parameters.sq_off.tail,
        ring_->parameters.cq_off.head, ring_->parameters.cq_off.tail}) {
    EXPECT_EQ(GpuLoadAcquire<uint32_t>(RingWord(offset)), request_count);
  }
  EXPECT_EQ(
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.sq_off.dropped)), 0u);
  EXPECT_EQ(
      GpuLoadAcquire<uint32_t>(RingWord(ring_->parameters.cq_off.overflow)),
      0u);
  ASSERT_NO_FATAL_FAILURE(VerifyFile(expected_file, test_case.mode));
  RecordProperty("file_staged_jobs", kJobCount);
  RecordProperty("file_staged_successful_jobs", successful_jobs);
  RecordProperty("file_staged_readers",
                 successful_jobs * test_case.reader_count);
  RecordProperty("file_staged_source_reuses_with_reader_pending",
                 overwritten_sources);
  RecordProperty("file_staged_requests", request_count);
  RecordProperty("file_staged_upload_cache_flags", upload_flags);
  RecordProperty("io_idle_wake_calls", std::to_string(wake_count));
}

class FileStagedOwnershipTest
    : public GpuFileStagedTest,
      public ::testing::WithParamInterface<StagedCase> {};

TEST_P(FileStagedOwnershipTest,
       CopyReleasesSourceAndEveryReaderReleasesDestination) {
  Run(GetParam());
}

std::vector<StagedCase> StagedCases() {
  std::vector<StagedCase> cases;
  // Equal credits, destination lookahead, source lookahead, and both pooled.
  constexpr std::array<std::array<uint32_t, 3>, 4> kCredits = {{
      {1, 1, 1},
      {1, 2, 2},
      {2, 1, 2},
      {2, 4, 2},
  }};
  for (auto memory_class :
       {AMDF_MEMORY_CLASS_SYSTEM, AMDF_MEMORY_CLASS_LOCAL}) {
    for (auto mode : {FileMode::kBuffered, FileMode::kDirect}) {
      for (uint32_t bytes : {4096u, 65536u}) {
        for (const auto& credits : kCredits) {
          cases.push_back({mode, memory_class, credits[0], credits[1],
                           credits[2], bytes, FileFailure::kNone});
        }
      }
    }
    for (auto failure : {FileFailure::kPartialEof, FileFailure::kInvalidFile}) {
      for (const auto& credits : {kCredits[1], kCredits[2]}) {
        cases.push_back({FileMode::kBuffered, memory_class, credits[0],
                         credits[1], credits[2], 4096, failure});
      }
    }
  }
  return cases;
}

INSTANTIATE_TEST_SUITE_P(
    FileToSdma, FileStagedOwnershipTest, ::testing::ValuesIn(StagedCases()),
    [](const ::testing::TestParamInfo<StagedCase>& info) {
      const auto& item = info.param;
      std::string name =
          item.mode == FileMode::kBuffered ? "Buffered" : "Direct";
      name += item.memory_class == AMDF_MEMORY_CLASS_LOCAL ? "Local" : "System";
      name += std::to_string(item.byte_length / 1024) + "KiB";
      name += "S" + std::to_string(item.source_count);
      name += "D" + std::to_string(item.destination_count);
      name += "R" + std::to_string(item.reader_count);
      if (item.failure == FileFailure::kPartialEof) {
        name += "PartialEof";
      } else if (item.failure == FileFailure::kInvalidFile) {
        name += "InvalidFile";
      }
      return name;
    });

}  // namespace
