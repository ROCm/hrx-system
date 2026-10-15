// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/cts/gpu/recipes/device_sdma_staged_fixture.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "libamdf/cts/gpu/kernels/device_sdma_consumer_kernels.h"
#include "libamdf/cts/gpu/kernels/device_sdma_staged.h"
#include "libamdf/cts/gpu/kernels/device_sdma_transfer_kernels.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"

void DeviceGeneratedSdmaStagedTest::Run(const DeviceSdmaStagedCase& test_case) {
  namespace staged = kernels::device_sdma_staged;
  if (test_case.memory_class == AMDF_MEMORY_CLASS_LOCAL &&
      (features_ & AMDF_GPU_DEVICE_FEATURE_LOCAL_MEMORY) == 0) {
    GTEST_SKIP() << "LOCAL placement is not advertised by this device";
  }
  constexpr uint64_t kComputeRingByteLength = 65536;
  ASSERT_NE(family_.ring_byte_length_alignment, 0u);
  if (kComputeRingByteLength < family_.minimum_ring_byte_length ||
      kComputeRingByteLength > family_.maximum_ring_byte_length ||
      kComputeRingByteLength % family_.ring_byte_length_alignment != 0) {
    GTEST_SKIP() << "finite batch requires a 64-KiB AQL ring";
  }
  const auto* transfer_kernel =
      kernels::device_sdma_transfer::kKernels.Find(gpu_endpoint_info_);
  const auto* consumer_kernel =
      kernels::device_sdma_consumer::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(transfer_kernel, nullptr);
  ASSERT_NE(consumer_kernel, nullptr);
  ASSERT_EQ(transfer_kernel->workgroup_size(), 1u);
  ASSERT_EQ(consumer_kernel->workgroup_size(), 64u);
  ASSERT_LE(transfer_kernel->arguments.byte_length,
            sizeof(staged::TransferArguments));
  ASSERT_LE(consumer_kernel->arguments.byte_length,
            sizeof(staged::ConsumerArguments));
  for (const auto* kernel : {transfer_kernel, consumer_kernel}) {
    ASSERT_EQ(kernel->private_segment_byte_length, 0u);
    ASSERT_EQ(kernel->group_segment_byte_length, 0u);
  }

  constexpr uint32_t kSourceGuard = 0x75db8163u;
  constexpr uint32_t kPayloadGuard = 0x32a49bc7u;
  constexpr uint32_t kControlGuard = 0x69ca714bu;
  constexpr uint32_t kRecordGuard = 0xab3265c9u;
  constexpr uint32_t kSeed = 0x91e10da5u;
  constexpr uint32_t kPayloadWordOffset = 16;
  constexpr uint64_t kCompletionByteOffset = 64;
  constexpr uint64_t kStateByteOffset = 128;
  constexpr uint64_t kLengthsByteOffset = 192;
  constexpr uint64_t kArgumentStride = 512;
  constexpr uint64_t kConsumerArgumentOffset = 160;
  constexpr uint64_t kDownloadArgumentOffset = 256;
  constexpr amdf_memory_access_t kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  const auto align_page = [](uint64_t bytes) {
    return (bytes + 4095) & ~UINT64_C(4095);
  };
  const uint32_t maximum_word_count = *std::max_element(
      test_case.word_counts.begin(), test_case.word_counts.end());
  ASSERT_LE(maximum_word_count, 262144u);
  const uint64_t slot_bytes =
      align_page(uint64_t{maximum_word_count} * 4 + 128);
  ASSERT_LT(slot_bytes, UINT64_C(1) << 22);
  const uint64_t slot_words = slot_bytes / sizeof(uint32_t);
  const uint64_t payload_bytes = test_case.slot_count * slot_bytes;
  const uint64_t readback_bytes = test_case.job_count * slot_bytes;
  GpuMemory* source = nullptr;
  GpuMemory* seed = nullptr;
  GpuMemory* input = nullptr;
  GpuMemory* output = nullptr;
  GpuMemory* readback = nullptr;
  GpuMemory* selections = nullptr;
  GpuMemory* control = nullptr;
  GpuMemory* arguments = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 8 * slot_bytes, &source));
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, slot_bytes, &seed));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, readback_bytes, &readback));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      kReadWrite, align_page(test_case.job_count * sizeof(staged::Selection)),
      &selections));
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
        .byte_length = payload_bytes,
        .minimum_alignment = 4096,
        .accesses = &attachment,
    };
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &input));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(local_scope_, creation, &output));
    for (const auto* memory : {input, output}) {
      ASSERT_EQ(memory->info.memory_class, AMDF_MEMORY_CLASS_LOCAL);
      ASSERT_NE(memory->info.flags & AMDF_MEMORY_FLAG_DEVICE_LOCAL, 0u);
      ASSERT_EQ(memory->info.flags & AMDF_MEMORY_FLAG_HOST_VISIBLE, 0u);
      ASSERT_EQ(memory->host.pointer, nullptr);
      ASSERT_EQ(memory->mapping, nullptr);
    }
  } else {
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, payload_bytes, &input));
    ASSERT_NO_FATAL_FAILURE(CreateMemory(kReadWrite, payload_bytes, &output));
  }
  uint32_t upload_flags = 0;
  uint32_t download_flags = 0;
  uint32_t seed_flags = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(source, input, selections, &upload_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryDownloadCacheFlags(output, readback, &download_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, input, selections, &seed_flags));
  ASSERT_NO_FATAL_FAILURE(
      QueryTransferCacheFlags(seed, output, selections, &seed_flags));
  RecordProperty("staged_payload_memory_class", input->info.memory_class);
  RecordProperty("staged_payload_flags", std::to_string(input->info.flags));
  RecordProperty("staged_upload_cache_flags", upload_flags);
  RecordProperty("staged_download_cache_flags", download_flags);
  RecordProperty("staged_slot_count", test_case.slot_count);
  RecordProperty("staged_job_count", test_case.job_count);
  RecordProperty("staged_slot_byte_length", std::to_string(slot_bytes));

  GpuUserQueue* compute = nullptr;
  GpuUserQueue* transfer = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(
      family_, &compute, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
      AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER, kComputeRingByteLength));
  ASSERT_NO_FATAL_FAILURE(
      CreateQueue(sdma_family_, &transfer, AMDF_QUEUE_PRODUCER_MODE_SINGLE, {},
                  AMDF_USER_QUEUE_CAPABILITY_HOST_PRODUCER |
                      AMDF_USER_QUEUE_CAPABILITY_DEVICE_PRODUCER));
  const auto& mapping = transfer->producer.info;
  uint64_t packet_index = 0;
  uint64_t transfer_descriptor = 0;
  uint64_t consumer_descriptor = 0;
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*compute, *transfer_kernel,
                                        "staged_transfer", &packet_index,
                                        &transfer_descriptor));
  ASSERT_NO_FATAL_FAILURE(PublishKernel(*compute, *consumer_kernel,
                                        "staged_consumer", &packet_index,
                                        &consumer_descriptor));

  std::vector<uint32_t> expected_source(8 * slot_words, kSourceGuard);
  for (uint32_t page = 0; page < 8; ++page) {
    for (uint32_t word = 0; word < maximum_word_count; ++word) {
      expected_source[page * slot_words + kPayloadWordOffset + word] =
          0x31415927u + page * 0x243f6a89u + word * 0x1020305u;
    }
  }
  std::memcpy(source->host.pointer, expected_source.data(),
              source->info.byte_length);
  std::fill_n(static_cast<uint32_t*>(seed->host.pointer), slot_words,
              kPayloadGuard);
  std::fill_n(static_cast<uint32_t*>(readback->host.pointer),
              readback_bytes / sizeof(uint32_t), kRecordGuard);
  std::vector<uint32_t> expected_selections(
      selections->info.byte_length / sizeof(uint32_t), kRecordGuard);
  std::memcpy(selections->host.pointer, expected_selections.data(),
              selections->info.byte_length);
  std::array<uint32_t, 1024> expected_control;
  expected_control.fill(kControlGuard);
  const aql::Signal final_signal = aql::UserSignal(1);
  std::memcpy(expected_control.data(), &final_signal, sizeof(final_signal));
  expected_control[kCompletionByteOffset / 4] = 0;
  expected_control[kStateByteOffset / 4 + 2] = kSeed;
  std::memcpy(expected_control.data() + kLengthsByteOffset / 4,
              test_case.word_counts.data(), sizeof(test_case.word_counts));
  std::memcpy(control->host.pointer, expected_control.data(),
              control->info.byte_length);
  auto& signal = *static_cast<aql::Signal*>(control->host.pointer);
  const uint64_t completion_host =
      reinterpret_cast<uintptr_t>(control->host.pointer) +
      kCompletionByteOffset;
  const uint64_t completion_address =
      control->device_address + kCompletionByteOffset;

  // Cold initialization is the host's only SDMA publication. Its completion
  // and RPTR join transfer ownership to the GPU before any workload starts.
  std::array<uint32_t, 96> seed_commands = {};
  SdmaCommandWriter seed_encoder(seed_commands.data(),
                                 sdma_family_.format_features);
  if ((seed_flags & 1) != 0) {
    seed_encoder.AcquireFromSystem();
  }
  for (uint32_t slot = 0; slot < test_case.slot_count; ++slot) {
    for (const auto* memory : {input, output}) {
      seed_encoder.CopyLinear(seed->device_address,
                              memory->device_address + slot * slot_bytes,
                              static_cast<uint32_t>(slot_bytes));
    }
  }
  if ((seed_flags & 2) != 0) {
    seed_encoder.ReleaseToSystem();
  }
  seed_encoder.Fence32(completion_address, 1);
  uint64_t frontier = seed_encoder.word_count() * sizeof(uint32_t);
  ASSERT_LT(frontier, mapping.ring_byte_length);
  std::memcpy(reinterpret_cast<void*>(transfer->host.ring_address),
              seed_commands.data(), frontier);
  transfer->PublishStream(frontier);
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(completion_host, 1));
  ASSERT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  GpuStoreRelease<uint32_t>(completion_host, 0);
  std::memcpy(static_cast<uint8_t*>(control->host.pointer) + kStateByteOffset,
              &frontier, sizeof(frontier));

  std::array<uint32_t, 11> encoding;
  SdmaCommandWriter encoder(encoding.data(), sdma_family_.format_features);
  encoder.CopyLinear(source->device_address, input->device_address, 4);
  encoder.Fence32(completion_address, 1);
  ASSERT_EQ(encoder.word_count(), encoding.size());
  std::vector<uint8_t> expected_arguments(arguments->info.byte_length, 0);
  std::vector<aql::Packet> packets;
  packets.reserve(test_case.job_count * 3);
  std::vector<uint32_t> expected_payload(payload_bytes / 4, kPayloadGuard);
  std::vector<uint32_t> expected_readback(readback_bytes / 4, kRecordGuard);
  uint32_t state = kSeed;
  uint32_t selected_pages = 0;
  uint32_t selected_slots = 0;
  uint32_t selected_lengths = 0;
  uint64_t uploaded_bytes = 0;
  auto advance_frontier = [&](uint32_t flags) {
    const uint64_t chain_bytes =
        44 + ((flags & 1) != 0 ? 20 : 0) + ((flags & 2) != 0 ? 20 : 0);
    const uint64_t tail =
        mapping.ring_byte_length - frontier % mapping.ring_byte_length;
    if (tail < chain_bytes) {
      frontier += tail;
    }
    frontier += chain_bytes;
  };
  for (uint32_t job = 0; job < test_case.job_count; ++job) {
    const uint64_t selection_address =
        selections->device_address + job * sizeof(staged::Selection);
    const uint64_t readback_address =
        readback->device_address + job * slot_bytes;
    const uint64_t argument_offset = job * kArgumentStride;
    staged::TransferArguments transfer_arguments = {
        .ring = mapping.ring_address,
        .read_index = mapping.read_index_address,
        .write_index = mapping.write_index_address,
        .notification = mapping.doorbell_address,
        .completion = completion_address,
        .state = control->device_address + kStateByteOffset,
        .selection = selection_address,
        .lengths = control->device_address + kLengthsByteOffset,
        .readback = readback_address,
        .source_address = source->device_address,
        .input_address = input->device_address,
        .output_address = output->device_address,
        .readback_address = readback_address,
        .completion_address = completion_address,
        .capacity = mapping.ring_byte_length,
        .slot_byte_length = slot_bytes,
        .job_index = job,
        .slot_count = test_case.slot_count,
        .phase = staged::Phase::kUpload,
        .copy_control = encoding[2],
        .fence_header = encoding[7],
        .cache_flags = upload_flags,
    };
    std::memcpy(expected_arguments.data() + argument_offset,
                &transfer_arguments, transfer_kernel->arguments.byte_length);
    const staged::ConsumerArguments consumer_arguments = {
        .input = input->device_address,
        .output = output->device_address,
        .selection = selection_address,
        .slot_byte_length = slot_bytes,
    };
    std::memcpy(
        expected_arguments.data() + argument_offset + kConsumerArgumentOffset,
        &consumer_arguments, consumer_kernel->arguments.byte_length);
    transfer_arguments.phase = staged::Phase::kDownload;
    transfer_arguments.cache_flags = download_flags;
    std::memcpy(
        expected_arguments.data() + argument_offset + kDownloadArgumentOffset,
        &transfer_arguments, transfer_kernel->arguments.byte_length);
    const uint64_t argument_address =
        arguments->device_address + argument_offset;
    packets.push_back(aql::Dispatch(aql::HeaderBarrier::kEnabled,
                                    {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
                                    transfer_descriptor, argument_address, 0));
    const uint32_t grid_words = (maximum_word_count + 63) & ~63u;
    packets.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {64, 1, 1}, {grid_words, 1, 1}}, 0, 0,
        consumer_descriptor, argument_address + kConsumerArgumentOffset, 0));
    packets.push_back(aql::Dispatch(
        aql::HeaderBarrier::kEnabled, {1, {1, 1, 1}, {1, 1, 1}}, 0, 0,
        transfer_descriptor, argument_address + kDownloadArgumentOffset,
        job + 1 == test_case.job_count ? control->device_address : 0));

    staged::Selection selection = {};
    std::fill_n(selection.guards, 6, kRecordGuard);
    selection.page = state & 7;
    selection.slot = (state >> 3) & (test_case.slot_count - 1);
    const uint32_t length_index = (state >> 8) & 3;
    selection.word_count = test_case.word_counts[length_index];
    selection.prior_state = state;
    selection.addend = state ^ ((job + 1) * 0x1020305u);
    selected_pages |= 1u << selection.page;
    selected_slots |= 1u << selection.slot;
    selected_lengths |= 1u << length_index;
    const uint64_t source_start =
        selection.page * slot_words + kPayloadWordOffset;
    const uint64_t output_start =
        selection.slot * slot_words + kPayloadWordOffset;
    for (uint32_t word = 0; word < selection.word_count; ++word) {
      expected_payload[output_start + word] =
          expected_source[source_start + word] * 3u + selection.addend;
    }
    std::copy_n(expected_payload.begin() + selection.slot * slot_words,
                slot_words, expected_readback.begin() + job * slot_words);
    state = expected_payload[output_start] ^
            (expected_payload[output_start + selection.word_count - 1] +
             (job + 1) * 0x9e3779b1u);
    selection.next_state = state;
    advance_frontier(upload_flags);
    selection.upload_frontier = frontier;
    advance_frontier(download_flags);
    selection.download_frontier = frontier;
    std::memcpy(expected_selections.data() + job * sizeof(selection) / 4,
                &selection, sizeof(selection));
    uploaded_bytes += uint64_t{selection.word_count} * sizeof(uint32_t);
  }
  ASSERT_EQ(selected_pages, 0xffu);
  ASSERT_EQ(selected_slots, (1u << test_case.slot_count) - 1);
  ASSERT_EQ(selected_lengths, 0xfu);
  if (test_case.job_count == 97) {
    ASSERT_GT(frontier / mapping.ring_byte_length, 1u);
  }
  std::memcpy(arguments->host.pointer, expected_arguments.data(),
              expected_arguments.size());
  expected_control[2] = 0;  // Low word of the final AQL signal value.
  expected_control[kCompletionByteOffset / 4] = test_case.job_count * 2;
  std::memcpy(expected_control.data() + kStateByteOffset / 4, &frontier,
              sizeof(frontier));
  expected_control[kStateByteOffset / 4 + 2] = state;

  // Write every later packet while the first header is still INVALID. This
  // prevents hardware polling from starting work before the complete finite
  // batch is ready, independently of when its doorbell is observed.
  const uint64_t capacity =
      compute->host.ring_byte_length / sizeof(aql::Packet);
  ASSERT_LT(packets.size(), capacity);
  ASSERT_EQ(GpuLoadAcquire<uint64_t>(compute->host.read_index_address),
            packet_index);
  auto* ring = reinterpret_cast<aql::Packet*>(compute->host.ring_address);
  auto& first_packet = ring[packet_index & (capacity - 1)];
  ASSERT_EQ(first_packet[0] & 0xffu, 1u);
  std::memcpy(first_packet.data() + 1, packets.front().data() + 1,
              sizeof(aql::Packet) - sizeof(uint32_t));
  for (size_t i = 1; i < packets.size(); ++i) {
    auto& target = ring[(packet_index + i) & (capacity - 1)];
    std::memcpy(target.data(), packets[i].data(), sizeof(target));
  }
  const uint64_t end_index = packet_index + packets.size();
  GpuStoreRelease(compute->host.write_index_address, end_index);
  GpuStoreRelease(reinterpret_cast<uintptr_t>(first_packet.data()),
                  packets.front()[0]);
  GpuRingDoorbell(compute->host.doorbell_address, end_index - 1);

  // No per-job CPU service or joins. Capture the actual result before native
  // consumption/teardown can obscure the GPU-to-SDMA-to-consumer dataflow.
  ASSERT_TRUE(
      GpuWaitEqual<int64_t>(reinterpret_cast<uintptr_t>(&signal.value), 0));
  std::vector<uint32_t> observed_readback(expected_readback.size());
  std::memcpy(observed_readback.data(), readback->host.pointer, readback_bytes);
  EXPECT_EQ(observed_readback, expected_readback);
  EXPECT_EQ(std::memcmp(selections->host.pointer, expected_selections.data(),
                        selections->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(source->host.pointer, expected_source.data(),
                        source->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(arguments->host.pointer, expected_arguments.data(),
                        arguments->info.byte_length),
            0);
  EXPECT_EQ(std::memcmp(control->host.pointer, expected_control.data(),
                        control->info.byte_length),
            0);
  EXPECT_EQ(GpuLoadAcquire<uint64_t>(transfer->host.write_index_address),
            frontier);
  EXPECT_NO_FATAL_FAILURE(compute->WaitConsumed(api_, end_index));
  EXPECT_NO_FATAL_FAILURE(transfer->WaitConsumed(api_, frontier));
  RecordProperty("staged_uploaded_bytes", std::to_string(uploaded_bytes));
  RecordProperty("staged_readback_bytes", std::to_string(readback_bytes));
  RecordProperty("staged_sdma_frontier", std::to_string(frontier));
  RecordProperty("staged_sdma_ring_wraps",
                 std::to_string(frontier / mapping.ring_byte_length));
}

std::string DeviceSdmaStagedCaseName(
    const ::testing::TestParamInfo<DeviceSdmaStagedCase>& info) {
  return info.param.name;
}
