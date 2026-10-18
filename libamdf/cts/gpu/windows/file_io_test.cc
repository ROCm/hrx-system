// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if !defined(_WIN32_WINNT) || _WIN32_WINNT < 0x0602
#undef _WIN32_WINNT
#define _WIN32_WINNT 0x0602
#endif  // _WIN32_WINNT
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif  // WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif  // NOMINMAX
#include <windows.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

#include "libamdf/cts/gpu/kernels/transform.h"
#include "libamdf/cts/gpu/kernels/transform_kernels.h"
#include "libamdf/cts/gpu/pm4/encoding/commands.h"
#include "libamdf/cts/gpu/sdma/encoding/commands.h"
#include "libamdf/cts/gpu/util/command_fixture.h"
#include "util/wait.h"

namespace {

enum class Backing { kAllocated, kRegistered };
enum class FileMode { kBuffered, kUnbuffered };

class GpuFileIoTest
    : public GpuCommandTest,
      public ::testing::WithParamInterface<
          std::tuple<amdf_queue_command_type_t, Backing, FileMode>> {
 protected:
  GpuFileIoTest()
      : GpuCommandTest(QueueRequirements(std::get<0>(GetParam()))) {}

  static GpuQueueRequirements QueueRequirements(
      amdf_queue_command_type_t type) {
    GpuQueueRequirements requirements = {
        .command_type = type,
        .roles = AMDF_QUEUE_ROLE_TRANSFER,
        .publication_modes = AMDF_QUEUE_PUBLICATION_MODE_USER |
                             AMDF_QUEUE_PUBLICATION_MODE_KERNEL,
    };
    if (type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
      requirements.roles |=
          AMDF_QUEUE_ROLE_COMPUTE | AMDF_QUEUE_ROLE_CACHE_CONTROL;
      requirements.format_features =
          AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR;
      requirements.cache_operations = AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM |
                                      AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM;
      requirements.cache_transition_kinds = AMDF_CACHE_TRANSITION_KINDS_GLOBAL;
    }
    return requirements;
  }

  amdf_status_t MatchGpuEndpoint(amdf_endpoint_t* endpoint,
                                 bool* out_matches) override {
    if (std::get<0>(GetParam()) == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4) {
      amdf_gpu_endpoint_info_t info = {
          .type = AMDF_STRUCTURE_TYPE_GPU_ENDPOINT_INFO,
          .structure_size = sizeof(info),
      };
      const auto status = gpu_api_->endpoint_query_info(endpoint, &info);
      if (!amdf_status_is_ok(status)) {
        return status;
      }
      pm4_profile_ = Pm4CommandProfile::Find(info);
      if (!pm4_profile_) {
        *out_matches = false;
        return AMDF_STATUS_OK;
      }
    }
    return GpuCommandTest::MatchGpuEndpoint(endpoint, out_matches);
  }

  void TearDown() override {
    if (operation_ != nullptr) {
      ASSERT_NO_FATAL_FAILURE(CancelTransfer());
    }
    // Retire the last shader reader before removing its aliases. Keep backing
    // and borrowed caller pages alive if native queue or alias teardown fails.
    if (pool_queue_ != nullptr) {
      ASSERT_TRUE(pool_queue_->Release(api_));
    }
    if (pool_reservation_ != nullptr) {
      ASSERT_EQ(api_->memory_reservation_destroy(pool_reservation_),
                AMDF_STATUS_OK);
      pool_reservation_ = nullptr;
    }
    // Queue retirement and checked native detach precede freeing caller pages.
    ASSERT_NO_FATAL_FAILURE(GpuCommandTest::TearDown());
    for (void* allocation : host_allocations_) {
      EXPECT_TRUE(VirtualFree(allocation, 0, MEM_RELEASE)) << GetLastError();
    }
    if (file_ != INVALID_HANDLE_VALUE) {
      EXPECT_TRUE(CloseHandle(file_)) << GetLastError();
    }
    if (port_) {
      EXPECT_TRUE(CloseHandle(port_)) << GetLastError();
    }
    if (path_[0]) {
      EXPECT_TRUE(DeleteFileW(path_)) << GetLastError();
    }
  }

  void OpenFile(FileMode mode) {
    wchar_t directory[MAX_PATH] = {};
    const DWORD directory_length = GetTempPathW(MAX_PATH, directory);
    ASSERT_GT(directory_length, 0u) << GetLastError();
    ASSERT_LT(directory_length, MAX_PATH);
    ASSERT_NE(GetTempFileNameW(directory, L"amf", 0, path_), 0u)
        << GetLastError();
    file_ = CreateFileW(
        path_, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED |
            (mode == FileMode::kUnbuffered ? FILE_FLAG_NO_BUFFERING : 0),
        nullptr);
    ASSERT_NE(file_, INVALID_HANDLE_VALUE) << GetLastError();
    // File buffering and durability are separate choices. WRITE_THROUGH is
    // not required for this completed-transfer visibility witness.
    if (mode == FileMode::kUnbuffered) {
      FILE_ALIGNMENT_INFO alignment = {};
      FILE_STORAGE_INFO storage = {};
      ASSERT_TRUE(GetFileInformationByHandleEx(file_, FileAlignmentInfo,
                                               &alignment, sizeof(alignment)))
          << GetLastError();
      ASSERT_TRUE(GetFileInformationByHandleEx(file_, FileStorageInfo, &storage,
                                               sizeof(storage)))
          << GetLastError();
      buffer_alignment_ = std::max<uint64_t>(
          static_cast<uint64_t>(alignment.AlignmentRequirement) + 1,
          std::max(storage.PhysicalBytesPerSectorForAtomicity,
                   storage.LogicalBytesPerSector));
      transfer_alignment_ = storage.LogicalBytesPerSector;
      ASSERT_TRUE(std::has_single_bit(buffer_alignment_));
      ASSERT_TRUE(std::has_single_bit(transfer_alignment_));
    }
    RecordProperty("file_buffer_alignment", std::to_string(buffer_alignment_));
    RecordProperty("file_transfer_alignment",
                   std::to_string(transfer_alignment_));
    port_ = CreateIoCompletionPort(file_, nullptr, kCompletionKey, 1);
    ASSERT_NE(port_, nullptr) << GetLastError();
  }

  void AllocateHostStorage(uint64_t byte_length, uint64_t alignment,
                           uint8_t** out_pointer) {
    ASSERT_TRUE(std::has_single_bit(alignment));
    ASSERT_LE(byte_length, SIZE_MAX - (alignment - 1));
    void* allocation = VirtualAlloc(nullptr, byte_length + alignment - 1,
                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    ASSERT_NE(allocation, nullptr) << GetLastError();
    host_allocations_.push_back(allocation);
    const uintptr_t base = reinterpret_cast<uintptr_t>(allocation);
    *out_pointer =
        reinterpret_cast<uint8_t*>((base + alignment - 1) & ~(alignment - 1));
  }

  void CreatePayload(Backing backing, uint64_t byte_length, uint64_t alignment,
                     GpuMemory** out_memory,
                     amdf_memory_flags_t flags = AMDF_MEMORY_FLAG_HOST_COHERENT,
                     amdf_memory_access_t required_access = 0) {
    const bool registered = backing == Backing::kRegistered;
    const amdf_memory_flags_t backing_flags =
        AMDF_MEMORY_FLAG_HOST_VISIBLE |
        (registered ? 0 : AMDF_MEMORY_FLAG_HOST_FILE_IO);
    const amdf_memory_device_access_t access = {
        .device = device_,
        .requirements =
            {
                .access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                          required_access,
                .flags = flags | AMDF_MEMORY_FLAG_DEVICE_ADDRESS,
            },
    };
    const auto ordinal = FindMemoryProfileOrdinal(
        system_scope_,
        AMDF_MEMORY_PROFILE_ROLE_HOST_MAP |
            (registered ? AMDF_MEMORY_PROFILE_ROLE_REGISTER
                        : AMDF_MEMORY_PROFILE_ROLE_CREATE),
        backing_flags, access.requirements);
    ASSERT_NE(ordinal, AMDF_MEMORY_PROFILE_ORDINAL_UNKNOWN);
    amdf_memory_profile_t profile = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE,
        .structure_size = sizeof(profile),
    };
    amdf_memory_access_capabilities_t capabilities = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_ACCESS_CAPABILITIES,
        .structure_size = sizeof(capabilities),
    };
    ASSERT_EQ(QueryMemoryProfile(system_scope_, ordinal, access.requirements,
                                 &profile, &capabilities),
              AMDF_STATUS_OK);
    ASSERT_EQ(profile.availability, AMDF_STATUS_OK);
    ASSERT_EQ(profile.host_mapping.features & AMDF_HOST_MAPPING_FEATURE_FILE_IO,
              registered ? 0u : AMDF_HOST_MAPPING_FEATURE_FILE_IO);
    const auto& geometry =
        registered ? profile.registration : profile.allocation;
    alignment = std::max(alignment, geometry.minimum_alignment);
    if (registered) {
      alignment =
          std::max(alignment, geometry.registered_host_pointer_alignment);
      ASSERT_EQ(geometry.registered_host_cacheability,
                AMDF_HOST_CACHEABILITY_WRITE_BACK);
    }
    ASSERT_LE(alignment, geometry.maximum_alignment);
    ASSERT_GT(geometry.byte_length_granularity, 0u);
    byte_length = (byte_length + geometry.byte_length_granularity - 1) /
                  geometry.byte_length_granularity *
                  geometry.byte_length_granularity;
    ASSERT_LE(byte_length, geometry.maximum_byte_length);
    amdf_memory_create_info_t creation = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_CREATE_INFO,
        .structure_size = sizeof(creation),
        .memory_profile_ordinal = ordinal,
        .access_count = 1,
        .required_flags = backing_flags,
        .byte_length = byte_length,
        .minimum_alignment = alignment,
        .accesses = &access,
    };
    if (registered) {
      uint8_t* pointer = nullptr;
      ASSERT_NO_FATAL_FAILURE(
          AllocateHostStorage(byte_length, alignment, &pointer));
      creation.registered_host_pointer = pointer;
      creation.registered_host_cacheability = AMDF_HOST_CACHEABILITY_WRITE_BACK;
    }
    ASSERT_NO_FATAL_FAILURE(CreateMemory(system_scope_, creation, out_memory));
    ASSERT_EQ((*out_memory)->host.features, profile.host_mapping.features);
    ASSERT_EQ(reinterpret_cast<uintptr_t>((*out_memory)->host.pointer) %
                  buffer_alignment_,
              0u);
  }

  void BeginTransfer(bool write, void* buffer, DWORD byte_length,
                     uint64_t file_offset) {
    ASSERT_EQ(reinterpret_cast<uintptr_t>(buffer) % buffer_alignment_, 0u);
    ASSERT_EQ(byte_length % transfer_alignment_, 0u);
    ASSERT_EQ(file_offset % transfer_alignment_, 0u);
    ASSERT_EQ(operation_, nullptr);
    operation_ = new OVERLAPPED{};
    operation_->Offset = static_cast<DWORD>(file_offset);
    operation_->OffsetHigh = static_cast<DWORD>(file_offset >> 32);
    const BOOL accepted =
        write ? WriteFile(file_, buffer, byte_length, nullptr, operation_)
              : ReadFile(file_, buffer, byte_length, nullptr, operation_);
    const DWORD immediate_error = accepted ? ERROR_SUCCESS : GetLastError();
    if (!accepted && immediate_error != ERROR_IO_PENDING) {
      delete operation_;
      operation_ = nullptr;
    }
    ASSERT_TRUE(accepted || immediate_error == ERROR_IO_PENDING)
        << immediate_error;
  }

  void CompleteTransfer(DWORD byte_length) {
    ASSERT_NE(operation_, nullptr);
    DWORD completed_length = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* completed = nullptr;
    const BOOL success =
        GetQueuedCompletionStatus(port_, &completed_length, &key, &completed,
                                  amdf::cts::WaitDeadline().remaining_ms());
    const DWORD completion_error = success ? ERROR_SUCCESS : GetLastError();
    ASSERT_EQ(completed, operation_)
        << "waiting for file I/O: " << completion_error;
    delete operation_;
    operation_ = nullptr;
    ASSERT_EQ(key, kCompletionKey);
    ASSERT_TRUE(success) << completion_error;
    ASSERT_EQ(completed_length, byte_length);
  }

  void Transfer(bool write, void* buffer, DWORD byte_length,
                uint64_t file_offset) {
    ASSERT_NO_FATAL_FAILURE(
        BeginTransfer(write, buffer, byte_length, file_offset));
    ASSERT_NO_FATAL_FAILURE(CompleteTransfer(byte_length));
  }

  void CancelTransfer() {
    ASSERT_NE(operation_, nullptr);
    const BOOL cancelled = CancelIoEx(file_, operation_);
    const DWORD error = cancelled ? ERROR_SUCCESS : GetLastError();
    EXPECT_TRUE(cancelled || error == ERROR_NOT_FOUND) << error;
    // Cancellation only requests completion. Retain the OVERLAPPED and all
    // payload owners if no completion packet proves native access has ended.
    DWORD completed_length = 0;
    ULONG_PTR key = 0;
    OVERLAPPED* completed = nullptr;
    const BOOL success =
        GetQueuedCompletionStatus(port_, &completed_length, &key, &completed,
                                  amdf::cts::WaitDeadline().remaining_ms());
    const DWORD completion_error = success ? ERROR_SUCCESS : GetLastError();
    ASSERT_EQ(completed, operation_)
        << "I/O cancellation did not retire: " << completion_error;
    delete operation_;
    operation_ = nullptr;
    EXPECT_EQ(key, kCompletionKey);
    EXPECT_TRUE(success || completion_error == ERROR_OPERATION_ABORTED)
        << completion_error;
    RecordProperty("cancelled_io_completion_status", completion_error);
  }

  void ResolveQueueTransition(const amdf_cache_transition_t& transition,
                              amdf_cache_operation_t operation,
                              amdf_cache_operations_t* out_operations) {
    if (transition.kind == AMDF_CACHE_TRANSITION_KIND_NONE) {
      return;
    }
    ASSERT_EQ(transition.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
    ASSERT_EQ(transition.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
    ASSERT_EQ(transition.operation, operation);
    amdf_queue_format_features_t feature =
        AMDF_GPU_PM4_FORMAT_FEATURE_ACQUIRE_MEM_GCR;
    if (family_.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
      feature = AMDF_GPU_SDMA_FORMAT_FEATURE_USER_GCR;
    }
    ASSERT_NE(family_.format_features & feature, 0u);
    ASSERT_NE(family_.cache_operations & (UINT64_C(1) << operation), 0u);
    ASSERT_NE(
        family_.cache_transition_kinds & AMDF_CACHE_TRANSITION_KINDS_GLOBAL,
        0u);
    *out_operations |= UINT64_C(1) << operation;
  }

  void QueryCacheOperations(GpuMemory* source, GpuMemory* target,
                            amdf_cache_operations_t* out_operations) {
    *out_operations = 0;
    const auto source_host = source->HostSite();
    const auto source_device = source->DeviceSite(family_.ordinal);
    const auto target_host = target->HostSite();
    const auto target_device = target->DeviceSite(family_.ordinal);
    amdf_memory_pair_info_t ingress = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(ingress),
    };
    amdf_memory_pair_info_t egress = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(egress),
    };
    ASSERT_EQ(
        api_->memory_query_pair_info(&source_host, &source_device, &ingress),
        AMDF_STATUS_OK);
    ASSERT_EQ(
        api_->memory_query_pair_info(&target_device, &target_host, &egress),
        AMDF_STATUS_OK);
    ASSERT_NE(ingress.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
              0u);
    ASSERT_NE(egress.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
              0u);
    ASSERT_EQ(ingress.release.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_EQ(egress.acquire.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
    ASSERT_NO_FATAL_FAILURE(ResolveQueueTransition(
        ingress.acquire, AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM,
        out_operations));
    ASSERT_NO_FATAL_FAILURE(ResolveQueueTransition(
        egress.release, AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM,
        out_operations));
  }

  void ReadIndependentFile(std::vector<uint8_t>* contents) {
    ASSERT_TRUE(CloseHandle(file_)) << GetLastError();
    file_ = INVALID_HANDLE_VALUE;
    file_ = CreateFileW(path_, GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    ASSERT_NE(file_, INVALID_HANDLE_VALUE) << GetLastError();
    LARGE_INTEGER length = {};
    ASSERT_TRUE(GetFileSizeEx(file_, &length)) << GetLastError();
    ASSERT_EQ(length.QuadPart, contents->size());
    DWORD completed_length = 0;
    ASSERT_TRUE(ReadFile(file_, contents->data(),
                         static_cast<DWORD>(contents->size()),
                         &completed_length, nullptr))
        << GetLastError();
    ASSERT_EQ(completed_length, contents->size());
  }

  void CopySwappedBlocks(GpuCommandQueue* queue, GpuMemory* source,
                         GpuMemory* target, GpuMemory* completion,
                         uint32_t block_byte_length,
                         amdf_cache_operations_t cache_operations) {
    ASSERT_GE(queue->words().size(), 512u);
    if (family_.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA) {
      SdmaCommandWriter commands(queue->words().data(),
                                 family_.format_features);
      if (cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
        commands.AcquireFromSystem();
      }
      commands.CopyLinear(source->device_address + block_byte_length,
                          target->device_address + 2 * block_byte_length,
                          block_byte_length);
      commands.CopyLinear(source->device_address + 2 * block_byte_length,
                          target->device_address + block_byte_length,
                          block_byte_length);
      if (cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
        commands.ReleaseToSystem();
      }
      commands.Fence32(completion->device_address, 1);
      ASSERT_NO_FATAL_FAILURE(
          queue->Publish(api_, gpu_api_, commands.word_count()));
    } else {
      Pm4CommandWriter commands(queue->words().data(), *pm4_profile_);
      if (cache_operations & AMDF_CACHE_OPERATIONS_ACQUIRE_FROM_SYSTEM) {
        commands.AcquireFromSystem();
      }
      // Use the same conservative CP-DMA chunk policy as the PM4 corpus.
      for (uint32_t block = 0; block < 2; ++block) {
        for (uint32_t offset = 0; offset < block_byte_length;) {
          const uint32_t length =
              std::min<uint32_t>(32736, block_byte_length - offset);
          commands.DmaCopy(
              source->device_address + (block + 1) * block_byte_length + offset,
              target->device_address + (2 - block) * block_byte_length + offset,
              length);
          offset += length;
        }
      }
      commands.WaitDma();
      if (cache_operations & AMDF_CACHE_OPERATIONS_RELEASE_TO_SYSTEM) {
        commands.SystemBarrier();
      }
      commands.WriteData32(completion->device_address, 1);
      commands.PadToEightWords();
      ASSERT_NO_FATAL_FAILURE(
          queue->Publish(api_, gpu_api_, commands.word_count()));
    }
  }

  // Private port identity; every operation is consumed before the next begins.
  static constexpr ULONG_PTR kCompletionKey = 17;
  // Native file closed before independent readback or fixture cleanup.
  HANDLE file_ = INVALID_HANDLE_VALUE;
  // Private IOCP retaining no outstanding requests at teardown.
  HANDLE port_ = nullptr;
  // One outstanding native request, retained with its payload until an IOCP
  // packet establishes completion, including after cancellation or timeout.
  OVERLAPPED* operation_ = nullptr;
  // Owned temporary file, removed only after all handles close.
  wchar_t path_[MAX_PATH] = {};
  // Caller-owned native allocations retained through registration teardown.
  std::vector<void*> host_allocations_;
  // Queried native buffer-address alignment in bytes.
  uint64_t buffer_alignment_ = 1;
  // Queried native file offset and transfer-length alignment in bytes.
  uint64_t transfer_alignment_ = 1;
  // Static PM4 encoding profile, absent for the independent SDMA path.
  const Pm4CommandProfile* pm4_profile_ = nullptr;
  // Shader queue retired before its remappable addresses are released.
  GpuCommandQueue* pool_queue_ = nullptr;
  // GPU address owner; every referenced backing remains fixture-owned.
  amdf_memory_reservation_t* pool_reservation_ = nullptr;
};

TEST_P(GpuFileIoTest, IocpAndGpuShareBackingWithoutHostPayloadCopies) {
  const auto [command_type, backing, mode] = GetParam();
  ASSERT_NO_FATAL_FAILURE(OpenFile(mode));
  const uint64_t block_byte_length =
      std::max({UINT64_C(65536), buffer_alignment_, transfer_alignment_});
  // Each half fits one conservative SDMA copy packet; the complete file fits
  // one Windows DWORD-sized transfer. This bounds the witness's resource use.
  ASSERT_LE(block_byte_length, UINT64_C(1) << 20);
  const auto payload_byte_length = static_cast<DWORD>(2 * block_byte_length);
  const auto file_byte_length = static_cast<DWORD>(8 * block_byte_length);
  constexpr uint8_t kGuard = 0xa5;
  std::vector<uint8_t> expected_file(file_byte_length, kGuard);
  for (uint64_t i = block_byte_length; i < 3 * block_byte_length; ++i) {
    expected_file[i] = static_cast<uint8_t>(i * 73 + (i >> 9) * 19 + 7);
  }
  uint8_t* seed = nullptr;
  ASSERT_NO_FATAL_FAILURE(
      AllocateHostStorage(file_byte_length, buffer_alignment_, &seed));
  std::memcpy(seed, expected_file.data(), expected_file.size());
  ASSERT_NO_FATAL_FAILURE(Transfer(true, seed, file_byte_length, 0));

  GpuMemory* source = nullptr;
  GpuMemory* target = nullptr;
  GpuMemory* completion = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreatePayload(backing, 4 * block_byte_length,
                                        buffer_alignment_, &source));
  ASSERT_NO_FATAL_FAILURE(CreatePayload(backing, 4 * block_byte_length,
                                        buffer_alignment_, &target));
  ASSERT_NO_FATAL_FAILURE(CreateMemory(
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE, 4096, &completion));
  amdf_cache_operations_t cache_operations = 0;
  ASSERT_NO_FATAL_FAILURE(
      QueryCacheOperations(source, target, &cache_operations));
  std::memset(source->host.pointer, kGuard, source->host.byte_length);
  std::memset(target->host.pointer, kGuard, target->host.byte_length);
  *static_cast<uint32_t*>(completion->host.pointer) = 0;
  auto* input = static_cast<uint8_t*>(source->host.pointer);
  auto* output = static_cast<uint8_t*>(target->host.pointer);
  GpuCommandQueue* queue = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&queue));

  // No CPU payload access occurs between native read completion, GPU
  // consumption/production and the file write from GPU-produced backing.
  ASSERT_NO_FATAL_FAILURE(Transfer(false, input + block_byte_length,
                                   payload_byte_length, block_byte_length));
  ASSERT_NO_FATAL_FAILURE(CopySwappedBlocks(
      queue, source, target, completion,
      static_cast<uint32_t>(block_byte_length), cache_operations));
  ASSERT_TRUE(GpuWaitEqual<uint32_t>(
      reinterpret_cast<uintptr_t>(completion->host.pointer), 1));
  ASSERT_NO_FATAL_FAILURE(Transfer(true, output + block_byte_length,
                                   payload_byte_length, 5 * block_byte_length));
  ASSERT_NO_FATAL_FAILURE(queue->WaitRetired(api_));

  std::vector<uint8_t> expected_source(source->host.byte_length, kGuard);
  std::vector<uint8_t> expected_target(target->host.byte_length, kGuard);
  std::copy_n(expected_file.data() + block_byte_length, payload_byte_length,
              expected_source.data() + block_byte_length);
  std::copy_n(expected_file.data() + 2 * block_byte_length, block_byte_length,
              expected_target.data() + block_byte_length);
  std::copy_n(expected_file.data() + block_byte_length, block_byte_length,
              expected_target.data() + 2 * block_byte_length);
  std::copy_n(expected_target.data() + block_byte_length, payload_byte_length,
              expected_file.data() + 5 * block_byte_length);
  std::vector<uint8_t> observed_file(file_byte_length);
  ASSERT_NO_FATAL_FAILURE(ReadIndependentFile(&observed_file));
  EXPECT_EQ(observed_file, expected_file);
  EXPECT_EQ(std::memcmp(input, expected_source.data(), expected_source.size()),
            0);
  EXPECT_EQ(std::memcmp(output, expected_target.data(), expected_target.size()),
            0);
}

class GpuFilePoolTest : public GpuFileIoTest {};

TEST_P(GpuFilePoolTest,
       CachedShadersConsumeReboundSlotsWhileNextReadIsPending) {
  const auto [command_type, backing, mode] = GetParam();
  ASSERT_EQ(command_type, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  ASSERT_NO_FATAL_FAILURE(OpenFile(mode));
  const auto* product = kernels::transform::kKernels.Find(gpu_endpoint_info_);
  ASSERT_NE(product, nullptr)
      << "missing compiled transform kernel for endpoint";
  const auto& kernel = *product;
  RecordProperty("transform_kernel_target", kernel.target);
  RecordProperty("transform_kernel_image_sha256", kernel.executable.sha256);

  constexpr uint32_t kGenerationCount = 4;
  constexpr auto kReadWrite =
      AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE;
  constexpr uint32_t kGuard = 0xa5a5a5a5u;
  amdf_memory_reservation_capabilities_t capabilities = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CAPABILITIES,
      .structure_size = sizeof(capabilities),
  };
  ASSERT_EQ(api_->device_query_memory_reservation_capabilities(device_, 0,
                                                               &capabilities),
            AMDF_STATUS_OK);
  const uint64_t unit = std::max(
      {UINT64_C(65536), buffer_alignment_, transfer_alignment_,
       capabilities.mapping_granularity, capabilities.reservation_granularity});
  ASSERT_LE(unit, UINT64_C(1) << 20);
  ASSERT_TRUE(std::has_single_bit(unit));
  const auto block_length = static_cast<DWORD>(unit);
  const auto word_count = static_cast<uint32_t>(unit / sizeof(uint32_t));
  const auto file_length = static_cast<DWORD>(12 * unit);
  const auto access = kReadWrite | capabilities.required_access;
  const amdf_memory_reservation_access_t consumer = {
      device_, capabilities.address_domain_ordinal, access, unit};
  const amdf_memory_reservation_create_info_t reservation = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_CREATE_INFO,
      .structure_size = sizeof(reservation),
      .byte_length = 2 * unit,
      .minimum_alignment = unit,
      .access_count = 1,
      .accesses = &consumer,
  };
  ASSERT_EQ(api_->memory_reservation_create(instance_, &reservation,
                                            &pool_reservation_),
            AMDF_STATUS_OK);
  amdf_memory_reservation_access_info_t projection = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_RESERVATION_ACCESS_INFO,
      .structure_size = sizeof(projection),
  };
  ASSERT_EQ(api_->memory_reservation_query_access_info(pool_reservation_, 0,
                                                       &projection),
            AMDF_STATUS_OK);

  GpuMemory* pool = nullptr;
  GpuMemory* arguments = nullptr;
  GpuMemory* code = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreatePayload(backing, 8 * unit, unit, &pool,
                                        AMDF_MEMORY_FLAG_MAPPING_SOURCE,
                                        capabilities.required_access));
  ASSERT_EQ(pool->host.cacheability, AMDF_HOST_CACHEABILITY_WRITE_BACK);
  ASSERT_EQ(pool->access_info.flags & AMDF_MEMORY_FLAG_HOST_COHERENT, 0u);
  ASSERT_NE(pool->access_info.flags & AMDF_MEMORY_FLAG_MAPPING_SOURCE, 0u);
  ASSERT_NE(projection.address, pool->device_address);
  RecordProperty("payload_is_host_coherent", 0);
  // A caller-owned pointer does not communicate its page provenance. This
  // fixture knows its registered pages came from VirtualAlloc; only the owned
  // construction can establish the portable file-view contract itself.
  auto file_site = pool->HostSite();
  file_site.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
  auto gpu_site = pool->DeviceSite(family_.ordinal);
  amdf_memory_profile_pair_query_t pair_query = {
      .type = AMDF_STRUCTURE_TYPE_MEMORY_PROFILE_PAIR_QUERY,
      .structure_size = sizeof(pair_query),
      .memory_profile_ordinal = pool->creation.memory_profile_ordinal,
      .access_count = pool->creation.access_count,
      .required_flags = pool->creation.required_flags,
      .accesses = pool->creation.accesses,
      .registered_host_cacheability =
          pool->creation.registered_host_cacheability,
  };
  pair_query.producer.kind = AMDF_MEMORY_SITE_KIND_FILE_IO;
  pair_query.producer.value.host_access =
      AMDF_MEMORY_MAP_FLAG_READ | AMDF_MEMORY_MAP_FLAG_WRITE;
  pair_query.consumer.kind = AMDF_MEMORY_SITE_KIND_DEVICE;
  pair_query.consumer.value.device = {0, family_.ordinal};
  for (uint32_t direction = 0; direction < 2; ++direction) {
    SCOPED_TRACE(direction);
    amdf_memory_pair_info_t prospective = {
        .type = AMDF_STRUCTURE_TYPE_MEMORY_PAIR_INFO,
        .structure_size = sizeof(prospective),
    };
    auto concrete = prospective;
    const auto expected_status =
        backing == Backing::kAllocated
            ? AMDF_STATUS_OK
            : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED);
    ASSERT_EQ(api_->memory_scope_query_pair_info(system_scope_, &pair_query,
                                                 &prospective),
              expected_status);
    ASSERT_EQ(api_->memory_query_pair_info(&file_site, &gpu_site, &concrete),
              expected_status);
    if (backing == Backing::kAllocated) {
      ASSERT_NE(concrete.flags & AMDF_MEMORY_PAIR_FLAG_SHARED_BACKING_REACHABLE,
                0u);
      EXPECT_EQ(concrete.flags, prospective.flags);
      EXPECT_EQ(concrete.atomic_reach.scope_32, AMDF_ATOMIC_SCOPE_NONE);
      EXPECT_EQ(concrete.atomic_reach.scope_64, AMDF_ATOMIC_SCOPE_NONE);
      for (const auto* pair : {&prospective, &concrete}) {
        const auto& file = direction == 0 ? pair->release : pair->acquire;
        EXPECT_EQ(file.kind, AMDF_CACHE_TRANSITION_KIND_NONE);
        EXPECT_EQ(file.executor, AMDF_CACHE_TRANSITION_EXECUTOR_NONE);
        const auto& gpu = direction == 0 ? pair->acquire : pair->release;
        EXPECT_EQ(gpu.kind, AMDF_CACHE_TRANSITION_KIND_GLOBAL);
        EXPECT_EQ(gpu.executor, AMDF_CACHE_TRANSITION_EXECUTOR_QUEUE);
        EXPECT_EQ(gpu.operation, direction == 0
                                     ? AMDF_CACHE_OPERATION_ACQUIRE_FROM_SYSTEM
                                     : AMDF_CACHE_OPERATION_RELEASE_TO_SYSTEM);
      }
    }
    std::swap(pair_query.producer, pair_query.consumer);
    std::swap(file_site, gpu_site);
  }
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ, 4096, &arguments));
  const uint64_t code_length =
      (pm4_profile_->CodeByteLength(kernel.executable.byte_length,
                                    kernel.entry_byte_offset,
                                    kernel.program.resource3) +
       4095u) &
      ~UINT64_C(4095);
  ASSERT_NO_FATAL_FAILURE(
      CreateMemory(AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_EXECUTE,
                   code_length, &code));
  std::memset(code->host.pointer, 0, code_length);
  std::memcpy(code->host.pointer, kernel.executable.words,
              kernel.executable.byte_length);
  ASSERT_EQ(api_->host_mapping_cache_control(
                code->mapping, AMDF_HOST_CACHE_OPERATION_FLUSH, 0, code_length),
            AMDF_STATUS_OK);
  const Pm4ComputeProgram program = {
      code->device_address + kernel.entry_byte_offset,
      kernel.program.resource1,
      kernel.program.resource2,
      kernel.program.resource3,
      kernel.group_segment_byte_length,
      kernel.wavefront_size,
      {kernel.workgroup_size(), 1, 1},
  };
  ASSERT_EQ(program.entry_address % 256, 0u);
  ASSERT_EQ(arguments->device_address % kernel.arguments.alignment, 0u);

  std::vector<uint32_t> expected_file(file_length / sizeof(uint32_t), kGuard);
  for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
    for (uint32_t word = 0; word < word_count; ++word) {
      expected_file[(1 + generation) * word_count + word] =
          0xf17b35a9u + generation * 0x01030507u + word * 0x9e3779b9u;
    }
  }
  uint8_t* seed = nullptr;
  ASSERT_NO_FATAL_FAILURE(AllocateHostStorage(file_length, unit, &seed));
  std::memcpy(seed, expected_file.data(), file_length);
  ASSERT_NO_FATAL_FAILURE(Transfer(true, seed, file_length, 0));
  std::vector<uint32_t> expected_pool(pool->info.byte_length / sizeof(uint32_t),
                                      kGuard);
  std::memcpy(pool->host.pointer, expected_pool.data(), pool->info.byte_length);
  ASSERT_EQ(api_->host_mapping_cache_control(pool->mapping,
                                             AMDF_HOST_CACHE_OPERATION_FLUSH, 0,
                                             pool->info.byte_length),
            AMDF_STATUS_OK);
  auto* bytes = static_cast<uint8_t*>(pool->host.pointer);
  ASSERT_NO_FATAL_FAILURE(Transfer(false, bytes + unit, block_length, unit));
  ASSERT_NO_FATAL_FAILURE(CreateQueue(&pool_queue_));
  // Windows uses native submission completion. No device wait depends on a
  // host-issued signal or a future file operation.
  ASSERT_EQ(pool_queue_->publication_mode(),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
  Pm4CommandWriter commands(pool_queue_->words().data(), *pm4_profile_);
  for (uint32_t generation = 0; generation < kGenerationCount; ++generation) {
    SCOPED_TRACE(generation);
    const uint32_t input_slot = 1 + 2 * (generation % 2);
    const uint32_t output_slot = 5 + 2 * (generation % 2);
    const uint32_t count = word_count - 13 - generation * 11;
    const uint32_t addend = 0x80000023u + generation * 0x01010101u;
    const std::array<amdf_memory_mapping_update_t, 2> updates = {{
        {.target_byte_offset = 0,
         .byte_length = unit,
         .memory = pool->memory,
         .access = access,
         .source_byte_offset = input_slot * unit},
        {.target_byte_offset = unit,
         .byte_length = unit,
         .memory = pool->memory,
         .access = access,
         .source_byte_offset = output_slot * unit},
    }};
    // The previous iteration retired its final GPU reader before rebinding.
    ASSERT_EQ(api_->memory_reservation_update(pool_reservation_, 0,
                                              updates.size(), updates.data()),
              AMDF_STATUS_OK);
    const kernels::transform::Arguments values = {
        projection.address, projection.address + unit, count, addend};
    std::memcpy(arguments->host.pointer, &values, sizeof(values));
    ASSERT_EQ(api_->host_mapping_cache_control(arguments->mapping,
                                               AMDF_HOST_CACHE_OPERATION_FLUSH,
                                               0, sizeof(values)),
              AMDF_STATUS_OK);
    // Keep a read to a disjoint slot outstanding across GPU publication. The
    // original host view remains unchanged while only GPU aliases are rebound.
    // This checks independent ownership, not a claim about physical overlap.
    if (generation + 1 < kGenerationCount) {
      const uint32_t next_slot = 1 + 2 * ((generation + 1) % 2);
      ASSERT_NO_FATAL_FAILURE(BeginTransfer(false, bytes + next_slot * unit,
                                            block_length,
                                            (generation + 2) * unit));
    }
    commands.SystemBarrier();
    commands.BindCompute(program, arguments->device_address,
                         family_.format_features);
    const uint32_t grid = (count + kernel.workgroup_size() - 1) /
                          kernel.workgroup_size() * kernel.workgroup_size();
    commands.Dispatch(program, grid, 1, 1);
    commands.SystemBarrier();
    commands.PadToEightWords();
    ASSERT_NO_FATAL_FAILURE(
        pool_queue_->Publish(api_, gpu_api_, commands.word_count()));
    if (generation + 1 < kGenerationCount) {
      ASSERT_NO_FATAL_FAILURE(CompleteTransfer(block_length));
    }
    ASSERT_NO_FATAL_FAILURE(pool_queue_->WaitRetired(api_));
    // Native file writes consume the completed, system-released shader output
    // directly. The CPU does not inspect or rewrite any payload in this loop.
    ASSERT_NO_FATAL_FAILURE(Transfer(true, bytes + output_slot * unit,
                                     block_length, (7 + generation) * unit));
    std::copy_n(expected_file.data() + (1 + generation) * word_count,
                word_count, expected_pool.data() + input_slot * word_count);
    for (uint32_t word = 0; word < count; ++word) {
      expected_pool[output_slot * word_count + word] =
          expected_file[(1 + generation) * word_count + word] * 3u + addend;
    }
    std::copy_n(expected_pool.data() + output_slot * word_count, word_count,
                expected_file.data() + (7 + generation) * word_count);
  }
  ASSERT_EQ(api_->host_mapping_cache_control(
                pool->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                pool->info.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pool->host.pointer, expected_pool.data(),
                        pool->info.byte_length),
            0);
  // A cancellation request may race successful completion. Collect the actual
  // terminal packet before reusing the same slot and OVERLAPPED storage.
  ASSERT_NO_FATAL_FAILURE(
      BeginTransfer(false, bytes + unit, block_length, unit));
  ASSERT_NO_FATAL_FAILURE(CancelTransfer());
  ASSERT_NO_FATAL_FAILURE(
      Transfer(false, bytes + unit, block_length, 2 * unit));
  std::copy_n(expected_file.data() + 2 * word_count, word_count,
              expected_pool.data() + word_count);
  ASSERT_EQ(api_->host_mapping_cache_control(
                pool->mapping, AMDF_HOST_CACHE_OPERATION_INVALIDATE, 0,
                pool->info.byte_length),
            AMDF_STATUS_OK);
  EXPECT_EQ(std::memcmp(pool->host.pointer, expected_pool.data(),
                        pool->info.byte_length),
            0);
  std::vector<uint8_t> observed(file_length);
  ASSERT_NO_FATAL_FAILURE(ReadIndependentFile(&observed));
  EXPECT_EQ(std::memcmp(observed.data(), expected_file.data(), file_length), 0);
  RecordProperty("completed_generations", kGenerationCount);
}

INSTANTIATE_TEST_SUITE_P(
    BackingAndMode, GpuFilePoolTest,
    ::testing::Combine(
        ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4),
        ::testing::Values(Backing::kAllocated, Backing::kRegistered),
        ::testing::Values(FileMode::kBuffered, FileMode::kUnbuffered)),
    [](const auto& info) {
      return std::string(std::get<1>(info.param) == Backing::kAllocated
                             ? "Allocated"
                             : "Registered") +
             (std::get<2>(info.param) == FileMode::kBuffered ? "Buffered"
                                                             : "Unbuffered");
    });

INSTANTIATE_TEST_SUITE_P(
    EngineBackingAndMode, GpuFileIoTest,
    ::testing::Combine(
        ::testing::Values(AMDF_QUEUE_COMMAND_TYPE_GPU_PM4,
                          AMDF_QUEUE_COMMAND_TYPE_GPU_SDMA),
        ::testing::Values(Backing::kAllocated, Backing::kRegistered),
        ::testing::Values(FileMode::kBuffered, FileMode::kUnbuffered)),
    [](const auto& info) {
      return std::string(std::get<0>(info.param) ==
                                 AMDF_QUEUE_COMMAND_TYPE_GPU_PM4
                             ? "Pm4"
                             : "Sdma") +
             (std::get<1>(info.param) == Backing::kAllocated ? "Allocated"
                                                             : "Registered") +
             (std::get<2>(info.param) == FileMode::kBuffered ? "Buffered"
                                                             : "Unbuffered");
    });

}  // namespace
