// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/mcdm/kernel_execution.h"

#include <array>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/memory.h"

namespace {

struct Allocation {
  // Private allocation kind received by KMT.
  uint32_t type = 0;
  // Native allocation extent requested by the provider.
  uint64_t byte_length = 0;
  // Actual page-aligned host storage supplied by the native dependency.
  void* pointer = nullptr;
  // GPU VA supplied by mapping this exact allocation.
  uint64_t device_address = 0;
  // Whether the native dependency has made this allocation resident.
  bool resident = false;
  // Number of native final-release attempts, including failed attempts.
  uint32_t release_count = 0;
  // Last range published to this allocation through the native cache API.
  struct {
    // Allocation-relative byte offset of the published range.
    uint64_t byte_offset = 0;
    // Byte length of the published range.
    uint64_t byte_length = 0;
  } publication;
};

struct NativeState {
  // Coupled native interface established before creating the execution owner.
  amdf_windows_xdna_protocol_t protocol = AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  // Handle-indexed native allocations, with zero reserved as invalid.
  std::vector<Allocation> allocations = {Allocation{}};
  // Allocation attempt rejected by KMT, or SIZE_MAX when none is rejected.
  size_t failed_allocation = SIZE_MAX;
  // Allocation whose native release fails, or SIZE_MAX when all succeed.
  size_t failed_release = SIZE_MAX;
  // Native execution request retained until the test supplies completion.
  struct Command {
    // Device-readable packet bytes in the exact submitted allocation.
    const uint8_t* packet = nullptr;
    // Independently addressed driver response cell.
    uint64_t* response = nullptr;
    // Instruction address captured at acceptance.
    uint64_t instruction_address = 0;
  };
  // Every native execution request, without a library-side command copy.
  std::vector<Command> commands;
  // Independent firmware address returned for the private instruction heap.
  uint64_t firmware_address = UINT64_C(0x8000000);
  // Queried policy controlling native completion-buffer sharing.
  bool shared_kernel_buffers = false;
  // CPU-visible native progress fence.
  volatile uint64_t progress = 0;
  // Native opcodes observed in publication order.
  std::vector<uint64_t> opcodes;
  // Firmware result written into the initialization response cell.
  uint64_t initialize_result = 1;
  // Firmware result written into the execution response cell before its fence.
  uint64_t execution_result = 4;
  // Captured instruction address from the last execution packet.
  uint64_t instruction_address = 0;
  // Captured instruction length in words from the execution packet.
  uint32_t instruction_word_count = 0;
  // Native opcode whose acceptance must remain unretired for a lifetime test.
  uint64_t deferred_opcode = 0;
  // Accepted progress value not yet exposed through the native fence.
  uint64_t pending_submission = 0;
  // Native wait dependency retained across finite caller timeouts.
  struct {
    // Number of attempted native event registrations.
    uint32_t count = 0;
    // Event retained by an accepted wait whose completion is supplied later.
    HANDLE event = nullptr;
  } deferred_wait;
  // Captured caller-owned notification request at the KMT boundary.
  struct {
    // Number of explicit native registrations.
    uint32_t count = 0;
    // Exact native fence value requested by the caller.
    uint64_t submission = 0;
    // Borrowed event handle copied by the simulated native registration.
    HANDLE event = nullptr;
    // Registration result supplied independently of execution progress.
    NTSTATUS result = 0;
  } notification;
  // Publication of the admission PDI within the private instruction backing.
  struct {
    // Number of cache publications requested for this allocation.
    uint32_t count = 0;
    // Native publication start within the allocation, in bytes.
    uint64_t byte_offset = 0;
    // Native publication extent, in bytes.
    uint64_t byte_length = 0;
    // Dependency result returned by the native cache publication operation.
    NTSTATUS result = 0;
  } bootstrap_publication;
};

// One dependency state per test, never shared with production code.
NativeState* native_state = nullptr;

uint64_t ReadU64(const void* bytes, size_t offset) {
  uint64_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

uint32_t ReadU32(const void* bytes, size_t offset) {
  uint32_t value = 0;
  std::memcpy(&value, static_cast<const uint8_t*>(bytes) + offset,
              sizeof(value));
  return value;
}

NTSTATUS APIENTRY CreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  if (native_state->allocations.size() == native_state->failed_allocation) {
    return static_cast<NTSTATUS>(0xC0000017u);
  }
  auto* info = create->pAllocationInfo2;
  EXPECT_EQ(create->NumAllocations, 1u);
  EXPECT_EQ(info->PrivateDriverDataSize, 56u);
  Allocation allocation;
  allocation.type = ReadU32(info->pPrivateDriverData, 0x1C);
  allocation.byte_length = ReadU64(info->pPrivateDriverData, 0x10);
  allocation.pointer = VirtualAlloc(nullptr, allocation.byte_length,
                                    MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
  if (allocation.pointer == nullptr) {
    return static_cast<NTSTATUS>(0xC0000017u);
  }
  if (allocation.type == 0x332C) {
    EXPECT_EQ(native_state->protocol, AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT);
    EXPECT_EQ(ReadU32(info->pPrivateDriverData, 0x20), 2u);
    EXPECT_EQ(ReadU32(info->pPrivateDriverData, 0x28), 0x02000000u);
    EXPECT_EQ(create->Flags.CreateResource,
              native_state->shared_kernel_buffers);
    EXPECT_EQ(create->Flags.CreateShared, native_state->shared_kernel_buffers);
  } else if (allocation.type == 0x332B) {
    EXPECT_NE(native_state->protocol, AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT);
    EXPECT_EQ(ReadU32(info->pPrivateDriverData, 0x20), 0u);
    EXPECT_EQ(ReadU32(info->pPrivateDriverData, 0x28), 0u);
    EXPECT_TRUE(create->Flags.CreateResource);
    EXPECT_TRUE(create->Flags.CreateShared);
  }
  if (allocation.type == 0x3323) {
    EXPECT_EQ(allocation.byte_length, AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE);
    EXPECT_EQ(ReadU32(info->pPrivateDriverData, 0x28), 0x01000001u);
    std::memcpy(static_cast<uint8_t*>(info->pPrivateDriverData) + 0x30,
                &native_state->firmware_address, sizeof(uint64_t));
    // Native storage has no zero-fill contract. Poison both the reserved
    // prefix and the start of the application-visible range.
    std::memset(allocation.pointer, 0xA5, 32768 + 64);
  }
  info->hAllocation =
      static_cast<D3DKMT_HANDLE>(native_state->allocations.size());
  if (create->Flags.CreateResource) {
    create->hResource = info->hAllocation;
  }
  native_state->allocations.push_back(allocation);
  return 0;
}

NTSTATUS APIENTRY DestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  const auto handle = destroy->hResource != 0 ? destroy->hResource
                                              : destroy->phAllocationList[0];
  auto& allocation = native_state->allocations[handle];
  ++allocation.release_count;
  if (handle == native_state->failed_release) {
    return static_cast<NTSTATUS>(0xC0000001u);
  }
  EXPECT_TRUE(VirtualFree(allocation.pointer, 0, MEM_RELEASE));
  allocation.pointer = nullptr;
  return 0;
}

NTSTATUS APIENTRY MapAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  map->VirtualAddress =
      UINT64_C(0x100000000) + uint64_t(map->hAllocation) * 0x4000000;
  map->PagingFenceValue = 0;
  native_state->allocations[map->hAllocation].device_address =
      map->VirtualAddress;
  return 0;
}

NTSTATUS APIENTRY MakeResident(D3DDDI_MAKERESIDENT* resident) {
  EXPECT_EQ(resident->Flags.MustSucceed, 0u);
  resident->PagingFenceValue = 0;
  for (uint32_t i = 0; i < resident->NumAllocations; ++i) {
    native_state->allocations[resident->AllocationList[i]].resident = true;
  }
  return 0;
}

NTSTATUS APIENTRY Lock(D3DKMT_LOCK2* lock) {
  lock->pData = native_state->allocations[lock->hAllocation].pointer;
  return 0;
}

NTSTATUS APIENTRY CreateQueue(D3DKMT_CREATEHWQUEUE* create) {
  create->hHwQueue = 41;
  create->hHwQueueProgressFence = 42;
  create->HwQueueProgressFenceCPUVirtualAddress =
      const_cast<uint64_t*>(&native_state->progress);
  create->HwQueueProgressFenceGPUVirtualAddress = 0x300000;
  return 0;
}

NTSTATUS APIENTRY Submit(const D3DKMT_SUBMITCOMMANDTOHWQUEUE* submit) {
  const void* bytes = submit->pPrivateDriverData;
  const uint64_t opcode = ReadU64(bytes, 0);
  const bool direct =
      native_state->protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  const size_t header_length =
      direct ? 120
             : (native_state->protocol == AMDF_WINDOWS_XDNA_PROTOCOL_METADATA
                    ? 104
                    : 88);
  const size_t response_address_offset = direct ? 0x40 : 0x38;
  native_state->opcodes.push_back(opcode);
  if (opcode == 5 || opcode == 3) {
    const auto& response_allocation =
        native_state->allocations[ReadU64(bytes, 0x28)];
    EXPECT_EQ(response_allocation.type, direct ? 0x332Cu : 0x332Bu);
    EXPECT_EQ(response_allocation.resident, direct);
    if (direct) {
      EXPECT_NE(response_allocation.device_address, 0u);
      EXPECT_EQ(ReadU64(bytes, 0x30), response_allocation.device_address);
    } else {
      EXPECT_EQ(response_allocation.device_address, 0u);
    }
    EXPECT_EQ(ReadU64(bytes, response_address_offset),
              reinterpret_cast<uintptr_t>(response_allocation.pointer) +
                  ReadU32(bytes, response_address_offset - 8));
    EXPECT_LE(uint64_t{ReadU32(bytes, response_address_offset - 8)} + 8,
              response_allocation.byte_length);
  }
  if (opcode == 2 || opcode == 9) {
    EXPECT_EQ(submit->PrivateDriverDataSize, header_length);
    EXPECT_EQ(ReadU64(bytes, 0x10), AMDF_WINDOWS_XDNA_PRIVATE_APERTURE_SIZE);
    if (opcode == 9 && direct) {
      EXPECT_EQ(ReadU64(bytes, 0x08), 0u);
    } else {
      EXPECT_EQ(native_state->allocations[ReadU64(bytes, 0x08)].type, 0x3323u);
    }
  } else if (opcode == 5) {
    EXPECT_EQ(submit->PrivateDriverDataSize, header_length + 520);
    auto* response =
        reinterpret_cast<uint64_t*>(ReadU64(bytes, response_address_offset));
    EXPECT_EQ(response[1], native_state->firmware_address);
    const auto* configuration =
        static_cast<const uint8_t*>(bytes) + header_length;
    EXPECT_EQ(ReadU32(configuration, 0), 1u);
    EXPECT_EQ(ReadU64(configuration, 8), native_state->firmware_address);
    // The interpreter's CU function is zero, independent of its PDI size.
    EXPECT_EQ(ReadU32(configuration, 16), 0u);
    response[0] = native_state->initialize_result;
  } else if (opcode == 3) {
    EXPECT_EQ(submit->CommandLength, 4096u + header_length);
    native_state->instruction_address = ReadU64(bytes, header_length + 0x10);
    native_state->instruction_word_count = ReadU32(bytes, header_length + 0x18);
    EXPECT_EQ(submit->PrivateDriverDataSize, header_length + 512);
    auto* response =
        reinterpret_cast<uint64_t*>(ReadU64(bytes, response_address_offset));
    EXPECT_EQ(*response, 0u);
    *response = native_state->execution_result;
    const auto& allocation = native_state->allocations[ReadU64(bytes, 0x08)];
    if (submit->CommandBuffer < allocation.device_address ||
        submit->CommandLength > allocation.byte_length ||
        submit->CommandBuffer - allocation.device_address >
            allocation.byte_length - submit->CommandLength) {
      ADD_FAILURE() << "Command range exceeds its resident native backing";
      return static_cast<NTSTATUS>(0xC000000Du);
    }
    const uint64_t byte_offset =
        submit->CommandBuffer - allocation.device_address;
    const auto* packet =
        static_cast<const uint8_t*>(allocation.pointer) + byte_offset;
    EXPECT_EQ(allocation.type, 0x3328u);
    EXPECT_TRUE(allocation.resident);
    EXPECT_EQ(allocation.publication.byte_offset, byte_offset);
    EXPECT_EQ(allocation.publication.byte_length, ReadU64(bytes, 0x10));
    EXPECT_EQ(
        std::memcmp(packet, static_cast<const uint8_t*>(bytes) + header_length,
                    ReadU64(bytes, 0x10)),
        0);
    native_state->commands.push_back(
        {packet, response, native_state->instruction_address});
  } else {
    ADD_FAILURE() << "Unexpected native opcode " << opcode;
  }
  if (opcode == native_state->deferred_opcode) {
    native_state->pending_submission = submit->HwQueueProgressFenceId;
  } else {
    native_state->progress = submit->HwQueueProgressFenceId;
  }
  return 0;
}

class WindowsXdnaKernelExecutionTest
    : public ::testing::TestWithParam<amdf_windows_xdna_adapter_info_t> {
 protected:
  void SetUp() override {
    native_state = &native_;
    native_.protocol = GetParam().protocol;
    native_.shared_kernel_buffers = GetParam().shared_kernel_buffers;
    kmt_.create_allocation = CreateAllocation;
    kmt_.destroy_allocation = DestroyAllocation;
    kmt_.map_gpu_virtual_address = MapAddress;
    kmt_.make_resident = MakeResident;
    kmt_.lock = Lock;
    kmt_.unlock = [](const D3DKMT_UNLOCK2*) -> NTSTATUS { return 0; };
    kmt_.invalidate_cache =
        [](const D3DKMT_INVALIDATECACHE* invalidate) -> NTSTATUS {
      auto& allocation = native_state->allocations[invalidate->hAllocation];
      EXPECT_LE(invalidate->Offset + invalidate->Length,
                allocation.byte_length);
      allocation.publication.byte_offset = invalidate->Offset;
      allocation.publication.byte_length = invalidate->Length;
      if (allocation.type == 0x3323) {
        auto& publication = native_state->bootstrap_publication;
        ++publication.count;
        publication.byte_offset = invalidate->Offset;
        publication.byte_length = invalidate->Length;
        return publication.result;
      }
      return 0;
    };
    kmt_.create_hardware_queue = CreateQueue;
    kmt_.destroy_hardware_queue = [](const D3DKMT_DESTROYHWQUEUE*) -> NTSTATUS {
      return 0;
    };
    kmt_.submit_command_to_hardware_queue = Submit;
    kmt_.wait_from_cpu =
        [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
      EXPECT_NE(native_state->pending_submission, 0u);
      EXPECT_EQ(wait->FenceValueArray[0], native_state->pending_submission);
      return static_cast<NTSTATUS>(0xC0000001u);
    };
    // The native device and context are already live. These entries establish
    // their capability contract but must not be invoked by memory allocation.
    kmt_.create_device = [](D3DKMT_CREATEDEVICE*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    kmt_.destroy_device = [](const D3DKMT_DESTROYDEVICE*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    kmt_.get_device_state = [](D3DKMT_GETDEVICESTATE* query) -> NTSTATUS {
      query->ExecutionState = D3DKMT_DEVICEEXECUTION_ACTIVE;
      return 0;
    };
    kmt_.create_paging_queue = [](D3DKMT_CREATEPAGINGQUEUE*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    kmt_.destroy_paging_queue = [](D3DDDI_DESTROYPAGINGQUEUE*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    kmt_.create_context_virtual = [](D3DKMT_CREATECONTEXTVIRTUAL*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    kmt_.destroy_context = [](const D3DKMT_DESTROYCONTEXT*) -> NTSTATUS {
      ADD_FAILURE();
      return -1;
    };
    amdf_endpoint_info_t endpoint = {};
    endpoint.pci.vendor_id = 0x1022;
    endpoint.pci.device_id = 0x17F0;
    endpoint.pci.revision_id = 0x11;
    endpoint.engine_kind = AMDF_ENGINE_KIND_XDNA;
    ASSERT_TRUE(amdf_xdna_device_profile_initialize(&endpoint, &device_info_,
                                                    &device_profile_));
    device_.profile = &device_profile_;
    device_.kmt = &kmt_;
    device_.host_allocator = amdf_allocator_system();
    device_.device = 10;
    device_.paging_queue = 11;
    device_.paging_sync_object = 12;
    context_.device = &device_;
    context_.handle = 13;
    context_.command_aperture_cookie = 0;
    context_.adapter_info = GetParam();
    ASSERT_EQ(amdf_windows_xdna_kernel_execution_create(
                  &context_, &context_.kernel_execution),
              AMDF_STATUS_OK);
    amdf_xdna_umd_context_query_memory_profile(&context_, &profile_);
    create_.byte_length = profile_.allocation.byte_length_granularity;
    create_.minimum_alignment = profile_.allocation.minimum_alignment;
    create_.device_access = AMDF_MEMORY_ACCESS_READ | AMDF_MEMORY_ACCESS_WRITE |
                            AMDF_MEMORY_ACCESS_EXECUTE;
  }

  void TearDown() override {
    if (memory_ != nullptr) {
      EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), AMDF_STATUS_OK);
    }
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_prepare_context_destroy(
                  context_.kernel_execution),
              AMDF_STATUS_OK);
    EXPECT_EQ(
        amdf_windows_xdna_kernel_execution_destroy(context_.kernel_execution),
        AMDF_STATUS_OK);
    for (const auto& allocation : native_.allocations) {
      EXPECT_EQ(allocation.pointer, nullptr);
    }
    if (event_ != nullptr) {
      EXPECT_TRUE(CloseHandle(event_));
    }
    native_state = nullptr;
  }

  void CreateNotificationEvent() {
    event_ = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    ASSERT_NE(event_, nullptr);
    kmt_.wait_from_cpu =
        [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
      EXPECT_EQ(wait->hDevice, 10u);
      EXPECT_EQ(wait->ObjectCount, 1u);
      EXPECT_EQ(wait->ObjectHandleArray[0], 42u);
      EXPECT_EQ(wait->Flags.Value, 0u);
      auto& notification = native_state->notification;
      ++notification.count;
      notification.submission = wait->FenceValueArray[0];
      notification.event = wait->hAsyncEvent;
      return notification.result;
    };
  }

  // Native allocation and submission observations.
  NativeState native_;
  // Procedures supplied at the existing platform dependency boundary.
  amdf_kmt_api_t kmt_ = {};
  // Architecture encodings borrowed by native execution.
  amdf_xdna_device_profile_t device_profile_ = {};
  // Instruction limits retained with the native owner.
  amdf_xdna_device_info_t device_info_ = {};
  // Explicitly live native device borrowed by memory and execution.
  amdf_xdna_umd_device_t device_ = {};
  // Context retaining native transport state, not private instruction memory.
  amdf_xdna_umd_context_t context_ = {};
  // Cached private allocation contract.
  amdf_memory_native_profile_t profile_ = {};
  // One full-aperture request from the private scope.
  amdf_memory_native_create_info_t create_ = {};
  // Actual memory owner, including unpublished preparation state on failure.
  amdf_xdna_umd_memory_t* memory_ = nullptr;
  // Caller-owned auto-reset wake destination, closed after native teardown.
  HANDLE event_ = nullptr;
};

TEST_P(WindowsXdnaKernelExecutionTest,
       NotificationsBorrowCallerEventWithoutConsumingResults) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 1),
            AMDF_STATUS_OK);
  native_.deferred_opcode = 3;
  uint64_t submission = 0;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  const size_t allocation_count = native_.allocations.size();
  {
    amdf_native_event_t event = {};
    event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
    event.payload.native_handle = event_;
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_request_notification(
                  execution, submission, &event),
              AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_.notification.count, 1u);
  EXPECT_EQ(native_.notification.submission, submission);
  EXPECT_EQ(native_.notification.event, event_);
  EXPECT_LT(amdf_windows_xdna_kernel_execution_query_progress(execution),
            submission);
  EXPECT_EQ(WaitForSingleObject(event_, 0), WAIT_TIMEOUT);
  EXPECT_EQ(native_.allocations.size(), allocation_count);

  // The dependency delivers after the borrowed descriptor has gone away.
  native_.progress = submission;
  EXPECT_TRUE(SetEvent(native_.notification.event));
  // KMT excludes NT status macros; a signaled Win32 wait returns zero.
  EXPECT_EQ(WaitForSingleObject(event_, 0), 0u);
  EXPECT_EQ(WaitForSingleObject(event_, 0), WAIT_TIMEOUT);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 0);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
  event.payload.native_handle = event_;
  for (uint32_t i = 0; i < 2; ++i) {
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_request_notification(
                  execution, submission, &event),
              AMDF_STATUS_OK);
    EXPECT_EQ(WaitForSingleObject(event_, 0), 0u);
    EXPECT_EQ(WaitForSingleObject(event_, 0), WAIT_TIMEOUT);
  }
  EXPECT_EQ(native_.notification.count, 1u);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       NotificationFailurePreservesAcceptedCommand) {
  ASSERT_NO_FATAL_FAILURE(CreateNotificationEvent());
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 1),
            AMDF_STATUS_OK);
  native_.deferred_opcode = 3;
  uint64_t submission = 0;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  native_.notification.result = static_cast<NTSTATUS>(0xC0000017u);
  amdf_native_event_t event = {};
  event.type = AMDF_NATIVE_EVENT_TYPE_WIN32_EVENT;
  event.payload.native_handle = event_;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_request_notification(
                execution, submission, &event),
            amdf_kmt_make_status(native_.notification.result));
  EXPECT_EQ(native_.commands.size(), 1u);
  EXPECT_EQ(native_.pending_submission, submission);
  EXPECT_LT(amdf_windows_xdna_kernel_execution_query_progress(execution),
            submission);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  EXPECT_EQ(WaitForSingleObject(event_, 0), WAIT_TIMEOUT);
  native_.progress = submission;
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(
      amdf_windows_xdna_kernel_execution_wait(execution, submission, &deadline),
      AMDF_STATUS_OK);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 0);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       OwnsOneApertureAndSubmitsImmutableRanges) {
  EXPECT_EQ(profile_.allocation.native_byte_length_prefix, 32768u);
  const uint64_t granularity =
      profile_.allocation.native_byte_length_granularity;
  const uint64_t native_byte_length =
      ((create_.byte_length + profile_.allocation.native_byte_length_prefix +
        granularity - 1) /
       granularity) *
      granularity;
  EXPECT_EQ(native_byte_length, UINT64_C(0x4000000));
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  EXPECT_EQ(result.device_address, native_.firmware_address + 32768);
  EXPECT_EQ(result.source_byte_offset, 32768u);
  EXPECT_EQ(result.byte_length, UINT64_C(0x4000000) - 32768);
  EXPECT_EQ(result.native_allocation_byte_length, UINT64_C(0x4000000));
  EXPECT_EQ(result.native_allocation_byte_length, native_byte_length);
  EXPECT_EQ(result.address_kinds, UINT64_C(1)
                                      << AMDF_MEMORY_ADDRESS_XDNA_FIRMWARE);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5, 9}));
  ASSERT_EQ(native_.allocations.size(), 3u);
  EXPECT_EQ(native_.bootstrap_publication.count, 1u);
  EXPECT_EQ(native_.bootstrap_publication.byte_offset, 0u);
  EXPECT_EQ(native_.bootstrap_publication.byte_length, 368u);
  const auto* bootstrap =
      static_cast<const uint8_t*>(native_.allocations[2].pointer);
  EXPECT_EQ(ReadU32(bootstrap, 340), 0x004F4443u);
  EXPECT_EQ(ReadU32(bootstrap, 348), 1u);
  EXPECT_EQ(ReadU32(bootstrap, 356), 0x111u);
  for (size_t i = 368; i < 32768 + 64; ++i) {
    ASSERT_EQ(bootstrap[i], 0xA5) << "byte " << i;
  }
  auto* instructions =
      static_cast<uint8_t*>(native_.allocations[2].pointer) + 32768;
  std::memset(instructions, 0xA7, 64);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  const size_t allocation_count = native_.allocations.size();
  uint64_t submission = 0;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  EXPECT_EQ(submission, 4u);
  EXPECT_EQ(native_.instruction_address, result.device_address);
  EXPECT_EQ(native_.instruction_word_count, 16u);
  for (size_t i = 0; i < 64; ++i) {
    EXPECT_EQ(instructions[i], 0xA7);
  }
  EXPECT_EQ(native_.allocations.size(), allocation_count);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 0);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), AMDF_STATUS_OK);
  memory_ = nullptr;
  EXPECT_EQ(native_.allocations[2].pointer, nullptr);
  EXPECT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_RESOURCE_EXHAUSTED));
  EXPECT_EQ(native_.allocations.size(), allocation_count);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       PendingCommandsOwnDistinctPacketsAndResults) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  const size_t allocation_count = native_.allocations.size();
  const uint64_t initial_progress = native_.progress;
  native_.deferred_opcode = 3;
  uint64_t last = 0;
  for (uint32_t i = 0; i < 3; ++i) {
    ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                  execution, i, result.device_address + i * 256, 64, &last),
              AMDF_STATUS_OK);
  }
  EXPECT_EQ(native_.progress, initial_progress);
  EXPECT_EQ(native_.allocations.size(), allocation_count);
  ASSERT_EQ(native_.commands.size(), 3u);
  for (size_t i = 0; i < native_.commands.size(); ++i) {
    const auto& command = native_.commands[i];
    EXPECT_EQ(ReadU64(command.packet, 0x10), result.device_address + i * 256);
    EXPECT_EQ(command.instruction_address, result.device_address + i * 256);
    EXPECT_EQ(*command.response, 4u);
    for (size_t j = 0; j < i; ++j) {
      EXPECT_NE(command.packet, native_.commands[j].packet);
      EXPECT_NE(command.response, native_.commands[j].response);
    }
  }
  native_.progress = last;
  // Inject a result into only the second command. The first result remains
  // independent, and consuming the third cannot erase the first failure.
  *native_.commands[1].response = 5;
  amdf_windows_xdna_kernel_execution_retire_command(execution, 0);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 1);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 2);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5));
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
  for (size_t i = 3; i < native_.allocations.size(); ++i) {
    EXPECT_EQ(native_.allocations[i].pointer, nullptr);
  }
}

TEST_P(WindowsXdnaKernelExecutionTest,
       FixedRingsRetainPendingSlotsWithoutPerSlotAllocations) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  // Cross a native page boundary and include both ends of the owned region.
  constexpr uint32_t capacity = 4096 / sizeof(uint64_t) + 1;
  const size_t initial_count = native_.allocations.size();
  // One packet ring and one result region must serve the whole window.
  native_.failed_allocation = initial_count + 2;
  ASSERT_EQ(
      amdf_windows_xdna_kernel_execution_acquire_queue(execution, capacity),
      AMDF_STATUS_OK);
  const size_t allocation_count = native_.allocations.size();
  native_.deferred_opcode = 3;
  const std::array<uint32_t, 3> slots = {0, capacity - 2, capacity - 1};
  uint64_t last = 0;
  for (uint32_t slot : slots) {
    ASSERT_EQ(
        amdf_windows_xdna_kernel_execution_submit(
            execution, slot, result.device_address + slot * 256, 64, &last),
        AMDF_STATUS_OK);
  }
  ASSERT_EQ(native_.commands.size(), slots.size());
  for (size_t i = 0; i < slots.size(); ++i) {
    EXPECT_EQ(ReadU64(native_.commands[i].packet, 0x10),
              result.device_address + slots[i] * 256);
    EXPECT_EQ(native_.commands[i].response,
              native_.commands[0].response + slots[i]);
    EXPECT_EQ(*native_.commands[i].response, 4u);
  }
  native_.progress = last;
  amdf_windows_xdna_kernel_execution_retire_command(execution, slots[0]);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  // Reusing the first slot must not overwrite another pending packet or result.
  *native_.commands[2].response = 5;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, slots[0], result.device_address + 512, 64, &last),
            AMDF_STATUS_OK);
  EXPECT_EQ(*native_.commands[2].response, 5u);
  EXPECT_EQ(ReadU64(native_.commands[2].packet, 0x10),
            result.device_address + slots[2] * 256);
  EXPECT_EQ(native_.commands.back().packet, native_.commands[0].packet);
  EXPECT_EQ(ReadU64(native_.commands.back().packet, 0x10),
            result.device_address + 512);
  native_.progress = last;
  amdf_windows_xdna_kernel_execution_retire_command(execution, slots[1]);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  amdf_windows_xdna_kernel_execution_retire_command(execution, slots[2]);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 5));
  amdf_windows_xdna_kernel_execution_retire_command(execution, slots[0]);
  EXPECT_EQ(native_.allocations.size(), allocation_count);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
  for (size_t i = initial_count; i < allocation_count; ++i) {
    EXPECT_EQ(native_.allocations[i].pointer, nullptr);
    EXPECT_EQ(native_.allocations[i].release_count, 1u);
  }
}

TEST_P(WindowsXdnaKernelExecutionTest,
       ResponseOffsetOverflowFailsBeforeNativePreparation) {
  const size_t initial_count = native_.allocations.size();
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(
                context_.kernel_execution, UINT32_MAX),
            amdf_make_api_status(AMDF_STATUS_CODE_OUT_OF_RANGE));
  EXPECT_EQ(native_.allocations.size(), initial_count);
  EXPECT_TRUE(native_.opcodes.empty());
}

TEST_P(WindowsXdnaKernelExecutionTest, PartialRingCreationReleasesTheLease) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  for (size_t failure_index : {0u, 1u}) {
    SCOPED_TRACE(failure_index);
    const size_t initial_count = native_.allocations.size();
    native_.failed_allocation = initial_count + failure_index;
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
              amdf_kmt_make_status(static_cast<NTSTATUS>(0xC0000017u)));
    for (size_t i = initial_count; i < native_.allocations.size(); ++i) {
      EXPECT_EQ(native_.allocations[i].pointer, nullptr);
    }
    native_.failed_allocation = SIZE_MAX;
    ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
              AMDF_STATUS_OK);
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
              AMDF_STATUS_OK);
  }
}

TEST_P(WindowsXdnaKernelExecutionTest,
       EarlierWaitCanProgressAfterLaterWaitTimesOut) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  native_.deferred_opcode = 3;
  std::array<uint64_t, 3> submissions;
  for (uint32_t i = 0; i < submissions.size(); ++i) {
    ASSERT_EQ(
        amdf_windows_xdna_kernel_execution_submit(
            execution, i, result.device_address + i * 256, 64, &submissions[i]),
        AMDF_STATUS_OK);
  }
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    ++native_state->deferred_wait.count;
    EXPECT_EQ(wait->FenceValueArray[0], native_state->pending_submission - 1);
    native_state->deferred_wait.event = wait->hAsyncEvent;
    return 0;
  };
  const auto timed_out =
      amdf_make_api_status(AMDF_STATUS_CODE_DEADLINE_EXCEEDED);
  // The timeout must retain its notification, but must not force an earlier
  // waiter to depend on the later command's completion.
  constexpr uint64_t timeout = UINT64_C(100000000);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_wait(execution, submissions[1],
                                                    &deadline),
            timed_out);
  ASSERT_NE(native_.deferred_wait.event, nullptr);
  EXPECT_EQ(native_.deferred_wait.count, 1u);
  EXPECT_LT(native_.progress, submissions[0]);
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_wait(execution, submissions[1],
                                                    &deadline),
            timed_out);
  EXPECT_EQ(native_.deferred_wait.count, 1u);

  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*) -> NTSTATUS {
    ++native_state->deferred_wait.count;
    return static_cast<NTSTATUS>(0xC0000017u);
  };
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_wait(execution, submissions[0],
                                                    &deadline),
            amdf_kmt_make_status(static_cast<NTSTATUS>(0xC0000017u)));
  EXPECT_EQ(native_.deferred_wait.count, 2u);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);

  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    ++native_state->deferred_wait.count;
    native_state->progress = wait->FenceValueArray[0];
    EXPECT_TRUE(SetEvent(wait->hAsyncEvent));
    return 0;
  };
  ASSERT_EQ(amdf_wait_deadline_initialize(timeout, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_wait(execution, submissions[0],
                                                    &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.progress, submissions[0]);
  EXPECT_EQ(native_.deferred_wait.count, 3u);

  // Deliver the original later notification, then use the event again. A
  // stale wake cannot replace the native fence check for the next command.
  native_.progress = submissions[1];
  EXPECT_TRUE(SetEvent(native_.deferred_wait.event));
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_wait(execution, submissions[2],
                                                    &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(native_.progress, submissions[2]);
  EXPECT_EQ(native_.deferred_wait.count, 4u);
  EXPECT_EQ(native_.commands.size(), 3u);
  for (uint32_t i = 0; i < submissions.size(); ++i) {
    amdf_windows_xdna_kernel_execution_retire_command(execution, i);
  }
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       QueueStorageReleaseFailureConsumesTheLease) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  for (size_t failure_index : {0u, 1u}) {
    SCOPED_TRACE(failure_index);
    const size_t initial_count = native_.allocations.size();
    ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
              AMDF_STATUS_OK);
    native_.failed_release = initial_count + failure_index;
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
              amdf_kmt_make_status(static_cast<NTSTATUS>(0xC0000001u)));
    for (size_t i = initial_count; i < native_.allocations.size(); ++i) {
      EXPECT_EQ(native_.allocations[i].release_count, 1u);
      if (i != native_.failed_release) {
        EXPECT_EQ(native_.allocations[i].pointer, nullptr);
      }
    }
    ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
              AMDF_STATUS_OK);
    EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
              AMDF_STATUS_OK);
    // Native failure can leak that allocation, but does not retain a public
    // lease or create a hidden retry owner. Release the dependency's simulated
    // leak explicitly; production teardown must not have retried it.
    auto& leaked = native_.allocations[native_.failed_release];
    EXPECT_EQ(leaked.release_count, 1u);
    EXPECT_NE(leaked.pointer, nullptr);
    EXPECT_TRUE(VirtualFree(leaked.pointer, 0, MEM_RELEASE));
    leaked.pointer = nullptr;
  }
}

TEST_P(WindowsXdnaKernelExecutionTest,
       RetiredFirmwareFailurePersistsWithoutPoisoningDevice) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  // The miniport signals progress successfully while the separate response
  // cell reports ERT timeout. No failing command reaches real hardware.
  native_.execution_result = 8;
  uint64_t submission = 0;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  EXPECT_EQ(submission, 4u);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_progress(execution),
            submission);
  // Fence observation establishes retirement; software retirement separately
  // consumes the exact command result before transport storage can be reused.
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            AMDF_STATUS_OK);
  amdf_windows_xdna_kernel_execution_retire_command(execution, 0);
  const auto failure = amdf_make_status(AMDF_STATUS_DOMAIN_FIRMWARE, 8);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            failure);
  EXPECT_EQ(amdf_kmt_device_status_query(&device_.status), AMDF_STATUS_OK);

  const size_t allocation_count = native_.allocations.size();
  const size_t submission_count = native_.opcodes.size();
  uint64_t rejected_submission = UINT64_MAX;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &rejected_submission),
            failure);
  EXPECT_EQ(rejected_submission, UINT64_MAX);
  EXPECT_EQ(native_.opcodes.size(), submission_count);
  EXPECT_EQ(native_.allocations.size(), allocation_count);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_progress(execution),
            submission);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            failure);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            failure);
  EXPECT_EQ(amdf_kmt_device_status_query(&device_.status), AMDF_STATUS_OK);
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), AMDF_STATUS_OK);
  memory_ = nullptr;
  EXPECT_EQ(native_.allocations[2].pointer, nullptr);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       InvalidFirmwareAddressNeverReachesBootstrap) {
  native_.firmware_address += 4096;
  amdf_xdna_umd_memory_result_t result = {};
  result.device_address = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_INTERNAL));
  EXPECT_EQ(result.device_address, UINT64_MAX);
  EXPECT_TRUE(native_.opcodes.empty());
}

TEST_P(WindowsXdnaKernelExecutionTest, ResetFenceDoesNotCompleteAcceptedWork) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  uint64_t submission = 0;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  native_.progress = UINT64_MAX;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_progress(execution), 0u);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            lost);
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  EXPECT_EQ(
      amdf_windows_xdna_kernel_execution_wait(execution, submission, &deadline),
      lost);
  const size_t accepted_count = native_.opcodes.size();
  uint64_t rejected_submission = 77;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &rejected_submission),
            lost);
  EXPECT_EQ(rejected_submission, 77u);
  EXPECT_EQ(native_.opcodes.size(), accepted_count);
  // This dependency executes no hardware work. End its simulated access
  // before the fixture releases memory; reset signaling did not prove this.
  native_.progress = submission;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       ResetDuringWaitDoesNotCompleteAcceptedWork) {
  amdf_xdna_umd_memory_result_t result = {};
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            AMDF_STATUS_OK);
  auto* execution = context_.kernel_execution;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_acquire_queue(execution, 3),
            AMDF_STATUS_OK);
  native_.deferred_opcode = 3;
  uint64_t submission = 0;
  ASSERT_EQ(amdf_windows_xdna_kernel_execution_submit(
                execution, 0, result.device_address, 64, &submission),
            AMDF_STATUS_OK);
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    EXPECT_EQ(wait->FenceValueArray[0], native_state->pending_submission);
    native_state->progress = UINT64_MAX;
    EXPECT_NE(wait->hAsyncEvent, nullptr);
    EXPECT_TRUE(SetEvent(wait->hAsyncEvent));
    return 0;
  };
  amdf_wait_deadline_t deadline;
  ASSERT_EQ(amdf_wait_deadline_initialize(AMDF_TIMEOUT_INFINITE, 0, &deadline),
            AMDF_STATUS_OK);
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(
      amdf_windows_xdna_kernel_execution_wait(execution, submission, &deadline),
      lost);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_progress(execution), 0u);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            lost);
  // End simulated device access independently of the reset indication.
  native_.progress = submission;
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_release_queue(execution),
            AMDF_STATUS_OK);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       ResetDuringBootstrapDoesNotPublishOrReleaseBacking) {
  native_.deferred_opcode = 5;
  kmt_.wait_from_cpu =
      [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* wait) -> NTSTATUS {
    EXPECT_EQ(wait->FenceValueArray[0], native_state->pending_submission);
    EXPECT_EQ(wait->hAsyncEvent, nullptr);
    native_state->progress = UINT64_MAX;
    return 0;
  };
  amdf_xdna_umd_memory_result_t result = {};
  result.device_address = 77;
  const auto lost = amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST);
  EXPECT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            lost);
  EXPECT_EQ(result.device_address, 77u);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5}));
  ASSERT_NE(memory_, nullptr);
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), lost);
  EXPECT_NE(native_.allocations[2].pointer, nullptr);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_prepare_context_destroy(
                context_.kernel_execution),
            amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  // The fake's last access ends here, separately from its failed wait. No
  // reset or failed-device recovery claim follows from this cleanup step.
  native_.progress = native_.pending_submission;
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), AMDF_STATUS_OK);
  memory_ = nullptr;
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5}));
  EXPECT_EQ(amdf_kmt_device_status_query(&device_.status), lost);
}

TEST_P(WindowsXdnaKernelExecutionTest, FailedBootstrapDoesNotPublishMemory) {
  native_.initialize_result = 0;
  amdf_xdna_umd_memory_result_t result = {};
  result.device_address = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            amdf_make_api_status(AMDF_STATUS_CODE_DEVICE_LOST));
  EXPECT_EQ(result.device_address, UINT64_MAX);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5}));
}

TEST_P(WindowsXdnaKernelExecutionTest, FailedPublicationNeverAdmitsContext) {
  native_.bootstrap_publication.result = static_cast<NTSTATUS>(0xC0000001u);
  const auto failure =
      amdf_kmt_make_status(native_.bootstrap_publication.result);
  amdf_xdna_umd_memory_result_t result = {};
  result.device_address = UINT64_MAX;
  EXPECT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            failure);
  EXPECT_EQ(result.device_address, UINT64_MAX);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2}));
  EXPECT_EQ(native_.bootstrap_publication.count, 1u);
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(
                context_.kernel_execution),
            failure);
}

TEST_P(WindowsXdnaKernelExecutionTest,
       FailedWaitRetainsBootstrapBackingUntilNativeRetirement) {
  native_.deferred_opcode = 5;
  const auto failure = amdf_kmt_make_status(static_cast<NTSTATUS>(0xC0000001u));
  amdf_xdna_umd_memory_result_t result = {};
  result.device_address = UINT64_MAX;
  ASSERT_EQ(amdf_xdna_umd_memory_prepare_private(&context_, &profile_, &create_,
                                                 &memory_, &result),
            failure);
  EXPECT_EQ(result.device_address, UINT64_MAX);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5}));
  ASSERT_EQ(native_.allocations.size(), 3u);
  EXPECT_EQ(native_.pending_submission, 2u);
  EXPECT_EQ(native_.progress, 1u);
  auto* execution = context_.kernel_execution;
  EXPECT_EQ(
      amdf_windows_xdna_kernel_execution_prepare_context_destroy(execution),
      amdf_make_api_status(AMDF_STATUS_CODE_BUSY));
  EXPECT_EQ(amdf_xdna_umd_memory_destroy(memory_), failure);
  EXPECT_NE(native_.allocations[2].pointer, nullptr);
  EXPECT_EQ(ReadU32(native_.allocations[2].pointer, 356), 0x111u);

  // The accepted native operation retires independently of the failed wait.
  // Releasing its backing does not retry admission or clear terminal failure.
  native_.progress = native_.pending_submission;
  ASSERT_EQ(amdf_xdna_umd_memory_destroy(memory_), AMDF_STATUS_OK);
  memory_ = nullptr;
  EXPECT_EQ(native_.allocations[2].pointer, nullptr);
  EXPECT_EQ(native_.opcodes, (std::vector<uint64_t>{2, 5}));
  EXPECT_EQ(amdf_windows_xdna_kernel_execution_query_terminal_status(execution),
            failure);
}

INSTANTIATE_TEST_SUITE_P(NativeInterfaces, WindowsXdnaKernelExecutionTest,
                         ::testing::Values(
                             amdf_windows_xdna_adapter_info_t{
                                 AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT, false},
                             amdf_windows_xdna_adapter_info_t{
                                 AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT, true},
                             amdf_windows_xdna_adapter_info_t{
                                 AMDF_WINDOWS_XDNA_PROTOCOL_METADATA, true},
                             amdf_windows_xdna_adapter_info_t{
                                 AMDF_WINDOWS_XDNA_PROTOCOL_METADATA_COMPACT,
                                 true}));

}  // namespace
