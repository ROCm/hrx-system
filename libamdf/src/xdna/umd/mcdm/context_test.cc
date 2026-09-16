// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/xdna/umd/context.h"

#include <malloc.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/xdna/device_profile.h"
#include "libamdf/src/xdna/target/npu4/bootstrap.h"
#include "libamdf/src/xdna/target/npu5/bootstrap.h"
#include "libamdf/src/xdna/umd/mcdm/context.h"
#include "libamdf/src/xdna/umd/mcdm/device.h"

namespace {

constexpr NTSTATUS kSuccess = 0;
constexpr NTSTATUS kFailure = static_cast<NTSTATUS>(0xC0000001u);

enum class Operation {
  kNone,
  kCreateKernelBuffer,
  kMapKernelBuffer,
  kMakeKernelBufferResident,
  kLockKernelBuffer,
  kCreateContext,
  kDestroyContext,
  kUnlockKernelBuffer,
  kDestroyKernelBuffer,
  kFreeHostAllocation,
};

struct KernelBuffer {
  // Page-aligned backing owned by the native allocation dependency.
  alignas(4096) std::array<uint8_t, 4096> bytes = {};
  // Whether the native host lock is still owned.
  bool locked = false;
};

struct FakeKmtState {
  // Coupled native interface exposed by this dependency.
  amdf_windows_xdna_protocol_t protocol = AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  // Partition width requested by the public caller.
  uint32_t expected_column_count = 1;
  // Native status returned by the required allocation-policy query.
  NTSTATUS query_status = kSuccess;
  // Native allocation policy reported by the query.
  uint8_t unshared_kernel_buffers = 0;
  // Native dependency failure injected before the selected operation accepts
  // work.
  Operation failure = Operation::kNone;
  // Independently owned native buffers, indexed by allocation handle.
  std::map<D3DKMT_HANDLE, KernelBuffer> kernel_buffers;
  // Next independent native allocation handle.
  D3DKMT_HANDLE next_allocation = 0x80;
  // Number of adapter allocation-policy queries.
  uint32_t query_count = 0;
  // Next context handle returned by native creation.
  D3DKMT_HANDLE next_context = 0x30;
  // Number of context destruction calls rejected before consumption.
  uint32_t destroy_failures_remaining = 0;
  // Context handles accepted by native creation.
  std::vector<D3DKMT_HANDLE> created_contexts;
  // Context handles consumed by native destruction.
  std::vector<D3DKMT_HANDLE> destroyed_contexts;
  // Ordered native and host releases observed during context teardown.
  std::vector<Operation> operations;
};

FakeKmtState* current_state = nullptr;

NTSTATUS APIENTRY FakeQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  ++current_state->query_count;
  EXPECT_EQ(query->hAdapter, 0x08u);
  if (current_state->query_status != kSuccess)
    return current_state->query_status;
  EXPECT_EQ(query->Type, KMTQAITYPE_UMDRIVERPRIVATE);
  if (current_state->protocol == AMDF_WINDOWS_XDNA_PROTOCOL_METADATA) {
    EXPECT_EQ(query->PrivateDriverDataSize, 8u);
    const uint32_t basic_info[2] = {0, 3};
    std::memcpy(query->pPrivateDriverData, basic_info, sizeof(basic_info));
    return kSuccess;
  }
  if (query->PrivateDriverDataSize == 8)
    return static_cast<NTSTATUS>(0xC0000023u);
  EXPECT_EQ(query->PrivateDriverDataSize, 12u);
  static_cast<uint8_t*>(query->pPrivateDriverData)[8] =
      current_state->unshared_kernel_buffers;
  return kSuccess;
}

NTSTATUS APIENTRY
FakeCreateContextVirtual(D3DKMT_CREATECONTEXTVIRTUAL* create) {
  current_state->operations.push_back(Operation::kCreateContext);
  const bool direct =
      current_state->protocol == AMDF_WINDOWS_XDNA_PROTOCOL_DIRECT;
  const auto* bytes = static_cast<const uint8_t*>(create->pPrivateDriverData);
  uint32_t columns = 0;
  if (direct) {
    EXPECT_EQ(create->PrivateDriverDataSize, 160u);
    uint64_t allocation_handle = 0;
    uint64_t host_address = 0;
    std::memcpy(&allocation_handle, bytes + 0x68, sizeof(allocation_handle));
    std::memcpy(&host_address, bytes + 0x78, sizeof(host_address));
    std::memcpy(&columns, bytes + 0x54, sizeof(columns));
    const auto& buffer = current_state->kernel_buffers.at(allocation_handle);
    EXPECT_TRUE(buffer.locked);
    EXPECT_EQ(host_address, reinterpret_cast<uintptr_t>(buffer.bytes.data()));
  } else {
    EXPECT_EQ(create->PrivateDriverDataSize, 312u);
    EXPECT_TRUE(current_state->kernel_buffers.empty());
    EXPECT_EQ(std::memcmp(bytes, amdf_xdna_npu5_bootstrap.context.uuid, 16), 0);
    uint64_t container_offset = 0;
    std::memcpy(&container_offset, bytes + 0x50, sizeof(container_offset));
    EXPECT_EQ(container_offset, 0x48u);
    uint64_t skipped_bytes = 0;
    std::memcpy(&skipped_bytes, bytes + container_offset + 0x90,
                sizeof(skipped_bytes));
    EXPECT_EQ(skipped_bytes, 0u);
    const auto* partition = bytes + container_offset + 0xA0 + skipped_bytes;
    uint32_t operations_per_cycle = 0;
    uint32_t candidate_count = 0;
    std::memcpy(&operations_per_cycle, partition + 0x40, 4);
    std::memcpy(&candidate_count, partition + 0x44, 4);
    std::memcpy(&columns, partition + 0x48, 4);
    EXPECT_EQ(operations_per_cycle,
              amdf_xdna_npu5_bootstrap.context.operations_per_cycle);
    EXPECT_EQ(candidate_count, 1u);
    EXPECT_EQ(partition + 0x50, bytes + create->PrivateDriverDataSize);
  }
  EXPECT_EQ(columns, current_state->expected_column_count);
  if (current_state->failure == Operation::kCreateContext) return kFailure;
  EXPECT_EQ(create->hDevice, 0x10u);
  EXPECT_EQ(create->NodeOrdinal, 0u);
  EXPECT_EQ(create->EngineAffinity, 1u);
  EXPECT_EQ(create->Flags.HwQueueSupported, 1u);
  EXPECT_NE(create->pPrivateDriverData, nullptr);
  EXPECT_GT(create->PrivateDriverDataSize, 0u);
  create->hContext = current_state->next_context++;
  const uint32_t cookie = create->hContext - 0x30;
  std::memcpy(static_cast<uint8_t*>(create->pPrivateDriverData) +
                  (direct ? 0x44 : 0x40),
              &cookie, sizeof(cookie));
  current_state->created_contexts.push_back(create->hContext);
  return kSuccess;
}

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* create) {
  current_state->operations.push_back(Operation::kCreateKernelBuffer);
  if (current_state->failure == Operation::kCreateKernelBuffer) return kFailure;
  EXPECT_EQ(create->Flags.CreateShared,
            current_state->unshared_kernel_buffers == 0);
  EXPECT_EQ(create->Flags.CreateResource, create->Flags.CreateShared);
  const auto allocation = current_state->next_allocation++;
  create->pAllocationInfo2->hAllocation = allocation;
  if (create->Flags.CreateResource) create->hResource = allocation + 0x1000;
  EXPECT_TRUE(current_state->kernel_buffers.try_emplace(allocation).second);
  return kSuccess;
}

NTSTATUS APIENTRY FakeMapAddress(D3DDDI_MAPGPUVIRTUALADDRESS* map) {
  current_state->operations.push_back(Operation::kMapKernelBuffer);
  if (current_state->failure == Operation::kMapKernelBuffer) return kFailure;
  EXPECT_EQ(current_state->kernel_buffers.count(map->hAllocation), 1u);
  map->VirtualAddress =
      UINT64_C(0x100000000) + uint64_t(map->hAllocation) * 4096;
  return kSuccess;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT*) {
  current_state->operations.push_back(Operation::kMakeKernelBufferResident);
  return current_state->failure == Operation::kMakeKernelBufferResident
             ? kFailure
             : kSuccess;
}

NTSTATUS APIENTRY FakeLock(D3DKMT_LOCK2* lock) {
  current_state->operations.push_back(Operation::kLockKernelBuffer);
  if (current_state->failure == Operation::kLockKernelBuffer) return kFailure;
  auto& buffer = current_state->kernel_buffers.at(lock->hAllocation);
  lock->pData = buffer.bytes.data();
  EXPECT_FALSE(buffer.locked);
  buffer.locked = true;
  return kSuccess;
}

NTSTATUS APIENTRY FakeUnlock(const D3DKMT_UNLOCK2* unlock) {
  current_state->operations.push_back(Operation::kUnlockKernelBuffer);
  auto& buffer = current_state->kernel_buffers.at(unlock->hAllocation);
  EXPECT_TRUE(buffer.locked);
  buffer.locked = false;
  return kSuccess;
}

NTSTATUS APIENTRY
FakeDestroyAllocation(const D3DKMT_DESTROYALLOCATION2* destroy) {
  current_state->operations.push_back(Operation::kDestroyKernelBuffer);
  const auto allocation = destroy->hResource != 0
                              ? destroy->hResource - 0x1000
                              : destroy->phAllocationList[0];
  EXPECT_FALSE(current_state->kernel_buffers.at(allocation).locked);
  EXPECT_EQ(current_state->kernel_buffers.erase(allocation), 1u);
  return kSuccess;
}

NTSTATUS APIENTRY FakeDestroyContext(const D3DKMT_DESTROYCONTEXT* destroy) {
  current_state->operations.push_back(Operation::kDestroyContext);
  if (current_state->destroy_failures_remaining != 0) {
    --current_state->destroy_failures_remaining;
    return kFailure;
  }
  current_state->destroyed_contexts.push_back(destroy->hContext);
  return kSuccess;
}

struct FaultAllocatorState {
  // One-based allocation call on which to report exhaustion.
  uint32_t failure_call = 0;
  // Number of allocation calls observed.
  uint32_t allocation_call_count = 0;
  // Number of allocations currently owned by the allocator.
  uint32_t live_allocation_count = 0;
  // Optional operation log receiving successful host releases.
  std::vector<Operation>* operations = nullptr;
};

void* AMDF_CALL FaultAllocate(void* user_data, uint64_t byte_length,
                              uint64_t minimum_alignment) {
  auto* state = static_cast<FaultAllocatorState*>(user_data);
  ++state->allocation_call_count;
  if (state->allocation_call_count == state->failure_call) return nullptr;
  void* pointer = _aligned_malloc(static_cast<size_t>(byte_length),
                                  static_cast<size_t>(minimum_alignment));
  if (pointer != nullptr) ++state->live_allocation_count;
  return pointer;
}

void AMDF_CALL FaultFree(void* user_data, void* allocation) {
  if (allocation == nullptr) return;
  auto* state = static_cast<FaultAllocatorState*>(user_data);
  EXPECT_NE(state->live_allocation_count, 0u);
  --state->live_allocation_count;
  if (state->operations != nullptr) {
    state->operations->push_back(Operation::kFreeHostAllocation);
  }
  _aligned_free(allocation);
}

class WindowsXdnaContextTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    kmt_.query_adapter_info = FakeQueryAdapterInfo;
    kmt_.create_context_virtual = FakeCreateContextVirtual;
    kmt_.destroy_context = FakeDestroyContext;
    kmt_.create_allocation = FakeCreateAllocation;
    kmt_.destroy_allocation = FakeDestroyAllocation;
    kmt_.map_gpu_virtual_address = FakeMapAddress;
    kmt_.make_resident = FakeMakeResident;
    kmt_.lock = FakeLock;
    kmt_.unlock = FakeUnlock;
    kmt_.wait_from_cpu =
        [](const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*) -> NTSTATUS {
      ADD_FAILURE() << "Paging dependency completes synchronously";
      return kFailure;
    };
    device_.host_allocator = amdf_allocator_system();
    device_.kmt = &kmt_;
    device_.adapter = 0x08;
    device_.device = 0x10;
    device_info_.array.column_count = 8;
    profile_.execution_capabilities =
        AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1;
    profile_.info = &device_info_;
    profile_.bootstrap = &amdf_xdna_npu5_bootstrap;
    device_.profile = &profile_;
    create_info_.logical_column_count = 1;
    create_info_.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
    create_info_.acceptable_scheduling_modes =
        AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  }

  void TearDown() override { current_state = nullptr; }

  // Native allocation and context observations.
  FakeKmtState state_;
  // Procedures supplied at the native dependency boundary.
  amdf_kmt_api_t kmt_ = {};
  // Live device borrowed by each context.
  amdf_xdna_umd_device_t device_ = {};
  // Hardware geometry used by the selected profile.
  amdf_xdna_device_info_t device_info_ = {};
  // Target capabilities and bootstrap image.
  amdf_xdna_device_profile_t profile_ = {};
  // Public partition and scheduling request.
  amdf_xdna_context_create_info_t create_info_ = {};
};

TEST_F(WindowsXdnaContextTest, CreatesTwoContextsAndDestroysIndependently) {
  amdf_xdna_umd_context_t* contexts[2] = {};
  amdf_xdna_umd_context_result_t results[2] = {};
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(amdf_xdna_umd_context_create(&device_, &create_info_,
                                           &contexts[i], &results[i]),
              AMDF_STATUS_OK);
    ASSERT_NE(contexts[i], nullptr);
    EXPECT_EQ(contexts[i]->device, &device_);
    EXPECT_EQ(results[i].physical_column_count, 0u);
  }
  EXPECT_NE(results[0].id.words[0], results[1].id.words[0]);
  EXPECT_EQ(state_.kernel_buffers.size(), 2u);
  EXPECT_EQ(state_.query_count, 4u);
  EXPECT_EQ(state_.created_contexts, (std::vector<D3DKMT_HANDLE>{0x30, 0x31}));

  ASSERT_EQ(amdf_xdna_umd_context_destroy(contexts[0]), AMDF_STATUS_OK);
  contexts[0] = nullptr;
  EXPECT_EQ(contexts[1]->handle, 0x31u);
  EXPECT_EQ(state_.kernel_buffers.size(), 1u);
  EXPECT_TRUE(
      state_.kernel_buffers.at(contexts[1]->kernel_buffer.allocation).locked);
  EXPECT_EQ(state_.destroyed_contexts, (std::vector<D3DKMT_HANDLE>{0x30}));
  ASSERT_EQ(amdf_xdna_umd_context_destroy(contexts[1]), AMDF_STATUS_OK);
  contexts[1] = nullptr;
  EXPECT_TRUE(state_.kernel_buffers.empty());
  EXPECT_EQ(state_.destroyed_contexts,
            (std::vector<D3DKMT_HANDLE>{0x30, 0x31}));
}

TEST_F(WindowsXdnaContextTest, CreatesContextWithZeroCookie) {
  profile_.bootstrap = &amdf_xdna_npu4_bootstrap;
  amdf_xdna_umd_context_t* context = nullptr;
  amdf_xdna_umd_context_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      AMDF_STATUS_OK);
  EXPECT_EQ(state_.query_count, 2u);
  EXPECT_EQ(context->command_aperture_cookie, 0u);
  EXPECT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroyed_contexts, (std::vector<D3DKMT_HANDLE>{0x30}));
}

TEST_F(WindowsXdnaContextTest, AdmitsMetadataPartitionsWithoutKernelBuffers) {
  state_.protocol = AMDF_WINDOWS_XDNA_PROTOCOL_METADATA;
  for (uint32_t width : {1u, 4u, 8u}) {
    state_.expected_column_count = width;
    create_info_.logical_column_count = width;
    state_.operations.clear();
    amdf_xdna_umd_context_t* context = nullptr;
    amdf_xdna_umd_context_result_t result = {};
    ASSERT_EQ(amdf_xdna_umd_context_create(&device_, &create_info_, &context,
                                           &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(context->adapter_info.protocol,
              AMDF_WINDOWS_XDNA_PROTOCOL_METADATA);
    EXPECT_EQ(context->kernel_buffer.allocation, 0u);
    EXPECT_EQ(context->command_aperture_cookie, context->handle - 0x30);
    EXPECT_EQ(result.physical_column_count, 0u);
    ASSERT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);
    EXPECT_EQ(state_.operations,
              (std::vector<Operation>{Operation::kCreateContext,
                                      Operation::kDestroyContext}));
  }
  EXPECT_EQ(state_.query_count, 3u);
}

TEST_F(WindowsXdnaContextTest, MetadataAdmissionFailureDoesNotPublish) {
  state_.protocol = AMDF_WINDOWS_XDNA_PROTOCOL_METADATA;
  state_.failure = Operation::kCreateContext;
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  auto* const sentinel =
      reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
  amdf_xdna_umd_context_t* context = sentinel;
  amdf_xdna_umd_context_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original = result;
  EXPECT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(context, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(allocator_state.live_allocation_count, 0u);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kCreateContext}));
}

TEST_F(WindowsXdnaContextTest, RetainsKernelBufferUntilContextDestruction) {
  for (uint8_t unshared : {uint8_t{0}, uint8_t{1}}) {
    state_.unshared_kernel_buffers = unshared;
    state_.operations.clear();
    amdf_xdna_umd_context_t* context = nullptr;
    amdf_xdna_umd_context_result_t result = {};
    ASSERT_EQ(amdf_xdna_umd_context_create(&device_, &create_info_, &context,
                                           &result),
              AMDF_STATUS_OK);
    EXPECT_EQ(state_.operations,
              (std::vector<Operation>{
                  Operation::kCreateKernelBuffer, Operation::kMapKernelBuffer,
                  Operation::kMakeKernelBufferResident,
                  Operation::kLockKernelBuffer, Operation::kCreateContext}));
    ASSERT_EQ(state_.kernel_buffers.size(), 1u);
    EXPECT_TRUE(state_.kernel_buffers.begin()->second.locked);
    state_.operations.clear();
    EXPECT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);
    EXPECT_EQ(state_.operations,
              (std::vector<Operation>{Operation::kDestroyContext,
                                      Operation::kUnlockKernelBuffer,
                                      Operation::kDestroyKernelBuffer}));
    EXPECT_TRUE(state_.kernel_buffers.empty());
  }
}

TEST_F(WindowsXdnaContextTest, RollsBackKernelBufferBeforePublication) {
  for (Operation failure :
       {Operation::kCreateKernelBuffer, Operation::kMapKernelBuffer,
        Operation::kMakeKernelBufferResident, Operation::kLockKernelBuffer,
        Operation::kCreateContext}) {
    state_.failure = failure;
    auto* const sentinel =
        reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
    amdf_xdna_umd_context_t* context = sentinel;
    amdf_xdna_umd_context_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original = result;
    EXPECT_EQ(amdf_xdna_umd_context_create(&device_, &create_info_, &context,
                                           &result),
              amdf_kmt_make_status(kFailure));
    EXPECT_EQ(context, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
    EXPECT_TRUE(state_.kernel_buffers.empty());
    EXPECT_TRUE(state_.created_contexts.empty());
  }
}

TEST_F(WindowsXdnaContextTest, ReleasesKernelBufferOnHostExhaustion) {
  for (uint32_t failure_call : {1u, 2u}) {
    FaultAllocatorState allocator_state = {.failure_call = failure_call};
    device_.host_allocator = {
        .user_data = &allocator_state,
        .allocate = FaultAllocate,
        .free = FaultFree,
    };
    auto* const sentinel =
        reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
    amdf_xdna_umd_context_t* context = sentinel;
    amdf_xdna_umd_context_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const auto original = result;
    EXPECT_EQ(amdf_status_code(amdf_xdna_umd_context_create(
                  &device_, &create_info_, &context, &result)),
              AMDF_STATUS_CODE_RESOURCE_EXHAUSTED);
    EXPECT_EQ(context, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
    EXPECT_EQ(allocator_state.live_allocation_count, 0u);
    EXPECT_TRUE(state_.kernel_buffers.empty());
    EXPECT_EQ(state_.destroyed_contexts, state_.created_contexts);
  }
}

TEST_F(WindowsXdnaContextTest,
       RetainsKernelBufferWhenNativeContextRemainsLive) {
  amdf_xdna_umd_context_t* context = nullptr;
  amdf_xdna_umd_context_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      AMDF_STATUS_OK);
  state_.destroy_failures_remaining = 1;
  EXPECT_EQ(amdf_xdna_umd_context_destroy(context),
            amdf_kmt_make_status(kFailure));
  ASSERT_EQ(state_.kernel_buffers.size(), 1u);
  EXPECT_TRUE(state_.kernel_buffers.begin()->second.locked);
  EXPECT_TRUE(state_.destroyed_contexts.empty());
  EXPECT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);
  EXPECT_TRUE(state_.kernel_buffers.empty());
  EXPECT_EQ(state_.destroyed_contexts, state_.created_contexts);
}

TEST_F(WindowsXdnaContextTest,
       RejectsUnrepresentableNativeCookieWithoutPublishing) {
  state_.next_context = 0x130;
  profile_.bootstrap = &amdf_xdna_npu4_bootstrap;
  auto* const sentinel =
      reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
  amdf_xdna_umd_context_t* context = sentinel;
  amdf_xdna_umd_context_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original = result;
  EXPECT_EQ(amdf_status_code(amdf_xdna_umd_context_create(
                &device_, &create_info_, &context, &result)),
            AMDF_STATUS_CODE_INTERNAL);
  EXPECT_EQ(context, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(state_.destroyed_contexts, (std::vector<D3DKMT_HANDLE>{0x130}));
}

TEST_F(WindowsXdnaContextTest,
       RejectsUnavailableInterpreterAndContextProceduresBeforePreparation) {
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  for (uint32_t unavailable : {0u, 1u, 2u}) {
    profile_.execution_capabilities =
        unavailable == 0
            ? 0
            : AMDF_XDNA_EXECUTION_CAPABILITY_TRANSACTION_INTERPRETER_V1;
    kmt_.create_context_virtual =
        unavailable == 1 ? nullptr : FakeCreateContextVirtual;
    kmt_.destroy_context = unavailable == 2 ? nullptr : FakeDestroyContext;
    auto* const sentinel =
        reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
    amdf_xdna_umd_context_t* context = sentinel;
    amdf_xdna_umd_context_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_context_result_t original = result;
    EXPECT_EQ(amdf_status_code(amdf_xdna_umd_context_create(
                  &device_, &create_info_, &context, &result)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(context, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  }
  EXPECT_EQ(state_.query_count, 0u);
  EXPECT_EQ(allocator_state.allocation_call_count, 0u);
  EXPECT_TRUE(state_.created_contexts.empty());
}

TEST_F(WindowsXdnaContextTest,
       RequiresAllocationPolicyBeforeContextPreparation) {
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  for (uint32_t failure : {0u, 1u, 2u}) {
    state_.query_status = failure == 0 ? kFailure : kSuccess;
    state_.unshared_kernel_buffers = failure == 1 ? 2 : UINT8_MAX;
    auto* const sentinel =
        reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
    amdf_xdna_umd_context_t* context = sentinel;
    amdf_xdna_umd_context_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_context_result_t original = result;
    EXPECT_EQ(amdf_xdna_umd_context_create(&device_, &create_info_, &context,
                                           &result),
              failure == 0
                  ? amdf_kmt_make_status(kFailure)
                  : amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));
    EXPECT_EQ(context, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  }
  EXPECT_EQ(state_.query_count, 5u);
  EXPECT_EQ(allocator_state.allocation_call_count, 0u);
  EXPECT_TRUE(state_.created_contexts.empty());
}

TEST_F(WindowsXdnaContextTest,
       RejectsExplicitPlacementBeforeNativeOrHostAllocation) {
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  for (uint32_t origin : {0u, 1u, 4u}) {
    create_info_.physical_column_origin = origin;
    auto* const sentinel =
        reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
    amdf_xdna_umd_context_t* context = sentinel;
    amdf_xdna_umd_context_result_t result;
    std::memset(&result, 0xA5, sizeof(result));
    const amdf_xdna_umd_context_result_t original_result = result;
    EXPECT_EQ(amdf_status_code(amdf_xdna_umd_context_create(
                  &device_, &create_info_, &context, &result)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(context, sentinel);
    EXPECT_EQ(std::memcmp(&result, &original_result, sizeof(result)), 0);
  }
  EXPECT_EQ(allocator_state.allocation_call_count, 0u);
  EXPECT_TRUE(state_.created_contexts.empty());
  EXPECT_TRUE(state_.destroyed_contexts.empty());
}

TEST_F(WindowsXdnaContextTest,
       DestroysNativeContextBeforePrivateExecutionStorage) {
  FaultAllocatorState allocator_state = {.operations = &state_.operations};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  amdf_xdna_umd_context_t* context = nullptr;
  amdf_xdna_umd_context_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      AMDF_STATUS_OK);
  ASSERT_NE(context, nullptr);
  EXPECT_EQ(allocator_state.live_allocation_count, 2u);
  state_.operations.clear();

  ASSERT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);

  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kDestroyContext, Operation::kFreeHostAllocation,
                Operation::kUnlockKernelBuffer, Operation::kDestroyKernelBuffer,
                Operation::kFreeHostAllocation}));
  EXPECT_EQ(allocator_state.live_allocation_count, 0u);
}

TEST_F(WindowsXdnaContextTest, ReleasesFailedRollbackWithoutPublishingOutputs) {
  FaultAllocatorState allocator_state = {.failure_call = 2};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  state_.destroy_failures_remaining = 1;
  auto* const context_sentinel =
      reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
  amdf_xdna_umd_context_t* context = context_sentinel;
  amdf_xdna_umd_context_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  amdf_xdna_umd_context_result_t expected_result = result;

  EXPECT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(context, context_sentinel);
  EXPECT_EQ(std::memcmp(&result, &expected_result, sizeof(result)), 0);
  EXPECT_EQ(allocator_state.live_allocation_count, 0u);
  EXPECT_TRUE(state_.destroyed_contexts.empty());
  EXPECT_EQ(
      state_.operations,
      (std::vector<Operation>{
          Operation::kCreateKernelBuffer, Operation::kMapKernelBuffer,
          Operation::kMakeKernelBufferResident, Operation::kLockKernelBuffer,
          Operation::kCreateContext, Operation::kDestroyContext}));
  ASSERT_EQ(state_.kernel_buffers.size(), 1u);
  EXPECT_TRUE(state_.kernel_buffers.begin()->second.locked);
}

TEST_F(WindowsXdnaContextTest, ReportsMalformedContextRollbackFailureLocally) {
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  state_.next_context = 0x130;
  state_.destroy_failures_remaining = 1;
  profile_.bootstrap = &amdf_xdna_npu4_bootstrap;
  auto* const sentinel =
      reinterpret_cast<amdf_xdna_umd_context_t*>(uintptr_t{1});
  amdf_xdna_umd_context_t* context = sentinel;
  amdf_xdna_umd_context_result_t result;
  std::memset(&result, 0xA5, sizeof(result));
  const auto original = result;

  EXPECT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(context, sentinel);
  EXPECT_EQ(std::memcmp(&result, &original, sizeof(result)), 0);
  EXPECT_EQ(allocator_state.live_allocation_count, 0u);
  EXPECT_TRUE(state_.destroyed_contexts.empty());
  EXPECT_EQ(
      state_.operations,
      (std::vector<Operation>{
          Operation::kCreateKernelBuffer, Operation::kMapKernelBuffer,
          Operation::kMakeKernelBufferResident, Operation::kLockKernelBuffer,
          Operation::kCreateContext, Operation::kDestroyContext}));
  ASSERT_EQ(state_.kernel_buffers.size(), 1u);
  EXPECT_TRUE(state_.kernel_buffers.begin()->second.locked);
}

TEST_F(WindowsXdnaContextTest, ExplicitDestroyFailureRetainsPublishedOwner) {
  FaultAllocatorState allocator_state = {};
  device_.host_allocator = {
      .user_data = &allocator_state,
      .allocate = FaultAllocate,
      .free = FaultFree,
  };
  amdf_xdna_umd_context_t* context = nullptr;
  amdf_xdna_umd_context_result_t result = {};
  ASSERT_EQ(
      amdf_xdna_umd_context_create(&device_, &create_info_, &context, &result),
      AMDF_STATUS_OK);
  state_.destroy_failures_remaining = 1;
  EXPECT_EQ(amdf_xdna_umd_context_destroy(context),
            amdf_kmt_make_status(kFailure));
  EXPECT_EQ(context->handle, 0x30u);
  EXPECT_NE(context->kernel_execution, nullptr);
  EXPECT_EQ(allocator_state.live_allocation_count, 2u);
  EXPECT_TRUE(state_.destroyed_contexts.empty());

  EXPECT_EQ(amdf_xdna_umd_context_destroy(context), AMDF_STATUS_OK);
  EXPECT_EQ(state_.destroyed_contexts, (std::vector<D3DKMT_HANDLE>{0x30}));
  EXPECT_EQ(allocator_state.live_allocation_count, 0u);
}

}  // namespace
