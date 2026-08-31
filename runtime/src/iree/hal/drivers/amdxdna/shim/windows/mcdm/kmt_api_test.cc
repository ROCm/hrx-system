// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <tuple>

#include "iree/testing/gtest.h"

namespace iree::hal::amdxdna::mcdm {
namespace {

int g_close_count = 0;
D3DKMT_HANDLE g_closed_adapters[4] = {};
NTSTATUS g_query_statuses[4] = {};
uint32_t g_query_outputs[4][3] = {};
UINT g_query_sizes[4] = {};
D3DKMT_HANDLE g_query_adapters[4] = {};
KMTQUERYADAPTERINFOTYPE g_query_types[4] = {};
bool g_query_data_was_zero[4] = {};
size_t g_query_count = 0;
enum class SetupCall { submit, make_resident, lock, unlock, invalidate, wait };
SetupCall g_setup_calls[16] = {};
enum class TeardownCall {
  unlock,
  destroy_allocation,
  destroy_hw_queue,
  destroy_context,
};
TeardownCall g_teardown_calls[8] = {};
size_t g_teardown_call_count = 0;
bool g_record_teardown = false;
uint32_t g_submit_opcodes[16] = {};
uint64_t g_submit_fences[16] = {};
uint64_t g_submit_offsets[16] = {};
uint64_t g_wait_fences[16] = {};
D3DKMT_HANDLE g_gpu_wait_context = 0;
D3DKMT_HANDLE g_gpu_wait_object = 0;
uint64_t g_gpu_wait_fence = 0;
size_t g_gpu_wait_count = 0;
size_t g_setup_call_count = 0;
size_t g_submit_count = 0;
size_t g_wait_count = 0;
size_t g_submit_failure_index = SIZE_MAX;
NTSTATUS g_submit_failure_status = static_cast<NTSTATUS>(0xC0000001u);
NTSTATUS g_wait_status = 0;
uint8_t g_invalidate_first_bytes[4] = {};
size_t g_invalidate_count = 0;
uint32_t* g_complete_on_wait = nullptr;
struct DestroyCall {
  D3DKMT_HANDLE resource = 0;
  D3DKMT_HANDLE allocation = 0;
  uint32_t allocation_count = 0;
  uint32_t flags = 0;
  size_t wait_count = 0;
};
DestroyCall g_destroy_calls[4] = {};
size_t g_destroy_call_count = 0;
size_t g_free_gpu_va_count = 0;
std::array<uint8_t, 0x40000> g_locked_aperture = {};

void ResetFakes() {
  g_close_count = 0;
  std::memset(g_closed_adapters, 0, sizeof(g_closed_adapters));
  std::memset(g_query_statuses, 0, sizeof(g_query_statuses));
  std::memset(g_query_outputs, 0, sizeof(g_query_outputs));
  std::memset(g_query_sizes, 0, sizeof(g_query_sizes));
  std::memset(g_query_adapters, 0, sizeof(g_query_adapters));
  std::memset(g_query_types, 0, sizeof(g_query_types));
  std::memset(g_query_data_was_zero, 0, sizeof(g_query_data_was_zero));
  g_query_count = 0;
  std::memset(g_setup_calls, 0, sizeof(g_setup_calls));
  std::memset(g_teardown_calls, 0, sizeof(g_teardown_calls));
  g_teardown_call_count = 0;
  g_record_teardown = false;
  std::memset(g_submit_opcodes, 0, sizeof(g_submit_opcodes));
  std::memset(g_submit_fences, 0, sizeof(g_submit_fences));
  std::memset(g_submit_offsets, 0, sizeof(g_submit_offsets));
  std::memset(g_wait_fences, 0, sizeof(g_wait_fences));
  g_gpu_wait_context = 0;
  g_gpu_wait_object = 0;
  g_gpu_wait_fence = 0;
  g_gpu_wait_count = 0;
  g_setup_call_count = 0;
  g_submit_count = 0;
  g_wait_count = 0;
  g_submit_failure_index = SIZE_MAX;
  g_submit_failure_status = static_cast<NTSTATUS>(0xC0000001u);
  g_wait_status = 0;
  std::memset(g_invalidate_first_bytes, 0,
              sizeof(g_invalidate_first_bytes));
  g_invalidate_count = 0;
  g_complete_on_wait = nullptr;
  std::memset(g_destroy_calls, 0, sizeof(g_destroy_calls));
  g_destroy_call_count = 0;
  g_free_gpu_va_count = 0;
  g_locked_aperture.fill(0);
}

NTSTATUS APIENTRY FakeQueryAdapterInfo(
    const D3DKMT_QUERYADAPTERINFO* args) {
  const size_t call = g_query_count++;
  if (call >= 4) return static_cast<NTSTATUS>(0xC0000001u);
  g_query_sizes[call] = args->PrivateDriverDataSize;
  g_query_adapters[call] = args->hAdapter;
  g_query_types[call] = args->Type;
  g_query_data_was_zero[call] = true;
  const auto* data = static_cast<const uint8_t*>(args->pPrivateDriverData);
  for (UINT i = 0; i < args->PrivateDriverDataSize; ++i) {
    g_query_data_was_zero[call] &= data[i] == 0;
  }
  if (g_query_statuses[call] == 0) {
    std::memcpy(args->pPrivateDriverData, g_query_outputs[call],
                args->PrivateDriverDataSize);
  }
  return g_query_statuses[call];
}

void WriteExpectedU32(std::array<uint8_t, kMaxMcdmPrivateDataSize>* data,
                      size_t offset, uint32_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void WriteExpectedU64(std::array<uint8_t, kMaxMcdmPrivateDataSize>* data,
                      size_t offset, uint64_t value) {
  std::memcpy(data->data() + offset, &value, sizeof(value));
}

void ExpectPrivateDataEquals(
    const McdmPrivateData& actual,
    const std::array<uint8_t, kMaxMcdmPrivateDataSize>& expected,
    uint32_t expected_size) {
  ASSERT_EQ(actual.size, expected_size);
  EXPECT_EQ(std::memcmp(actual.data, expected.data(), expected_size), 0);
}

void ExpectPrivatePacketsMatchSnapshot(McdmAbi mcdm_abi) {
  const McdmAbiInfo abi = GetMcdmAbiInfo(mcdm_abi);
  CommandAperture aperture = {};
  aperture.allocation = 0x11223344;
  aperture.gpu_allocation = 0x55667788;
  aperture.status_gpu_va = 0x1111222233334000ull;
  aperture.gpu_va = 0x5555666677778000ull;
  aperture.protocol_gpu_va = aperture.gpu_va;
  aperture.cpu_ptr = reinterpret_cast<void*>(0x123456789ABCull);

  std::array<uint8_t, kMaxMcdmPrivateDataSize> expected = {};
  WriteExpectedU32(&expected, 0x00, 5);
  WriteExpectedU64(&expected, 0x28, aperture.allocation);
  if (abi.status_has_gpu_va) {
    WriteExpectedU64(&expected, 0x30, aperture.status_gpu_va);
    WriteExpectedU32(&expected, 0x38, 0);
    WriteExpectedU32(&expected, 0x3c, 8);
    WriteExpectedU64(&expected, 0x40,
                     reinterpret_cast<uint64_t>(aperture.cpu_ptr));
  } else {
    WriteExpectedU32(&expected, 0x30, 0);
    WriteExpectedU32(&expected, 0x34, 8);
    WriteExpectedU64(&expected, 0x38,
                     reinterpret_cast<uint64_t>(aperture.cpu_ptr));
  }
  WriteExpectedU32(&expected, abi.submit_private_prefix_size, 1);
  WriteExpectedU64(&expected, abi.submit_private_prefix_size + 8,
                   aperture.gpu_va);
  ExpectPrivateDataEquals(BuildPathBSetupPrivateData(mcdm_abi, aperture),
                          expected, abi.setup_private_size);

  expected = {};
  constexpr uint64_t kSyncOffset = 0x24680;
  WriteExpectedU32(&expected, 0x00, 9);
  if (abi.sync_has_allocation_handle) {
    WriteExpectedU64(&expected, 0x08, aperture.gpu_allocation);
  }
  WriteExpectedU64(&expected, 0x10, kSyncOffset);
  ExpectPrivateDataEquals(
      BuildPathBSyncPrivateData(mcdm_abi, aperture, kSyncOffset),
      expected, abi.submit_private_prefix_size);

  Buffer exec_buffer = {};
  exec_buffer.allocation = 0x10203040;
  Buffer completion_ring = {};
  completion_ring.allocation = 0x50607080;
  completion_ring.gpu_va = 0xA0B0C0D0E000ull;
  constexpr uint32_t kSlotOffset = 0x18;
  void* const slot_cpu = reinterpret_cast<void*>(0xFEDCBA987650ull);
  const std::array<uint8_t, 16> ert_packet = {
      0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
      0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE};

  auto expect_submit_common = [&](uint32_t opcode) {
    expected = {};
    WriteExpectedU32(&expected, 0x00, opcode);
    WriteExpectedU64(&expected, 0x08, exec_buffer.allocation);
    WriteExpectedU64(&expected, 0x10, ert_packet.size());
    WriteExpectedU64(&expected, 0x28, completion_ring.allocation);
    if (abi.status_has_gpu_va) {
      WriteExpectedU64(&expected, 0x30, completion_ring.gpu_va);
      WriteExpectedU32(&expected, 0x38, kSlotOffset);
      WriteExpectedU32(&expected, 0x3c, 8);
      WriteExpectedU64(&expected, 0x40,
                       reinterpret_cast<uint64_t>(slot_cpu));
    } else {
      WriteExpectedU32(&expected, 0x30, kSlotOffset);
      WriteExpectedU32(&expected, 0x34, 8);
      WriteExpectedU64(&expected, 0x38,
                       reinterpret_cast<uint64_t>(slot_cpu));
    }
  };

  expect_submit_common(3);
  std::memcpy(expected.data() + abi.pathb_packet_offset, ert_packet.data(),
              ert_packet.size());
  ExpectPrivateDataEquals(
      BuildPathBSubmitPrivateData(
          mcdm_abi, exec_buffer, completion_ring, kSlotOffset, slot_cpu,
          ert_packet.data(), static_cast<uint32_t>(ert_packet.size()), 3,
          /*chain_info=*/nullptr),
      expected, abi.pathb_private_size);

  PathBChainSubmitInfo chain_info = {};
  chain_info.descriptor_gpu_va = 0x135724680000ull;
  chain_info.descriptor_bytes = 0x240;
  chain_info.command_count = 7;
  chain_info.first_child_opcode = 3;
  expect_submit_common(6);
  WriteExpectedU64(&expected, abi.chain_metadata_offset,
                   chain_info.descriptor_gpu_va);
  WriteExpectedU32(&expected, abi.chain_metadata_offset + 8,
                   chain_info.descriptor_bytes);
  WriteExpectedU32(&expected, abi.chain_metadata_offset + 12,
                   chain_info.command_count);
  WriteExpectedU32(&expected, abi.chain_metadata_offset + 16,
                   chain_info.first_child_opcode);
  std::memcpy(expected.data() + abi.pathb_packet_offset, ert_packet.data(),
              ert_packet.size());
  ExpectPrivateDataEquals(
      BuildPathBSubmitPrivateData(
          mcdm_abi, exec_buffer, completion_ring, kSlotOffset, slot_cpu,
          ert_packet.data(), static_cast<uint32_t>(ert_packet.size()), 3,
          &chain_info),
      expected, abi.pathb_private_size);
}

NTSTATUS APIENTRY FakeEnumTooManyAdapters(D3DKMT_ENUMADAPTERS3* args) {
  args->NumAdapters = 257;
  return 0;
}

NTSTATUS APIENTRY FakeCreateDeviceFails(D3DKMT_CREATEDEVICE* args) {
  args->hDevice = 0;
  return static_cast<NTSTATUS>(0xC0000001u);
}

NTSTATUS APIENTRY FakeCloseAdapter(CONST D3DKMT_CLOSEADAPTER* args) {
  if (g_close_count < static_cast<int>(sizeof(g_closed_adapters) /
                                       sizeof(g_closed_adapters[0]))) {
    g_closed_adapters[g_close_count] = args->hAdapter;
  }
  ++g_close_count;
  return 0;
}

NTSTATUS APIENTRY FakeSubmitCommandToHwQueue(
    CONST D3DKMT_SUBMITCOMMANDTOHWQUEUE* args) {
  const size_t call_index = g_submit_count;
  if (g_setup_call_count < std::size(g_setup_calls)) {
    g_setup_calls[g_setup_call_count] = SetupCall::submit;
  }
  ++g_setup_call_count;
  if (g_submit_count >= std::size(g_submit_opcodes)) {
    return static_cast<NTSTATUS>(0xC0000001u);
  }
  std::memcpy(&g_submit_opcodes[g_submit_count], args->pPrivateDriverData,
              sizeof(uint32_t));
  std::memcpy(&g_submit_offsets[g_submit_count],
              static_cast<const uint8_t*>(args->pPrivateDriverData) + 0x10,
              sizeof(uint64_t));
  g_submit_fences[g_submit_count++] = args->HwQueueProgressFenceId;
  if (call_index == g_submit_failure_index) return g_submit_failure_status;
  return 0;
}

NTSTATUS APIENTRY FakeMakeResident(D3DDDI_MAKERESIDENT* args) {
  g_setup_calls[g_setup_call_count++] = SetupCall::make_resident;
  args->PagingFenceValue = 11;
  return static_cast<NTSTATUS>(0x00000103u);
}

NTSTATUS APIENTRY FakeCreateAllocation(D3DKMT_CREATEALLOCATION* args) {
  args->pAllocationInfo2[0].hAllocation = 0x40;
  args->hResource = 0x41;
  return 0;
}

NTSTATUS APIENTRY FakeMapGpuVirtualAddress(
    D3DDDI_MAPGPUVIRTUALADDRESS* args) {
  args->VirtualAddress = 0x10000;
  args->PagingFenceValue = 10;
  return static_cast<NTSTATUS>(0x00000103u);
}

NTSTATUS APIENTRY FakeLock2(D3DKMT_LOCK2* args) {
  g_setup_calls[g_setup_call_count++] = SetupCall::lock;
  args->pData = g_locked_aperture.data();
  return 0;
}

NTSTATUS APIENTRY FakeUnlock2(CONST D3DKMT_UNLOCK2* args) {
  (void)args;
  if (g_record_teardown) {
    g_teardown_calls[g_teardown_call_count++] = TeardownCall::unlock;
  } else {
    g_setup_calls[g_setup_call_count++] = SetupCall::unlock;
  }
  return 0;
}

NTSTATUS APIENTRY FakeDestroyAllocation2(
    CONST D3DKMT_DESTROYALLOCATION2* args) {
  if (g_record_teardown) {
    g_teardown_calls[g_teardown_call_count++] =
        TeardownCall::destroy_allocation;
  }
  DestroyCall& call = g_destroy_calls[g_destroy_call_count++];
  call.resource = args->hResource;
  call.allocation_count = args->AllocationCount;
  call.flags = args->Flags.Value;
  call.wait_count = g_wait_count;
  if (args->AllocationCount && args->phAllocationList) {
    call.allocation = args->phAllocationList[0];
  }
  return 0;
}

NTSTATUS APIENTRY FakeDestroyHwQueue(CONST D3DKMT_DESTROYHWQUEUE* args) {
  (void)args;
  g_teardown_calls[g_teardown_call_count++] =
      TeardownCall::destroy_hw_queue;
  return 0;
}

NTSTATUS APIENTRY FakeDestroyContext(CONST D3DKMT_DESTROYCONTEXT* args) {
  (void)args;
  g_teardown_calls[g_teardown_call_count++] = TeardownCall::destroy_context;
  return 0;
}

NTSTATUS APIENTRY FakeFreeGpuVirtualAddress(
    CONST D3DKMT_FREEGPUVIRTUALADDRESS* args) {
  (void)args;
  ++g_free_gpu_va_count;
  return 0;
}

NTSTATUS APIENTRY FakeInvalidateCache(CONST D3DKMT_INVALIDATECACHE* args) {
  if (g_invalidate_count < std::size(g_invalidate_first_bytes)) {
    g_invalidate_first_bytes[g_invalidate_count] = g_locked_aperture[0];
  }
  ++g_invalidate_count;
  g_setup_calls[g_setup_call_count++] = SetupCall::invalidate;
  return 0;
}

NTSTATUS APIENTRY FakeWaitFromCpu(
    CONST D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* args) {
  g_setup_calls[g_setup_call_count++] = SetupCall::wait;
  g_wait_fences[g_wait_count++] = args->FenceValueArray[0];
  if (g_wait_status == 0 && g_complete_on_wait) *g_complete_on_wait = 1;
  return g_wait_status;
}

NTSTATUS APIENTRY FakeWaitFromGpu(
    CONST D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMGPU* args) {
  g_gpu_wait_context = args->hContext;
  g_gpu_wait_object = args->ObjectHandleArray[0];
  g_gpu_wait_fence = args->MonitoredFenceValueArray[0];
  ++g_gpu_wait_count;
  return 0;
}

TEST(KmtApiTest, ErrorMessageDefaultsWhenEmpty) {
  Error error = {};
  EXPECT_STREQ(ErrorMessage(&error), "unknown MCDM error");
}

TEST(KmtApiTest, LegacyAbiMatchesOriginalSubmitLayout) {
  McdmAbiInfo abi = GetMcdmAbiInfo(McdmAbi::legacy);
  EXPECT_EQ(abi.status_private_type, 0x332bu);
  EXPECT_EQ(abi.status_policy, 0u);
  EXPECT_EQ(abi.status_xcl_flags, 0u);
  EXPECT_EQ(abi.submit_private_prefix_size, 0x68u);
  EXPECT_EQ(abi.setup_private_size, 0x270u);
  EXPECT_EQ(abi.pathb_private_size, 0x268u);
  EXPECT_EQ(abi.pathb_packet_offset, 0x68u);
  EXPECT_EQ(abi.chain_metadata_offset, 0x48u);
  EXPECT_EQ(abi.pathb_bo_table_entry_count, 5u);
  EXPECT_FALSE(abi.status_has_gpu_va);
  EXPECT_TRUE(abi.sync_has_allocation_handle);
  EXPECT_EQ(abi.command_aperture_code_slot_size, 0x8000u);
  EXPECT_EQ(abi.command_aperture_write_publish_mode,
            CommandApertureWritePublishMode::kmt_invalidate);
  EXPECT_EQ(abi.command_aperture_code_publish_granularity, 0u);
  EXPECT_FALSE(abi.command_aperture_residency_after_bootstrap);
  EXPECT_TRUE(abi.command_aperture_remap_after_write);
  EXPECT_FALSE(abi.retain_command_aperture_mapping_across_contexts);
  EXPECT_FALSE(abi.retain_command_aperture_session_across_contexts);
  EXPECT_EQ(abi.shared_resource_destroy_flags, 0u);
  EXPECT_TRUE(abi.explicit_gpu_va_free_on_destroy);
}

TEST(KmtApiTest, CompactAbiMatchesXrt221SubmitLayout) {
  McdmAbiInfo abi = GetMcdmAbiInfo(McdmAbi::compact);
  EXPECT_EQ(abi.status_private_type, 0x332cu);
  EXPECT_EQ(abi.status_policy, 2u);
  EXPECT_EQ(abi.status_xcl_flags, 0x02000000u);
  EXPECT_EQ(abi.submit_private_prefix_size, 0x78u);
  EXPECT_EQ(abi.setup_private_size, 0x280u);
  EXPECT_EQ(abi.pathb_private_size, 0x278u);
  EXPECT_EQ(abi.pathb_packet_offset, 0x78u);
  EXPECT_EQ(abi.chain_metadata_offset, 0x58u);
  EXPECT_EQ(abi.pathb_bo_table_entry_count, 6u);
  EXPECT_TRUE(abi.status_has_gpu_va);
  EXPECT_FALSE(abi.sync_has_allocation_handle);
  EXPECT_EQ(abi.command_aperture_code_slot_size, 0x8000u);
  EXPECT_EQ(abi.command_aperture_write_publish_mode,
            CommandApertureWritePublishMode::cpu_cache_flush);
  EXPECT_EQ(abi.command_aperture_code_publish_granularity, 0x8000u);
  EXPECT_TRUE(abi.command_aperture_residency_after_bootstrap);
  EXPECT_FALSE(abi.command_aperture_remap_after_write);
  EXPECT_TRUE(abi.retain_command_aperture_mapping_across_contexts);
  EXPECT_TRUE(abi.retain_command_aperture_session_across_contexts);
  EXPECT_EQ(abi.shared_resource_destroy_flags, 0x3u);
  EXPECT_FALSE(abi.explicit_gpu_va_free_on_destroy);
}

TEST(KmtApiTest, SharedResourceDestroyFlagsMatchNegotiatedAbi) {
  for (const auto& [mcdm_abi, expected_buffer_flags,
                   expected_aperture_flags] :
       {std::tuple{McdmAbi::legacy, 0x1u, 0x2u},
        std::tuple{McdmAbi::compact, 0x3u, 0x3u}}) {
    ResetFakes();
    KmtApi api = {};
    api.destroy_allocation2 = FakeDestroyAllocation2;
    api.free_gpu_virtual_address = FakeFreeGpuVirtualAddress;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;

    Buffer buffer = {};
    buffer.allocation = 0x20;
    buffer.resource = 0x21;
    buffer.gpu_va = 0x10000;
    buffer.mapped_size = 0x1000;
    DestroyBuffer(api, device, &buffer);

    CommandAperture aperture = {};
    aperture.resource = 0x31;
    aperture.gpu_va = 0x20000;
    aperture.gpu_va_size = 0x1000;
    aperture.status_gpu_va = 0x30000;
    DestroyCommandAperture(api, device, &aperture);

    ASSERT_EQ(g_destroy_call_count, 2u);
    EXPECT_EQ(g_destroy_calls[0].resource, 0x21u);
    EXPECT_EQ(g_destroy_calls[0].allocation_count, 0u);
    EXPECT_EQ(g_destroy_calls[0].flags, expected_buffer_flags);
    EXPECT_EQ(g_destroy_calls[1].resource, 0x31u);
    EXPECT_EQ(g_destroy_calls[1].allocation_count, 0u);
    EXPECT_EQ(g_destroy_calls[1].flags, expected_aperture_flags);
    EXPECT_EQ(g_free_gpu_va_count,
              mcdm_abi == McdmAbi::legacy ? 3u : 0u);
  }
}

TEST(KmtApiTest, BufferDestroyWaitsForPendingPaging) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_cpu = FakeWaitFromCpu;
  api.unlock2 = FakeUnlock2;
  api.destroy_allocation2 = FakeDestroyAllocation2;
  Device device = {};
  device.device = 0x10;
  device.paging_sync_object = 0x11;
  uint64_t completed_fence = 6;
  device.paging_fence_cpu = &completed_fence;
  Buffer buffer = {};
  buffer.allocation = 0x20;
  buffer.cpu_ptr = reinterpret_cast<void*>(0x1000);
  buffer.paging_fence_value = 7;

  DestroyBuffer(api, device, &buffer);

  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 7u);
  ASSERT_EQ(g_destroy_call_count, 1u);
  EXPECT_EQ(g_destroy_calls[0].wait_count, 1u);
  EXPECT_EQ(g_destroy_calls[0].flags, 0x1u);
  EXPECT_EQ(buffer.allocation, 0u);
}

TEST(KmtApiTest, BufferDestroySkipsCompletedPagingWait) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_cpu = FakeWaitFromCpu;
  api.destroy_allocation2 = FakeDestroyAllocation2;
  Device device = {};
  device.device = 0x10;
  uint64_t completed_fence = 7;
  device.paging_fence_cpu = &completed_fence;
  Buffer buffer = {};
  buffer.allocation = 0x20;
  buffer.paging_fence_value = 7;

  DestroyBuffer(api, device, &buffer);

  EXPECT_EQ(g_wait_count, 0u);
  ASSERT_EQ(g_destroy_call_count, 1u);
  EXPECT_EQ(g_destroy_calls[0].wait_count, 0u);
  EXPECT_EQ(g_destroy_calls[0].flags, 0x1u);
  EXPECT_EQ(buffer.allocation, 0u);
}

TEST(KmtApiTest, BufferDestroyRetainsAllocationWhenPagingWaitFails) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_cpu = FakeWaitFromCpu;
  api.destroy_allocation2 = FakeDestroyAllocation2;
  Device device = {};
  device.device = 0x10;
  device.paging_sync_object = 0x11;
  uint64_t completed_fence = 6;
  device.paging_fence_cpu = &completed_fence;
  Buffer buffer = {};
  buffer.allocation = 0x20;
  buffer.paging_fence_value = 7;
  g_wait_status = static_cast<NTSTATUS>(0xC0000001u);

  DestroyBuffer(api, device, &buffer);

  EXPECT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_destroy_call_count, 0u);
  EXPECT_EQ(buffer.allocation, 0x20u);
}

TEST(KmtApiTest, CompactContextTeardownMatchesXrtOwnershipOrder) {
  ResetFakes();
  g_record_teardown = true;

  KmtApi api = {};
  api.unlock2 = FakeUnlock2;
  api.destroy_allocation2 = FakeDestroyAllocation2;
  api.destroy_hw_queue = FakeDestroyHwQueue;
  api.destroy_context = FakeDestroyContext;
  Device device = {};
  device.device = 0x10;
  device.mcdm_abi = McdmAbi::compact;

  CommandAperture aperture = {};
  aperture.gpu_allocation = 0x20;
  aperture.gpu_cpu_ptr = reinterpret_cast<void*>(0x1000);
  aperture.gpu_va = 0x04000000;
  aperture.gpu_va_size = 0x04000000;
  aperture.allocation = 0x21;
  aperture.resource = 0x22;
  aperture.cpu_ptr = reinterpret_cast<void*>(0x2000);

  Context context = {};
  context.hw_queue = 0x30;
  context.context = 0x31;
  context.context_private_buffer.allocation = 0x40;
  context.context_private_buffer.resource = 0x41;
  context.context_private_buffer.cpu_ptr = reinterpret_cast<void*>(0x3000);

  DestroyContextWithCommandAperture(api, device, &context, &aperture);

  const std::array expected = {
      TeardownCall::unlock,
      TeardownCall::destroy_allocation,
      TeardownCall::destroy_hw_queue,
      TeardownCall::unlock,
      TeardownCall::destroy_allocation,
      TeardownCall::unlock,
      TeardownCall::destroy_allocation,
      TeardownCall::destroy_context,
  };
  ASSERT_EQ(g_teardown_call_count, expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(g_teardown_calls[i], expected[i]);
  }
  ASSERT_EQ(g_destroy_call_count, 3u);
  EXPECT_EQ(g_destroy_calls[0].allocation, 0x20u);
  EXPECT_EQ(g_destroy_calls[0].flags, 0x3u);
  EXPECT_EQ(g_destroy_calls[1].resource, 0x22u);
  EXPECT_EQ(g_destroy_calls[1].flags, 0x3u);
  EXPECT_EQ(g_destroy_calls[2].resource, 0x41u);
  EXPECT_EQ(g_destroy_calls[2].flags, 0x3u);
}

TEST(KmtApiTest, CodeRangeUsesProtocolVaIndependentOfAllocatorMapping) {
  CommandAperture aperture = {};
  aperture.gpu_allocation = 0x20;
  aperture.gpu_va = 0x08000000;
  aperture.protocol_gpu_va = 0x04000000;
  aperture.gpu_va_size = 0x04000000;
  Error error = {};

  for (McdmAbi abi : {McdmAbi::legacy, McdmAbi::compact}) {
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        abi, 9952, &aperture, &error));
    EXPECT_EQ(aperture.code_offset, 0x8000u);
    EXPECT_EQ(aperture.code_gpu_va, 0x04008000u);
    EXPECT_EQ(aperture.code_size, 0x03ff8000u);

    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        abi, 107600, &aperture, &error));
    EXPECT_EQ(aperture.code_offset, 0x20000u);
    EXPECT_EQ(aperture.code_gpu_va, 0x04020000u);
    EXPECT_EQ(aperture.code_size, 0x03fe0000u);

    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        abi, 0x8000, &aperture, &error));
    EXPECT_EQ(aperture.code_offset, 0x8000u);
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        abi, 0x8001, &aperture, &error));
    EXPECT_EQ(aperture.code_offset, 0x10000u);
  }
}

TEST(KmtApiTest, CodeRangeRejectsPayloadWithoutRemainingCodeSpace) {
  CommandAperture aperture = {};
  aperture.gpu_va_size = 0x8000;
  Error error = {};

  for (McdmAbi abi : {McdmAbi::legacy, McdmAbi::compact}) {
    for (size_t payload_size : {size_t{0x8000}, size_t{0x8001}}) {
      EXPECT_FALSE(ConfigurePathBCodeRangeForSetupPayload(
          abi, payload_size, &aperture, &error));
      EXPECT_NE(std::strstr(ErrorMessage(&error),
                            "leaves no valid code range"),
                nullptr);
    }
  }
}

TEST(KmtApiTest, LegacyChainChildHandleIsMappedPacketPointer) {
  Device device = {};
  device.mcdm_abi = McdmAbi::legacy;
  Buffer buffer = {};
  buffer.cpu_ptr = reinterpret_cast<void*>(0x123456789abcull);
  EXPECT_EQ(GetPathBChainChildHandle(device, &buffer),
            0x123456789abcull);
}

TEST(KmtApiTest, CompactChainChildHandleMatchesXrt221BoPrefix) {
  Device device = {};
  device.mcdm_abi = McdmAbi::compact;
  Buffer buffer = {};
  buffer.kind = BufferKind::execbuf;
  buffer.requested_size = 0x1078;
  buffer.mapped_size = 0x2000;
  buffer.allocation = 0x40003280;
  buffer.gpu_va = 0x03f55000;
  buffer.cpu_ptr = reinterpret_cast<void*>(0x0000026f0bd3c000ull);

  std::array<uint8_t, kCompactPathBChainHandleSize> expected = {};
  auto write_u64 = [&](size_t offset, uint64_t value) {
    std::memcpy(expected.data() + offset, &value, sizeof(value));
  };
  write_u64(0x18, 0x1078);
  write_u64(0x20, 0x80000000);
  write_u64(0x38, 0x40003280);
  write_u64(0x58, 2);
  write_u64(0x60, 0x03f55000);
  write_u64(0x70, 0x0000026f0bd3c000ull);
  write_u64(0x80, 2);
  write_u64(0xc8, 0xffffffffu);
  write_u64(0xd0, 1);

  const uint64_t handle = GetPathBChainChildHandle(device, &buffer);
  EXPECT_EQ(handle,
            reinterpret_cast<uintptr_t>(&buffer.compact_chain_handle));
  EXPECT_EQ(std::memcmp(&buffer.compact_chain_handle, expected.data(),
                        expected.size()),
            0);
}

TEST(KmtApiTest, HardwareTypeNamesMatchMiniport) {
  EXPECT_STREQ(HardwareTypeName(0), "PHX");
  EXPECT_STREQ(HardwareTypeName(1), "STX");
  EXPECT_STREQ(HardwareTypeName(2), "STX2");
  EXPECT_STREQ(HardwareTypeName(3), "STXH");
  EXPECT_STREQ(HardwareTypeName(4), "KRK1");
  EXPECT_STREQ(HardwareTypeName(16), "UNKNOWN");
}

TEST(KmtApiTest, RejectsStxHardwareType) {
  ResetFakes();
  g_query_outputs[0][1] = 1;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "hw_type=STX"), nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, ProbesLegacyV2AbiFromStx2TwoDwordAdapterInfo) {
  ResetFakes();
  g_query_outputs[0][1] = 2;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy_v2);
  ASSERT_EQ(g_query_count, 1u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_adapters[0], 0x5678u);
  EXPECT_EQ(g_query_types[0], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_TRUE(g_query_data_was_zero[0]);
}

TEST(KmtApiTest, RecordsDiagnosticsForLegacyV2IdentityDisambiguation) {
  ResetFakes();
  g_query_outputs[0][1] = 2;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbiDiagnostics diagnostics = {};
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbiDiagnostics(api, 0x5678, &diagnostics, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(diagnostics.selected_abi, McdmAbi::legacy_v2);
  EXPECT_EQ(diagnostics.probed_abi, McdmAbi::legacy_v2);
  EXPECT_EQ(diagnostics.source, McdmAbiSource::identity_query);
  EXPECT_EQ(diagnostics.kmd_version, 0u);
  EXPECT_EQ(diagnostics.hw_type, static_cast<uint32_t>(HardwareType::stx2));
  EXPECT_FALSE(diagnostics.compact_adapter_info);
  EXPECT_EQ(diagnostics.identity_word_count, 2u);
  EXPECT_EQ(diagnostics.identity_words[0], 0u);
  EXPECT_EQ(diagnostics.identity_words[1], 2u);
  EXPECT_EQ(diagnostics.accepted_identity_count, 1u);
  EXPECT_TRUE(diagnostics.identity_accepted);
  EXPECT_TRUE(diagnostics.driver_version_disambiguation_required);
  EXPECT_FALSE(diagnostics.driver_version_disambiguation_available);
  EXPECT_TRUE(diagnostics.legacy_query_attempted);
  EXPECT_FALSE(diagnostics.compact_query_attempted);
  EXPECT_EQ(diagnostics.legacy_query_status, 0);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, ProbesLegacyAbiFromStxhAdapterInfo) {
  ResetFakes();
  g_query_outputs[0][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy);
  ASSERT_EQ(g_query_count, 1u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_adapters[0], 0x5678u);
  EXPECT_EQ(g_query_types[0], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_TRUE(g_query_data_was_zero[0]);
}

TEST(KmtApiTest, ProbesCompactAbiWhenLegacyShapeIsRejected) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 2;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::compact);
  ASSERT_EQ(g_query_count, 2u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_sizes[1], 3u * sizeof(uint32_t));
}

TEST(KmtApiTest, ProbesCompactAbiFromStxhAdapterInfo) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::compact);
  ASSERT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, ProbesCompactAbiFromKrk1AdapterInfo) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::compact);
  ASSERT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, RecordsDiagnosticsForStxhTwoDwordAdapterInfo) {
  // Halo 329 looks like this: same STXH hw_type, two-DWORD query.
  ResetFakes();
  g_query_outputs[0][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbiDiagnostics diagnostics = {};
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbiDiagnostics(api, 0x5678, &diagnostics, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(diagnostics.selected_abi, McdmAbi::legacy);
  EXPECT_EQ(diagnostics.probed_abi, McdmAbi::legacy);
  EXPECT_EQ(diagnostics.source, McdmAbiSource::identity_query);
  EXPECT_EQ(diagnostics.kmd_version, 0u);
  EXPECT_EQ(diagnostics.hw_type, static_cast<uint32_t>(HardwareType::stxh));
  EXPECT_FALSE(diagnostics.compact_adapter_info);
  EXPECT_EQ(diagnostics.identity_word_count, 2u);
  EXPECT_EQ(diagnostics.identity_words[0], 0u);
  EXPECT_EQ(diagnostics.identity_words[1], 3u);
  EXPECT_EQ(diagnostics.accepted_identity_count, 1u);
  EXPECT_TRUE(diagnostics.identity_accepted);
  EXPECT_TRUE(diagnostics.legacy_query_attempted);
  EXPECT_FALSE(diagnostics.compact_query_attempted);
  EXPECT_EQ(diagnostics.legacy_query_status, 0);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RecordsDiagnosticsForKrk1TwoDwordAdapterInfo) {
  ResetFakes();
  g_query_outputs[0][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbiDiagnostics diagnostics = {};
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbiDiagnostics(api, 0x5678, &diagnostics, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(diagnostics.selected_abi, McdmAbi::legacy);
  EXPECT_EQ(diagnostics.probed_abi, McdmAbi::legacy);
  EXPECT_EQ(diagnostics.source, McdmAbiSource::identity_query);
  EXPECT_EQ(diagnostics.kmd_version, 0u);
  EXPECT_EQ(diagnostics.hw_type, static_cast<uint32_t>(HardwareType::krk1));
  EXPECT_FALSE(diagnostics.compact_adapter_info);
  EXPECT_EQ(diagnostics.identity_word_count, 2u);
  EXPECT_EQ(diagnostics.identity_words[0], 0u);
  EXPECT_EQ(diagnostics.identity_words[1], 4u);
  EXPECT_EQ(diagnostics.accepted_identity_count, 1u);
  EXPECT_TRUE(diagnostics.identity_accepted);
  EXPECT_TRUE(diagnostics.legacy_query_attempted);
  EXPECT_FALSE(diagnostics.compact_query_attempted);
  EXPECT_EQ(diagnostics.legacy_query_status, 0);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RecordsDiagnosticsForStxhCompactAdapterInfo) {
  // Halo 3760 and 3930 both look like this: same STXH hw_type, compact query.
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbiDiagnostics diagnostics = {};
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbiDiagnostics(api, 0x5678, &diagnostics, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(diagnostics.selected_abi, McdmAbi::compact);
  EXPECT_EQ(diagnostics.probed_abi, McdmAbi::compact);
  EXPECT_EQ(diagnostics.source, McdmAbiSource::identity_query);
  EXPECT_EQ(diagnostics.kmd_version, 0u);
  EXPECT_EQ(diagnostics.hw_type, static_cast<uint32_t>(HardwareType::stxh));
  EXPECT_TRUE(diagnostics.compact_adapter_info);
  EXPECT_EQ(diagnostics.identity_word_count, 3u);
  EXPECT_EQ(diagnostics.identity_words[0], 0u);
  EXPECT_EQ(diagnostics.identity_words[1], 3u);
  EXPECT_EQ(diagnostics.identity_words[2], 0u);
  EXPECT_EQ(diagnostics.accepted_identity_count, 1u);
  EXPECT_TRUE(diagnostics.identity_accepted);
  EXPECT_TRUE(diagnostics.legacy_query_attempted);
  EXPECT_TRUE(diagnostics.compact_query_attempted);
  EXPECT_EQ(diagnostics.legacy_query_status,
            static_cast<NTSTATUS>(0xC0000023u));
  EXPECT_EQ(diagnostics.compact_query_status, 0);
  EXPECT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, RecordsDiagnosticsForKrk1CompactAdapterInfo) {
  // Compact KRK1 (PCI REV_20). Not what a Halo 3930 package reports.
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbiDiagnostics diagnostics = {};
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbiDiagnostics(api, 0x5678, &diagnostics, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(diagnostics.selected_abi, McdmAbi::compact);
  EXPECT_EQ(diagnostics.probed_abi, McdmAbi::compact);
  EXPECT_EQ(diagnostics.source, McdmAbiSource::identity_query);
  EXPECT_EQ(diagnostics.kmd_version, 0u);
  EXPECT_EQ(diagnostics.hw_type, static_cast<uint32_t>(HardwareType::krk1));
  EXPECT_TRUE(diagnostics.compact_adapter_info);
  EXPECT_EQ(diagnostics.identity_word_count, 3u);
  EXPECT_EQ(diagnostics.identity_words[0], 0u);
  EXPECT_EQ(diagnostics.identity_words[1], 4u);
  EXPECT_EQ(diagnostics.identity_words[2], 0u);
  EXPECT_EQ(diagnostics.accepted_identity_count, 1u);
  EXPECT_TRUE(diagnostics.identity_accepted);
  EXPECT_TRUE(diagnostics.legacy_query_attempted);
  EXPECT_TRUE(diagnostics.compact_query_attempted);
  EXPECT_EQ(diagnostics.legacy_query_status,
            static_cast<NTSTATUS>(0xC0000023u));
  EXPECT_EQ(diagnostics.compact_query_status, 0);
  EXPECT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, ProbesLegacyAbiFromKrk1TwoDwordAdapterInfo) {
  ResetFakes();
  g_query_outputs[0][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, ProbesLegacyV0AbiFromPhxAdapterInfo) {
  ResetFakes();
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  ASSERT_TRUE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy_v0);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, SelectsLegacyV2LayoutForPre314DriverVersions) {
  for (uint32_t revision : {240u, 280u, 313u}) {
    McdmAbi selected = McdmAbi::compact;
    Error error;
    ASSERT_TRUE(SelectMcdmAbiForDriverVersion(
        McdmAbi::legacy_v2, true, DriverVersion{32, 0, 203, revision},
        &selected, &error));
    EXPECT_EQ(selected, McdmAbi::legacy_v2);
  }
}

TEST(KmtApiTest, SubmissionPolicyFollowsNegotiatedAbiContract) {
  for (McdmAbi abi : {McdmAbi::legacy_v0, McdmAbi::legacy_v2}) {
    const McdmSubmissionPolicy policy = GetMcdmSubmissionPolicy(abi);
    EXPECT_FALSE(policy.supports_command_chaining);
    EXPECT_TRUE(policy.uses_shared_command_code_view);
    EXPECT_FALSE(policy.submit_completion_is_deferred);
    EXPECT_FALSE(SupportsHostBufferReuse(abi));
  }
  for (McdmAbi abi : {McdmAbi::legacy, McdmAbi::compact}) {
    const McdmSubmissionPolicy policy = GetMcdmSubmissionPolicy(abi);
    EXPECT_TRUE(policy.supports_command_chaining);
    EXPECT_FALSE(policy.uses_shared_command_code_view);
    EXPECT_TRUE(policy.submit_completion_is_deferred);
    EXPECT_TRUE(SupportsHostBufferReuse(abi));
  }
}

TEST(KmtApiTest, SelectsLegacyLayoutForPost280Stx2DriverVersions) {
  // INF revision is consulted only for STX2. Halo 32.0.203.329 reports STXH
  // and never enters this path.
  for (uint32_t revision : {314u, 329u}) {
    McdmAbi selected = McdmAbi::compact;
    Error error;
    ASSERT_TRUE(SelectMcdmAbiForDriverVersion(
        McdmAbi::legacy_v2, true, DriverVersion{32, 0, 203, revision},
        &selected, &error));
    EXPECT_EQ(selected, McdmAbi::legacy);
  }
}

TEST(KmtApiTest, RejectsAmbiguousLegacyV2WhenDriverVersionIsUnavailable) {
  McdmAbi selected = McdmAbi::compact;
  Error error;
  EXPECT_FALSE(SelectMcdmAbiForDriverVersion(
      McdmAbi::legacy_v2, false, DriverVersion{}, &selected, &error));
  EXPECT_NE(std::string(ErrorMessage(&error)).find("ambiguous"),
            std::string::npos);
}

TEST(KmtApiTest, RejectsAmbiguousLegacyV2ForUnknownDriverVersionFamily) {
  McdmAbi selected = McdmAbi::compact;
  Error error;
  EXPECT_FALSE(SelectMcdmAbiForDriverVersion(
      McdmAbi::legacy_v2, true, DriverVersion{32, 0, 204, 1}, &selected,
      &error));
  EXPECT_NE(std::string(ErrorMessage(&error)).find("unsupported"),
            std::string::npos);
}

TEST(KmtApiTest, DriverVersionSelectionLeavesOtherAbisUnchanged) {
  const DriverVersion legacy_v2_version{32, 0, 203, 280};
  const DriverVersion unknown_version{};
  for (McdmAbi abi :
       {McdmAbi::legacy_v0, McdmAbi::legacy, McdmAbi::compact}) {
    McdmAbi selected = McdmAbi::legacy_v2;
    Error error;
    ASSERT_TRUE(SelectMcdmAbiForDriverVersion(
        abi, true, legacy_v2_version, &selected, &error));
    EXPECT_EQ(selected, abi);
    ASSERT_TRUE(SelectMcdmAbiForDriverVersion(
        abi, false, unknown_version, &selected, &error));
    EXPECT_EQ(selected, abi);
  }
}

TEST(KmtApiTest, RejectsCompactZeroIdentityAfterLegacyShapeIsRejected) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported UMDRIVERPRIVATE compact adapter info"),
            nullptr);
  EXPECT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, RejectsUnknownTwoDwordAbiIdentity) {
  ResetFakes();
  g_query_outputs[0][0] = 1;
  g_query_outputs[0][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported UMDRIVERPRIVATE adapter info"),
            nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RejectsUnknownTwoDwordAbiMajor) {
  ResetFakes();
  g_query_outputs[0][0] = 1;
  g_query_outputs[0][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported UMDRIVERPRIVATE adapter info"),
            nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RejectsUnknownThreeDwordAbiIdentity) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_outputs[1][1] = 2;
  g_query_outputs[1][2] = 1;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported UMDRIVERPRIVATE compact adapter info"),
            nullptr);
  EXPECT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, RejectsBothAbiQueryShapesRejected) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_statuses[1] = static_cast<NTSTATUS>(0xC0000023u);
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x5678, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "rejected both three-dword and two-dword "
                        "UMDRIVERPRIVATE adapter-info"),
            nullptr);
  EXPECT_EQ(g_query_count, 2u);
}

TEST(KmtApiTest, RejectsUnknownAbiQueryFailure) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000001u);
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryProbedMcdmAbi(api, 0x9ABC, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "UMDRIVERPRIVATE legacy"),
            nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, LegacyPrivatePacketsMatchSnapshot) {
  ExpectPrivatePacketsMatchSnapshot(McdmAbi::legacy);
}

TEST(KmtApiTest, CompactPrivatePacketsMatchSnapshot) {
  ExpectPrivatePacketsMatchSnapshot(McdmAbi::compact);
}

TEST(KmtApiTest, CompactPathBSetupMakesApertureResidentAfterBootstrap) {
  ResetFakes();
  KmtApi api = {};
  api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
  api.make_resident = FakeMakeResident;
  api.lock2 = FakeLock2;
  api.invalidate_cache = FakeInvalidateCache;
  api.wait_from_cpu = FakeWaitFromCpu;

  Device device = {};
  device.device = 0x10;
  device.paging_queue = 0x11;
  device.mcdm_abi = McdmAbi::compact;
  Context context = {};
  context.hw_queue = 0x20;
  context.progress_fence = 0x30;
  context.next_fence_id = 7;
  CommandAperture aperture = {};
  aperture.allocation = 0x40;
  aperture.gpu_allocation = 0x50;
  aperture.gpu_va = 0x100000;
  aperture.protocol_gpu_va = aperture.gpu_va;
  aperture.gpu_va_size = g_locked_aperture.size();
  uint32_t setup_completion = 0xffffffffu;
  aperture.cpu_ptr = &setup_completion;
  g_complete_on_wait = &setup_completion;
  std::array<uint8_t, 107600> setup_payload;
  setup_payload.fill(0x5a);
  Error error = {};

  ASSERT_TRUE(SubmitAndWaitPathBSetup(api, device, &context, &aperture,
                                      setup_payload.data(),
                                      setup_payload.size(), &error))
      << ErrorMessage(&error);
  const SetupCall expected_calls[] = {
      SetupCall::submit, SetupCall::make_resident, SetupCall::lock,
      SetupCall::invalidate, SetupCall::wait, SetupCall::submit,
      SetupCall::wait};
  ASSERT_EQ(g_setup_call_count, std::size(expected_calls));
  for (size_t i = 0; i < std::size(expected_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], expected_calls[i]);
  }
  ASSERT_EQ(g_submit_count, 2u);
  EXPECT_EQ(g_submit_opcodes[0], 2u);
  EXPECT_EQ(g_submit_opcodes[1], 5u);
  EXPECT_EQ(g_submit_fences[0], 7u);
  EXPECT_EQ(g_submit_fences[1], 8u);
  ASSERT_EQ(g_wait_count, 2u);
  EXPECT_EQ(g_wait_fences[0], 11u);
  EXPECT_EQ(g_wait_fences[1], 8u);
  EXPECT_EQ(context.next_fence_id, 9u);
  EXPECT_EQ(aperture.gpu_cpu_ptr, g_locked_aperture.data());
  EXPECT_EQ(aperture.code_offset, 0x20000u);
  EXPECT_EQ(aperture.code_gpu_va, 0x120000u);
  EXPECT_EQ(aperture.code_cpu_ptr, g_locked_aperture.data() + 0x20000);
  EXPECT_EQ(aperture.code_size, 0x20000u);
  EXPECT_EQ(std::memcmp(g_locked_aperture.data(), setup_payload.data(),
                        setup_payload.size()),
            0);
}

TEST(KmtApiTest, LegacyPathBSetupPublishesPayloadAfterFinalWrite) {
  ResetFakes();
  KmtApi api = {};
  api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
  api.lock2 = FakeLock2;
  api.invalidate_cache = FakeInvalidateCache;
  api.wait_from_cpu = FakeWaitFromCpu;

  Device device = {};
  device.device = 0x10;
  device.mcdm_abi = McdmAbi::legacy;
  Context context = {};
  context.hw_queue = 0x20;
  context.progress_fence = 0x30;
  context.next_fence_id = 7;
  CommandAperture aperture = {};
  aperture.allocation = 0x40;
  aperture.gpu_allocation = 0x50;
  aperture.gpu_va = 0x100000;
  aperture.gpu_va_size = g_locked_aperture.size();
  uint32_t setup_completion = 0xffffffffu;
  aperture.cpu_ptr = &setup_completion;
  g_complete_on_wait = &setup_completion;
  std::array<uint8_t, 9952> setup_payload;
  setup_payload.fill(0x5a);
  Error error = {};

  ASSERT_TRUE(SubmitAndWaitPathBSetup(api, device, &context, &aperture,
                                      setup_payload.data(),
                                      setup_payload.size(), &error))
      << ErrorMessage(&error);
  const SetupCall expected_calls[] = {
      SetupCall::submit, SetupCall::lock, SetupCall::invalidate,
      SetupCall::invalidate, SetupCall::submit, SetupCall::wait};
  ASSERT_EQ(g_setup_call_count, std::size(expected_calls));
  for (size_t i = 0; i < std::size(expected_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], expected_calls[i]);
  }
  ASSERT_EQ(g_invalidate_count, 2u);
  EXPECT_EQ(g_invalidate_first_bytes[0], 0u);
  EXPECT_EQ(g_invalidate_first_bytes[1], 0x5au);
  EXPECT_EQ(std::memcmp(g_locked_aperture.data(), setup_payload.data(),
                        setup_payload.size()),
            0);
}

TEST(KmtApiTest,
     CreateHostBufferUsesOnlyDevicePagingStateAndPreservesResidencyFence) {
  const McdmAbi abis[] = {McdmAbi::legacy, McdmAbi::compact};
  for (McdmAbi mcdm_abi : abis) {
    ResetFakes();
    KmtApi api = {};
    api.create_allocation2 = FakeCreateAllocation;
    api.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
    api.make_resident = FakeMakeResident;
    api.lock2 = FakeLock2;
    Device device = {};
    device.device = 0x10;
    device.paging_queue = 0x11;
    device.mcdm_abi = mcdm_abi;
    Buffer buffer = {};
    Error error = {};

    ASSERT_TRUE(CreateBuffer(api, device, BufferKind::host_only, 4096, &buffer,
                             &error))
        << ErrorMessage(&error);
    EXPECT_EQ(buffer.allocation, 0x40u);
    EXPECT_EQ(buffer.gpu_va, 0x10000u);
    EXPECT_EQ(buffer.paging_fence_value, 11u);
    EXPECT_TRUE(std::all_of(g_locked_aperture.begin(),
                            g_locked_aperture.end(),
                            [](uint8_t value) { return value == 0; }));
  }
}

TEST(KmtApiTest, ExecBufferAllocationPreservesLogicalCommandSize) {
  for (McdmAbi mcdm_abi : {McdmAbi::legacy, McdmAbi::compact}) {
    for (uint64_t logical_size : {uint64_t{224}, uint64_t{4096}}) {
      ResetFakes();
      KmtApi api = {};
      api.create_allocation2 = FakeCreateAllocation;
      api.map_gpu_virtual_address = FakeMapGpuVirtualAddress;
      api.make_resident = FakeMakeResident;
      api.lock2 = FakeLock2;
      Device device = {};
      device.device = 0x10;
      device.paging_queue = 0x11;
      device.mcdm_abi = mcdm_abi;
      Buffer buffer = {};
      Error error = {};

      ASSERT_TRUE(CreateBuffer(api, device, BufferKind::execbuf, logical_size,
                               &buffer, &error))
          << ErrorMessage(&error);
      const uint64_t requested_size =
          logical_size + GetMcdmAbiInfo(mcdm_abi).submit_private_prefix_size;
      EXPECT_EQ(buffer.size, logical_size);
      EXPECT_EQ(buffer.requested_size, requested_size);
      EXPECT_EQ(buffer.mapped_size, (requested_size + 4095) & ~uint64_t{4095});
    }
  }
}

TEST(KmtApiTest, CompletedPathBSubmitStillRetiresParentFence) {
  auto run = [](uint64_t completed_fence) {
    ResetFakes();
    KmtApi api = {};
    api.wait_from_cpu = FakeWaitFromCpu;
    api.invalidate_cache = FakeInvalidateCache;
    Device device = {};
    device.device = 0x10;
    uint64_t progress_fence = completed_fence;
    Context context = {};
    context.hw_queue = 0x20;
    context.progress_fence = 0x21;
    context.progress_fence_cpu = &progress_fence;
    uint32_t slot_state = 4;
    uint32_t packet_header = 0;
    PathBPendingSubmit pending = {};
    pending.fence_id = 7;
    pending.slot_cpu = reinterpret_cast<uint8_t*>(&slot_state);
    pending.packet_header = &packet_header;
    pending.ring.allocation = 0x30;
    pending.ring.size = sizeof(slot_state);
    Error error = {};

    EXPECT_TRUE(
        WaitForPathBSubmits(api, device, &context, &pending, 1, &error))
        << ErrorMessage(&error);
    EXPECT_EQ(packet_header & 0xFu, 4u);
    return g_wait_count;
  };

  EXPECT_EQ(run(/*completed_fence=*/7), 1u);
  EXPECT_EQ(run(/*completed_fence=*/6), 1u);
}

TEST(KmtApiTest, PathBBatchRetirementMatchesXrtRunlistWaitOrder) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_cpu = FakeWaitFromCpu;
  api.invalidate_cache = FakeInvalidateCache;
  Device device = {};
  device.device = 0x10;
  uint64_t progress_fence = 10;
  Context context = {};
  context.hw_queue = 0x20;
  context.progress_fence = 0x21;
  context.progress_fence_cpu = &progress_fence;
  std::array<uint32_t, 6> ring_storage = {0, 0, 4, 0, 4, 0};
  std::array<uint32_t, 2> packet_headers = {};
  std::array<PathBPendingSubmit, 2> pending = {};
  for (size_t i = 0; i < pending.size(); ++i) {
    pending[i].fence_id = 11 + i;
    pending[i].slot_cpu =
        reinterpret_cast<uint8_t*>(&ring_storage[(i + 1) * 2]);
    pending[i].slot_offset = static_cast<uint32_t>((i + 1) * 8);
    pending[i].packet_header = &packet_headers[i];
    pending[i].ring.allocation = 0x30;
    pending[i].ring.cpu_ptr = ring_storage.data();
    pending[i].ring.size = sizeof(ring_storage);
  }
  Error error = {};

  ASSERT_TRUE(WaitForPathBSubmits(api, device, &context, pending.data(),
                                  pending.size(), &error))
      << ErrorMessage(&error);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 12u);
  EXPECT_EQ(packet_headers[0] & 0xFu, 4u);
  EXPECT_EQ(packet_headers[1] & 0xFu, 4u);
}

TEST(KmtApiTest, PathBWaitFailureDoesNotPublishCompletion) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_cpu = FakeWaitFromCpu;
  api.invalidate_cache = FakeInvalidateCache;
  Device device = {};
  device.device = 0x10;
  uint64_t progress_fence = 6;
  Context context = {};
  context.hw_queue = 0x20;
  context.progress_fence = 0x21;
  context.progress_fence_cpu = &progress_fence;
  uint32_t slot_state = 4;
  uint32_t packet_header = 0;
  PathBPendingSubmit pending = {};
  pending.fence_id = 7;
  pending.slot_cpu = reinterpret_cast<uint8_t*>(&slot_state);
  pending.packet_header = &packet_header;
  pending.ring.allocation = 0x30;
  pending.ring.size = sizeof(slot_state);
  g_wait_status = static_cast<NTSTATUS>(0xC0000001u);
  Error error = {};

  EXPECT_FALSE(
      WaitForPathBSubmits(api, device, &context, &pending, 1, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "pathb batch"), nullptr);
  EXPECT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_invalidate_count, 0u);
  EXPECT_EQ(packet_header, 0u);
}

TEST(KmtApiTest, FailedPathBSubmitDoesNotCreatePendingCommand) {
  ResetFakes();
  KmtApi api = {};
  api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
  Device device = {};
  device.device = 0x10;
  device.mcdm_abi = McdmAbi::legacy;
  alignas(64) std::array<uint8_t, 4096> ring_storage = {};
  Context context = {};
  context.hw_queue = 0x20;
  context.next_fence_id = 7;
  context.completion_ring_ready = true;
  context.completion_ring.allocation = 0x30;
  context.completion_ring.gpu_va = 0x100000;
  context.completion_ring.cpu_ptr = ring_storage.data();
  context.completion_ring.size = ring_storage.size();
  Buffer exec_buffer = {};
  exec_buffer.allocation = 0x40;
  exec_buffer.gpu_va = 0x200000;
  exec_buffer.size = 224;
  std::array<uint8_t, 16> ert_packet = {};
  uint32_t first_header = 0;
  uint32_t second_header = 0;
  PathBPendingSubmit first = {};
  PathBPendingSubmit second = {};
  Error error = {};

  ASSERT_TRUE(SubmitPathB(api, device, &context, exec_buffer,
                          ert_packet.data(), ert_packet.size(), 3,
                          /*completion_slot_offset=*/8, &first_header, &first,
                          &error))
      << ErrorMessage(&error);
  g_submit_failure_index = 1;
  EXPECT_FALSE(SubmitPathB(api, device, &context, exec_buffer,
                           ert_packet.data(), ert_packet.size(), 3,
                           /*completion_slot_offset=*/16, &second_header,
                           &second, &error));

  EXPECT_EQ(first.fence_id, 7u);
  EXPECT_NE(first.slot_cpu, nullptr);
  EXPECT_EQ(second.fence_id, 0u);
  EXPECT_EQ(second.slot_cpu, nullptr);
  EXPECT_EQ(g_submit_count, 2u);
  EXPECT_EQ(g_submit_fences[0], 7u);
  EXPECT_EQ(g_submit_fences[1], 8u);
}

TEST(KmtApiTest, PublishBufferCpuWritesValidatesRangeAndPreservesData) {
  alignas(64) std::array<uint8_t, 256> storage = {};
  std::fill(storage.begin(), storage.end(), 0x5a);
  const auto expected = storage;

  Buffer buffer = {};
  buffer.cpu_ptr = storage.data();
  buffer.size = storage.size();
  Error error = {};

  ASSERT_TRUE(PublishBufferCpuWrites(buffer, 3, 65, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(storage, expected);
  ASSERT_TRUE(InvalidateBufferCpuReads(buffer, 3, 65, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(storage, expected);
  EXPECT_TRUE(PublishBufferCpuWrites(buffer, buffer.size, 0, &error));
  EXPECT_TRUE(InvalidateBufferCpuReads(buffer, buffer.size, 0, &error));

  EXPECT_FALSE(PublishBufferCpuWrites(buffer, buffer.size, 1, &error));
  EXPECT_FALSE(PublishBufferCpuWrites(buffer, 200, 64, &error));
  EXPECT_FALSE(InvalidateBufferCpuReads(buffer, buffer.size, 1, &error));
  EXPECT_FALSE(InvalidateBufferCpuReads(buffer, 200, 64, &error));

  buffer.cpu_ptr = nullptr;
  EXPECT_FALSE(PublishBufferCpuWrites(buffer, 0, 1, &error));
  EXPECT_TRUE(PublishBufferCpuWrites(buffer, 0, 0, &error));
  EXPECT_FALSE(InvalidateBufferCpuReads(buffer, 0, 1, &error));
  EXPECT_TRUE(InvalidateBufferCpuReads(buffer, 0, 0, &error));
}

TEST(KmtApiTest, PathBCompletionCapacityFollowsAllocatedRing) {
  Context context = {};
  EXPECT_EQ(PathBCompletionCapacity(context), 0u);

  context.completion_ring.size = 8;
  EXPECT_EQ(PathBCompletionCapacity(context), 0u);

  context.completion_ring.size = 16;
  EXPECT_EQ(PathBCompletionCapacity(context), 1u);

  context.completion_ring.size = 4096;
  EXPECT_EQ(PathBCompletionCapacity(context), 511u);
}

TEST(KmtApiTest, ValidatesCallerOwnedCompletionSlotOffsets) {
  EXPECT_FALSE(IsValidPathBCompletionSlot(/*ring_size=*/4096, 0));
  EXPECT_FALSE(IsValidPathBCompletionSlot(/*ring_size=*/4096, 4));
  EXPECT_TRUE(IsValidPathBCompletionSlot(/*ring_size=*/4096, 8));
  EXPECT_TRUE(IsValidPathBCompletionSlot(/*ring_size=*/4096, 4088));
  EXPECT_FALSE(IsValidPathBCompletionSlot(/*ring_size=*/4096, 4096));
  EXPECT_FALSE(IsValidPathBCompletionSlot(/*ring_size=*/4, 8));
}

TEST(KmtApiTest, InitializesReservedCompletionSlotsAsOneBatch) {
  alignas(64) std::array<uint8_t, 4096> storage = {};
  std::fill(storage.begin(), storage.end(), 0x5a);
  Context context = {};
  context.completion_ring.cpu_ptr = storage.data();
  context.completion_ring.size = storage.size();
  context.completion_ring_ready = true;
  const uint32_t offsets[] = {4088, 8, 16};
  Error error = {};

  ASSERT_TRUE(InitializePathBCompletionSlots(
      &context, offsets, std::size(offsets), &error))
      << ErrorMessage(&error);
  for (uint32_t offset : offsets) {
    EXPECT_TRUE(std::all_of(storage.begin() + offset,
                            storage.begin() + offset + 8,
                            [](uint8_t value) { return value == 0; }));
  }
  EXPECT_EQ(storage[0], 0x5a);
  EXPECT_EQ(storage[24], 0x5a);
  EXPECT_EQ(storage[4087], 0x5a);
}

TEST(KmtApiTest, RejectsInvalidCompletionSlotSetsBeforeModification) {
  alignas(64) std::array<uint8_t, 128> storage = {};
  std::fill(storage.begin(), storage.end(), 0x5a);
  Context context = {};
  context.completion_ring.cpu_ptr = storage.data();
  context.completion_ring.size = storage.size();
  context.completion_ring_ready = true;
  Error error = {};

  const uint32_t duplicate_offsets[] = {8, 8};
  EXPECT_FALSE(InitializePathBCompletionSlots(
      &context, duplicate_offsets, std::size(duplicate_offsets), &error));
  EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                          [](uint8_t value) { return value == 0x5a; }));

  const uint32_t invalid_offsets[] = {8, 128};
  EXPECT_FALSE(InitializePathBCompletionSlots(
      &context, invalid_offsets, std::size(invalid_offsets), &error));
  EXPECT_TRUE(std::all_of(storage.begin(), storage.end(),
                          [](uint8_t value) { return value == 0x5a; }));
}

TEST(KmtApiTest, CopyAndCommitPathBCodeWritesCopiesAlignedRangesAndTails) {
  alignas(64) std::array<uint8_t, 512> storage = {};
  std::fill(storage.begin(), storage.end(), 0xcc);
  std::array<uint8_t, 193> source = {};
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = static_cast<uint8_t>((i * 17) & 0xff);
  }

  CommandAperture aperture = {};
  aperture.gpu_cpu_ptr = storage.data();
  aperture.gpu_va_size = storage.size();
  const CpuCopyRange range = {64, source.data(), source.size()};
  Error error = {};

  ASSERT_TRUE(CopyAndCommitPathBCodeWrites(aperture, &range, 1, &error))
      << ErrorMessage(&error);
  EXPECT_TRUE(std::equal(source.begin(), source.end(), storage.begin() + 64));
  EXPECT_TRUE(std::all_of(storage.begin(), storage.begin() + 64,
                          [](uint8_t value) { return value == 0xcc; }));
  EXPECT_EQ(storage[64 + source.size()], 0xcc);
}

TEST(KmtApiTest, CopyAndCommitPathBCodeWritesHandlesMultipleAndUnalignedRanges) {
  alignas(64) std::array<uint8_t, 512> storage = {};
  std::fill(storage.begin(), storage.end(), 0xcc);
  std::array<uint8_t, 67> first = {};
  std::array<uint8_t, 96> second = {};
  std::fill(first.begin(), first.end(), 0x35);
  std::fill(second.begin(), second.end(), 0xa7);

  CommandAperture aperture = {};
  aperture.gpu_cpu_ptr = storage.data();
  aperture.gpu_va_size = storage.size();
  const CpuCopyRange ranges[] = {
      {3, first.data(), first.size()},
      {256, second.data(), second.size()},
  };
  Error error = {};

  ASSERT_TRUE(CopyAndCommitPathBCodeWrites(aperture, ranges, 2, &error))
      << ErrorMessage(&error);
  EXPECT_TRUE(std::equal(first.begin(), first.end(), storage.begin() + 3));
  EXPECT_TRUE(std::equal(second.begin(), second.end(), storage.begin() + 256));
  EXPECT_EQ(storage[2], 0xcc);
  EXPECT_EQ(storage[70], 0xcc);
  EXPECT_EQ(storage[255], 0xcc);
  EXPECT_EQ(storage[352], 0xcc);
}

TEST(KmtApiTest, CopyAndCommitPathBCodeWritesRejectsInvalidRanges) {
  alignas(64) std::array<uint8_t, 128> storage = {};
  std::array<uint8_t, 16> source = {};
  CommandAperture aperture = {};
  aperture.gpu_cpu_ptr = storage.data();
  aperture.gpu_va_size = storage.size();
  Error error = {};

  EXPECT_TRUE(CopyAndCommitPathBCodeWrites(aperture, nullptr, 0, &error));
  EXPECT_FALSE(CopyAndCommitPathBCodeWrites(aperture, nullptr, 1, &error));

  CpuCopyRange range = {0, nullptr, 1};
  EXPECT_FALSE(CopyAndCommitPathBCodeWrites(aperture, &range, 1, &error));
  range = {120, source.data(), source.size()};
  EXPECT_FALSE(CopyAndCommitPathBCodeWrites(aperture, &range, 1, &error));
  range = {aperture.gpu_va_size, nullptr, 0};
  EXPECT_TRUE(CopyAndCommitPathBCodeWrites(aperture, &range, 1, &error));

  aperture.gpu_cpu_ptr = nullptr;
  range = {0, source.data(), source.size()};
  EXPECT_FALSE(CopyAndCommitPathBCodeWrites(aperture, &range, 1, &error));
}

TEST(KmtApiTest, BufferResidencyUsesNegotiatedPagingModel) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_gpu = FakeWaitFromGpu;
  Device device = {};
  device.paging_sync_object = 0x20;
  device.mcdm_abi = McdmAbi::legacy;
  Context context = {};
  context.context = 0x30;
  Buffer buffer = {};
  buffer.paging_fence_value = 0x1234;
  Error error = {};

  ASSERT_TRUE(WaitForBufferResidency(api, device, context, buffer, "test",
                                     &error))
      << ErrorMessage(&error);
  EXPECT_EQ(g_gpu_wait_count, 1u);
  EXPECT_EQ(g_gpu_wait_context, context.context);
  EXPECT_EQ(g_gpu_wait_object, device.paging_sync_object);
  EXPECT_EQ(g_gpu_wait_fence, buffer.paging_fence_value);

  device.mcdm_abi = McdmAbi::compact;
  ASSERT_TRUE(WaitForBufferResidency(api, device, context, buffer, "test",
                                     &error))
      << ErrorMessage(&error);
  EXPECT_EQ(g_gpu_wait_count, 1u);

  buffer.paging_fence_value = 0;
  ASSERT_TRUE(WaitForBufferResidency(api, device, context, buffer, "test",
                                     &error))
      << ErrorMessage(&error);
  EXPECT_EQ(g_gpu_wait_count, 1u);
}

TEST(KmtApiTest, CodeRangeLifecycleMatchesNegotiatedAbi) {
  auto run = [](McdmAbi mcdm_abi, size_t setup_payload_size) {
    ResetFakes();
    KmtApi api = {};
    api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
    api.invalidate_cache = FakeInvalidateCache;
    api.wait_from_cpu = FakeWaitFromCpu;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;
    Context context = {};
    context.hw_queue = 0x20;
    context.progress_fence = 0x21;
    context.next_fence_id = 7;
    CommandAperture aperture = {};
    aperture.gpu_allocation = 0x30;
    alignas(64) static std::array<uint8_t, 0x100000> aperture_storage = {};
    aperture.gpu_cpu_ptr = aperture_storage.data();
    aperture.gpu_va_size = aperture_storage.size();
    Error error = {};
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        mcdm_abi, setup_payload_size, &aperture, &error));

    ASSERT_TRUE(AcquirePathBCodeRange(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
    EXPECT_EQ(g_submit_count, 0u);
    ASSERT_TRUE(CommitPathBCodeWrite(api, device, aperture,
                                     aperture.code_offset, 0x10000, &error));
    EXPECT_EQ(g_submit_count, 0u);
    ASSERT_TRUE(PublishPathBCodeWrite(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
    EXPECT_EQ(g_submit_count, 2u);
    ASSERT_TRUE(ReleasePathBCodeRange(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
    ASSERT_TRUE(AcquirePathBCodeRange(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
    ASSERT_TRUE(PublishPathBCodeWrite(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
  };

  run(McdmAbi::compact, 9952);
  const SetupCall compact_calls[] = {
      SetupCall::submit, SetupCall::submit, SetupCall::submit,
      SetupCall::submit, SetupCall::wait, SetupCall::submit,
      SetupCall::submit};
  ASSERT_EQ(g_setup_call_count, std::size(compact_calls));
  size_t compact_submit_index = 0;
  for (size_t i = 0; i < std::size(compact_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], compact_calls[i]);
    if (compact_calls[i] == SetupCall::submit) {
      EXPECT_EQ(g_submit_opcodes[compact_submit_index++], 9u);
    }
  }
  ASSERT_EQ(g_submit_count, 6u);
  EXPECT_EQ(g_submit_offsets[0], 0x10000u);
  EXPECT_EQ(g_submit_offsets[1], 0x18000u);
  EXPECT_EQ(g_submit_offsets[2], 0x10000u);
  EXPECT_EQ(g_submit_offsets[3], 0x8000u);
  EXPECT_EQ(g_submit_offsets[4], 0x10000u);
  EXPECT_EQ(g_submit_offsets[5], 0x18000u);
  EXPECT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 10u);

  run(McdmAbi::compact, 107600);
  ASSERT_EQ(g_setup_call_count, std::size(compact_calls));
  ASSERT_EQ(g_submit_count, 6u);
  EXPECT_EQ(g_submit_offsets[0], 0x28000u);
  EXPECT_EQ(g_submit_offsets[1], 0x30000u);
  EXPECT_EQ(g_submit_offsets[2], 0x28000u);
  EXPECT_EQ(g_submit_offsets[3], 0x20000u);
  EXPECT_EQ(g_submit_offsets[4], 0x28000u);
  EXPECT_EQ(g_submit_offsets[5], 0x30000u);
  EXPECT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 10u);

  run(McdmAbi::legacy, 107600);
  ASSERT_EQ(g_setup_call_count, 7u);
  EXPECT_EQ(g_setup_calls[0], SetupCall::invalidate);
  EXPECT_EQ(g_setup_calls[1], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[2], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[3], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[4], SetupCall::wait);
  EXPECT_EQ(g_setup_calls[5], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[6], SetupCall::submit);
  ASSERT_EQ(g_submit_count, 5u);
  EXPECT_EQ(g_submit_opcodes[0], 9u);
  EXPECT_EQ(g_submit_offsets[0], 0x28000u);
  EXPECT_EQ(g_submit_offsets[1], 0x30000u);
  EXPECT_EQ(g_submit_offsets[2], 0x20000u);
  EXPECT_EQ(g_submit_offsets[3], 0x28000u);
  EXPECT_EQ(g_submit_offsets[4], 0x30000u);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 9u);
}

TEST(KmtApiTest, QueueCodeRangeReleasePreservesMarkersWithoutCpuWait) {
  auto run = [](McdmAbi mcdm_abi, size_t setup_payload_size,
                size_t expected_submits) {
    ResetFakes();
    KmtApi api = {};
    api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
    api.wait_from_cpu = FakeWaitFromCpu;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;
    Context context = {};
    context.hw_queue = 0x20;
    context.progress_fence = 0x21;
    context.next_fence_id = 7;
    CommandAperture aperture = {};
    aperture.gpu_allocation = 0x30;
    alignas(64) static std::array<uint8_t, 0x100000> aperture_storage = {};
    aperture.gpu_cpu_ptr = aperture_storage.data();
    aperture.gpu_va_size = aperture_storage.size();
    Error error = {};
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        mcdm_abi, setup_payload_size, &aperture, &error));

    ASSERT_TRUE(QueuePathBCodeRangeRelease(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error))
        << ErrorMessage(&error);
    EXPECT_EQ(g_submit_count, expected_submits);
    EXPECT_EQ(g_wait_count, 0u);
    for (size_t i = 0; i < g_submit_count; ++i) {
      EXPECT_EQ(g_submit_opcodes[i], 9u);
    }
  };

  run(McdmAbi::compact, 9952, 2u);
  run(McdmAbi::legacy, 107600, 1u);
}

TEST(KmtApiTest, PublishPathBCodeWriteRepeatsOpcode9WithoutRewrite) {
  // Opcode-9 is a queue-ordered happens-before, not a copy. Repeating
  // PublishPathBCodeWrite must emit the same end-marker sequence without
  // another CommitPathBCodeWrite at this KMT API layer. Native submit
  // restages separately before each Publish.
  auto run = [](McdmAbi mcdm_abi) {
    constexpr uint64_t kCodeBytes = 0x10000;
    constexpr size_t kPublishCount = 3;
    ResetFakes();
    KmtApi api = {};
    api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
    api.invalidate_cache = FakeInvalidateCache;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;
    Context context = {};
    context.hw_queue = 0x20;
    context.progress_fence = 0x21;
    context.next_fence_id = 7;
    CommandAperture aperture = {};
    aperture.gpu_allocation = 0x30;
    alignas(64) static std::array<uint8_t, 0x100000> aperture_storage = {};
    aperture.gpu_cpu_ptr = aperture_storage.data();
    aperture.gpu_va_size = aperture_storage.size();
    Error error = {};
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        mcdm_abi, 9952, &aperture, &error));
    ASSERT_TRUE(AcquirePathBCodeRange(api, device, &context, aperture,
                                      aperture.code_offset, kCodeBytes,
                                      &error));
    ASSERT_TRUE(CommitPathBCodeWrite(api, device, aperture,
                                     aperture.code_offset, kCodeBytes,
                                     &error));
    const size_t submits_after_commit = g_submit_count;
    const size_t invalidates_after_commit = g_invalidate_count;
    EXPECT_EQ(submits_after_commit, 0u);
    for (size_t i = 0; i < kPublishCount; ++i) {
      ASSERT_TRUE(PublishPathBCodeWrite(
          api, device, &context, aperture, aperture.code_offset, kCodeBytes,
          &error))
          << ErrorMessage(&error);
    }
    EXPECT_EQ(g_invalidate_count, invalidates_after_commit);
    ASSERT_GT(g_submit_count, submits_after_commit);
    ASSERT_EQ(g_submit_count % kPublishCount, 0u);
    const size_t markers_per_publish = g_submit_count / kPublishCount;
    ASSERT_GT(markers_per_publish, 0u);
    for (size_t i = 0; i < g_submit_count; ++i) {
      EXPECT_EQ(g_submit_opcodes[i], 9u);
    }
    for (size_t publish = 1; publish < kPublishCount; ++publish) {
      for (size_t marker = 0; marker < markers_per_publish; ++marker) {
        EXPECT_EQ(g_submit_offsets[publish * markers_per_publish + marker],
                  g_submit_offsets[marker]);
      }
    }
  };

  run(McdmAbi::compact);
  run(McdmAbi::legacy);
}

TEST(KmtApiTest, PublishPathBCodeEndMarkerSkipsInternalSlotBoundaries) {
  auto run = [](McdmAbi mcdm_abi, size_t setup_payload_size) {
    ResetFakes();
    KmtApi api = {};
    api.submit_command_to_hw_queue = FakeSubmitCommandToHwQueue;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;
    Context context = {};
    context.hw_queue = 0x20;
    context.progress_fence = 0x21;
    context.next_fence_id = 7;
    CommandAperture aperture = {};
    aperture.gpu_allocation = 0x30;
    alignas(64) static std::array<uint8_t, 0x100000> aperture_storage = {};
    aperture.gpu_cpu_ptr = aperture_storage.data();
    aperture.gpu_va_size = aperture_storage.size();
    Error error = {};
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        device.mcdm_abi, setup_payload_size, &aperture, &error));

    const uint64_t offset = aperture.code_offset + 0x123;
    const uint64_t length = 0x10000;
    ASSERT_TRUE(PublishPathBCodeEndMarker(api, device, &context, aperture,
                                          offset, length, &error));
    ASSERT_EQ(g_submit_count, 1u);
    EXPECT_EQ(g_submit_opcodes[0], 9u);
    const uint64_t slot_size =
        GetMcdmAbiInfo(mcdm_abi).command_aperture_code_slot_size;
    EXPECT_EQ(g_submit_offsets[0],
              (offset + length + slot_size - 1) & ~(slot_size - 1));
  };

  run(McdmAbi::compact, 9952);
  run(McdmAbi::legacy, 107600);
}

TEST(KmtApiTest, CodeWriteRemapMatchesNegotiatedAbi) {
  auto run = [](McdmAbi mcdm_abi) {
    ResetFakes();
    KmtApi api = {};
    api.unlock2 = FakeUnlock2;
    api.lock2 = FakeLock2;
    api.invalidate_cache = FakeInvalidateCache;
    Device device = {};
    device.device = 0x10;
    device.mcdm_abi = mcdm_abi;
    CommandAperture aperture = {};
    aperture.gpu_allocation = 0x20;
    aperture.gpu_cpu_ptr = reinterpret_cast<void*>(0x30);
    aperture.gpu_va_size = g_locked_aperture.size();
    Error error = {};
    ASSERT_TRUE(ConfigurePathBCodeRangeForSetupPayload(
        mcdm_abi, 9952, &aperture, &error));
    EXPECT_TRUE(RefreshPathBCodeMappingAfterWrite(
        api, device, &aperture, &error))
        << ErrorMessage(&error);
  };

  run(McdmAbi::compact);
  EXPECT_EQ(g_setup_call_count, 0u);

  run(McdmAbi::legacy);
  const SetupCall legacy_calls[] = {SetupCall::unlock, SetupCall::lock,
                                    SetupCall::invalidate};
  ASSERT_EQ(g_setup_call_count, std::size(legacy_calls));
  for (size_t i = 0; i < std::size(legacy_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], legacy_calls[i]);
  }
}

TEST(KmtApiTest, BoTableWritesBoundEntriesAndZerosUnusedCapacity) {
  auto run = [](McdmAbi mcdm_abi) {
    KmtApi api = {};
    Device device = {};
    device.mcdm_abi = mcdm_abi;
    const D3DGPU_VIRTUAL_ADDRESS real_vas[] = {0x13000, 0x14000, 0x15000};
    std::array<uint32_t, 23> words;
    words.fill(0xffffffffu);
    Error error = {};
    EXPECT_TRUE(PopulatePathBBoTable(device, words.data(),
                                     words.size() * sizeof(uint32_t), real_vas,
                                     std::size(real_vas), &error))
        << ErrorMessage(&error);
    return words;
  };
  auto read_entry = [](const std::array<uint32_t, 23>& words, size_t index) {
    return static_cast<uint64_t>(words[11 + 2 * index]) |
           (static_cast<uint64_t>(words[11 + 2 * index + 1]) << 32);
  };

  const auto compact = run(McdmAbi::compact);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(read_entry(compact, i), 0x13000u + i * 0x1000u);
  }
  for (size_t i = 3; i < 6; ++i) {
    EXPECT_EQ(read_entry(compact, i), 0u);
  }

  const auto legacy = run(McdmAbi::legacy);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(read_entry(legacy, i), 0x13000u + i * 0x1000u);
  }
  for (size_t i = 3; i < 6; ++i) {
    EXPECT_EQ(read_entry(legacy, i), 0u);
  }
}

TEST(KmtApiTest, BoTableRejectsEntriesBeyondNegotiatedCapacity) {
  const D3DGPU_VIRTUAL_ADDRESS real_vas[] = {
      0x1000, 0x2000, 0x3000, 0x4000, 0x5000, 0x6000, 0x7000};
  std::array<uint32_t, 25> words = {};
  for (const auto& [abi, count] :
       {std::pair{McdmAbi::legacy, size_t{6}},
        std::pair{McdmAbi::compact, size_t{7}}}) {
    Device device = {};
    device.mcdm_abi = abi;
    Error error = {};
    EXPECT_FALSE(PopulatePathBBoTable(
        device, words.data(), words.size() * sizeof(uint32_t), real_vas,
        count, &error));
    EXPECT_NE(std::strstr(ErrorMessage(&error), "ABI permits"), nullptr);
  }
}

TEST(KmtApiTest, FindNpuAdapterRejectsExcessiveAdapterCount) {
  KmtApi api = {};
  api.enum_adapters3 = FakeEnumTooManyAdapters;
  Adapter adapter = {};
  Error error = {};

  EXPECT_FALSE(FindNpuAdapter(api, &adapter, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "too many adapters"), nullptr);
}

TEST(KmtApiTest, CreateDeviceClosesRetainedHandlesOnCreateFailure) {
  ResetFakes();
  KmtApi api = {};
  api.create_device = FakeCreateDeviceFails;
  api.close_adapter = FakeCloseAdapter;

  Adapter adapter = {};
  adapter.handle = 0x10;
  adapter.retained_handles[0] = 0x20;
  adapter.retained_handles[1] = 0x30;
  adapter.retained_handle_count = 2;
  Device device = {};
  Error error = {};

  EXPECT_FALSE(CreateDevice(api, adapter, &device, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error), "D3DKMTCreateDevice"), nullptr);
  EXPECT_EQ(g_close_count, 2);
  EXPECT_EQ(g_closed_adapters[0], 0x20u);
  EXPECT_EQ(g_closed_adapters[1], 0x30u);
}

}  // namespace
}  // namespace iree::hal::amdxdna::mcdm
