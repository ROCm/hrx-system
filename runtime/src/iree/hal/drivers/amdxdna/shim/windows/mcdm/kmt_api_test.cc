// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "kmt_api.h"

#include <array>
#include <cstring>

#include "iree/testing/gtest.h"

namespace iree::hal::amdxdna::mcdm {
namespace {

int g_close_count = 0;
D3DKMT_HANDLE g_closed_adapters[4] = {};
NTSTATUS g_query_statuses[2] = {};
uint32_t g_query_outputs[2][3] = {};
UINT g_query_sizes[2] = {};
D3DKMT_HANDLE g_query_adapters[2] = {};
KMTQUERYADAPTERINFOTYPE g_query_types[2] = {};
bool g_query_data_was_zero[2] = {};
size_t g_query_count = 0;
enum class SetupCall { submit, make_resident, lock, unlock, invalidate, wait };
SetupCall g_setup_calls[16] = {};
uint32_t g_submit_opcodes[8] = {};
uint64_t g_submit_fences[8] = {};
uint64_t g_submit_offsets[8] = {};
uint64_t g_wait_fences[2] = {};
D3DKMT_HANDLE g_gpu_wait_context = 0;
D3DKMT_HANDLE g_gpu_wait_object = 0;
uint64_t g_gpu_wait_fence = 0;
size_t g_gpu_wait_count = 0;
size_t g_setup_call_count = 0;
size_t g_submit_count = 0;
size_t g_wait_count = 0;
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
  g_locked_aperture.fill(0);
}

NTSTATUS APIENTRY FakeQueryAdapterInfo(
    const D3DKMT_QUERYADAPTERINFO* args) {
  const size_t call = g_query_count++;
  if (call >= 2) return static_cast<NTSTATUS>(0xC0000001u);
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
  g_setup_calls[g_setup_call_count++] = SetupCall::submit;
  std::memcpy(&g_submit_opcodes[g_submit_count], args->pPrivateDriverData,
              sizeof(uint32_t));
  std::memcpy(&g_submit_offsets[g_submit_count],
              static_cast<const uint8_t*>(args->pPrivateDriverData) + 0x10,
              sizeof(uint64_t));
  g_submit_fences[g_submit_count++] = args->HwQueueProgressFenceId;
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
  g_setup_calls[g_setup_call_count++] = SetupCall::unlock;
  return 0;
}

NTSTATUS APIENTRY FakeInvalidateCache(CONST D3DKMT_INVALIDATECACHE* args) {
  g_setup_calls[g_setup_call_count++] = SetupCall::invalidate;
  return 0;
}

NTSTATUS APIENTRY FakeWaitFromCpu(
    CONST D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU* args) {
  g_setup_calls[g_setup_call_count++] = SetupCall::wait;
  g_wait_fences[g_wait_count++] = args->FenceValueArray[0];
  return 0;
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
  EXPECT_EQ(abi.command_aperture_code_publish_granularity, 0u);
  EXPECT_FALSE(abi.command_aperture_residency_after_bootstrap);
  EXPECT_TRUE(abi.command_aperture_remap_after_write);
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
  EXPECT_EQ(abi.command_aperture_code_publish_granularity, 0x8000u);
  EXPECT_TRUE(abi.command_aperture_residency_after_bootstrap);
  EXPECT_FALSE(abi.command_aperture_remap_after_write);
}

TEST(KmtApiTest, CodeRangeFollowsSetupPayloadAllocatorForBothAbis) {
  CommandAperture aperture = {};
  aperture.gpu_allocation = 0x20;
  aperture.gpu_va = 0x04000000;
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

TEST(KmtApiTest, NegotiatesLegacyAbiFromTwoDwordQuery) {
  ResetFakes();
  g_query_statuses[0] = 0;
  g_query_outputs[0][1] = 2;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbi(api, 0x1234, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy);
  ASSERT_EQ(g_query_count, 1u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_adapters[0], 0x1234u);
  EXPECT_EQ(g_query_types[0], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_TRUE(g_query_data_was_zero[0]);
}

TEST(KmtApiTest, NegotiatesLegacyV3AbiFromTwoDwordQuery) {
  ResetFakes();
  g_query_statuses[0] = 0;
  g_query_outputs[0][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::compact;
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbi(api, 0x1234, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::legacy);
  ASSERT_EQ(g_query_count, 1u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_adapters[0], 0x1234u);
  EXPECT_EQ(g_query_types[0], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_TRUE(g_query_data_was_zero[0]);
}

TEST(KmtApiTest, NegotiatesCompactAbiAfterLegacyShapeIsTooSmall) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_statuses[1] = 0;
  g_query_outputs[1][1] = 2;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  ASSERT_TRUE(QueryMcdmAbi(api, 0x5678, &abi, &error))
      << ErrorMessage(&error);
  EXPECT_EQ(abi, McdmAbi::compact);
  ASSERT_EQ(g_query_count, 2u);
  EXPECT_EQ(g_query_sizes[0], 2u * sizeof(uint32_t));
  EXPECT_EQ(g_query_sizes[1], 3u * sizeof(uint32_t));
  EXPECT_EQ(g_query_adapters[0], 0x5678u);
  EXPECT_EQ(g_query_adapters[1], 0x5678u);
  EXPECT_EQ(g_query_types[0], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_EQ(g_query_types[1], KMTQAITYPE_UMDRIVERPRIVATE);
  EXPECT_TRUE(g_query_data_was_zero[0]);
  EXPECT_TRUE(g_query_data_was_zero[1]);
}

TEST(KmtApiTest, RejectsUnknownTwoDwordAbiIdentity) {
  ResetFakes();
  g_query_statuses[0] = 0;
  g_query_outputs[0][1] = 4;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported two-dword MCDM identity"),
            nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RejectsUnknownTwoDwordAbiMajor) {
  ResetFakes();
  g_query_statuses[0] = 0;
  g_query_outputs[0][0] = 1;
  g_query_outputs[0][1] = 3;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryMcdmAbi(api, 0x1234, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported two-dword MCDM identity"),
            nullptr);
  EXPECT_EQ(g_query_count, 1u);
}

TEST(KmtApiTest, RejectsUnknownThreeDwordAbiIdentity) {
  ResetFakes();
  g_query_statuses[0] = static_cast<NTSTATUS>(0xC0000023u);
  g_query_statuses[1] = 0;
  g_query_outputs[1][1] = 2;
  g_query_outputs[1][2] = 1;
  KmtApi api = {};
  api.query_adapter_info = FakeQueryAdapterInfo;
  McdmAbi abi = McdmAbi::legacy;
  Error error = {};

  EXPECT_FALSE(QueryMcdmAbi(api, 0x5678, &abi, &error));
  EXPECT_NE(std::strstr(ErrorMessage(&error),
                        "unsupported three-dword MCDM identity"),
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

  EXPECT_FALSE(QueryMcdmAbi(api, 0x9ABC, &abi, &error));
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
  aperture.gpu_va_size = g_locked_aperture.size();
  aperture.cpu_ptr = reinterpret_cast<void*>(0x60);
  std::array<uint8_t, 107600> setup_payload;
  setup_payload.fill(0x5a);
  Error error = {};

  ASSERT_TRUE(SubmitAndWaitPathBSetup(api, device, &context, &aperture,
                                      setup_payload.data(),
                                      setup_payload.size(), &error))
      << ErrorMessage(&error);
  const SetupCall expected_calls[] = {
      SetupCall::submit, SetupCall::make_resident, SetupCall::lock,
      SetupCall::invalidate, SetupCall::submit, SetupCall::wait};
  ASSERT_EQ(g_setup_call_count, std::size(expected_calls));
  for (size_t i = 0; i < std::size(expected_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], expected_calls[i]);
  }
  ASSERT_EQ(g_submit_count, 2u);
  EXPECT_EQ(g_submit_opcodes[0], 2u);
  EXPECT_EQ(g_submit_opcodes[1], 5u);
  EXPECT_EQ(g_submit_fences[0], 7u);
  EXPECT_EQ(g_submit_fences[1], 8u);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 8u);
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

TEST(KmtApiTest, CreateBufferPreservesPendingResidencyFenceForBothAbis) {
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
  }
}

TEST(KmtApiTest, BufferResidencyAddsGpuContextDependency) {
  ResetFakes();
  KmtApi api = {};
  api.wait_from_gpu = FakeWaitFromGpu;
  Device device = {};
  device.paging_sync_object = 0x20;
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
    ASSERT_TRUE(CommitPathBCodeWrite(api, device, aperture,
                                     aperture.code_offset, 0x10000, &error));
    ASSERT_TRUE(PublishPathBCodeWrite(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
    ASSERT_TRUE(ReleasePathBCodeRange(
        api, device, &context, aperture, aperture.code_offset, 0x10000,
        &error));
  };

  run(McdmAbi::compact, 9952);
  const SetupCall compact_calls[] = {
      SetupCall::submit, SetupCall::submit, SetupCall::wait,
      SetupCall::submit, SetupCall::submit};
  ASSERT_EQ(g_setup_call_count, std::size(compact_calls));
  size_t compact_submit_index = 0;
  for (size_t i = 0; i < std::size(compact_calls); ++i) {
    EXPECT_EQ(g_setup_calls[i], compact_calls[i]);
    if (compact_calls[i] == SetupCall::submit) {
      EXPECT_EQ(g_submit_opcodes[compact_submit_index++], 9u);
    }
  }
  ASSERT_EQ(g_submit_count, 4u);
  EXPECT_EQ(g_submit_offsets[0], 0x10000u);
  EXPECT_EQ(g_submit_offsets[1], 0x18000u);
  EXPECT_EQ(g_submit_offsets[2], 0x10000u);
  EXPECT_EQ(g_submit_offsets[3], 0x8000u);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 8u);

  run(McdmAbi::compact, 107600);
  ASSERT_EQ(g_setup_call_count, std::size(compact_calls));
  ASSERT_EQ(g_submit_count, 4u);
  EXPECT_EQ(g_submit_offsets[0], 0x28000u);
  EXPECT_EQ(g_submit_offsets[1], 0x30000u);
  EXPECT_EQ(g_submit_offsets[2], 0x28000u);
  EXPECT_EQ(g_submit_offsets[3], 0x20000u);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 8u);

  run(McdmAbi::legacy, 107600);
  ASSERT_EQ(g_setup_call_count, 5u);
  EXPECT_EQ(g_setup_calls[0], SetupCall::invalidate);
  EXPECT_EQ(g_setup_calls[1], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[2], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[3], SetupCall::submit);
  EXPECT_EQ(g_setup_calls[4], SetupCall::wait);
  ASSERT_EQ(g_submit_count, 3u);
  EXPECT_EQ(g_submit_opcodes[0], 9u);
  EXPECT_EQ(g_submit_offsets[0], 0x28000u);
  EXPECT_EQ(g_submit_offsets[1], 0x30000u);
  EXPECT_EQ(g_submit_offsets[2], 0x20000u);
  ASSERT_EQ(g_wait_count, 1u);
  EXPECT_EQ(g_wait_fences[0], 9u);
}

TEST(KmtApiTest, SingleCodeWriteRemapMatchesNegotiatedAbi) {
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
    EXPECT_TRUE(RefreshPathBSingleCodeMappingAfterWrite(
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
  for (const auto [abi, count] :
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
