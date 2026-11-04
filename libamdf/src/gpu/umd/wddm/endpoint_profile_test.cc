// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/wddm/provider/fake_api.h"
#include "libamdf/src/platform/windows/endpoint.h"

namespace {

constexpr NTSTATUS kSuccess = 0;
constexpr NTSTATUS kFailure = static_cast<NTSTATUS>(0xC0000001u);

struct FakeKmtState {
  // Number of successfully closed endpoint adapters.
  uint32_t adapter_close_success_count = 0;
  // Native result returned by memory-capability qualification.
  NTSTATUS memory_capabilities_status = kSuccess;
  // Native virtual-address width returned by the GPU MMU capability query.
  uint32_t virtual_address_bit_count = 48;
  // Number of native memory-capability queries.
  uint32_t memory_query_count = 0;
  // Exact native local segment capacity for the selected physical adapter.
  uint64_t local_capacity = UINT64_C(8) << 30;
  // Current per-process native residency budget.
  uint64_t budget = UINT64_C(6) << 30;
  // Current resident usage charged against the native budget.
  uint64_t usage = UINT64_C(2) << 30;
  // Native result returned by the explicit residency observation.
  NTSTATUS observation_status = kSuccess;
};

FakeKmtState* current_state = nullptr;

NTSTATUS APIENTRY FakeQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  if (query->Type == KMTQAITYPE_GETSEGMENTGROUPSIZE) {
    EXPECT_EQ(query->hAdapter, 0x08u);
    EXPECT_EQ(query->PrivateDriverDataSize,
              sizeof(D3DKMT_SEGMENTGROUPSIZEINFO));
    auto* sizes =
        static_cast<D3DKMT_SEGMENTGROUPSIZEINFO*>(query->pPrivateDriverData);
    EXPECT_EQ(sizes->PhysicalAdapterIndex, 0u);
    sizes->LocalMemory = current_state->local_capacity;
    return kSuccess;
  }
  ++current_state->memory_query_count;
  EXPECT_EQ(query->hAdapter, 0x08u);
  EXPECT_EQ(query->Type, KMTQAITYPE_QUERY_GPUMMU_CAPS);
  EXPECT_EQ(query->PrivateDriverDataSize, sizeof(D3DKMT_QUERY_GPUMMU_CAPS));
  if (current_state->memory_capabilities_status != kSuccess) {
    return current_state->memory_capabilities_status;
  }
  auto* gpu_mmu =
      static_cast<D3DKMT_QUERY_GPUMMU_CAPS*>(query->pPrivateDriverData);
  EXPECT_EQ(gpu_mmu->PhysicalAdapterIndex, 0u);
  gpu_mmu->Caps.VirtualAddressBitCount =
      current_state->virtual_address_bit_count;
  gpu_mmu->Caps.Flags.ReadOnlyMemorySupported = 1;
  gpu_mmu->Caps.Flags.NoExecuteMemorySupported = 1;
  gpu_mmu->Caps.Flags.CacheCoherentMemorySupported = 1;
  return kSuccess;
}

NTSTATUS APIENTRY FakeQueryVideoMemoryInfo(D3DKMT_QUERYVIDEOMEMORYINFO* query) {
  EXPECT_EQ(query->hProcess, GetCurrentProcess());
  EXPECT_EQ(query->hAdapter, 0x08u);
  EXPECT_EQ(query->PhysicalAdapterIndex, 0u);
  EXPECT_EQ(query->MemorySegmentGroup, D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL);
  query->Budget = current_state->budget;
  query->CurrentUsage = current_state->usage;
  return current_state->observation_status;
}

// These native dependencies are present but must never be invoked by discovery.
template <typename Arguments>
NTSTATUS APIENTRY UnexpectedMemoryOperation(Arguments*) {
  ADD_FAILURE() << "memory acquisition is not part of endpoint qualification";
  return kFailure;
}

NTSTATUS APIENTRY FakeCloseAdapter(const D3DKMT_CLOSEADAPTER* close) {
  EXPECT_EQ(close->hAdapter, 0x08u);
  ++current_state->adapter_close_success_count;
  return kSuccess;
}

class WindowsGpuEndpointProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    instance_.host_allocator = amdf_allocator_system();
    instance_.kmt.close_adapter = FakeCloseAdapter;
    instance_.kmt.query_adapter_info = FakeQueryAdapterInfo;
    instance_.kmt.query_video_memory_info = FakeQueryVideoMemoryInfo;

    ASSERT_EQ(amdf_calloc(instance_.host_allocator, sizeof(*endpoint_),
                          amdf_alignof(amdf_platform_endpoint_t),
                          reinterpret_cast<void**>(&endpoint_)),
              AMDF_STATUS_OK);
    endpoint_->instance = &instance_;
    endpoint_->adapter = 0x08;
    endpoint_->physical_adapter_index = 0;

    amdf_test_wddm_provider_reset();
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
      endpoint_ = nullptr;
    }
    current_state = nullptr;
  }

  void EnableMemoryOperations() {
    instance_.kmt.query_adapter_info = FakeQueryAdapterInfo;
    instance_.kmt.create_allocation = UnexpectedMemoryOperation;
    instance_.kmt.destroy_allocation = UnexpectedMemoryOperation;
    instance_.kmt.map_gpu_virtual_address = UnexpectedMemoryOperation;
    instance_.kmt.make_resident = UnexpectedMemoryOperation;
    instance_.kmt.wait_from_cpu = UnexpectedMemoryOperation;
    instance_.kmt.reserve_gpu_virtual_address = UnexpectedMemoryOperation;
    instance_.kmt.free_gpu_virtual_address = UnexpectedMemoryOperation;
    instance_.kmt.invalidate_cache = UnexpectedMemoryOperation;
  }

  // Native dependency observations for the active test.
  FakeKmtState state_;
  // Platform instance owning the injected KMT table.
  amdf_platform_instance_t instance_ = {};
  // Live query endpoint, with no ownership of failed qualification.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
};

TEST_F(WindowsGpuEndpointProfileTest,
       FailedProbeReleaseLeavesNoEndpointCleanupObligation) {
  auto* profile = reinterpret_cast<amdf_gpu_endpoint_profile_t*>(uintptr_t{1});
  const auto* sentinel_profile = profile;

  const amdf_status_t status = amdf_gpu_umd_create_endpoint_profile(
      endpoint_, instance_.host_allocator, &profile);

  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_BUSY);
  EXPECT_EQ(profile, sentinel_profile);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_open_success_count(), 1u);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_attempt_count(), 1u);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_success_count(), 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 0u);

  EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
  endpoint_ = nullptr;
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_attempt_count(), 1u);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_success_count(), 0u);
  EXPECT_EQ(state_.adapter_close_success_count, 1u);
}

TEST_F(WindowsGpuEndpointProfileTest,
       PublishesQualifiedProfileAfterProbeRelease) {
  amdf_test_wddm_provider_set_adapter_close_failures(0);
  amdf_gpu_endpoint_profile_t* profile = nullptr;

  EXPECT_EQ(amdf_gpu_umd_create_endpoint_profile(
                endpoint_, instance_.host_allocator, &profile),
            AMDF_STATUS_OK);

  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->info.gfx_ip.major, 11u);
  EXPECT_EQ(profile->info.gfx_ip.minor, 5u);
  EXPECT_EQ(profile->info.gfx_ip.stepping, 1u);
  EXPECT_EQ(profile->info.compute.wavefront_size, 32u);
  amdf_free(instance_.host_allocator, profile);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_open_success_count(), 1u);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_attempt_count(), 1u);
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_success_count(), 1u);

  EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
  endpoint_ = nullptr;
  EXPECT_EQ(state_.adapter_close_success_count, 1u);
}

TEST_F(WindowsGpuEndpointProfileTest,
       DiscoveryDoesNotAcquireMemoryOrQueryGpuMmu) {
  amdf_test_wddm_provider_set_adapter_close_failures(0);
  EnableMemoryOperations();
  state_.memory_capabilities_status = kFailure;
  state_.virtual_address_bit_count = 0;
  instance_.kmt.update_gpu_virtual_address = UnexpectedMemoryOperation;
  instance_.kmt.wait_on_hardware_queue = UnexpectedMemoryOperation;
  amdf_gpu_endpoint_profile_t* profile = nullptr;
  ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                endpoint_, instance_.host_allocator, &profile),
            AMDF_STATUS_OK);
  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->info.gfx_ip.major, 11u);
  EXPECT_EQ(state_.memory_query_count, 0u);
  ASSERT_GT(profile->queue_family_count, 0u);
  for (uint32_t i = 0; i < profile->queue_family_count; ++i) {
    const auto& family = profile->queue_families[i];
    // The injected adapter has HWS compute and SWS SDMA. KMT entry points
    // alone cannot give the latter a hardware-queue dependency handle.
    const bool supports_mapping =
        family.command_type == AMDF_QUEUE_COMMAND_TYPE_GPU_PM4;
    EXPECT_EQ((family.kernel_queue_capabilities &
               AMDF_KERNEL_QUEUE_CAPABILITY_VIRTUAL_MEMORY) != 0,
              supports_mapping);
    EXPECT_EQ((family.roles & AMDF_QUEUE_ROLE_VIRTUAL_MEMORY) != 0,
              supports_mapping);
  }
  EXPECT_EQ(amdf_test_wddm_provider_query_adapter_close_success_count(), 1u);
  amdf_free(instance_.host_allocator, profile);
}

TEST_F(WindowsGpuEndpointProfileTest,
       LocalObservationUsesTheNativeResidencyBudget) {
  amdf_memory_scope_observation_t observation = {};
  ASSERT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      AMDF_STATUS_OK);
  EXPECT_EQ(observation.capacity_byte_length, state_.local_capacity);
  EXPECT_EQ(observation.available_byte_length, UINT64_C(4) << 30);
  state_.usage = state_.budget + 1;
  ASSERT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      AMDF_STATUS_OK);
  EXPECT_EQ(observation.available_byte_length, 0u);
}

TEST_F(WindowsGpuEndpointProfileTest,
       UnifiedObservationSharesSystemPhysicalCapacity) {
  amdf_memory_scope_observation_t observation = {};
  ASSERT_EQ(amdf_gpu_umd_sample_memory_scope(
                endpoint_,
                AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                    AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES,
                &observation),
            AMDF_STATUS_OK);
  EXPECT_GT(observation.capacity_byte_length, 0u);
  EXPECT_LE(observation.available_byte_length,
            observation.capacity_byte_length);
  EXPECT_LE(observation.available_byte_length, state_.budget - state_.usage);
}

TEST_F(WindowsGpuEndpointProfileTest, FailedObservationPreservesCallerOutput) {
  amdf_memory_scope_observation_t observation = {
      .capacity_byte_length = UINT64_MAX,
      .available_byte_length = UINT64_MAX,
  };
  const auto expected = observation;
  state_.observation_status = kFailure;
  EXPECT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      amdf_kmt_make_status(kFailure));
  EXPECT_EQ(std::memcmp(&observation, &expected, sizeof(observation)), 0);
}

}  // namespace
