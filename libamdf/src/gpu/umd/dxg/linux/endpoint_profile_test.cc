// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/gpu/umd/endpoint_profile.h"

#include <sys/ioctl.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/gpu/umd/dxg/rocdxg/loader.h"
#include "libamdf/src/platform/linux/dxg/endpoint.h"

namespace {

constexpr uint32_t kNativeFailure = 0xC0000001u;

enum class Operation {
  kLoadBridge,
  kQueryBridgeApi,
  kCreateDevice,
  kOpenBridgeAdapter,
  kCloseBridgeAdapter,
  kDestroyDevice,
  kUnloadBridge,
  kSegmentQuery,
  kResidencyQuery,
  kHostMemorySample,
};

struct FakeState {
  // Bridge result returned while opening private adapter state.
  amdf_dxg_bridge_result_t bridge_open_result = AMDF_DXG_BRIDGE_RESULT_SUCCESS;
  // Native result returned by the local-memory capacity query.
  uint32_t segment_query_result = 0;
  // Native result returned by the process residency-budget query.
  uint32_t residency_query_result = 0;
  // Native logical device handle returned during qualification.
  D3DKMT_HANDLE device = 0x10;
  // Exact local-memory capacity exposed by the selected adapter.
  uint64_t local_capacity = UINT64_C(8) << 30;
  // Current process residency budget for local memory.
  uint64_t budget = UINT64_C(6) << 30;
  // Current process usage charged against the residency budget.
  uint64_t usage = UINT64_C(2) << 30;
  // Host-memory observation used by a system-backed scope.
  amdf_memory_scope_observation_t host_observation = {
      .capacity_byte_length = UINT64_C(16) << 30,
      .available_byte_length = UINT64_C(7) << 30,
  };
  // Private provider properties returned for the selected adapter.
  amdf_dxg_bridge_gpu_properties_t provider_properties = {
      .gfx_ip_major = 11,
      .gfx_ip_minor = 5,
      .gfx_ip_stepping = 0,
      .asic_revision = 3,
      .wavefront_size = 32,
      .compute_unit_count = 20,
      .maximum_wave_count_per_compute_unit = 32,
      .maximum_scratch_wave_count_per_compute_unit = 32,
      .local_data_share_byte_length = UINT64_C(64) << 10,
      .xcc_count = 1,
      .shader_engine_count = 2,
      .supports_pm4_kernel_queue = 1,
      .is_discrete = 1,
      .platform_atomic_support = 1,
  };
  // Ordered loader, bridge, and native operations.
  std::vector<Operation> operations;
};

FakeState* current_state = nullptr;
amdf_platform_endpoint_t* current_endpoint = nullptr;
amdf_linux_dxg_connection_t* current_connection = nullptr;
amdf_linux_dxg_kmt_t* current_kmt = nullptr;

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL FakeAdapterOpen(
    const amdf_dxg_bridge_gpu_adapter_open_info_t* open_info,
    const amdf_allocator_t*, amdf_dxg_bridge_gpu_adapter_t** out_adapter,
    amdf_dxg_bridge_gpu_properties_t* out_properties,
    uint32_t* out_native_status) {
  current_state->operations.push_back(Operation::kOpenBridgeAdapter);
  EXPECT_EQ(open_info->adapter_handle, current_endpoint->adapter);
  EXPECT_EQ(open_info->device_handle, current_state->device);
  EXPECT_EQ(open_info->physical_adapter_index,
            current_endpoint->physical_adapter_index);
  EXPECT_EQ(open_info->adapter_luid,
            current_endpoint->info.native_identity.value.windows_adapter.luid);
  EXPECT_EQ(open_info->native_transport.context, current_kmt);
  EXPECT_EQ(open_info->native_transport.invoke, amdf_linux_dxg_kmt_invoke);
  *out_native_status = 0;
  if (current_state->bridge_open_result != AMDF_DXG_BRIDGE_RESULT_SUCCESS) {
    return current_state->bridge_open_result;
  }
  *out_adapter = reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(uintptr_t{1});
  *out_properties = current_state->provider_properties;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

amdf_dxg_bridge_result_t AMDF_DXG_BRIDGE_CALL FakeAdapterClose(
    amdf_dxg_bridge_gpu_adapter_t* adapter, uint32_t* out_native_status) {
  current_state->operations.push_back(Operation::kCloseBridgeAdapter);
  EXPECT_EQ(adapter,
            reinterpret_cast<amdf_dxg_bridge_gpu_adapter_t*>(uintptr_t{1}));
  *out_native_status = 0;
  return AMDF_DXG_BRIDGE_RESULT_SUCCESS;
}

const amdf_dxg_bridge_api_t kBridgeApi = {
    .structure_size = sizeof(amdf_dxg_bridge_api_t),
    .abi_version = AMDF_DXG_BRIDGE_ABI_VERSION_13,
    .gpu_adapter_open = FakeAdapterOpen,
    .gpu_adapter_close = FakeAdapterClose,
};

class LinuxDxgEndpointProfileTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_state = &state_;
    current_endpoint = &endpoint_;
    current_connection = &instance_.connection;
    current_kmt = &instance_.kmt;
    instance_.host_allocator = amdf_allocator_system();
    instance_.connection.descriptor = 1;
    instance_.kmt.connection = &instance_.connection;
    instance_.kmt.host_allocator = &instance_.host_allocator;
    endpoint_.instance = &instance_;
    endpoint_.adapter = 0x08;
    endpoint_.physical_adapter_index = 2;
    endpoint_.info.native_identity.type =
        AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER;
    endpoint_.info.native_identity.value.windows_adapter.luid =
        UINT64_C(0x1122334455667788);
    endpoint_.info.native_identity.value.windows_adapter
        .physical_adapter_index = endpoint_.physical_adapter_index;
  }

  void TearDown() override {
    current_kmt = nullptr;
    current_connection = nullptr;
    current_endpoint = nullptr;
    current_state = nullptr;
  }

  // Native behavior and observations selected by the active test.
  FakeState state_;
  // Platform instance owning the typed Linux DXG transport.
  amdf_platform_instance_t instance_ = {};
  // Query endpoint retaining the native adapter and immutable identity.
  amdf_platform_endpoint_t endpoint_ = {};
};

TEST_F(LinuxDxgEndpointProfileTest,
       PublishesProfileAfterRetiringTemporaryProbeState) {
  amdf_gpu_endpoint_profile_t* profile = nullptr;

  ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                &endpoint_, instance_.host_allocator, &profile),
            AMDF_STATUS_OK);

  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->info.gfx_ip.major, 11u);
  EXPECT_EQ(profile->info.gfx_ip.minor, 5u);
  EXPECT_EQ(profile->info.compute.compute_unit_count, 20u);
  ASSERT_EQ(profile->queue_family_count, 1u);
  EXPECT_EQ(profile->queue_families[0].command_type,
            AMDF_QUEUE_COMMAND_TYPE_GPU_PM4);
  EXPECT_EQ(profile->memory.flags, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES);
  EXPECT_EQ(profile->memory.capacity_byte_length, state_.local_capacity);
  amdf_free(instance_.host_allocator, profile);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kLoadBridge, Operation::kQueryBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCloseBridgeAdapter, Operation::kDestroyDevice,
                Operation::kUnloadBridge, Operation::kSegmentQuery}));
}

TEST_F(LinuxDxgEndpointProfileTest,
       RejectedAdapterRetiresDeviceAndPreservesOutput) {
  state_.bridge_open_result = AMDF_DXG_BRIDGE_RESULT_UNSUPPORTED;
  auto* profile = reinterpret_cast<amdf_gpu_endpoint_profile_t*>(uintptr_t{1});
  const auto* expected_profile = profile;

  EXPECT_EQ(amdf_gpu_umd_create_endpoint_profile(
                &endpoint_, instance_.host_allocator, &profile),
            amdf_make_api_status(AMDF_STATUS_CODE_UNSUPPORTED));

  EXPECT_EQ(profile, expected_profile);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kLoadBridge, Operation::kQueryBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kDestroyDevice, Operation::kUnloadBridge}));
}

TEST_F(LinuxDxgEndpointProfileTest,
       IntegratedProfileUsesTheInstanceMemoryScope) {
  state_.provider_properties.is_discrete = 0;
  amdf_gpu_endpoint_profile_t* profile = nullptr;

  ASSERT_EQ(amdf_gpu_umd_create_endpoint_profile(
                &endpoint_, instance_.host_allocator, &profile),
            AMDF_STATUS_OK);

  ASSERT_NE(profile, nullptr);
  EXPECT_EQ(profile->memory.flags, AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                                       AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES);
  EXPECT_EQ(profile->memory.capacity_byte_length, 0u);
  amdf_free(instance_.host_allocator, profile);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kLoadBridge, Operation::kQueryBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCloseBridgeAdapter, Operation::kDestroyDevice,
                Operation::kUnloadBridge}));
}

TEST_F(LinuxDxgEndpointProfileTest,
       FailedCapacityQueryPreservesOutputAfterProbeRetires) {
  state_.segment_query_result = kNativeFailure;
  auto* profile = reinterpret_cast<amdf_gpu_endpoint_profile_t*>(uintptr_t{1});
  const auto* expected_profile = profile;

  EXPECT_EQ(amdf_gpu_umd_create_endpoint_profile(
                &endpoint_, instance_.host_allocator, &profile),
            amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, kNativeFailure));

  EXPECT_EQ(profile, expected_profile);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{
                Operation::kLoadBridge, Operation::kQueryBridgeApi,
                Operation::kCreateDevice, Operation::kOpenBridgeAdapter,
                Operation::kCloseBridgeAdapter, Operation::kDestroyDevice,
                Operation::kUnloadBridge, Operation::kSegmentQuery}));
}

TEST_F(LinuxDxgEndpointProfileTest,
       LocalObservationUsesCurrentProcessResidencyBudget) {
  amdf_memory_scope_observation_t observation = {};

  ASSERT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          &endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      AMDF_STATUS_OK);

  EXPECT_EQ(observation.capacity_byte_length, state_.local_capacity);
  EXPECT_EQ(observation.available_byte_length, UINT64_C(4) << 30);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kResidencyQuery,
                                    Operation::kSegmentQuery}));
}

TEST_F(LinuxDxgEndpointProfileTest,
       SystemObservationCapsHostHeadroomWithResidencyBudget) {
  amdf_memory_scope_observation_t observation = {};

  ASSERT_EQ(amdf_gpu_umd_sample_memory_scope(
                &endpoint_,
                AMDF_MEMORY_SCOPE_FLAG_SYSTEM_BACKED |
                    AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES,
                &observation),
            AMDF_STATUS_OK);

  EXPECT_EQ(observation.capacity_byte_length,
            state_.host_observation.capacity_byte_length);
  EXPECT_EQ(observation.available_byte_length, UINT64_C(4) << 30);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kResidencyQuery,
                                    Operation::kHostMemorySample}));
}

TEST_F(LinuxDxgEndpointProfileTest, FailedObservationPreservesCallerOutput) {
  state_.residency_query_result = kNativeFailure;
  amdf_memory_scope_observation_t observation = {
      .capacity_byte_length = UINT64_MAX,
      .available_byte_length = UINT64_MAX,
  };
  const auto expected_observation = observation;

  EXPECT_EQ(
      amdf_gpu_umd_sample_memory_scope(
          &endpoint_, AMDF_MEMORY_SCOPE_FLAG_AVAILABLE_BYTES, &observation),
      amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, kNativeFailure));

  EXPECT_EQ(
      std::memcmp(&observation, &expected_observation, sizeof(observation)), 0);
  EXPECT_EQ(state_.operations,
            (std::vector<Operation>{Operation::kResidencyQuery}));
}

}  // namespace

extern "C" amdf_status_t __wrap_amdf_gpu_dxg_rocdxg_loader_initialize(
    amdf_allocator_t, amdf_gpu_dxg_rocdxg_loader_t* out_loader) {
  current_state->operations.push_back(Operation::kLoadBridge);
  *out_loader = {.module = reinterpret_cast<void*>(uintptr_t{1})};
  return AMDF_STATUS_OK;
}

extern "C" amdf_status_t __wrap_amdf_gpu_dxg_rocdxg_loader_query_api(
    const amdf_gpu_dxg_rocdxg_loader_t* loader,
    const amdf_dxg_bridge_api_t** out_api) {
  current_state->operations.push_back(Operation::kQueryBridgeApi);
  EXPECT_EQ(loader->module, reinterpret_cast<void*>(uintptr_t{1}));
  *out_api = &kBridgeApi;
  return AMDF_STATUS_OK;
}

extern "C" void __wrap_amdf_gpu_dxg_rocdxg_loader_deinitialize(
    amdf_gpu_dxg_rocdxg_loader_t* loader) {
  current_state->operations.push_back(Operation::kUnloadBridge);
  EXPECT_EQ(loader->module, reinterpret_cast<void*>(uintptr_t{1}));
  *loader = {};
}

extern "C" amdf_status_t __wrap_amdf_host_memory_sample_observation(
    amdf_memory_scope_observation_t* out_observation) {
  current_state->operations.push_back(Operation::kHostMemorySample);
  *out_observation = current_state->host_observation;
  return AMDF_STATUS_OK;
}

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, current_connection);
  switch (_IOC_NR(request)) {
    case 0x02: {
      current_state->operations.push_back(Operation::kCreateDevice);
      auto* create = static_cast<D3DKMT_CREATEDEVICE*>(arguments);
      EXPECT_EQ(create->hAdapter, current_endpoint->adapter);
      create->hDevice = current_state->device;
      return 0;
    }
    case 0x09: {
      current_state->operations.push_back(Operation::kSegmentQuery);
      auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
      EXPECT_EQ(query->hAdapter, current_endpoint->adapter);
      EXPECT_EQ(query->Type, KMTQAITYPE_GETSEGMENTGROUPSIZE);
      EXPECT_EQ(query->PrivateDriverDataSize,
                sizeof(D3DKMT_SEGMENTGROUPSIZEINFO));
      auto* sizes =
          static_cast<D3DKMT_SEGMENTGROUPSIZEINFO*>(query->pPrivateDriverData);
      EXPECT_EQ(sizes->PhysicalAdapterIndex,
                current_endpoint->physical_adapter_index);
      if (current_state->segment_query_result == 0) {
        sizes->LocalMemory = current_state->local_capacity;
      }
      return current_state->segment_query_result;
    }
    case 0x0A: {
      current_state->operations.push_back(Operation::kResidencyQuery);
      auto* query = static_cast<D3DKMT_QUERYVIDEOMEMORYINFO*>(arguments);
      EXPECT_EQ(query->hProcess, nullptr);
      EXPECT_EQ(query->hAdapter, current_endpoint->adapter);
      EXPECT_EQ(query->PhysicalAdapterIndex,
                current_endpoint->physical_adapter_index);
      EXPECT_EQ(query->MemorySegmentGroup, D3DKMT_MEMORY_SEGMENT_GROUP_LOCAL);
      if (current_state->residency_query_result == 0) {
        query->Budget = current_state->budget;
        query->CurrentUsage = current_state->usage;
      }
      return current_state->residency_query_result;
    }
    case 0x19: {
      current_state->operations.push_back(Operation::kDestroyDevice);
      const auto* destroy = static_cast<D3DKMT_DESTROYDEVICE*>(arguments);
      EXPECT_EQ(destroy->hDevice, current_state->device);
      return 0;
    }
    default:
      ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
      return kNativeFailure;
  }
}
