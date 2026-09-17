// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/windows/endpoint_snapshot.h"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/endpoint.h"
#include "libamdf/src/platform/windows/endpoint.h"
#include "libamdf/src/platform/windows/instance.h"

namespace {

constexpr NTSTATUS kSuccess = 0;
constexpr NTSTATUS kFailure = static_cast<NTSTATUS>(0xC0000001u);
constexpr NTSTATUS kCloseFailure = static_cast<NTSTATUS>(0xC000000Du);
constexpr D3DKMT_HANDLE kAmdAdapter = 1;
constexpr LONG kAmdAdapterLuidHigh = static_cast<LONG>(0x87654321u);
constexpr D3DKMT_HANDLE kOtherAdapter = 2;

amdf_endpoint_summary_t MakeSentinelSummary() {
  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  return summary;
}

enum class FailurePoint {
  kNone,
  kOpen,
  kOpenWithHandle,
  kFirstEnumeration,
  kSecondEnumeration,
  kPhysicalAdapterCount,
  kDeviceIds,
  kAdapterType,
  kDescription,
  kClose,
};

struct HostAllocationState {
  // Backing allocator used by the counted callbacks.
  amdf_allocator_t system = amdf_allocator_system();
  // Host allocations still owned by the implementation under test.
  uint32_t live_count = 0;

  static void* AMDF_CALL Allocate(void* user_data, uint64_t byte_length,
                                  uint64_t minimum_alignment) {
    auto* state = static_cast<HostAllocationState*>(user_data);
    void* pointer = state->system.allocate(state->system.user_data, byte_length,
                                           minimum_alignment);
    if (pointer != nullptr) {
      ++state->live_count;
    }
    return pointer;
  }

  static void AMDF_CALL Free(void* user_data, void* pointer) {
    auto* state = static_cast<HostAllocationState*>(user_data);
    if (pointer == nullptr) {
      return;
    }
    --state->live_count;
    state->system.free(state->system.user_data, pointer);
  }

  amdf_allocator_t MakeAllocator() {
    return {.user_data = this, .allocate = Allocate, .free = Free};
  }
};

struct FakeKmt {
  // Native operation whose failure is injected by the test.
  FailurePoint failure_point = FailurePoint::kNone;
  // Number of direct adapter-open calls.
  uint32_t open_call_count = 0;
  // Number of physical adapters reported for the AMD adapter handle.
  uint32_t amd_physical_adapter_count = 1;
  // PCI vendor used for endpoint classification.
  uint32_t amd_vendor_id = 0x1002u;
  // PCI device identity, shared by every AMD physical adapter in the fixture.
  uint32_t amd_device_id = 0x1000u;
  // PCI revision used for endpoint classification.
  uint32_t amd_revision_id = 1;
  // Native adapter capability distinguishing compute-only AMD endpoints.
  bool amd_compute_only = false;
  // Number of calls in the native two-call snapshot protocol.
  uint32_t enumeration_call_count = 0;
  // Native release fault and consumption evidence.
  struct {
    // Number of attempted adapter closes, including failures.
    uint32_t call_count = 0;
    // Adapter whose close fails independently of other injected failures.
    D3DKMT_HANDLE blocked_adapter = 0;
    // Successful closes per native adapter handle.
    std::array<uint32_t, 3> successful = {};
  } close;
};

FakeKmt* current_fake = nullptr;

NTSTATUS APIENTRY
FakeOpenAdapterFromLuid(D3DKMT_OPENADAPTERFROMLUID* open_adapter) {
  ++current_fake->open_call_count;
  if (open_adapter->AdapterLuid.LowPart != kAmdAdapter ||
      open_adapter->AdapterLuid.HighPart != kAmdAdapterLuidHigh) {
    return kFailure;
  }
  if (current_fake->failure_point == FailurePoint::kOpen) {
    return kFailure;
  }
  open_adapter->hAdapter = kAmdAdapter;
  return current_fake->failure_point == FailurePoint::kOpenWithHandle
             ? kFailure
             : kSuccess;
}

NTSTATUS APIENTRY FakeEnumerateAdapters(D3DKMT_ENUMADAPTERS3* enumeration) {
  ++current_fake->enumeration_call_count;
  const bool include_amd_adapter =
      !current_fake->amd_compute_only || enumeration->Filter.IncludeComputeOnly;
  const uint32_t adapter_count = include_amd_adapter ? 2 : 1;
  if (current_fake->enumeration_call_count == 1) {
    if (current_fake->failure_point == FailurePoint::kFirstEnumeration) {
      return kFailure;
    }
    enumeration->NumAdapters = adapter_count;
    return kSuccess;
  }

  if (enumeration->NumAdapters < adapter_count ||
      enumeration->pAdapters == nullptr) {
    return kFailure;
  }
  enumeration->NumAdapters = adapter_count;
  if (include_amd_adapter) {
    enumeration->pAdapters[0].hAdapter = kAmdAdapter;
    enumeration->pAdapters[0].AdapterLuid.LowPart = kAmdAdapter;
    enumeration->pAdapters[0].AdapterLuid.HighPart = kAmdAdapterLuidHigh;
  }
  enumeration->pAdapters[adapter_count - 1].hAdapter = kOtherAdapter;
  enumeration->pAdapters[adapter_count - 1].AdapterLuid.LowPart = kOtherAdapter;
  if (current_fake->failure_point == FailurePoint::kSecondEnumeration) {
    return kFailure;
  }
  return kSuccess;
}

NTSTATUS APIENTRY FakeQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  switch (query->Type) {
    case KMTQAITYPE_PHYSICALADAPTERCOUNT: {
      if (current_fake->failure_point == FailurePoint::kPhysicalAdapterCount) {
        return kFailure;
      }
      auto* count = static_cast<D3DKMT_PHYSICAL_ADAPTER_COUNT*>(
          query->pPrivateDriverData);
      count->Count = query->hAdapter == kAmdAdapter
                         ? current_fake->amd_physical_adapter_count
                         : 1;
      return kSuccess;
    }
    case KMTQAITYPE_PHYSICALADAPTERDEVICEIDS: {
      if (current_fake->failure_point == FailurePoint::kDeviceIds) {
        return kFailure;
      }
      auto* ids =
          static_cast<D3DKMT_QUERY_DEVICE_IDS*>(query->pPrivateDriverData);
      ids->DeviceIds.VendorID = query->hAdapter == kAmdAdapter
                                    ? current_fake->amd_vendor_id
                                    : 0x8086u;
      ids->DeviceIds.DeviceID = query->hAdapter == kAmdAdapter
                                    ? current_fake->amd_device_id
                                    : 0x1000u + ids->PhysicalAdapterIndex;
      ids->DeviceIds.SubVendorID = ids->DeviceIds.VendorID;
      ids->DeviceIds.SubSystemID = 0x2000u;
      ids->DeviceIds.RevisionID =
          query->hAdapter == kAmdAdapter ? current_fake->amd_revision_id : 1;
      ids->DeviceIds.BusType = 1;
      return kSuccess;
    }
    case KMTQAITYPE_ADAPTERTYPE: {
      if (current_fake->failure_point == FailurePoint::kAdapterType) {
        return kFailure;
      }
      auto* type = static_cast<D3DKMT_ADAPTERTYPE*>(query->pPrivateDriverData);
      type->Value = 0;
      type->ComputeOnly =
          query->hAdapter == kAmdAdapter && current_fake->amd_compute_only;
      type->RenderSupported = !type->ComputeOnly;
      return kSuccess;
    }
    case KMTQAITYPE_DRIVER_DESCRIPTION: {
      if (current_fake->failure_point == FailurePoint::kDescription) {
        return kFailure;
      }
      auto* description =
          static_cast<D3DKMT_DRIVER_DESCRIPTION*>(query->pPrivateDriverData);
      static constexpr wchar_t kName[] = L"Fake adapter";
      std::memcpy(description->DriverDescription, kName, sizeof(kName));
      return kSuccess;
    }
    default:
      return kFailure;
  }
}

NTSTATUS APIENTRY FakeCloseAdapter(const D3DKMT_CLOSEADAPTER* close) {
  EXPECT_TRUE(close->hAdapter == kAmdAdapter ||
              close->hAdapter == kOtherAdapter);
  ++current_fake->close.call_count;
  if (current_fake->failure_point == FailurePoint::kClose ||
      close->hAdapter == current_fake->close.blocked_adapter) {
    return kCloseFailure;
  }
  ++current_fake->close.successful[close->hAdapter];
  return kSuccess;
}

class EndpointSnapshotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    current_fake = &fake_;
    const amdf_allocator_t host_allocator = host_allocations_.MakeAllocator();
    ASSERT_EQ(amdf_calloc(host_allocator, sizeof(*platform_instance_),
                          amdf_alignof(amdf_platform_instance_t),
                          reinterpret_cast<void**>(&platform_instance_)),
              AMDF_STATUS_OK);
    platform_instance_->host_allocator = host_allocator;
    platform_instance_->kmt.enumerate_adapters = FakeEnumerateAdapters;
    platform_instance_->kmt.open_adapter_from_luid = FakeOpenAdapterFromLuid;
    platform_instance_->kmt.query_adapter_info = FakeQueryAdapterInfo;
    platform_instance_->kmt.close_adapter = FakeCloseAdapter;
  }

  void TearDown() override {
    if (platform_instance_ != nullptr) {
      EXPECT_TRUE(amdf_status_is_ok(DestroyInstance()));
    }
    EXPECT_EQ(host_allocations_.live_count, 0u);
    current_fake = nullptr;
  }

  amdf_status_t DestroyInstance() {
    const amdf_status_t status =
        amdf_platform_instance_destroy(platform_instance_);
    if (amdf_status_is_ok(status)) {
      platform_instance_ = nullptr;
    }
    return status;
  }

  amdf_endpoint_id_t EnumerateAmdEndpointId() {
    amdf_endpoint_summary_t summary = {};
    uint32_t endpoint_count = 0;
    const amdf_status_t status = amdf_platform_endpoint_enumerate(
        platform_instance_, 1, &summary, &endpoint_count);
    EXPECT_TRUE(amdf_status_is_ok(status));
    EXPECT_EQ(endpoint_count, 1u);
    fake_.enumeration_call_count = 0;
    fake_.close.call_count = 0;
    return summary.id;
  }

  // Injected KMT operations and native ownership accounting.
  FakeKmt fake_;
  // Host bookkeeping must be released even when a native close fails.
  HostAllocationState host_allocations_;
  // Real platform owner containing only the injected native procedure table.
  amdf_platform_instance_t* platform_instance_ = nullptr;
};

TEST_F(EndpointSnapshotTest, ReturnsOnlyAmdEndpointsAndClosesSnapshot) {
  amdf_endpoint_summary_t summaries[2] = {};
  uint32_t endpoint_count = 0;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 2, summaries, &endpoint_count);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_EQ(endpoint_count, 1u);
  EXPECT_STREQ(summaries[0].name, "Fake adapter");
  EXPECT_EQ(summaries[0].engine_kind, AMDF_ENGINE_KIND_GPU);
  EXPECT_EQ(fake_.enumeration_call_count, 2u);
  EXPECT_EQ(fake_.close.call_count, 2u);
}

TEST_F(EndpointSnapshotTest,
       MissingDiscoveryProcedureRejectsEnumerationAndDirectOpen) {
  enum class MissingProcedure {
    kEnumerateAdapters,
    kOpenAdapterFromLuid,
    kQueryAdapterInfo,
    kCloseAdapter,
  };
  const amdf_kmt_api_t complete_api = platform_instance_->kmt;
  for (MissingProcedure missing : {
           MissingProcedure::kEnumerateAdapters,
           MissingProcedure::kOpenAdapterFromLuid,
           MissingProcedure::kQueryAdapterInfo,
           MissingProcedure::kCloseAdapter,
       }) {
    fake_ = {};
    platform_instance_->kmt = complete_api;
    switch (missing) {
      case MissingProcedure::kEnumerateAdapters:
        platform_instance_->kmt.enumerate_adapters = nullptr;
        break;
      case MissingProcedure::kOpenAdapterFromLuid:
        platform_instance_->kmt.open_adapter_from_luid = nullptr;
        break;
      case MissingProcedure::kQueryAdapterInfo:
        platform_instance_->kmt.query_adapter_info = nullptr;
        break;
      case MissingProcedure::kCloseAdapter:
        platform_instance_->kmt.close_adapter = nullptr;
        break;
    }

    amdf_endpoint_summary_t summary = MakeSentinelSummary();
    const amdf_endpoint_summary_t expected_summary = summary;
    uint32_t endpoint_count = 123;
    EXPECT_EQ(amdf_status_code(amdf_platform_endpoint_enumerate(
                  platform_instance_, 1, &summary, &endpoint_count)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
    EXPECT_EQ(endpoint_count, 123u);

    const amdf_endpoint_id_t id = {};
    amdf_platform_endpoint_t* endpoint =
        reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
    amdf_endpoint_info_t info = {};
    EXPECT_EQ(amdf_status_code(amdf_platform_endpoint_open(
                  platform_instance_, &id, &endpoint, &info)),
              AMDF_STATUS_CODE_UNSUPPORTED);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
    EXPECT_EQ(fake_.enumeration_call_count, 0u);
    EXPECT_EQ(fake_.open_call_count, 0u);
    EXPECT_EQ(fake_.close.call_count, 0u);
  }
  platform_instance_->kmt = complete_api;
}

TEST_F(EndpointSnapshotTest, ClassifiesKnownXdnaEndpoint) {
  fake_.amd_vendor_id = 0x1022u;
  fake_.amd_device_id = 0x17F0u;
  fake_.amd_revision_id = 0x10u;
  fake_.amd_compute_only = true;
  amdf_endpoint_summary_t summary = {};
  uint32_t endpoint_count = 0;

  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 1, &summary, &endpoint_count);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_EQ(endpoint_count, 1u);
  EXPECT_EQ(summary.engine_kind, AMDF_ENGINE_KIND_XDNA);
}

TEST_F(EndpointSnapshotTest, ClassifiesUnrecognizedComputeOnlyIdentity) {
  fake_.amd_vendor_id = 0x1022u;
  fake_.amd_device_id = 0xABCDu;
  fake_.amd_revision_id = 0x12u;
  fake_.amd_compute_only = true;
  amdf_endpoint_summary_t summary = {};
  uint32_t endpoint_count = 0;

  ASSERT_EQ(amdf_platform_endpoint_enumerate(platform_instance_, 1, &summary,
                                             &endpoint_count),
            AMDF_STATUS_OK);
  ASSERT_EQ(endpoint_count, 1u);
  EXPECT_EQ(summary.engine_kind, AMDF_ENGINE_KIND_XDNA);
}

TEST_F(EndpointSnapshotTest, LeavesNonComputeAmdIdentityUnclassified) {
  fake_.amd_vendor_id = 0x1022u;
  fake_.amd_device_id = 0x17F0u;
  fake_.amd_revision_id = 0x12u;
  amdf_endpoint_summary_t summary = {};
  uint32_t endpoint_count = 0;

  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 1, &summary, &endpoint_count);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_EQ(endpoint_count, 1u);
  EXPECT_EQ(summary.engine_kind, AMDF_ENGINE_KIND_UNKNOWN);
}

TEST_F(EndpointSnapshotTest, CountsWithoutOutputStorage) {
  uint32_t endpoint_count = 0;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 0, nullptr, &endpoint_count);

  EXPECT_TRUE(amdf_status_is_ok(status));
  EXPECT_EQ(endpoint_count, 1u);
  EXPECT_EQ(fake_.close.call_count, 2u);
}

TEST_F(EndpointSnapshotTest, EmitsDistinctPhysicalAdapterEndpoints) {
  fake_.amd_physical_adapter_count = 2;
  amdf_endpoint_summary_t summaries[2] = {};
  uint32_t endpoint_count = 0;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 2, summaries, &endpoint_count);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_EQ(endpoint_count, 2u);
  EXPECT_FALSE(amdf_endpoint_id_is_equal(&summaries[0].id, &summaries[1].id));
  EXPECT_EQ(fake_.close.call_count, 2u);
  for (uint32_t i = 0; i < endpoint_count; ++i) {
    amdf_platform_endpoint_t* endpoint = nullptr;
    amdf_endpoint_info_t info = {};
    ASSERT_EQ(amdf_platform_endpoint_open(platform_instance_, &summaries[i].id,
                                          &endpoint, &info),
              AMDF_STATUS_OK);
    EXPECT_EQ(info.pci.device_id, fake_.amd_device_id);
    EXPECT_EQ(info.native_identity.type,
              AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER);
    EXPECT_EQ(info.native_identity.value.windows_adapter.luid,
              UINT64_C(0x8765432100000001));
    EXPECT_EQ(info.native_identity.value.windows_adapter.physical_adapter_index,
              i);
    EXPECT_EQ(amdf_platform_endpoint_close(endpoint), AMDF_STATUS_OK);
  }
  EXPECT_EQ(fake_.close.call_count, 4u);
}

TEST_F(EndpointSnapshotTest, WritesAvailablePrefixAndReportsTotal) {
  fake_.amd_physical_adapter_count = 2;
  amdf_endpoint_summary_t summary = {};
  uint32_t endpoint_count = 0;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 1, &summary, &endpoint_count);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_BUFFER_TOO_SMALL);
  EXPECT_EQ(endpoint_count, 2u);
  EXPECT_STREQ(summary.name, "Fake adapter");
  EXPECT_EQ(fake_.close.call_count, 2u);
}

TEST_F(EndpointSnapshotTest, ClosesReturnedHandlesWhenEnumerationFails) {
  fake_.failure_point = FailurePoint::kSecondEnumeration;
  amdf_endpoint_summary_t summary = MakeSentinelSummary();
  const amdf_endpoint_summary_t expected_summary = summary;
  uint32_t endpoint_count = 123;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 1, &summary, &endpoint_count);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
  EXPECT_EQ(fake_.close.call_count, 2u);
}

TEST_F(EndpointSnapshotTest, ReturnsInitialEnumerationFailureWithoutCleanup) {
  fake_.failure_point = FailurePoint::kFirstEnumeration;
  uint32_t endpoint_count = 123;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 0, nullptr, &endpoint_count);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(fake_.close.call_count, 0u);
}

TEST_F(EndpointSnapshotTest, ClosesEveryHandleAfterQueryFailures) {
  const FailurePoint failure_points[] = {
      FailurePoint::kPhysicalAdapterCount,
      FailurePoint::kDeviceIds,
      FailurePoint::kAdapterType,
      FailurePoint::kDescription,
  };
  for (FailurePoint failure_point : failure_points) {
    fake_ = {};
    fake_.failure_point = failure_point;
    amdf_endpoint_summary_t summary = MakeSentinelSummary();
    const amdf_endpoint_summary_t expected_summary = summary;
    uint32_t endpoint_count = 123;
    const amdf_status_t status = amdf_platform_endpoint_enumerate(
        platform_instance_, 1, &summary, &endpoint_count);
    EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
    EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
    EXPECT_EQ(endpoint_count, 123u);
    EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
    EXPECT_EQ(fake_.close.call_count, 2u);
  }
}

TEST_F(EndpointSnapshotTest, SurfacesCloseFailureAfterClosingEveryHandle) {
  fake_.failure_point = FailurePoint::kClose;
  amdf_endpoint_summary_t summary = MakeSentinelSummary();
  const amdf_endpoint_summary_t expected_summary = summary;
  uint32_t endpoint_count = 123;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 1, &summary, &endpoint_count);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kCloseFailure));
  EXPECT_EQ(endpoint_count, 123u);
  EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
  EXPECT_EQ(fake_.close.call_count, 2u);
  EXPECT_EQ(host_allocations_.live_count, 1u);
}

TEST_F(EndpointSnapshotTest, OpensIdentityDirectlyAndCachesProperties) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  amdf_platform_endpoint_t* endpoint = nullptr;
  amdf_endpoint_info_t info = {};
  const amdf_status_t status =
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info);

  EXPECT_TRUE(amdf_status_is_ok(status));
  ASSERT_NE(endpoint, nullptr);
  EXPECT_TRUE(amdf_endpoint_id_is_equal(&id, &info.id));
  EXPECT_EQ(fake_.enumeration_call_count, 0u);
  EXPECT_EQ(fake_.close.call_count, 0u);
  EXPECT_TRUE(amdf_status_is_ok(amdf_platform_endpoint_close(endpoint)));
  EXPECT_EQ(fake_.close.call_count, 1u);
}

TEST_F(EndpointSnapshotTest, RejectsStaleIdentityAndClosesOpenedAdapter) {
  amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  id.words[1] ^= UINT64_C(1) << 63;
  amdf_platform_endpoint_t* endpoint =
      reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_info_t info = {};
  const amdf_status_t status =
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_API);
  EXPECT_EQ(amdf_status_code(status), AMDF_STATUS_CODE_NOT_FOUND);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
  EXPECT_EQ(fake_.enumeration_call_count, 0u);
  EXPECT_EQ(fake_.close.call_count, 1u);
}

TEST_F(EndpointSnapshotTest, ClosesHandleReturnedByFailedOpen) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  fake_.failure_point = FailurePoint::kOpenWithHandle;
  amdf_platform_endpoint_t* endpoint =
      reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_info_t info = {};
  const amdf_status_t status =
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
  EXPECT_EQ(fake_.close.call_count, 1u);
}

TEST_F(EndpointSnapshotTest, ReturnsOpenFailureWithoutCleanup) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  fake_.failure_point = FailurePoint::kOpen;
  amdf_platform_endpoint_t* endpoint =
      reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_info_t info = {};
  const amdf_status_t status =
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
  EXPECT_EQ(fake_.close.call_count, 0u);
}

TEST_F(EndpointSnapshotTest, ClosesOpenedAdapterAfterQueryFailure) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  fake_.failure_point = FailurePoint::kAdapterType;
  amdf_platform_endpoint_t* endpoint =
      reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_info_t info = {};
  const amdf_status_t status =
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info);

  EXPECT_EQ(amdf_status_domain(status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(status), static_cast<uint32_t>(kFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
  EXPECT_EQ(fake_.close.call_count, 1u);
}

TEST_F(EndpointSnapshotTest, LeavesEndpointLiveWhenCloseFails) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  amdf_platform_endpoint_t* endpoint = nullptr;
  amdf_endpoint_info_t info = {};
  ASSERT_TRUE(amdf_status_is_ok(
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info)));
  ASSERT_NE(endpoint, nullptr);

  fake_.failure_point = FailurePoint::kClose;
  const amdf_status_t close_status = amdf_platform_endpoint_close(endpoint);
  EXPECT_EQ(amdf_status_domain(close_status), AMDF_STATUS_DOMAIN_NTSTATUS);
  EXPECT_EQ(amdf_status_code(close_status),
            static_cast<uint32_t>(kCloseFailure));

  fake_.failure_point = FailurePoint::kNone;
  EXPECT_TRUE(amdf_status_is_ok(amdf_platform_endpoint_close(endpoint)));
  EXPECT_EQ(fake_.close.call_count, 2u);
}

TEST_F(EndpointSnapshotTest,
       ReportsFailedSnapshotCloseWithoutRetainingStorage) {
  fake_.close.blocked_adapter = kAmdAdapter;
  uint32_t endpoint_count = 37;
  const amdf_status_t status = amdf_platform_endpoint_enumerate(
      platform_instance_, 0, nullptr, &endpoint_count);
  EXPECT_EQ(status, amdf_kmt_make_status(kCloseFailure));
  EXPECT_EQ(endpoint_count, 37u);
  EXPECT_EQ(fake_.close.call_count, 2u);
  EXPECT_EQ(fake_.close.successful[kAmdAdapter], 0u);
  EXPECT_EQ(fake_.close.successful[kOtherAdapter], 1u);

  EXPECT_EQ(host_allocations_.live_count, 1u);
  ASSERT_EQ(DestroyInstance(), AMDF_STATUS_OK);
  EXPECT_EQ(host_allocations_.live_count, 0u);
  EXPECT_EQ(fake_.close.call_count, 2u);
  EXPECT_EQ(fake_.close.successful[kAmdAdapter], 0u);
  EXPECT_EQ(fake_.close.successful[kOtherAdapter], 1u);
}

TEST_F(EndpointSnapshotTest, ReportsCleanupFailureAfterEnumerationFailure) {
  fake_.failure_point = FailurePoint::kSecondEnumeration;
  fake_.close.blocked_adapter = kAmdAdapter;
  uint32_t endpoint_count = 37;
  EXPECT_EQ(amdf_platform_endpoint_enumerate(platform_instance_, 0, nullptr,
                                             &endpoint_count),
            amdf_kmt_make_status(kCloseFailure));
  EXPECT_EQ(endpoint_count, 37u);
  EXPECT_EQ(fake_.close.call_count, 2u);
  EXPECT_EQ(fake_.close.successful[kOtherAdapter], 1u);
  EXPECT_EQ(host_allocations_.live_count, 1u);
  ASSERT_EQ(DestroyInstance(), AMDF_STATUS_OK);
  EXPECT_EQ(host_allocations_.live_count, 0u);
  EXPECT_EQ(fake_.close.call_count, 2u);
  EXPECT_EQ(fake_.close.successful[kAmdAdapter], 0u);
  EXPECT_EQ(fake_.close.successful[kOtherAdapter], 1u);
}

TEST_F(EndpointSnapshotTest, ReportsEndpointRollbackFailureWithoutRetention) {
  const amdf_endpoint_id_t id = EnumerateAmdEndpointId();
  fake_.failure_point = FailurePoint::kAdapterType;
  fake_.close.blocked_adapter = kAmdAdapter;
  fake_.close.successful = {};
  amdf_platform_endpoint_t* endpoint =
      reinterpret_cast<amdf_platform_endpoint_t*>(uintptr_t{1});
  amdf_endpoint_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_endpoint_info_t original_info = info;
  EXPECT_EQ(
      amdf_platform_endpoint_open(platform_instance_, &id, &endpoint, &info),
      amdf_kmt_make_status(kCloseFailure));
  EXPECT_EQ(reinterpret_cast<uintptr_t>(endpoint), uintptr_t{1});
  EXPECT_EQ(std::memcmp(&info, &original_info, sizeof(info)), 0);
  EXPECT_EQ(fake_.close.call_count, 1u);
  EXPECT_EQ(fake_.close.successful[kAmdAdapter], 0u);

  EXPECT_EQ(host_allocations_.live_count, 1u);
  ASSERT_EQ(DestroyInstance(), AMDF_STATUS_OK);
  EXPECT_EQ(host_allocations_.live_count, 0u);
  EXPECT_EQ(fake_.close.call_count, 1u);
  EXPECT_EQ(fake_.close.successful[kAmdAdapter], 0u);
}

}  // namespace

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
