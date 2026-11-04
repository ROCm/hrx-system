// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "libamdf/src/platform/linux/dxg/endpoint.h"

#include <sys/ioctl.h>

#include <cstdint>
#include <cstring>

#include "gtest/gtest.h"
#include "libamdf/src/allocator.h"
#include "libamdf/src/platform/dxg/endpoint_identity.h"

namespace {

constexpr D3DKMT_HANDLE kAmdAdapter = 11;
constexpr D3DKMT_HANDLE kOtherAdapter = 12;
constexpr uint64_t kAmdLuid = UINT64_C(0x1234567887654321);
constexpr uint64_t kOtherLuid = UINT64_C(0xABCDEF0101020304);
constexpr uint32_t kStatusFailure = 0xC0000001u;
constexpr uint32_t kStatusCloseFailure = 0xC000000Du;

enum class FailurePoint {
  kNone,
  kOpenWithHandle,
  kDeviceIds,
  kCloseAmdAdapter,
};

LUID DecodeLuid(uint64_t value) {
  return {
      .LowPart = static_cast<DWORD>(value),
      .HighPart = static_cast<LONG>(static_cast<uint32_t>(value >> 32)),
  };
}

uint64_t EncodeLuid(LUID luid) {
  return (static_cast<uint64_t>(static_cast<uint32_t>(luid.HighPart)) << 32) |
         luid.LowPart;
}

amdf_dxg_device_ids_t MakeDeviceIds(uint32_t physical_adapter_index) {
  return {
      .vendor_id = 0x1002,
      .device_id = 0x744C + physical_adapter_index,
      .subsystem_vendor_id = 0x1DA2,
      .subsystem_device_id = 0xE471,
      .revision_id = 0xC8,
      .bus_type = 0x12,
  };
}

struct FakeDxg {
  // Native operation whose failure is injected by the test.
  FailurePoint failure_point = FailurePoint::kNone;
  // Number of calls in the native two-call enumeration protocol.
  uint32_t enumeration_call_count = 0;
  // Number of adapters released by discovery or endpoint teardown.
  uint32_t close_call_count = 0;
  // Number of direct open-by-LUID calls.
  uint32_t open_call_count = 0;
  // Most recent LUID supplied to direct endpoint opening.
  uint64_t opened_luid = 0;

  uint32_t Invoke(uint64_t request, void* arguments) {
    switch (_IOC_NR(request)) {
      case 0x01: {
        ++open_call_count;
        auto* open = static_cast<D3DKMT_OPENADAPTERFROMLUID*>(arguments);
        opened_luid = EncodeLuid(open->AdapterLuid);
        if (opened_luid != kAmdLuid) {
          return 0xC0000034u;  // STATUS_OBJECT_NAME_NOT_FOUND
        }
        open->hAdapter = kAmdAdapter;
        return failure_point == FailurePoint::kOpenWithHandle ? kStatusFailure
                                                              : 0;
      }
      case 0x09: {
        auto* query = static_cast<D3DKMT_QUERYADAPTERINFO*>(arguments);
        switch (query->Type) {
          case KMTQAITYPE_PHYSICALADAPTERCOUNT: {
            EXPECT_EQ(query->PrivateDriverDataSize,
                      sizeof(D3DKMT_PHYSICAL_ADAPTER_COUNT));
            auto* count = static_cast<D3DKMT_PHYSICAL_ADAPTER_COUNT*>(
                query->pPrivateDriverData);
            count->Count = query->hAdapter == kAmdAdapter ? 2 : 1;
            return 0;
          }
          case KMTQAITYPE_PHYSICALADAPTERDEVICEIDS: {
            if (failure_point == FailurePoint::kDeviceIds) {
              return kStatusFailure;
            }
            EXPECT_EQ(query->PrivateDriverDataSize,
                      sizeof(D3DKMT_QUERY_DEVICE_IDS));
            auto* ids = static_cast<D3DKMT_QUERY_DEVICE_IDS*>(
                query->pPrivateDriverData);
            const amdf_dxg_device_ids_t values =
                MakeDeviceIds(ids->PhysicalAdapterIndex);
            ids->DeviceIds.VendorID =
                query->hAdapter == kAmdAdapter ? values.vendor_id : 0x10DE;
            ids->DeviceIds.DeviceID = values.device_id;
            ids->DeviceIds.SubVendorID = values.subsystem_vendor_id;
            ids->DeviceIds.SubSystemID = values.subsystem_device_id;
            ids->DeviceIds.RevisionID = values.revision_id;
            ids->DeviceIds.BusType = values.bus_type;
            return 0;
          }
          case KMTQAITYPE_ADAPTERTYPE: {
            EXPECT_EQ(query->PrivateDriverDataSize, sizeof(D3DKMT_ADAPTERTYPE));
            auto* type =
                static_cast<D3DKMT_ADAPTERTYPE*>(query->pPrivateDriverData);
            type->RenderSupported = 1;
            return 0;
          }
          default:
            ADD_FAILURE() << "unexpected query type " << query->Type;
            return 0xC000000Du;  // STATUS_INVALID_PARAMETER
        }
      }
      case 0x14: {
        ++enumeration_call_count;
        auto* enumeration = static_cast<D3DKMT_ENUMADAPTERS2*>(arguments);
        if (enumeration->pAdapters == nullptr) {
          enumeration->NumAdapters = 2;
          return 0;
        }
        EXPECT_EQ(enumeration->NumAdapters, 2u);
        enumeration->pAdapters[0] = {
            .hAdapter = kAmdAdapter,
            .AdapterLuid = DecodeLuid(kAmdLuid),
        };
        enumeration->pAdapters[1] = {
            .hAdapter = kOtherAdapter,
            .AdapterLuid = DecodeLuid(kOtherLuid),
        };
        enumeration->NumAdapters = 2;
        return 0;
      }
      case 0x15: {
        ++close_call_count;
        auto* close = static_cast<D3DKMT_CLOSEADAPTER*>(arguments);
        return failure_point == FailurePoint::kCloseAmdAdapter &&
                       close->hAdapter == kAmdAdapter
                   ? kStatusCloseFailure
                   : 0;
      }
      default:
        ADD_FAILURE() << "unexpected DXG operation " << _IOC_NR(request);
        return 0xC00000BBu;  // STATUS_NOT_SUPPORTED
    }
  }
};

FakeDxg* fake_dxg = nullptr;
const amdf_linux_dxg_connection_t* expected_connection = nullptr;

class DxgEndpointTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fake_dxg = &native_;
    instance_.host_allocator = amdf_allocator_system();
    instance_.connection.descriptor = 17;
    instance_.kmt.connection = &instance_.connection;
    instance_.kmt.host_allocator = &instance_.host_allocator;
    expected_connection = &instance_.connection;
  }

  void TearDown() override {
    if (endpoint_ != nullptr) {
      EXPECT_EQ(amdf_platform_endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    expected_connection = nullptr;
    fake_dxg = nullptr;
  }

  // Platform instance carrying the explicit synthetic DXG connection.
  amdf_platform_instance_t instance_ = {};
  // Native call model used by the production endpoint implementation.
  FakeDxg native_;
  // Published endpoint retained by tests until teardown.
  amdf_platform_endpoint_t* endpoint_ = nullptr;
};

TEST_F(DxgEndpointTest, EnumeratesPhysicalAmdGpuEndpointsAndClosesSnapshot) {
  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  uint32_t count = UINT32_MAX;
  EXPECT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            amdf_make_api_status(AMDF_STATUS_CODE_BUFFER_TOO_SMALL));
  EXPECT_EQ(count, 2u);
  EXPECT_EQ(summary.engine_kind, AMDF_ENGINE_KIND_GPU);
  EXPECT_STREQ(summary.name, "AMD GPU 1002:744c:c8 (DXG 1234567887654321/0)");
  EXPECT_EQ(native_.enumeration_call_count, 2u);
  EXPECT_EQ(native_.close_call_count, 2u);
}

TEST_F(DxgEndpointTest, OpensAndRevalidatesOnePhysicalAdapter) {
  const amdf_dxg_device_ids_t device_ids = MakeDeviceIds(1);
  const amdf_endpoint_id_t id =
      amdf_dxg_endpoint_id_encode(kAmdLuid, 1, &device_ids);
  amdf_endpoint_info_t info = {};
  ASSERT_EQ(amdf_platform_endpoint_open(&instance_, &id, &endpoint_, &info),
            AMDF_STATUS_OK);
  EXPECT_EQ(info.id.words[0], id.words[0]);
  EXPECT_EQ(info.id.words[1], id.words[1]);
  EXPECT_EQ(info.pci.device_id, 0x744Du);
  EXPECT_EQ(info.native_identity.type,
            AMDF_ENDPOINT_NATIVE_IDENTITY_TYPE_WINDOWS_ADAPTER);
  EXPECT_EQ(info.native_identity.value.windows_adapter.luid, kAmdLuid);
  EXPECT_EQ(info.native_identity.value.windows_adapter.physical_adapter_index,
            1u);
  EXPECT_EQ(native_.open_call_count, 1u);
  EXPECT_EQ(native_.opened_luid, kAmdLuid);
  EXPECT_EQ(amdf_platform_endpoint_host_numa_node(endpoint_), UINT32_MAX);
  EXPECT_EQ(amdf_platform_endpoint_pci_address(endpoint_)->available, 0u);
  EXPECT_EQ(amdf_platform_endpoint_query_queue_publication_modes(
                endpoint_, AMDF_QUEUE_COMMAND_TYPE_GPU_PM4),
            AMDF_QUEUE_PUBLICATION_MODE_KERNEL);
}

TEST_F(DxgEndpointTest, RejectsStaleIdentityWithoutPublishingOutput) {
  const amdf_dxg_device_ids_t device_ids = MakeDeviceIds(0);
  amdf_endpoint_id_t id = amdf_dxg_endpoint_id_encode(kAmdLuid, 0, &device_ids);
  id.words[1] ^= UINT64_C(1) << 32;
  auto* const sentinel = reinterpret_cast<amdf_platform_endpoint_t*>(1);
  amdf_platform_endpoint_t* endpoint = sentinel;
  amdf_endpoint_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  amdf_endpoint_info_t expected_info;
  std::memset(&expected_info, 0xA5, sizeof(expected_info));
  EXPECT_EQ(amdf_platform_endpoint_open(&instance_, &id, &endpoint, &info),
            amdf_make_api_status(AMDF_STATUS_CODE_NOT_FOUND));
  EXPECT_EQ(endpoint, sentinel);
  EXPECT_EQ(std::memcmp(&info, &expected_info, sizeof(info)), 0);
  EXPECT_EQ(native_.close_call_count, 1u);
}

TEST_F(DxgEndpointTest, QueryFailureLeavesOutputsUnchangedAndClosesSnapshot) {
  native_.failure_point = FailurePoint::kDeviceIds;
  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  const amdf_endpoint_summary_t expected_summary = summary;
  uint32_t count = 37;

  EXPECT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, kStatusFailure));
  EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
  EXPECT_EQ(count, 37u);
  EXPECT_EQ(native_.close_call_count, 2u);
}

TEST_F(DxgEndpointTest, CloseFailureStillAttemptsEverySnapshotHandle) {
  native_.failure_point = FailurePoint::kCloseAmdAdapter;
  amdf_endpoint_summary_t summary;
  std::memset(&summary, 0xA5, sizeof(summary));
  const amdf_endpoint_summary_t expected_summary = summary;
  uint32_t count = 37;

  EXPECT_EQ(amdf_platform_endpoint_enumerate(&instance_, 1, &summary, &count),
            amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, kStatusCloseFailure));
  EXPECT_EQ(std::memcmp(&summary, &expected_summary, sizeof(summary)), 0);
  EXPECT_EQ(count, 37u);
  EXPECT_EQ(native_.close_call_count, 2u);
}

TEST_F(DxgEndpointTest, FailedOpenWithHandleClosesUnpublishedAdapter) {
  const amdf_dxg_device_ids_t device_ids = MakeDeviceIds(0);
  const amdf_endpoint_id_t id =
      amdf_dxg_endpoint_id_encode(kAmdLuid, 0, &device_ids);
  native_.failure_point = FailurePoint::kOpenWithHandle;
  auto* const sentinel = reinterpret_cast<amdf_platform_endpoint_t*>(1);
  amdf_platform_endpoint_t* endpoint = sentinel;
  amdf_endpoint_info_t info;
  std::memset(&info, 0xA5, sizeof(info));
  const amdf_endpoint_info_t expected_info = info;

  EXPECT_EQ(amdf_platform_endpoint_open(&instance_, &id, &endpoint, &info),
            amdf_make_status(AMDF_STATUS_DOMAIN_NTSTATUS, kStatusFailure));
  EXPECT_EQ(endpoint, sentinel);
  EXPECT_EQ(std::memcmp(&info, &expected_info, sizeof(info)), 0);
  EXPECT_EQ(native_.close_call_count, 1u);
}

}  // namespace

extern "C" uint32_t __wrap_amdf_linux_dxg_connection_invoke(void* context,
                                                            uint64_t request,
                                                            void* arguments) {
  EXPECT_EQ(context, expected_connection);
  EXPECT_NE(fake_dxg, nullptr);
  return fake_dxg->Invoke(request, arguments);
}
