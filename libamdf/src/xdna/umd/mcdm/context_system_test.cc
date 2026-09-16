// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iomanip>
#include <vector>

#include "amdf/xdna.h"
#include "gtest/gtest.h"
#include "libamdf/src/instance.h"
#include "libamdf/src/platform/windows/endpoint.h"

namespace {

// Unmodified native procedures forwarded by this single-threaded fixture.
amdf_kmt_api_t native_kmt = {};

// Each observer preserves the real request, result and cleanup behavior. A
// failure names its native boundary even when constructor cleanup follows it.
#define AMDF_OBSERVE_KMT_CALL(field, name, argument_type)             \
  NTSTATUS APIENTRY Observe##name(argument_type argument) {           \
    const NTSTATUS status = native_kmt.field(argument);               \
    EXPECT_GE(status, 0) << #name " returned NTSTATUS 0x" << std::hex \
                         << static_cast<uint32_t>(status);            \
    return status;                                                    \
  }

NTSTATUS APIENTRY
ObserveD3DKMTQueryAdapterInfo(const D3DKMT_QUERYADAPTERINFO* query) {
  const NTSTATUS status = native_kmt.query_adapter_info(query);
  // The direct interface requests extended storage before writing its policy.
  const bool negotiating_size = query->Type == KMTQAITYPE_UMDRIVERPRIVATE &&
                                query->PrivateDriverDataSize == 8 &&
                                status == static_cast<NTSTATUS>(0xC0000023u);
  EXPECT_TRUE(status >= 0 || negotiating_size)
      << "D3DKMTQueryAdapterInfo returned NTSTATUS 0x" << std::hex
      << static_cast<uint32_t>(status);
  return status;
}

AMDF_OBSERVE_KMT_CALL(create_allocation, D3DKMTCreateAllocation2,
                      D3DKMT_CREATEALLOCATION*)
AMDF_OBSERVE_KMT_CALL(map_gpu_virtual_address, D3DKMTMapGpuVirtualAddress,
                      D3DDDI_MAPGPUVIRTUALADDRESS*)
AMDF_OBSERVE_KMT_CALL(make_resident, D3DKMTMakeResident, D3DDDI_MAKERESIDENT*)
AMDF_OBSERVE_KMT_CALL(wait_from_cpu, D3DKMTWaitForSynchronizationObjectFromCpu,
                      const D3DKMT_WAITFORSYNCHRONIZATIONOBJECTFROMCPU*)
AMDF_OBSERVE_KMT_CALL(lock, D3DKMTLock2, D3DKMT_LOCK2*)
AMDF_OBSERVE_KMT_CALL(create_context_virtual, D3DKMTCreateContextVirtual,
                      D3DKMT_CREATECONTEXTVIRTUAL*)
AMDF_OBSERVE_KMT_CALL(destroy_context, D3DKMTDestroyContext,
                      const D3DKMT_DESTROYCONTEXT*)
AMDF_OBSERVE_KMT_CALL(unlock, D3DKMTUnlock2, const D3DKMT_UNLOCK2*)
AMDF_OBSERVE_KMT_CALL(evict, D3DKMTEvict, D3DKMT_EVICT*)
AMDF_OBSERVE_KMT_CALL(free_gpu_virtual_address, D3DKMTFreeGpuVirtualAddress,
                      const D3DKMT_FREEGPUVIRTUALADDRESS*)
AMDF_OBSERVE_KMT_CALL(destroy_allocation, D3DKMTDestroyAllocation2,
                      const D3DKMT_DESTROYALLOCATION2*)

#undef AMDF_OBSERVE_KMT_CALL

class WindowsXdnaContextSystemTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(
        amdf_query_api(AMDF_ABI_VERSION_1, AMDF_ABI_VERSION_LATEST, &api_),
        AMDF_STATUS_OK);
    const void* extension = nullptr;
    ASSERT_EQ(api_->query_extension(
                  AMDF_EXTENSION_XDNA, AMDF_XDNA_EXTENSION_VERSION_1,
                  AMDF_XDNA_EXTENSION_VERSION_LATEST, &extension),
              AMDF_STATUS_OK);
    xdna_api_ = static_cast<const amdf_xdna_api_t*>(extension);
    amdf_instance_create_info_t instance_info = {};
    instance_info.type = AMDF_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.structure_size = sizeof(instance_info);
    instance_info.native_lifetime = AMDF_NATIVE_LIFETIME_INSTANCE;
    ASSERT_EQ(api_->instance_create(&instance_info, &instance_),
              AMDF_STATUS_OK);

    uint32_t count = 0;
    ASSERT_EQ(api_->endpoint_enumerate(instance_, 0, nullptr, &count),
              AMDF_STATUS_OK);
    std::vector<amdf_endpoint_summary_t> summaries(count);
    ASSERT_EQ(
        api_->endpoint_enumerate(instance_, count, summaries.data(), &count),
        AMDF_STATUS_OK);
    for (const auto& summary : summaries) {
      if (summary.engine_kind != AMDF_ENGINE_KIND_XDNA) continue;
      ASSERT_EQ(api_->endpoint_open(instance_, &summary.id, &endpoint_),
                AMDF_STATUS_OK);
      break;
    }
    ASSERT_NE(endpoint_, nullptr) << "required native XDNA endpoint is absent";

    amdf_xdna_device_create_info_t device_info = {};
    device_info.type = AMDF_STRUCTURE_TYPE_XDNA_DEVICE_CREATE_INFO;
    device_info.structure_size = sizeof(device_info);
    ASSERT_EQ(xdna_api_->device_create(endpoint_, &device_info, &device_),
              AMDF_STATUS_OK);
  }

  void TearDown() override {
    if (context_) {
      ASSERT_EQ(xdna_api_->context_destroy(context_), AMDF_STATUS_OK);
    }
    if (device_) {
      ASSERT_EQ(api_->device_destroy(device_), AMDF_STATUS_OK);
    }
    if (endpoint_) {
      ASSERT_EQ(api_->endpoint_close(endpoint_), AMDF_STATUS_OK);
    }
    if (instance_) {
      ASSERT_EQ(api_->instance_destroy(instance_), AMDF_STATUS_OK);
    }
    native_kmt = {};
  }

  // Core API borrowed from the statically linked provider.
  const amdf_api_t* api_ = nullptr;
  // XDNA extension borrowed from the same provider.
  const amdf_xdna_api_t* xdna_api_ = nullptr;
  // Case-owned instance; every native dependency lives within this owner.
  amdf_instance_t* instance_ = nullptr;
  // Case-owned endpoint retained through device destruction.
  amdf_endpoint_t* endpoint_ = nullptr;
  // Real native device retained through context destruction.
  amdf_device_t* device_ = nullptr;
  // Real context released by teardown, including after assertion failures.
  amdf_xdna_context_t* context_ = nullptr;
};

TEST_F(WindowsXdnaContextSystemTest, CreatesAndReleasesNativeContext) {
  // Observe only context ownership operations after real device activation.
  // Missing procedures stay missing so observation cannot invent support.
  auto& kmt = instance_->platform->kmt;
  native_kmt = kmt;
#define AMDF_INSTALL_KMT_OBSERVER(field, name) \
  if (kmt.field != nullptr) kmt.field = Observe##name;

  AMDF_INSTALL_KMT_OBSERVER(query_adapter_info, D3DKMTQueryAdapterInfo)
  AMDF_INSTALL_KMT_OBSERVER(create_allocation, D3DKMTCreateAllocation2)
  AMDF_INSTALL_KMT_OBSERVER(map_gpu_virtual_address, D3DKMTMapGpuVirtualAddress)
  AMDF_INSTALL_KMT_OBSERVER(make_resident, D3DKMTMakeResident)
  AMDF_INSTALL_KMT_OBSERVER(wait_from_cpu,
                            D3DKMTWaitForSynchronizationObjectFromCpu)
  AMDF_INSTALL_KMT_OBSERVER(lock, D3DKMTLock2)
  AMDF_INSTALL_KMT_OBSERVER(create_context_virtual, D3DKMTCreateContextVirtual)
  AMDF_INSTALL_KMT_OBSERVER(destroy_context, D3DKMTDestroyContext)
  AMDF_INSTALL_KMT_OBSERVER(unlock, D3DKMTUnlock2)
  AMDF_INSTALL_KMT_OBSERVER(evict, D3DKMTEvict)
  AMDF_INSTALL_KMT_OBSERVER(free_gpu_virtual_address,
                            D3DKMTFreeGpuVirtualAddress)
  AMDF_INSTALL_KMT_OBSERVER(destroy_allocation, D3DKMTDestroyAllocation2)

#undef AMDF_INSTALL_KMT_OBSERVER

  amdf_xdna_context_create_info_t create_info = {};
  create_info.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_CREATE_INFO;
  create_info.structure_size = sizeof(create_info);
  create_info.logical_column_count = 1;
  create_info.physical_column_origin = AMDF_XDNA_PHYSICAL_COLUMN_ORIGIN_ANY;
  create_info.acceptable_scheduling_modes =
      AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED;
  ASSERT_EQ(xdna_api_->context_create(device_, &create_info, &context_),
            AMDF_STATUS_OK);

  amdf_xdna_context_info_t info = {};
  info.type = AMDF_STRUCTURE_TYPE_XDNA_CONTEXT_INFO;
  info.structure_size = sizeof(info);
  ASSERT_EQ(xdna_api_->context_query_info(context_, &info), AMDF_STATUS_OK);
  EXPECT_EQ(info.logical_column_count, 1u);
  EXPECT_EQ(info.scheduling_mode, AMDF_XDNA_SCHEDULING_MODE_TIME_SLICED);
}

}  // namespace
