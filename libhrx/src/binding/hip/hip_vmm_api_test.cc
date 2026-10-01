// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

template <typename Cleanup>
class ScopeExit {
 public:
  explicit ScopeExit(Cleanup cleanup) : cleanup_(std::move(cleanup)) {}
  ~ScopeExit() { cleanup_(); }

  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

 private:
  // Cleanup operation run when the owning scope exits.
  Cleanup cleanup_;
};

template <typename Cleanup>
ScopeExit(Cleanup) -> ScopeExit<Cleanup>;

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipDeviceSynchronizeFn = hipError_t (*)();
using HipMemsetFn = hipError_t (*)(void* destination, int value, size_t size);
using HipMemcpyFn = hipError_t (*)(void* destination, const void* source,
                                   size_t size, hipMemcpyKind kind);
using HipMemAddressReserveFn = hipError_t (*)(void** pointer, size_t size,
                                              size_t alignment, void* address,
                                              unsigned long long flags);
using HipMemAddressFreeFn = hipError_t (*)(void* pointer, size_t size);
using HipMemCreateFn = hipError_t (*)(hipMemGenericAllocationHandle_t* handle,
                                      size_t size,
                                      const hipMemAllocationProp* properties,
                                      unsigned long long flags);
using HipMemReleaseFn = hipError_t (*)(hipMemGenericAllocationHandle_t handle);
using HipMemMapFn = hipError_t (*)(void* pointer, size_t size, size_t offset,
                                   hipMemGenericAllocationHandle_t handle,
                                   unsigned long long flags);
using HipMemUnmapFn = hipError_t (*)(void* pointer, size_t size);
using HipMemSetAccessFn = hipError_t (*)(void* pointer, size_t size,
                                         const hipMemAccessDesc* descriptors,
                                         size_t descriptor_count);
using HipMemGetAccessFn = hipError_t (*)(unsigned long long* flags,
                                         const hipMemLocation* location,
                                         void* pointer);
using HipMemGetAllocationGranularityFn =
    hipError_t (*)(size_t* granularity, const hipMemAllocationProp* properties,
                   hipMemAllocationGranularity_flags option);
using HipMemGetAllocationPropertiesFn = hipError_t (*)(
    hipMemAllocationProp* properties, hipMemGenericAllocationHandle_t handle);

TEST(HipVmmApiTest, UncachedHostAllocationUsesDefaultPhysicalPlacement) {
  hrx::hip::testing::HipDso dso;
  ASSERT_TRUE(dso.Open()) << dso.error();
  HipInitFn init = dso.Resolve<HipInitFn>("hipInit");
  HipGetDeviceFn get_device = dso.Resolve<HipGetDeviceFn>("hipGetDevice");
  HipDeviceSynchronizeFn device_synchronize =
      dso.Resolve<HipDeviceSynchronizeFn>("hipDeviceSynchronize");
  HipMemsetFn memset = dso.Resolve<HipMemsetFn>("hipMemset");
  HipMemcpyFn memcpy = dso.Resolve<HipMemcpyFn>("hipMemcpy");
  HipMemAddressReserveFn address_reserve =
      dso.Resolve<HipMemAddressReserveFn>("hipMemAddressReserve");
  HipMemAddressFreeFn address_free =
      dso.Resolve<HipMemAddressFreeFn>("hipMemAddressFree");
  HipMemCreateFn create = dso.Resolve<HipMemCreateFn>("hipMemCreate");
  HipMemReleaseFn release = dso.Resolve<HipMemReleaseFn>("hipMemRelease");
  HipMemMapFn map = dso.Resolve<HipMemMapFn>("hipMemMap");
  HipMemUnmapFn unmap = dso.Resolve<HipMemUnmapFn>("hipMemUnmap");
  HipMemSetAccessFn set_access =
      dso.Resolve<HipMemSetAccessFn>("hipMemSetAccess");
  HipMemGetAccessFn get_access =
      dso.Resolve<HipMemGetAccessFn>("hipMemGetAccess");
  HipMemGetAllocationGranularityFn get_granularity =
      dso.Resolve<HipMemGetAllocationGranularityFn>(
          "hipMemGetAllocationGranularity");
  HipMemGetAllocationPropertiesFn get_properties =
      dso.Resolve<HipMemGetAllocationPropertiesFn>(
          "hipMemGetAllocationPropertiesFromHandle");
  ASSERT_NE(nullptr, init) << dso.error();
  ASSERT_NE(nullptr, get_device) << dso.error();
  ASSERT_NE(nullptr, device_synchronize) << dso.error();
  ASSERT_NE(nullptr, memset) << dso.error();
  ASSERT_NE(nullptr, memcpy) << dso.error();
  ASSERT_NE(nullptr, address_reserve) << dso.error();
  ASSERT_NE(nullptr, address_free) << dso.error();
  ASSERT_NE(nullptr, create) << dso.error();
  ASSERT_NE(nullptr, release) << dso.error();
  ASSERT_NE(nullptr, map) << dso.error();
  ASSERT_NE(nullptr, unmap) << dso.error();
  ASSERT_NE(nullptr, set_access) << dso.error();
  ASSERT_NE(nullptr, get_access) << dso.error();
  ASSERT_NE(nullptr, get_granularity) << dso.error();
  ASSERT_NE(nullptr, get_properties) << dso.error();
  ASSERT_EQ(hipSuccess, init(/*flags=*/0));

  int device = -1;
  ASSERT_EQ(hipSuccess, get_device(&device));

  // AMDGPU cannot satisfy an uncached request with host-owned physical backing,
  // but the same host placement is valid with its default cache policy. This
  // combination therefore deterministically requires fallback.
  hipMemAllocationProp properties = {};
  properties.type = hipMemAllocationTypeUncached;
  properties.requestedHandleType = hipMemHandleTypeNone;
  properties.location.type = hipMemLocationTypeHost;
  properties.location.id = 0;

  size_t allocation_size = 0;
  ASSERT_EQ(hipSuccess,
            get_granularity(&allocation_size, &properties,
                            hipMemAllocationGranularityRecommended));
  ASSERT_NE(0u, allocation_size);

  hipMemGenericAllocationHandle_t handle = nullptr;
  ASSERT_EQ(hipSuccess,
            create(&handle, allocation_size, &properties, /*flags=*/0));
  ASSERT_NE(nullptr, handle);

  void* pointer = nullptr;
  bool is_mapped = false;
  ScopeExit cleanup([&]() {
    if (is_mapped) {
      EXPECT_EQ(hipSuccess, unmap(pointer, allocation_size));
    }
    if (pointer) {
      EXPECT_EQ(hipSuccess, address_free(pointer, allocation_size));
    }
    if (handle) {
      EXPECT_EQ(hipSuccess, release(handle));
    }
  });

  hipMemAllocationProp observed_properties = {};
  ASSERT_EQ(hipSuccess, get_properties(&observed_properties, handle));
  EXPECT_EQ(hipMemAllocationTypeUncached, observed_properties.type);
  EXPECT_EQ(hipMemHandleTypeNone, observed_properties.requestedHandleType);
  EXPECT_EQ(hipMemLocationTypeHost, observed_properties.location.type);
  EXPECT_EQ(0, observed_properties.location.id);

  ASSERT_EQ(hipSuccess,
            address_reserve(&pointer, allocation_size, /*alignment=*/0,
                            /*address=*/nullptr, /*flags=*/0));
  ASSERT_NE(nullptr, pointer);
  ASSERT_EQ(hipSuccess,
            map(pointer, allocation_size, /*offset=*/0, handle, /*flags=*/0));
  is_mapped = true;

  hipMemAccessDesc access = {};
  access.location.type = hipMemLocationTypeDevice;
  access.location.id = device;
  access.flags = hipMemAccessFlagsProtReadWrite;
  ASSERT_EQ(hipSuccess, set_access(pointer, allocation_size, &access,
                                   /*descriptor_count=*/1));
  unsigned long long observed_access = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess,
            get_access(&observed_access, &access.location, pointer));
  EXPECT_EQ(hipMemAccessFlagsProtReadWrite, observed_access);

  constexpr uint8_t kFillValue = 0xA5;
  ASSERT_EQ(hipSuccess, memset(pointer, kFillValue, allocation_size));
  ASSERT_EQ(hipSuccess, device_synchronize());
  std::array<uint8_t, 64> readback = {};
  ASSERT_EQ(hipSuccess, memcpy(readback.data(), pointer, readback.size(),
                               hipMemcpyDeviceToHost));
  for (uint8_t value : readback) {
    EXPECT_EQ(kFillValue, value);
  }

  ASSERT_EQ(hipSuccess, unmap(pointer, allocation_size));
  is_mapped = false;
  ASSERT_EQ(hipSuccess, address_free(pointer, allocation_size));
  pointer = nullptr;
  ASSERT_EQ(hipSuccess, release(handle));
  handle = nullptr;
}

}  // namespace
