// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <array>
#include <cstring>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

class HipVirtualMemoryApiTest : public testing::Test {
 protected:
  static void SetUpTestSuite() {
    ASSERT_TRUE(dso_.Open()) << dso_.error();
    const auto initialize = dso_.Resolve<decltype(&hipInit)>("hipInit");
    ASSERT_NE(initialize, nullptr);
    ASSERT_EQ(hipSuccess, initialize(0));
  }

  static void TearDownTestSuite() {
    if (dso_.is_open()) {
      const auto deinitialize = dso_.Resolve<hipError_t (*)()>("hipHALDeinit");
      ASSERT_NE(deinitialize, nullptr);
      EXPECT_EQ(hipSuccess, deinitialize());
      EXPECT_TRUE(dso_.Close()) << dso_.error();
    }
  }

  void SetUp() override {
    ASSERT_TRUE(dso_.is_open());
    const auto get_device =
        dso_.Resolve<decltype(&hipGetDevice)>("hipGetDevice");
    const auto get_attribute =
        dso_.Resolve<decltype(&hipDeviceGetAttribute)>("hipDeviceGetAttribute");
    ASSERT_NE(get_device, nullptr);
    ASSERT_NE(get_attribute, nullptr);
    ASSERT_EQ(hipSuccess, get_device(&device_));
    ASSERT_EQ(hipSuccess,
              get_attribute(&supported_,
                            hipDeviceAttributeVirtualMemoryManagementSupported,
                            device_));
    ASSERT_EQ(supported_, 1)
        << "The selected HIP GPU backend must support VMM.";
    reserve_ =
        dso_.Resolve<decltype(&hipMemAddressReserve)>("hipMemAddressReserve");
    close_ = dso_.Resolve<decltype(&hipMemAddressFree)>("hipMemAddressFree");
    create_ = dso_.Resolve<decltype(&hipMemCreate)>("hipMemCreate");
    release_ = dso_.Resolve<decltype(&hipMemRelease)>("hipMemRelease");
    map_ = dso_.Resolve<decltype(&hipMemMap)>("hipMemMap");
    unmap_ = dso_.Resolve<decltype(&hipMemUnmap)>("hipMemUnmap");
    set_access_ = dso_.Resolve<decltype(&hipMemSetAccess)>("hipMemSetAccess");
    get_access_ = dso_.Resolve<decltype(&hipMemGetAccess)>("hipMemGetAccess");
    granularity_ = dso_.Resolve<decltype(&hipMemGetAllocationGranularity)>(
        "hipMemGetAllocationGranularity");
    get_properties_ =
        dso_.Resolve<decltype(&hipMemGetAllocationPropertiesFromHandle)>(
            "hipMemGetAllocationPropertiesFromHandle");
    copy_ = dso_.Resolve<decltype(&hipMemcpy)>("hipMemcpy");
    ASSERT_NE(reserve_, nullptr);
    ASSERT_NE(close_, nullptr);
    ASSERT_NE(create_, nullptr);
    ASSERT_NE(release_, nullptr);
    ASSERT_NE(map_, nullptr);
    ASSERT_NE(unmap_, nullptr);
    ASSERT_NE(set_access_, nullptr);
    ASSERT_NE(get_access_, nullptr);
    ASSERT_NE(granularity_, nullptr);
    ASSERT_NE(get_properties_, nullptr);
    ASSERT_NE(copy_, nullptr);
  }

  void TearDown() override {
    for (size_t i = 0; i < mapped_.size(); ++i) {
      if (mapped_[i]) {
        EXPECT_EQ(hipSuccess, unmap_(Address(i), unit_));
      }
    }
    if (address_) {
      EXPECT_EQ(hipSuccess, close_(address_, 3 * unit_));
    }
    for (auto handle : handles_) {
      if (handle) {
        EXPECT_EQ(hipSuccess, release_(handle));
      }
    }
  }

  hipMemAllocationProp Properties() const {
    return hipMemAllocationProp{
        .type = hipMemAllocationTypePinned,
        .location = {.type = hipMemLocationTypeDevice, .id = device_},
    };
  }

  void* Address(size_t index) {
    return static_cast<uint8_t*>(address_) + index * unit_;
  }

  // One runtime/driver incarnation shared by the suite's public API cases.
  static inline hrx::hip::testing::HipDso dso_;
  // Current physical placement device.
  int device_ = 0;
  // Public capability value required by this native GPU test.
  int supported_ = 0;
  // Native allocation and mapping unit for this device.
  size_t unit_ = 0;
  // Explicit address owner, released after installed aliases.
  void* address_ = nullptr;
  // Physical handles owned by this test until explicitly released.
  std::array<hipMemGenericAllocationHandle_t, 2> handles_ = {};
  // Successful bindings requiring teardown if an assertion stops the test.
  std::array<bool, 3> mapped_ = {};
  // Creates an explicit address reservation.
  decltype(&hipMemAddressReserve) reserve_ = nullptr;
  // Consumes the reservation on success, retaining it on failure.
  decltype(&hipMemAddressFree) close_ = nullptr;
  // Allocates initially unmapped physical backing behind a HIP handle.
  decltype(&hipMemCreate) create_ = nullptr;
  // Releases one public physical handle reference.
  decltype(&hipMemRelease) release_ = nullptr;
  // Installs one inaccessible alias.
  decltype(&hipMemMap) map_ = nullptr;
  // Removes a mapped range before backing reuse.
  decltype(&hipMemUnmap) unmap_ = nullptr;
  // Updates permissions over all physical pieces of an interval.
  decltype(&hipMemSetAccess) set_access_ = nullptr;
  // Queries the HIP permission record for an address.
  decltype(&hipMemGetAccess) get_access_ = nullptr;
  // Queries required physical and virtual mapping geometry.
  decltype(&hipMemGetAllocationGranularity) granularity_ = nullptr;
  // Returns the public properties retained by a physical allocation handle.
  decltype(&hipMemGetAllocationPropertiesFromHandle) get_properties_ = nullptr;
  // Performs completed copies through the public HIP pointer lookup path.
  decltype(&hipMemcpy) copy_ = nullptr;
};

TEST_F(HipVirtualMemoryApiTest, CapabilityMatchesConstruction) {
  EXPECT_EQ(hipErrorInvalidValue, reserve_(nullptr, 65536, 0, nullptr, 0));
  EXPECT_EQ(hipErrorInvalidValue, set_access_(nullptr, 65536, nullptr, 0));
  const auto properties = Properties();

  ASSERT_EQ(hipSuccess, granularity_(&unit_, &properties,
                                     hipMemAllocationGranularityMinimum));
  EXPECT_GT(unit_, 0u);
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
}

TEST_F(HipVirtualMemoryApiTest, PermissionsPreserveAliasesAndHandleLifetime) {
  const auto properties = Properties();
  ASSERT_EQ(hipSuccess, granularity_(&unit_, &properties,
                                     hipMemAllocationGranularityMinimum));
  ASSERT_GT(unit_, 0u);
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
  for (size_t i = 0; i < handles_.size(); ++i) {
    ASSERT_EQ(hipSuccess, create_(&handles_[i], unit_, &properties, 0));
    ASSERT_EQ(hipSuccess, map_(Address(i), unit_, 0, handles_[i], 0));
    mapped_[i] = true;
  }
  ASSERT_EQ(hipSuccess, map_(Address(2), unit_, 0, handles_[1], 0));
  mapped_[2] = true;
  hipMemAccessDesc access = {
      .location = properties.location,
      .flags = hipMemAccessFlagsProtReadWrite,
  };
  ASSERT_EQ(hipSuccess, set_access_(address_, 3 * unit_, &access, 1));
  for (auto& handle : handles_) {
    ASSERT_EQ(hipSuccess, release_(handle));
    handle = nullptr;
  }
  const std::array<uint32_t, 4> expected = {7, 13, 19, 23};
  ASSERT_EQ(hipSuccess, copy_(Address(1), expected.data(), sizeof(expected),
                              hipMemcpyHostToDevice));
  std::array<uint32_t, 4> observed = {};
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(2), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
  access.flags = hipMemAccessFlagsProtNone;
  ASSERT_EQ(hipSuccess, set_access_(address_, 2 * unit_, &access, 1));
  access.flags = hipMemAccessFlagsProtRead;
  ASSERT_EQ(hipSuccess, set_access_(address_, 2 * unit_, &access, 1));
  unsigned long long observed_access = 0;
  ASSERT_EQ(hipSuccess,
            get_access_(&observed_access, &access.location, Address(1)));
  EXPECT_EQ(observed_access, hipMemAccessFlagsProtRead);
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(1), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
  ASSERT_EQ(hipSuccess, unmap_(address_, 2 * unit_));
  mapped_[0] = mapped_[1] = false;
  EXPECT_EQ(hipErrorInvalidValue, close_(address_, 3 * unit_));
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(2), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
}

TEST_F(HipVirtualMemoryApiTest,
       NativeHostAccessPreservesTheSinglePointerContract) {
  auto properties = Properties();
  properties.location = {.type = hipMemLocationTypeHost, .id = 0};
  ASSERT_EQ(hipSuccess, granularity_(&unit_, &properties,
                                     hipMemAllocationGranularityMinimum));
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
  ASSERT_EQ(hipSuccess, create_(&handles_[0], unit_, &properties, 0));
  for (size_t i = 0; i < 2; ++i) {
    ASSERT_EQ(hipSuccess, map_(Address(i), unit_, 0, handles_[0], 0));
    mapped_[i] = true;
  }
  const hipMemAccessDesc access[] = {
      {
          .location = {.type = hipMemLocationTypeHost, .id = 0},
          .flags = hipMemAccessFlagsProtReadWrite,
      },
      {
          .location = {.type = hipMemLocationTypeDevice, .id = device_},
          .flags = hipMemAccessFlagsProtReadWrite,
      },
  };
  ASSERT_EQ(hipSuccess, set_access_(address_, 2 * unit_, access, 2));
  ASSERT_EQ(hipSuccess, release_(handles_[0]));
  handles_[0] = nullptr;
  const std::array<uint32_t, 4> expected = {11, 17, 29, 47};
  std::memcpy(Address(0), expected.data(), sizeof(expected));
  std::array<uint32_t, 4> observed = {};
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(1), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
  const std::array<uint32_t, 4> update = {53, 71, 89, 97};
  ASSERT_EQ(hipSuccess, copy_(Address(1), update.data(), sizeof(update),
                              hipMemcpyHostToDevice));
  std::memcpy(observed.data(), Address(0), sizeof(observed));
  EXPECT_EQ(observed, update);
}

TEST_F(HipVirtualMemoryApiTest,
       NativeUncachedHostAccessPreservesTheSinglePointerContract) {
  auto properties = Properties();
  properties.type = hipMemAllocationTypeUncached;
  properties.location = {.type = hipMemLocationTypeHost, .id = 0};
  ASSERT_EQ(hipSuccess, granularity_(&unit_, &properties,
                                     hipMemAllocationGranularityMinimum));
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
  ASSERT_EQ(hipSuccess, create_(&handles_[0], unit_, &properties, 0));
  hipMemAllocationProp observed_properties = {};
  ASSERT_EQ(hipSuccess, get_properties_(&observed_properties, handles_[0]));
  EXPECT_EQ(observed_properties.type, hipMemAllocationTypeUncached);
  EXPECT_EQ(observed_properties.requestedHandleType, hipMemHandleTypeNone);
  EXPECT_EQ(observed_properties.location.type, hipMemLocationTypeHost);
  EXPECT_EQ(observed_properties.location.id, 0);
  ASSERT_EQ(hipSuccess, map_(address_, unit_, 0, handles_[0], 0));
  mapped_[0] = true;
  const hipMemAccessDesc access[] = {
      {
          .location = {.type = hipMemLocationTypeHost, .id = 0},
          .flags = hipMemAccessFlagsProtReadWrite,
      },
      {
          .location = {.type = hipMemLocationTypeDevice, .id = device_},
          .flags = hipMemAccessFlagsProtReadWrite,
      },
  };
  ASSERT_EQ(hipSuccess, set_access_(address_, unit_, access, 2));
  ASSERT_EQ(hipSuccess, release_(handles_[0]));
  handles_[0] = nullptr;
  const std::array<uint32_t, 4> expected = {59, 61, 67, 71};
  std::memcpy(address_, expected.data(), sizeof(expected));
  std::array<uint32_t, 4> observed = {};
  ASSERT_EQ(hipSuccess, copy_(observed.data(), address_, sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
  const std::array<uint32_t, 4> update = {73, 79, 83, 89};
  ASSERT_EQ(hipSuccess, copy_(address_, update.data(), sizeof(update),
                              hipMemcpyHostToDevice));
  std::memcpy(observed.data(), address_, sizeof(observed));
  EXPECT_EQ(observed, update);
}

TEST_F(HipVirtualMemoryApiTest,
       NativeMixedBackingKeepsCpuPermissionsOnSharedRanges) {
  std::array<hipMemAllocationProp, 2> properties = {Properties(), Properties()};
  properties[1].location = {.type = hipMemLocationTypeHost, .id = 0};
  for (const auto& property : properties) {
    size_t granularity = 0;
    ASSERT_EQ(hipSuccess, granularity_(&granularity, &property,
                                       hipMemAllocationGranularityMinimum));
    unit_ = std::max(unit_, granularity);
  }
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
  for (size_t i = 0; i < handles_.size(); ++i) {
    ASSERT_EQ(hipSuccess, create_(&handles_[i], unit_, &properties[i], 0));
    ASSERT_EQ(hipSuccess, map_(Address(i), unit_, 0, handles_[i], 0));
    mapped_[i] = true;
  }
  ASSERT_EQ(hipSuccess, map_(Address(2), unit_, 0, handles_[1], 0));
  mapped_[2] = true;
  const hipMemAccessDesc gpu = {
      .location = properties[0].location,
      .flags = hipMemAccessFlagsProtReadWrite,
  };
  ASSERT_EQ(hipSuccess, set_access_(address_, 3 * unit_, &gpu, 1));
  const hipMemAccessDesc host = {
      .location = properties[1].location,
      .flags = hipMemAccessFlagsProtReadWrite,
  };
  // Host permissions apply only to the selected virtual ranges, independently
  // of whether the adjacent device-local backing also permits host mappings.
  ASSERT_EQ(hipSuccess, set_access_(Address(1), 2 * unit_, &host, 1));
  for (size_t i = 0; i < mapped_.size(); ++i) {
    unsigned long long access = 0;
    ASSERT_EQ(hipSuccess, get_access_(&access, &host.location, Address(i)));
    EXPECT_EQ(access, i == 0 ? hipMemAccessFlagsProtNone
                             : hipMemAccessFlagsProtReadWrite);
    ASSERT_EQ(hipSuccess, get_access_(&access, &gpu.location, Address(i)));
    EXPECT_EQ(access, hipMemAccessFlagsProtReadWrite);
  }
  for (auto& handle : handles_) {
    ASSERT_EQ(hipSuccess, release_(handle));
    handle = nullptr;
  }
  const std::array<uint32_t, 4> expected = {41, 43, 47, 53};
  std::memcpy(Address(1), expected.data(), sizeof(expected));
  ASSERT_EQ(hipSuccess, copy_(Address(0), Address(2), sizeof(expected),
                              hipMemcpyDeviceToDevice));
  std::array<uint32_t, 4> observed = {};
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(0), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
  ASSERT_EQ(hipSuccess, unmap_(Address(1), unit_));
  mapped_[1] = false;
  std::memcpy(observed.data(), Address(2), sizeof(observed));
  EXPECT_EQ(observed, expected);
}

TEST_F(HipVirtualMemoryApiTest,
       NativePartialUnmapRetainsTheRemainingBackingPieces) {
  const auto properties = Properties();
  ASSERT_EQ(hipSuccess, granularity_(&unit_, &properties,
                                     hipMemAllocationGranularityMinimum));
  ASSERT_EQ(hipSuccess, reserve_(&address_, 3 * unit_, unit_, nullptr, 0));
  ASSERT_EQ(hipSuccess, create_(&handles_[0], 3 * unit_, &properties, 0));
  ASSERT_EQ(hipSuccess, map_(address_, 3 * unit_, 0, handles_[0], 0));
  mapped_.fill(true);
  hipMemAccessDesc access = {
      .location = properties.location,
      .flags = hipMemAccessFlagsProtReadWrite,
  };
  ASSERT_EQ(hipSuccess, set_access_(address_, 3 * unit_, &access, 1));
  ASSERT_EQ(hipSuccess, release_(handles_[0]));
  handles_[0] = nullptr;
  const std::array<uint32_t, 4> expected = {3, 5, 7, 11};
  ASSERT_EQ(hipSuccess, copy_(Address(2), expected.data(), sizeof(expected),
                              hipMemcpyHostToDevice));
  ASSERT_EQ(hipSuccess, unmap_(Address(1), unit_));
  mapped_[1] = false;
  access.flags = hipMemAccessFlagsProtRead;
  ASSERT_EQ(hipSuccess, set_access_(Address(2), unit_, &access, 1));
  std::array<uint32_t, 4> observed = {};
  ASSERT_EQ(hipSuccess, copy_(observed.data(), Address(2), sizeof(observed),
                              hipMemcpyDeviceToHost));
  EXPECT_EQ(observed, expected);
}

}  // namespace
