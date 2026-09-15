// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <climits>

#include "api.h"
#include "hip_device_attribute_validation_test_shim.h"
#include "hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

constexpr int kOutputSentinel = 0x5a5a5a5a;

using HipInitFn = hipError_t (*)(unsigned int flags);
using HipHalDeinitFn = hipError_t (*)(void);
using HipGetDeviceFn = hipError_t (*)(int* device);
using HipSetDeviceFn = hipError_t (*)(int device);
using HipGetDeviceCountFn = hipError_t (*)(int* count);
using HipDeviceGetAttributeFn = hipError_t (*)(int* value,
                                               hipDeviceAttribute_t attribute,
                                               int device);
using HipGetLastErrorFn = hipError_t (*)(void);
using HipPeekAtLastErrorFn = hipError_t (*)(void);

// Owns the RTLD_LOCAL handle and exact public entry points exercised by this
// test.
struct HipRuntimeApi {
  // Initializes the loaded HIP runtime instance.
  HipInitFn init = nullptr;
  // Deinitializes the loaded HRX runtime instance before unloading its DSO.
  HipHalDeinitFn hal_deinit = nullptr;
  // Reports the calling thread's current device.
  HipGetDeviceFn get_device = nullptr;
  // Selects the calling thread's current device.
  HipSetDeviceFn set_device = nullptr;
  // Reports every device visible to the loaded runtime instance.
  HipGetDeviceCountFn get_device_count = nullptr;
  // Reports one public device attribute for one visible device.
  HipDeviceGetAttributeFn device_get_attribute = nullptr;
  // Returns and clears the calling thread's last HIP error.
  HipGetLastErrorFn get_last_error = nullptr;
  // Returns without clearing the calling thread's last HIP error.
  HipPeekAtLastErrorFn peek_at_last_error = nullptr;
};

class HipDeviceAttributeValidationApiTest : public testing::Test {
 protected:
  void SetUp() override {
    if (!dso_.is_open()) {
      ASSERT_TRUE(dso_.Open()) << dso_.error();
      api_.init = dso_.Resolve<HipInitFn>("hipInit");
      api_.hal_deinit = dso_.Resolve<HipHalDeinitFn>("hipHALDeinit");
      api_.get_device = dso_.Resolve<HipGetDeviceFn>("hipGetDevice");
      api_.set_device = dso_.Resolve<HipSetDeviceFn>("hipSetDevice");
      api_.get_device_count =
          dso_.Resolve<HipGetDeviceCountFn>("hipGetDeviceCount");
      api_.device_get_attribute =
          dso_.Resolve<HipDeviceGetAttributeFn>("hipDeviceGetAttribute");
      api_.get_last_error = dso_.Resolve<HipGetLastErrorFn>("hipGetLastError");
      api_.peek_at_last_error =
          dso_.Resolve<HipPeekAtLastErrorFn>("hipPeekAtLastError");
      ASSERT_TRUE(dso_.error().empty()) << dso_.error();
    }

    ASSERT_NE(nullptr, api_.init);
    ASSERT_NE(nullptr, api_.hal_deinit);
    ASSERT_NE(nullptr, api_.get_device);
    ASSERT_NE(nullptr, api_.set_device);
    ASSERT_NE(nullptr, api_.get_device_count);
    ASSERT_NE(nullptr, api_.device_get_attribute);
    ASSERT_NE(nullptr, api_.get_last_error);
    ASSERT_NE(nullptr, api_.peek_at_last_error);

    ASSERT_EQ(hipSuccess, api_.init(/*flags=*/0));
    ASSERT_EQ(hipSuccess, api_.get_device_count(&device_count_));
    ASSERT_GT(device_count_, 0);

    current_device_ = device_count_ - 1;
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    ASSERT_EQ(hipSuccess, api_.set_device(current_device_));
    ExpectCurrentDevice();
    ASSERT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  static void TearDownTestSuite() {
    if (!dso_.is_open()) return;
    ASSERT_NE(nullptr, api_.hal_deinit);
    EXPECT_EQ(hipSuccess, api_.hal_deinit());
    api_ = {};
    EXPECT_TRUE(dso_.Close()) << dso_.error();
  }

  void ExpectCurrentDevice() {
    int actual_device = -1;
    ASSERT_EQ(hipSuccess, api_.get_device(&actual_device));
    EXPECT_EQ(current_device_, actual_device);
  }

  hipError_t GetDeviceAttribute(int* value, int attribute, int device) {
    return hrx_test_hip_device_get_attribute(api_.device_get_attribute, value,
                                             attribute, device);
  }

  template <typename Callable>
  void ExpectErrorState(hipError_t expected, Callable&& callable) {
    ASSERT_EQ(hipSuccess, api_.get_last_error());
    EXPECT_EQ(expected, callable());
    EXPECT_EQ(expected, api_.peek_at_last_error());
    ExpectCurrentDevice();
    EXPECT_EQ(expected, api_.peek_at_last_error());
    EXPECT_EQ(expected, api_.get_last_error());
    EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
  }

  // Process-lifetime runtime instance loaded from the exact built DSO.
  static HipRuntimeApi api_;
  // Exact DSO owner shared by every test in this fixture.
  static hrx::hip::testing::HipDso dso_;
  // Number of compatible devices visible to the loaded runtime.
  int device_count_ = 0;
  // Device selected in calling-thread TLS before each validation query.
  int current_device_ = -1;
};

HipRuntimeApi HipDeviceAttributeValidationApiTest::api_;
hrx::hip::testing::HipDso HipDeviceAttributeValidationApiTest::dso_;

TEST_F(HipDeviceAttributeValidationApiTest,
       SupportedQueryPreservesPendingErrorAndCurrentDevice) {
  ASSERT_EQ(hipErrorInvalidValue,
            GetDeviceAttribute(nullptr, hipDeviceAttributeMaxThreadsPerBlock,
                               /*device=*/0));
  ASSERT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());

  int attribute_value = kOutputSentinel;
  EXPECT_EQ(hipSuccess, GetDeviceAttribute(&attribute_value,
                                           hipDeviceAttributeMaxThreadsPerBlock,
                                           /*device=*/0));
  EXPECT_GT(attribute_value, 0);
  ExpectCurrentDevice();
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
}

TEST_F(HipDeviceAttributeValidationApiTest,
       ZeroAccessPolicyWindowCapabilityIsSupported) {
  ASSERT_EQ(
      hipErrorInvalidValue,
      GetDeviceAttribute(nullptr, hipDeviceAttributeAccessPolicyMaxWindowSize,
                         /*device=*/0));
  ASSERT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());

  int attribute_value = kOutputSentinel;
  EXPECT_EQ(hipSuccess,
            GetDeviceAttribute(&attribute_value,
                               hipDeviceAttributeAccessPolicyMaxWindowSize,
                               /*device=*/0));
  EXPECT_EQ(0, attribute_value);
  ExpectCurrentDevice();
  EXPECT_EQ(hipErrorInvalidValue, api_.peek_at_last_error());
  EXPECT_EQ(hipErrorInvalidValue, api_.get_last_error());
  EXPECT_EQ(hipSuccess, api_.peek_at_last_error());
}

TEST_F(HipDeviceAttributeValidationApiTest,
       CompleteDeviceAttributeSetReturnsValues) {
  const int attributes[] = {
      hipDeviceAttributeEccEnabled,
      hipDeviceAttributeAsyncEngineCount,
      hipDeviceAttributeComputeMode,
      hipDeviceAttributeComputePreemptionSupported,
      hipDeviceAttributeDeviceOverlap,
      hipDeviceAttributeGlobalL1CacheSupported,
      hipDeviceAttributeIntegrated,
      hipDeviceAttributeIsMultiGpuBoard,
      hipDeviceAttributeKernelExecTimeout,
      hipDeviceAttributeLocalL1CacheSupported,
      hipDeviceAttributeLuidDeviceNodeMask,
      hipDeviceAttributeMultiGpuBoardGroupID,
      hipDeviceAttributePageableMemoryAccess,
      hipDeviceAttributePageableMemoryAccessUsesHostPageTables,
      hipDeviceAttributeMaxRegistersPerMultiprocessor,
      hipDeviceAttributeReservedSharedMemPerBlock,
      hipDeviceAttributeSingleToDoublePrecisionPerfRatio,
      hipDeviceAttributeSurfaceAlignment,
      hipDeviceAttributeTccDriver,
      hipDeviceAttributeTextureAlignment,
      hipDeviceAttributeTexturePitchAlignment,
      hipDeviceAttributeVirtualMemoryManagementSupported,
      hipDeviceAttributeMemoryPoolSupportedHandleTypes,
      hipDeviceAttributeHostNumaId,
      hipDeviceAttributeDmaBufSupported,
      hipDeviceAttributeGPUDirectRDMAWithHipVMMSupported,
      hipDeviceAttributeClockInstructionRate,
      hipDeviceAttributeCooperativeMultiDeviceUnmatchedFunc,
      hipDeviceAttributeCooperativeMultiDeviceUnmatchedGridDim,
      hipDeviceAttributeCooperativeMultiDeviceUnmatchedBlockDim,
      hipDeviceAttributeCooperativeMultiDeviceUnmatchedSharedMem,
      hipDeviceAttributeAsicRevision,
      hipDeviceAttributeMaxAvailableVgprsPerThread,
      hipDeviceAttributePciChipId,
      hipDeviceAttributeExpertSchedMode,
      hipDeviceAttributeMaxDynDataPrefetchRegions,
  };
  for (int attribute : attributes) {
    SCOPED_TRACE(attribute);
    int attribute_value = kOutputSentinel;
    EXPECT_EQ(hipSuccess,
              GetDeviceAttribute(&attribute_value, attribute, /*device=*/0));
    EXPECT_NE(kOutputSentinel, attribute_value);
  }

  const int pointer_attributes[] = {
      hipDeviceAttributeHdpMemFlushCntl,
      hipDeviceAttributeHdpRegFlushCntl,
  };
  for (int attribute : pointer_attributes) {
    SCOPED_TRACE(attribute);
    uintptr_t attribute_value = UINTPTR_MAX;
    EXPECT_EQ(hipSuccess,
              GetDeviceAttribute(reinterpret_cast<int*>(&attribute_value),
                                 attribute, /*device=*/0));
    EXPECT_EQ(0u, attribute_value);
  }
}

TEST_F(HipDeviceAttributeValidationApiTest,
       NullOutputPrecedesDeviceAndAttributeValidation) {
  const int devices[] = {0, -1, device_count_, INT_MIN, INT_MAX};
  const int attributes[] = {
      hipDeviceAttributeMaxThreadsPerBlock,
      INT_MAX,
  };
  for (int device : devices) {
    for (int attribute : attributes) {
      SCOPED_TRACE(device);
      SCOPED_TRACE(attribute);
      ExpectErrorState(hipErrorInvalidValue, [&] {
        return GetDeviceAttribute(nullptr, attribute, device);
      });
    }
  }
}

TEST_F(HipDeviceAttributeValidationApiTest,
       InvalidDevicePrecedesAttributeValidationAndPreservesOutput) {
  const int invalid_devices[] = {-1, device_count_, INT_MIN, INT_MAX};
  const int attributes[] = {
      hipDeviceAttributeMaxThreadsPerBlock,
      INT_MAX,
  };
  for (int device : invalid_devices) {
    for (int attribute : attributes) {
      SCOPED_TRACE(device);
      SCOPED_TRACE(attribute);
      int attribute_value = kOutputSentinel;
      ExpectErrorState(hipErrorInvalidDevice, [&] {
        return GetDeviceAttribute(&attribute_value, attribute, device);
      });
      EXPECT_EQ(kOutputSentinel, attribute_value);
    }
  }
}

TEST_F(HipDeviceAttributeValidationApiTest,
       DeclaredButUnsupportedAttributesPreserveOutput) {
  const int attributes[] = {
      hipDeviceAttributeLuid,    hipDeviceAttributeMaxThreadsDim,
      hipDeviceAttributeUnused1, hipDeviceAttributeUnused2,
      hipDeviceAttributeUnused3, hipDeviceAttributeUnused4,
      hipDeviceAttributeUnused5,
  };
  for (int attribute : attributes) {
    SCOPED_TRACE(attribute);
    int attribute_value = kOutputSentinel;
    ExpectErrorState(hipErrorInvalidValue, [&] {
      return GetDeviceAttribute(&attribute_value, attribute, /*device=*/0);
    });
    EXPECT_EQ(kOutputSentinel, attribute_value);
  }
}

TEST_F(HipDeviceAttributeValidationApiTest,
       GapsMarkersAndBoundsPreserveOutput) {
  const int attributes[] = {
      INT_MIN,
      -1,
      95,
      9998,
      hipDeviceAttributeCudaCompatibleEnd,
      10023,
      19998,
      hipDeviceAttributeAmdSpecificEnd,
      20000,
      INT_MAX,
  };
  for (int attribute : attributes) {
    SCOPED_TRACE(attribute);
    int attribute_value = kOutputSentinel;
    ExpectErrorState(hipErrorInvalidValue, [&] {
      return GetDeviceAttribute(&attribute_value, attribute, /*device=*/0);
    });
    EXPECT_EQ(kOutputSentinel, attribute_value);
  }
}

}  // namespace
