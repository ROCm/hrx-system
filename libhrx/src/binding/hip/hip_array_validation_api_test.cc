// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <limits>

#include "binding/hip/api.h"
#include "binding/hip/hip_dso_test_util.h"
#include "iree/testing/gtest.h"

namespace {

constexpr size_t kMaximumDimension = std::numeric_limits<int>::max();
constexpr hipChannelFormatDesc kChannelDescriptor = {
    .x = 32,
    .y = 32,
    .z = 32,
    .w = 32,
    .f = hipChannelFormatKindUnsigned,
};

// On 64-bit hosts these overflow the slice pitch and total size, respectively.
constexpr hipExtent kOverflowExtents[] = {
    {kMaximumDimension, kMaximumDimension, 0},
    {1, kMaximumDimension, kMaximumDimension},
};

class HipArrayValidationApiTest : public ::testing::Test {
 protected:
  void SetUp() override { ASSERT_TRUE(dso_.Open()) << dso_.error(); }

  // Exact production library used by every public API call in the test.
  hrx::hip::testing::HipDso dso_;
};

TEST_F(HipArrayValidationApiTest, ReportsStorageOverflowAsExhaustion) {
  const auto malloc_array =
      dso_.Resolve<decltype(&hipMallocArray)>("hipMallocArray");
  ASSERT_NE(nullptr, malloc_array) << dso_.error();

  hipArray_t array = reinterpret_cast<hipArray_t>(1);

  EXPECT_EQ(hipErrorOutOfMemory,
            malloc_array(&array, &kChannelDescriptor, kMaximumDimension,
                         kMaximumDimension, hipArrayDefault));
  EXPECT_EQ(nullptr, array);

  EXPECT_EQ(hipErrorInvalidValue,
            malloc_array(&array, &kChannelDescriptor, kMaximumDimension + 1,
                         kMaximumDimension, hipArrayDefault));
  auto malformed = kChannelDescriptor;
  malformed.w = 16;
  EXPECT_EQ(hipErrorInvalidValue,
            malloc_array(&array, &malformed, kMaximumDimension,
                         kMaximumDimension, hipArrayDefault));
}

TEST_F(HipArrayValidationApiTest, Malloc3DArrayStorageOverflow) {
  const auto malloc_array =
      dso_.Resolve<decltype(&hipMalloc3DArray)>("hipMalloc3DArray");
  ASSERT_NE(nullptr, malloc_array) << dso_.error();

  for (const auto extent : kOverflowExtents) {
    SCOPED_TRACE(extent.depth);
    hipArray_t array = reinterpret_cast<hipArray_t>(1);
    EXPECT_EQ(hipErrorOutOfMemory, malloc_array(&array, &kChannelDescriptor,
                                                extent, hipArrayDefault));
    EXPECT_EQ(nullptr, array);
  }

  hipArray_t array = nullptr;
  EXPECT_EQ(hipErrorInvalidValue,
            malloc_array(&array, &kChannelDescriptor,
                         {kMaximumDimension + 1, kMaximumDimension, 1},
                         hipArrayDefault));
  auto malformed = kChannelDescriptor;
  malformed.w = 16;
  EXPECT_EQ(
      hipErrorInvalidValue,
      malloc_array(&array, &malformed,
                   {kMaximumDimension, kMaximumDimension, 1}, hipArrayDefault));
}

TEST_F(HipArrayValidationApiTest, ArrayCreateStorageOverflow) {
  const auto create_array =
      dso_.Resolve<decltype(&hipArrayCreate)>("hipArrayCreate");
  ASSERT_NE(nullptr, create_array) << dso_.error();

  HIP_ARRAY_DESCRIPTOR descriptor = {
      .Width = kMaximumDimension,
      .Height = kMaximumDimension,
      .Format = HIP_AD_FORMAT_UNSIGNED_INT32,
      .NumChannels = 4,
  };
  hipArray_t array = reinterpret_cast<hipArray_t>(1);
  EXPECT_EQ(hipErrorOutOfMemory, create_array(&array, &descriptor));
  EXPECT_EQ(nullptr, array);

  descriptor.Width = kMaximumDimension + 1;
  EXPECT_EQ(hipErrorInvalidValue, create_array(&array, &descriptor));
  descriptor.Width = kMaximumDimension;
  descriptor.NumChannels = 3;
  EXPECT_EQ(hipErrorInvalidValue, create_array(&array, &descriptor));
}

TEST_F(HipArrayValidationApiTest, Array3DCreateStorageOverflow) {
  const auto create_array =
      dso_.Resolve<decltype(&hipArray3DCreate)>("hipArray3DCreate");
  ASSERT_NE(nullptr, create_array) << dso_.error();

  HIP_ARRAY3D_DESCRIPTOR descriptor = {
      .Format = HIP_AD_FORMAT_UNSIGNED_INT32,
      .NumChannels = 4,
      .Flags = hipArrayDefault,
  };
  for (const auto extent : kOverflowExtents) {
    SCOPED_TRACE(extent.depth);
    descriptor.Width = extent.width;
    descriptor.Height = extent.height;
    descriptor.Depth = extent.depth;
    hipArray_t array = reinterpret_cast<hipArray_t>(1);
    EXPECT_EQ(hipErrorOutOfMemory, create_array(&array, &descriptor));
    EXPECT_EQ(nullptr, array);
  }

  hipArray_t array = nullptr;
  descriptor.Width = kMaximumDimension + 1;
  EXPECT_EQ(hipErrorInvalidValue, create_array(&array, &descriptor));
  descriptor.Width = 1;
  descriptor.NumChannels = 3;
  EXPECT_EQ(hipErrorInvalidValue, create_array(&array, &descriptor));
}

}  // namespace
