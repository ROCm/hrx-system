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

using HipMallocArrayFn = hipError_t (*)(hipArray_t* array,
                                        const hipChannelFormatDesc* descriptor,
                                        size_t width, size_t height,
                                        unsigned int flags);

TEST(HipArrayValidationApiTest, ReportsStorageOverflowAsExhaustion) {
  hrx::hip::testing::HipDso dso;
  ASSERT_TRUE(dso.Open()) << dso.error();
  HipMallocArrayFn malloc_array =
      dso.Resolve<HipMallocArrayFn>("hipMallocArray");
  ASSERT_NE(nullptr, malloc_array) << dso.error();

  const hipChannelFormatDesc descriptor = {
      /*.x=*/32,
      /*.y=*/32,
      /*.z=*/32,
      /*.w=*/32,
      /*.f=*/hipChannelFormatKindUnsigned,
  };
  const size_t maximum_dimension =
      static_cast<size_t>(std::numeric_limits<int>::max());
  hipArray_t array = nullptr;

  EXPECT_EQ(hipErrorOutOfMemory,
            malloc_array(&array, &descriptor, maximum_dimension,
                         maximum_dimension, /*flags=*/0));
  EXPECT_EQ(nullptr, array);
}

}  // namespace
