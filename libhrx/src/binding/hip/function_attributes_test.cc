// Copyright 2026 The HRX Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "binding/hip/function_attributes.h"

#include <cstdint>
#include <cstring>

#include "common/context.h"
#include "common/device.h"
#include "iree/testing/gtest.h"

namespace {

constexpr iree_hal_streaming_function_attribute_flags_t kResourceFlags =
    IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK |
    IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY |
    IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY |
    IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT |
    IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY;

class HipFunctionAttributesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    device_.compute_capability_major = 11;
    device_.compute_capability_minor = 5;
    context_.device_entry = &device_;
    module_.context = &context_;
    symbol_.module = &module_;
    auto& attributes = symbol_.function_attributes;
    attributes.provided_flags = kResourceFlags;
    attributes.maximum_threads_per_block = 768;
    attributes.fixed_shared_memory_size = 4096;
    attributes.fixed_local_memory_size = 64;
    attributes.register_count = 40;
    attributes.maximum_configurable_dynamic_shared_memory_size = 92 * 1024;
    iree_atomic_store(&attributes.configured_dynamic_shared_memory_size,
                      60 * 1024, iree_memory_order_relaxed);
    iree_atomic_store(&symbol_.preferred_shared_memory_carveout, -1,
                      iree_memory_order_relaxed);
  }

  // Device identity consumed by the architecture-version compatibility fields.
  iree_hal_streaming_device_t device_ = {};
  // Context that owns the symbol's device identity.
  iree_hal_streaming_context_t context_ = {};
  // Loaded-module owner retained by public query callers.
  iree_hal_streaming_module_t module_ = {};
  // Cached function facts supplied by module metadata extraction.
  iree_hal_streaming_symbol_t symbol_ = {};
};

TEST_F(HipFunctionAttributesTest, MapsEverySingleAttribute) {
  const struct {
    // Public attribute requested by this case.
    hipFuncAttribute_t attribute;
    // Expected HIP compatibility value.
    int expected;
  } cases[] = {
      {hipFuncAttributeMaxThreadsPerBlock, 768},
      {hipFuncAttributeSharedSizeBytes, 4096},
      {hipFuncAttributeConstSizeBytes, -1},
      {hipFuncAttributeLocalSizeBytes, 64},
      {hipFuncAttributeNumRegs, 40},
      {hipFuncAttributePtxVersion, 115},
      {hipFuncAttributeBinaryVersion, 115},
      {hipFuncAttributeCacheModeCA, 0},
      {hipFuncAttributeMaxDynamicSharedSizeBytes, 60 * 1024},
      {hipFuncAttributePreferredSharedMemoryCarveout, -1},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.attribute);
    int value = 123;
    EXPECT_EQ(hipSuccess, iree_hip_function_get_attribute(
                              &symbol_, test_case.attribute, &value));
    EXPECT_EQ(test_case.expected, value);
  }
}

TEST_F(HipFunctionAttributesTest, MapsEveryAggregateField) {
  hipFuncAttributes attributes = {};
  ASSERT_EQ(hipSuccess,
            iree_hip_function_get_attributes(&symbol_, &attributes));
  EXPECT_EQ(115, attributes.binaryVersion);
  EXPECT_EQ(0, attributes.cacheModeCA);
  EXPECT_EQ(SIZE_MAX, attributes.constSizeBytes);
  EXPECT_EQ(64u, attributes.localSizeBytes);
  EXPECT_EQ(60 * 1024, attributes.maxDynamicSharedSizeBytes);
  EXPECT_EQ(768, attributes.maxThreadsPerBlock);
  EXPECT_EQ(40, attributes.numRegs);
  EXPECT_EQ(-1, attributes.preferredShmemCarveout);
  EXPECT_EQ(115, attributes.ptxVersion);
  EXPECT_EQ(4096u, attributes.sharedSizeBytes);
}

TEST_F(HipFunctionAttributesTest, PreservesUnavailableResourceFactsAndOutputs) {
  const struct {
    // Public resource attribute requested by this case.
    hipFuncAttribute_t attribute;
    // Metadata availability bit removed for this case.
    iree_hal_streaming_function_attribute_flags_t required_flag;
  } cases[] = {
      {hipFuncAttributeMaxThreadsPerBlock,
       IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_MAXIMUM_THREADS_PER_BLOCK},
      {hipFuncAttributeSharedSizeBytes,
       IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_SHARED_MEMORY},
      {hipFuncAttributeLocalSizeBytes,
       IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_FIXED_LOCAL_MEMORY},
      {hipFuncAttributeNumRegs,
       IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_REGISTER_COUNT},
      {hipFuncAttributeMaxDynamicSharedSizeBytes,
       IREE_HAL_STREAMING_FUNCTION_ATTRIBUTE_FLAG_DYNAMIC_SHARED_MEMORY},
  };
  for (const auto& test_case : cases) {
    SCOPED_TRACE(test_case.attribute);
    symbol_.function_attributes.provided_flags =
        kResourceFlags & ~test_case.required_flag;
    int value = 123;
    EXPECT_EQ(hipErrorNotSupported, iree_hip_function_get_attribute(
                                        &symbol_, test_case.attribute, &value));
    EXPECT_EQ(123, value);
    hipFuncAttributes attributes;
    std::memset(&attributes, 0xA5, sizeof(attributes));
    unsigned char expected[sizeof(attributes)];
    std::memcpy(expected, &attributes, sizeof(attributes));
    EXPECT_EQ(hipErrorNotSupported,
              iree_hip_function_get_attributes(&symbol_, &attributes));
    EXPECT_EQ(0, std::memcmp(expected, &attributes, sizeof(attributes)));
  }
}

TEST_F(HipFunctionAttributesTest, DistinguishesKnownZeroFromUnavailable) {
  symbol_.function_attributes.fixed_shared_memory_size = 0;
  symbol_.function_attributes.fixed_local_memory_size = 0;
  symbol_.function_attributes.register_count = 0;
  ASSERT_TRUE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &symbol_.function_attributes, 0));
  hipFuncAttributes attributes = {};
  ASSERT_EQ(hipSuccess,
            iree_hip_function_get_attributes(&symbol_, &attributes));
  EXPECT_EQ(0u, attributes.sharedSizeBytes);
  EXPECT_EQ(0u, attributes.localSizeBytes);
  EXPECT_EQ(0, attributes.numRegs);
  EXPECT_EQ(0, attributes.maxDynamicSharedSizeBytes);
  EXPECT_EQ(SIZE_MAX, attributes.constSizeBytes);
}

TEST_F(HipFunctionAttributesTest, ReadsCurrentMutableAttributes) {
  ASSERT_TRUE(
      iree_hal_streaming_function_attributes_try_set_dynamic_shared_memory_size(
          &symbol_.function_attributes, 92 * 1024));
  iree_atomic_store(&symbol_.preferred_shared_memory_carveout, 100,
                    iree_memory_order_relaxed);
  hipFuncAttributes attributes = {};
  ASSERT_EQ(hipSuccess,
            iree_hip_function_get_attributes(&symbol_, &attributes));
  EXPECT_EQ(92 * 1024, attributes.maxDynamicSharedSizeBytes);
  EXPECT_EQ(100, attributes.preferredShmemCarveout);
  int value = 0;
  ASSERT_EQ(hipSuccess,
            iree_hip_function_get_attribute(
                &symbol_, hipFuncAttributeMaxDynamicSharedSizeBytes, &value));
  EXPECT_EQ(attributes.maxDynamicSharedSizeBytes, value);
  ASSERT_EQ(
      hipSuccess,
      iree_hip_function_get_attribute(
          &symbol_, hipFuncAttributePreferredSharedMemoryCarveout, &value));
  EXPECT_EQ(attributes.preferredShmemCarveout, value);
}

TEST_F(HipFunctionAttributesTest, RejectsInvalidAttributesWithoutWriting) {
  for (const auto attribute :
       {static_cast<hipFuncAttribute_t>(-1), hipFuncAttributeMax}) {
    int value = 123;
    EXPECT_EQ(hipErrorInvalidValue,
              iree_hip_function_get_attribute(&symbol_, attribute, &value));
    EXPECT_EQ(123, value);
  }
}

}  // namespace
