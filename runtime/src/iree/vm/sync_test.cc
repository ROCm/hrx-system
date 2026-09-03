// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/sync.h"

#include <array>
#include <cstdint>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/invocation_test_module.h"

namespace {

constexpr iree_host_size_t kInvocationStorageSize = 16 * 1024;

template <size_t N>
iree_vm_variant_span_t MakeVariantSpan(
    std::array<iree_vm_variant_t, N>& variants) {
  return iree_vm_variant_span_from_ptr(variants.data(), variants.size());
}

class VMSyncTest : public ::testing::Test {
 protected:
  void SetUp() override {
    IREE_ASSERT_OK(
        iree_vm_invocation_test_module_initialize(&counters_, &module_));
    IREE_ASSERT_OK(
        iree_vm_program_create({&module_.base, iree_vm_module_span_empty()},
                               iree_allocator_system(), &program_));
    IREE_ASSERT_OK(iree_vm_invocation_initialize(
        iree_make_byte_span(invocation_storage_.data(),
                            invocation_storage_.size()),
        &invocation_));
    IREE_ASSERT_OK(iree_vm_process_create(program_, invocation_,
                                          iree_vm_variant_span_empty(),
                                          iree_allocator_system(), &process_));
    ASSERT_NE(process_, nullptr);
  }

  void TearDown() override {
    iree_vm_process_release(process_);
    iree_vm_invocation_deinitialize(invocation_);
    iree_vm_program_release(program_);
    iree_vm_module_release(&module_.base);
    EXPECT_EQ(counters_.destroy_count, 1);
  }

  iree_vm_function_t LookupFunction(iree_string_view_t name) {
    iree_vm_function_t function = iree_vm_function_null();
    IREE_EXPECT_OK(iree_vm_process_lookup_function(
        process_, IREE_SV("invocation.test"), name, &function));
    return function;
  }

  void ExpectI32(iree_vm_variant_t variant, int32_t expected_value) {
    int32_t value = 0;
    IREE_ASSERT_OK(iree_vm_i32_from_variant(variant, &value));
    EXPECT_EQ(value, expected_value);
  }

  iree_vm_invocation_test_counters_t counters_ = {};
  iree_vm_invocation_test_module_t module_ = {};
  iree_vm_program_t* program_ = nullptr;
  iree_vm_process_t* process_ = nullptr;
  alignas(iree_max_align_t)
      std::array<uint8_t, kInvocationStorageSize> invocation_storage_ = {};
  iree_vm_invocation_t* invocation_ = nullptr;
};

TEST_F(VMSyncTest, InvokesImmediateFunction) {
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  IREE_ASSERT_OK(iree_vm_invoke(invocation_, LookupFunction(IREE_SV("add")),
                                MakeVariantSpan(arguments),
                                MakeVariantSpan(results)));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  ExpectI32(results[0], 42);
  EXPECT_EQ(counters_.start_count, 1);
  EXPECT_EQ(counters_.resume_count, 0);
}

TEST_F(VMSyncTest, PreservesWakesPublishedBeforeEachWait) {
  std::array<iree_vm_variant_t, 1> arguments = {
      iree_vm_variant_from_i32(31),
  };
  std::array<iree_vm_variant_t, 1> results = {iree_vm_variant_empty()};
  IREE_ASSERT_OK(
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("yield_twice")),
                     MakeVariantSpan(arguments), MakeVariantSpan(results)));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  ExpectI32(results[0], 32);
  EXPECT_EQ(counters_.start_count, 1);
  EXPECT_EQ(counters_.resume_count, 2);
  EXPECT_EQ(counters_.cleanup_count, 1);
}

TEST_F(VMSyncTest, FailureLeavesResultsUntouched) {
  std::array<iree_vm_variant_t, 2> arguments = {
      iree_vm_variant_from_i32(19),
      iree_vm_variant_from_i32(23),
  };
  const iree_vm_variant_t untouched_result =
      iree_vm_variant_from_i64(INT64_C(0x12345678));
  std::array<iree_vm_variant_t, 1> results = {untouched_result};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      iree_vm_invoke(invocation_, LookupFunction(IREE_SV("fail")),
                     MakeVariantSpan(arguments), MakeVariantSpan(results)));

  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[0]));
  EXPECT_TRUE(iree_vm_variant_is_empty(arguments[1]));
  EXPECT_EQ(results[0].payload, untouched_result.payload);
  EXPECT_EQ(results[0].metadata, untouched_result.metadata);
  EXPECT_EQ(counters_.start_count, 1);
  EXPECT_EQ(counters_.resume_count, 0);
}

}  // namespace
