// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/vm/native_call.h"

#include <array>
#include <cstdint>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace {

const iree_vm_native_call_field_t kPairFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_REF, 0, nullptr},
    {IREE_VM_NATIVE_CALL_FIELD_VALUE, 1, nullptr},
};
const iree_vm_native_call_schema_t kPairSchema = {2, 2, kPairFields, 16};
const iree_vm_native_call_field_t kResultField[] = {
    {IREE_VM_NATIVE_CALL_FIELD_RESULT_REF, 0, nullptr},
};
const iree_vm_native_call_schema_t kResultSchema = {1, 1, kResultField, 8};
const iree_vm_native_call_field_t kFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0, &kPairSchema},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 2, &kPairSchema},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 4, &kResultSchema},
};
const iree_vm_native_call_schema_t kSchema = {6, 3, kFields, 0};
const iree_vm_native_export_t kExport = {IREE_SV("submit"), 7, &kSchema};

class NativeCallTest : public ::testing::Test {
 protected:
  iree_status_t Verify(const std::vector<uint32_t>& words,
                       iree_host_size_t* out_size) {
    const iree_vm_native_callsite_t callsite = {
        IREE_SV("native"),
        IREE_SV("submit"),
        7,
        0,
        words.data(),
        static_cast<uint32_t>(words.size()),
        {4, 4, 24}};
    return iree_vm_native_call_verify(&kExport, &callsite,
                                      iree_allocator_system(), out_size);
  }
  // Two independent semaphore-like lists and two result slots. Results may
  // replace any input; only result/result overlap is prohibited.
  std::vector<uint32_t> words_ = {2, 6, 1, 10, 2, 12, 0, 0, 1, 1, 2, 2, 0, 1};
};

TEST_F(NativeCallTest, SizesIndependentRangesAndAliasedResults) {
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(Verify(words_, &size));
  EXPECT_EQ(size, 3 * 16 + 2 * 8);
}

TEST_F(NativeCallTest, RejectsEveryTruncatedRecord) {
  for (size_t count = 0; count < words_.size(); ++count) {
    auto truncated = words_;
    truncated.resize(count);
    iree_host_size_t size = 123;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          Verify(truncated, &size));
    EXPECT_EQ(size, 123u);
  }
}

TEST_F(NativeCallTest, RejectsOverlappingOrUncoveredRanges) {
  for (uint32_t offset : {0u, 5u, 7u, UINT32_MAX}) {
    auto words = words_;
    words[1] = offset;
    iree_host_size_t size = 0;
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Verify(words, &size));
  }
  words_.push_back(0);
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Verify(words_, &size));
}

TEST_F(NativeCallTest, RejectsInvalidSelectorsAndDuplicateResults) {
  iree_host_size_t size = 0;
  words_[6] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Verify(words_, &size));
  words_[6] = 0;
  words_[7] = 4;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Verify(words_, &size));
  words_[7] = 0;
  words_[13] = words_[12];
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Verify(words_, &size));
}

TEST_F(NativeCallTest, AdmitsWholeLocalScalarCells) {
  iree_host_size_t size = 0;
  words_[7] = IREE_VM_NATIVE_CALL_LOCAL_VALUE | 16;
  IREE_ASSERT_OK(Verify(words_, &size));
  words_[7] = IREE_VM_NATIVE_CALL_LOCAL_VALUE | 17;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE, Verify(words_, &size));
}

TEST_F(NativeCallTest, EmptyRangesHaveCanonicalOffsets) {
  words_ = {0, 6, 0, 6, 0, 6};
  iree_host_size_t size = 123;
  IREE_ASSERT_OK(Verify(words_, &size));
  EXPECT_EQ(size, 0u);
}

TEST(NativeCall, NestedRangesAccumulateNativeScratch) {
  const iree_vm_native_call_field_t fields[] = {
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0, &kSchema}};
  const iree_vm_native_call_schema_t schema = {2, 1, fields, 32};
  const iree_vm_native_export_t entry = {IREE_SV("nested"), 0, &schema};
  // Reserve both parent records, then each parent's descendants in order.
  const uint32_t words[] = {2,  2, 1,  14, 0,  16, 0, 16, 0,
                            16, 1, 16, 0,  18, 0,  0, 1,  1};
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"),     IREE_SV("nested"), 0, 0, words,
      IREE_ARRAYSIZE(words), {2, 2, 0}};
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(iree_vm_native_call_verify(&entry, &callsite,
                                            iree_allocator_system(), &size));
  EXPECT_EQ(size, 64u);
}

TEST(NativeCall, LocalScalarResultsCannotPartiallyOverlap) {
  const iree_vm_native_call_field_t fields[] = {
      {IREE_VM_NATIVE_CALL_FIELD_RESULT_VALUE, 0, nullptr},
      {IREE_VM_NATIVE_CALL_FIELD_RESULT_VALUE, 1, nullptr},
  };
  const iree_vm_native_call_schema_t schema = {2, 2, fields, 0};
  const iree_vm_native_export_t entry = {IREE_SV("results"), 0, &schema};
  uint32_t words[] = {IREE_VM_NATIVE_CALL_LOCAL_VALUE,
                      IREE_VM_NATIVE_CALL_LOCAL_VALUE | 4};
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"), IREE_SV("results"), 0, 0, words, 2, {0, 0, 16}};
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        iree_vm_native_call_verify(
                            &entry, &callsite, iree_allocator_system(), &size));
  words[1] = IREE_VM_NATIVE_CALL_LOCAL_VALUE | 8;
  IREE_ASSERT_OK(iree_vm_native_call_verify(&entry, &callsite,
                                            iree_allocator_system(), &size));
}

TEST(NativeCall, ScalarAccessPreservesCompleteUnalignedCells) {
  uint64_t values[2] = {};
  uint8_t local_bytes[16] = {};
  iree_vm_native_call_t call = {};
  call.values = values;
  call.local_bytes = local_bytes;
  for (uint32_t selector : {1u, IREE_VM_NATIVE_CALL_LOCAL_VALUE | 3}) {
    iree_vm_native_call_store_value(&call, selector,
                                    UINT64_C(0xFEDCBA9876543210));
    EXPECT_EQ(iree_vm_native_call_load_value(&call, selector),
              UINT64_C(0xFEDCBA9876543210));
  }
}

}  // namespace
