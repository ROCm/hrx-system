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
#include "iree/vm/buffer_provider.h"

namespace {

const iree_vm_ref_type_t kRefTypes[] = {iree_vm_ref_type_storage_at(
    iree_vm_buffer_provider_table()->types, IREE_VM_REF_TYPE_BUFFER)};

iree_status_t VerifyAndLink(const iree_vm_native_export_t* entry,
                            const iree_vm_native_callsite_t* callsite,
                            iree_host_size_t* out_size) {
  IREE_RETURN_IF_ERROR(
      iree_vm_native_call_verify(callsite, iree_allocator_system()));
  return iree_vm_native_call_link(entry, kRefTypes, callsite, kRefTypes,
                                  iree_allocator_system(), out_size);
}

const iree_vm_native_call_field_t kFields[] = {
    {IREE_VM_NATIVE_CALL_FIELD_REF, 0},
    {IREE_VM_SCALAR_TYPE_I64, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_NATIVE_CALL_FIELD_REF, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
    {IREE_VM_NATIVE_CALL_FIELD_RANGE, 1},
};
const iree_vm_native_call_layout_t kLayouts[] = {
    {0, 2, 2, 0}, {2, 1, 1, 0}, {3, 6, 3, 1}};
const iree_vm_native_call_schema_t kSchema = {kLayouts, kFields, 3, 6, 2};
const uint32_t kScratchSizes[] = {16, 8, 0};
const iree_vm_native_export_t kExport = {
    .name = IREE_SV("submit"),
    .version = 7,
    .schema = &kSchema,
    .scratch_sizes = kScratchSizes,
};

class NativeCallTest : public ::testing::Test {
 protected:
  iree_status_t Verify(const std::vector<uint32_t>& words,
                       iree_host_size_t* out_size) {
    const iree_vm_native_callsite_t callsite = {
        IREE_SV("native"),
        IREE_SV("submit"),
        7,
        kSchema,
        0,
        words.data(),
        static_cast<uint32_t>(words.size()),
        {4, 4, 24}};
    return VerifyAndLink(&kExport, &callsite, out_size);
  }
  // Two independent semaphore-like lists and two result slots. Results may
  // replace any input; only result/result overlap is prohibited.
  std::vector<uint32_t> words_ = {2, 6, 1, 10, 2, 12, 0, 0, 1, 1, 2, 2, 0, 1};
};

TEST_F(NativeCallTest, SizesIndependentRangesAndAliasedResults) {
  IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&kSchema, 1));
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
  std::vector<iree_vm_native_call_field_t> fields(std::begin(kFields),
                                                  std::end(kFields));
  fields.push_back({IREE_VM_NATIVE_CALL_FIELD_RANGE, 2});
  std::vector<iree_vm_native_call_layout_t> layouts(std::begin(kLayouts),
                                                    std::end(kLayouts));
  layouts.push_back({6, 2, 1, 2});
  const iree_vm_native_call_schema_t schema = {layouts.data(), fields.data(), 4,
                                               7, 3};
  const uint32_t scratch_sizes[] = {16, 8, 0, 32};
  const iree_vm_native_export_t entry = {
      .name = IREE_SV("nested"),
      .schema = &schema,
      .scratch_sizes = scratch_sizes,
  };
  // Reserve both parent records, then each parent's descendants in order.
  const uint32_t words[] = {2,  2, 1,  14, 0,  16, 0, 16, 0,
                            16, 1, 16, 0,  18, 0,  0, 1,  1};
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"),     IREE_SV("nested"), 0, schema, 0, words,
      IREE_ARRAYSIZE(words), {2, 2, 0}};
  iree_host_size_t size = 0;
  IREE_ASSERT_OK(VerifyAndLink(&entry, &callsite, &size));
  EXPECT_EQ(size, 64u);
}

TEST(NativeCall, LocalScalarResultsCannotPartiallyOverlap) {
  const iree_vm_native_call_field_t fields[] = {
      {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_SCALAR_TYPE_I64, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_SCALAR_TYPE_I64, 0},
  };
  const iree_vm_native_call_layout_t layout = {0, 2, 2, 0};
  const iree_vm_native_call_schema_t schema = {&layout, fields, 1, 2, 0};
  const iree_vm_native_export_t entry = {
      .name = IREE_SV("results"),
      .schema = &schema,
  };
  uint32_t words[] = {IREE_VM_NATIVE_CALL_LOCAL_VALUE,
                      IREE_VM_NATIVE_CALL_LOCAL_VALUE | 4};
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"), IREE_SV("results"), 0, schema, 0, words, 2,
      {0, 0, 16}};
  iree_host_size_t size = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        VerifyAndLink(&entry, &callsite, &size));
  words[1] = IREE_VM_NATIVE_CALL_LOCAL_VALUE | 8;
  IREE_ASSERT_OK(VerifyAndLink(&entry, &callsite, &size));
}

TEST(NativeCall, ScalarAccessPreservesCompleteUnalignedCells) {
  uint64_t values[2] = {};
  uint8_t local_bytes[16] = {};
  iree_vm_native_call_t call = {
      .values = values,
      .local_bytes = local_bytes,
  };
  for (uint32_t selector : {1u, IREE_VM_NATIVE_CALL_LOCAL_VALUE | 3}) {
    iree_vm_native_call_store_value(&call, selector,
                                    UINT64_C(0xFEDCBA9876543210));
    EXPECT_EQ(iree_vm_native_call_load_value(&call, selector),
              UINT64_C(0xFEDCBA9876543210));
  }
}

TEST(NativeCallSchema, ExactScalarKindsAndResultDirections) {
  for (uint16_t kind = IREE_VM_SCALAR_TYPE_I8; kind <= IREE_VM_SCALAR_TYPE_F64;
       ++kind) {
    const iree_vm_native_call_field_t fields[] = {
        {kind, 0},
        {static_cast<uint16_t>(kind | IREE_VM_NATIVE_CALL_FIELD_RESULT), 0}};
    const iree_vm_native_call_layout_t layout = {0, 2, 2, 0};
    const iree_vm_native_call_schema_t schema = {&layout, fields, 1, 2, 0};
    IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 0));
  }
}

TEST(NativeCallSchema, RejectsInvalidFieldKindsAndOrdinals) {
  const iree_vm_native_call_field_t invalid_fields[] = {
      {0, 0},
      {IREE_VM_SCALAR_TYPE_F64 + 1, 0},
      {IREE_VM_SCALAR_TYPE_I64, 1},
      {IREE_VM_NATIVE_CALL_FIELD_REF, 1},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
      {0x0400, 0}};
  const iree_vm_native_call_layout_t layout = {0, 1, 1, 0};
  for (const auto& field : invalid_fields) {
    const iree_vm_native_call_schema_t schema = {&layout, &field, 1, 1, 0};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_vm_native_call_schema_validate(&schema, 1));
  }
}

TEST(NativeCallSchema, RejectsIncorrectGeometry) {
  const iree_vm_native_call_field_t field = {IREE_VM_SCALAR_TYPE_I64, 0};
  const iree_vm_native_call_layout_t invalid_layouts[] = {
      {1, 1, 1, 0}, {0, 0, 1, 0}, {0, 1, 2, 0}, {0, 1, 1, 1}, {0, 0, 0, 0}};
  for (const auto& layout : invalid_layouts) {
    const iree_vm_native_call_schema_t schema = {&layout, &field, 1, 1, 0};
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          iree_vm_native_call_schema_validate(&schema, 0));
  }
}

TEST(NativeCallSchema, SharedChildrenDoNotExpandEmptyRanges) {
  // Two references per parent describe exponentially many paths, but the
  // declaration and its empty root record require only linear verification.
  std::vector<iree_vm_native_call_layout_t> layouts = {{0, 1, 1, 0}};
  std::vector<iree_vm_native_call_field_t> fields = {
      {IREE_VM_SCALAR_TYPE_I64, 0}};
  for (uint16_t i = 1; i < 128; ++i) {
    layouts.push_back({static_cast<uint32_t>(fields.size()), 4, 2, i});
    fields.push_back(
        {IREE_VM_NATIVE_CALL_FIELD_RANGE, static_cast<uint16_t>(i - 1)});
    fields.push_back(fields.back());
  }
  const iree_vm_native_call_schema_t schema = {
      layouts.data(), fields.data(), static_cast<uint32_t>(layouts.size()),
      static_cast<uint32_t>(fields.size()), 127};
  IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 0));
  const iree_vm_native_export_t entry = {
      .name = IREE_SV("empty"),
      .schema = &schema,
  };
  const uint32_t words[] = {0, 4, 0, 4};
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"), IREE_SV("empty"), 0, schema, 0, words, 4, {}};
  iree_host_size_t size = 1;
  IREE_ASSERT_OK(VerifyAndLink(&entry, &callsite, &size));
  EXPECT_EQ(size, 0u);
}

TEST(NativeCallSchema, DeepRecordsUseAnExplicitStack) {
  constexpr uint16_t kDepth = 16384;
  std::vector<iree_vm_native_call_layout_t> layouts = {{0, 1, 1, 0}};
  std::vector<iree_vm_native_call_field_t> fields = {
      {IREE_VM_SCALAR_TYPE_I64, 0}};
  std::vector<uint32_t> words;
  for (uint16_t i = 1; i <= kDepth; ++i) {
    layouts.push_back({i, 2, 1, i});
    fields.push_back(
        {IREE_VM_NATIVE_CALL_FIELD_RANGE, static_cast<uint16_t>(i - 1)});
    words.push_back(1);
    words.push_back(2u * i);
  }
  words.push_back(0);
  const iree_vm_native_call_schema_t schema = {
      layouts.data(), fields.data(), static_cast<uint32_t>(layouts.size()),
      static_cast<uint32_t>(fields.size()), kDepth};
  IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 0));
  const iree_vm_native_export_t entry = {
      .name = IREE_SV("deep"),
      .schema = &schema,
  };
  const iree_vm_native_callsite_t callsite = {
      IREE_SV("native"),
      IREE_SV("deep"),
      0,
      schema,
      0,
      words.data(),
      static_cast<uint32_t>(words.size()),
      {1, 0, 0}};
  iree_host_size_t size = 1;
  IREE_ASSERT_OK(VerifyAndLink(&entry, &callsite, &size));
  EXPECT_EQ(size, 0u);
}

TEST(NativeCallLink, EmptyRangesStillRequireExactLeafTypesAndDirections) {
  const uint32_t words[] = {0, 6, 0, 6, 0, 6};
  const iree_vm_native_callsite_t callsite = {
      .version = 7,
      .schema = kSchema,
      .record = words,
      .word_count = IREE_ARRAYSIZE(words),
  };
  IREE_ASSERT_OK(
      iree_vm_native_call_verify(&callsite, iree_allocator_system()));
  for (uint16_t kind : std::array<uint16_t, 3>{
           IREE_VM_SCALAR_TYPE_I32, IREE_VM_SCALAR_TYPE_F64,
           IREE_VM_NATIVE_CALL_FIELD_RESULT | IREE_VM_SCALAR_TYPE_I64}) {
    std::vector<iree_vm_native_call_field_t> fields(std::begin(kFields),
                                                    std::end(kFields));
    fields[1].kind = kind;
    auto schema = kSchema;
    schema.fields = fields.data();
    IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 1));
    auto entry = kExport;
    entry.schema = &schema;
    iree_host_size_t size = 123;
    IREE_EXPECT_STATUS_IS(
        IREE_STATUS_INCOMPATIBLE,
        iree_vm_native_call_link(&entry, kRefTypes, &callsite, kRefTypes,
                                 iree_allocator_system(), &size));
    EXPECT_EQ(size, 123u);
  }
}

TEST(NativeCallLink, RefAndScalarPermutationsCannotShareAnEncoding) {
  const iree_vm_native_call_layout_t layout = {0, 2, 2, 0};
  const iree_vm_native_call_field_t caller_fields[] = {
      {IREE_VM_SCALAR_TYPE_I64, 0}, {IREE_VM_NATIVE_CALL_FIELD_REF, 0}};
  const iree_vm_native_call_field_t target_fields[] = {
      {IREE_VM_NATIVE_CALL_FIELD_REF, 0}, {IREE_VM_SCALAR_TYPE_I64, 0}};
  const iree_vm_native_call_schema_t target_schema = {&layout, target_fields, 1,
                                                      2, 0};
  const uint32_t words[] = {0, 0};
  const iree_vm_native_callsite_t callsite = {
      .schema = {&layout, caller_fields, 1, 2, 0},
      .record = words,
      .word_count = 2,
      .storage = {1, 1, 0},
  };
  const iree_vm_native_export_t entry = {.schema = &target_schema};
  IREE_ASSERT_OK(
      iree_vm_native_call_verify(&callsite, iree_allocator_system()));
  iree_host_size_t size = 123;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_vm_native_call_link(&entry, kRefTypes, &callsite, kRefTypes,
                               iree_allocator_system(), &size));
  EXPECT_EQ(size, 123u);
}

TEST(NativeCallLink, RefOrdinalsAreLocalButResolvedIdentitiesAreExact) {
  const iree_vm_native_call_layout_t layout = {0, 1, 1, 0};
  const iree_vm_native_call_field_t caller_field = {
      IREE_VM_NATIVE_CALL_FIELD_REF, 0};
  const iree_vm_native_call_field_t target_field = {
      IREE_VM_NATIVE_CALL_FIELD_REF, 1};
  const iree_vm_native_call_schema_t target_schema = {&layout, &target_field, 1,
                                                      1, 0};
  const uint32_t word = 0;
  const iree_vm_native_callsite_t callsite = {
      .schema = {&layout, &caller_field, 1, 1, 0},
      .record = &word,
      .word_count = 1,
      .storage = {0, 1, 0},
  };
  const iree_vm_native_export_t entry = {.schema = &target_schema};
  const iree_vm_ref_type_descriptor_t same_name_different_identity =
      *kRefTypes[0];
  iree_vm_ref_type_t target_types[] = {&same_name_different_identity,
                                       kRefTypes[0]};
  IREE_ASSERT_OK(
      iree_vm_native_call_verify(&callsite, iree_allocator_system()));
  iree_host_size_t size = 123;
  IREE_ASSERT_OK(iree_vm_native_call_link(&entry, target_types, &callsite,
                                          kRefTypes, iree_allocator_system(),
                                          &size));
  EXPECT_EQ(size, 0u);
  std::swap(target_types[0], target_types[1]);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INCOMPATIBLE,
      iree_vm_native_call_link(&entry, target_types, &callsite, kRefTypes,
                               iree_allocator_system(), &size));
}

TEST(NativeCallLink, SharedAndDuplicatedChildrenHaveTheSameLayout) {
  const iree_vm_native_call_layout_t layouts[] = {
      {0, 2, 2, 0}, {2, 2, 2, 0}, {4, 1, 1, 0}, {5, 6, 3, 1}};
  const iree_vm_native_call_field_t fields[] = {
      kFields[0],
      kFields[1],
      kFields[0],
      kFields[1],
      kFields[2],
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 0},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 1},
      {IREE_VM_NATIVE_CALL_FIELD_RANGE, 2},
  };
  const iree_vm_native_call_schema_t schema = {layouts, fields, 4, 8, 3};
  IREE_ASSERT_OK(iree_vm_native_call_schema_validate(&schema, 1));
  const uint32_t words[] = {0, 6, 0, 6, 0, 6};
  const iree_vm_native_callsite_t callsite = {
      .version = 7,
      .schema = schema,
      .record = words,
      .word_count = IREE_ARRAYSIZE(words),
  };
  iree_host_size_t size = 123;
  IREE_ASSERT_OK(VerifyAndLink(&kExport, &callsite, &size));
  EXPECT_EQ(size, 0u);
}

}  // namespace
