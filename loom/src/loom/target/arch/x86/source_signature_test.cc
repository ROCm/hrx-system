// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/x86/source_signature.h"

#include <initializer_list>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/func/ops.h"

namespace loom {
namespace {

struct TestField {
  const char* name;
  uint32_t type;
  uint32_t byte_offset;
};

class X86SourceSignatureTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    iree_arena_initialize(&block_pool_, &result_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t op_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_func_dialect_vtables(&op_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_FUNC,
                                                 vtables, (uint16_t)op_count));
    iree_host_size_t attr_count = 0;
    const loom_parameterized_attr_descriptor_t* attrs =
        loom_func_dialect_parameterized_attrs(&attr_count);
    IREE_ASSERT_OK(loom_context_register_parameterized_attrs(
        &context_, LOOM_DIALECT_FUNC, attrs, attr_count));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(
        &context_, IREE_SV("source_signature_test"), &block_pool_, NULL,
        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&result_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint32_t AddType(
      loom_func_abi_c_type_attr_build_flags_t flags,
      loom_func_abi_c_type_kind_t kind, int64_t byte_length,
      int64_t byte_alignment, int64_t component_count, int64_t bitwidth = 0,
      loom_func_abi_c_type_format_t format = (loom_func_abi_c_type_format_t)0,
      int64_t element_type = 0, int64_t element_count = 0,
      int64_t element_stride = 0,
      loom_parameterized_attr_array_t fields =
          loom_parameterized_attr_array_empty()) {
    loom_attribute_t type;
    IREE_CHECK_OK(loom_func_abi_c_type_attr_make(
        module_, flags, kind, byte_length, byte_alignment, component_count,
        bitwidth, format, /*address_space=*/0, element_type, element_count,
        element_stride, LOOM_STRING_ID_INVALID, fields, &type));
    types_.push_back(type);
    return (uint32_t)types_.size() - 1u;
  }

  uint32_t AddInteger(uint32_t bitwidth, bool is_unsigned = false) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_BITWIDTH,
                   is_unsigned ? LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER
                               : LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER,
                   bitwidth / 8u, bitwidth / 8u, 1, bitwidth);
  }

  uint32_t AddFloat(loom_func_abi_c_type_format_t format,
                    uint32_t byte_length) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FORMAT,
                   LOOM_FUNC_ABI_C_TYPE_KIND_FLOATING, byte_length, byte_length,
                   1, /*bitwidth=*/0, format);
  }

  uint32_t AddPointer() {
    return AddType(/*flags=*/0, LOOM_FUNC_ABI_C_TYPE_KIND_POINTER,
                   /*byte_length=*/8, /*byte_alignment=*/8,
                   /*component_count=*/2);
  }

  uint32_t AddVector(uint32_t element_type, uint32_t element_count,
                     uint32_t byte_length, uint32_t byte_alignment) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT,
                   LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR, byte_length,
                   byte_alignment, 1, /*bitwidth=*/0,
                   (loom_func_abi_c_type_format_t)0, element_type,
                   element_count);
  }

  uint32_t AddArray(uint32_t element_type, uint32_t element_count,
                    uint32_t element_stride, uint32_t byte_alignment,
                    uint32_t component_count) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_STRIDE,
                   LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY,
                   element_count * element_stride, byte_alignment,
                   component_count, /*bitwidth=*/0,
                   (loom_func_abi_c_type_format_t)0, element_type,
                   element_count, element_stride);
  }

  uint32_t AddRecord(uint32_t byte_length, uint32_t byte_alignment,
                     uint32_t component_count,
                     std::initializer_list<TestField> fields) {
    std::vector<loom_attribute_t> field_attrs;
    for (const TestField& field : fields) {
      loom_string_id_t name = LOOM_STRING_ID_INVALID;
      IREE_CHECK_OK(loom_module_intern_string(
          module_, iree_make_cstring_view(field.name), &name));
      loom_attribute_t field_attr;
      IREE_CHECK_OK(loom_func_abi_c_field_attr_make(
          module_, name, field.type, field.byte_offset, &field_attr));
      field_attrs.push_back(field_attr);
    }
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_FIELDS,
                   LOOM_FUNC_ABI_C_TYPE_KIND_RECORD, byte_length,
                   byte_alignment, component_count, /*bitwidth=*/0,
                   (loom_func_abi_c_type_format_t)0, /*element_type=*/0,
                   /*element_count=*/0, /*element_stride=*/0,
                   loom_make_parameterized_attr_array(field_attrs.data(),
                                                      field_attrs.size()));
  }

  loom_x86_source_signature_classification_t Classify(
      uint32_t root, uint16_t maximum_vector_bytes) {
    const int64_t root_type = root;
    const loom_func_c_abi_signature_t signature = {
        .arguments = loom_make_i64_array(&root_type, 1),
        .types =
            loom_make_parameterized_attr_array(types_.data(), types_.size()),
    };
    loom_c_abi_signature_layout_t layout;
    IREE_CHECK_OK(loom_c_abi_signature_layout_build(&signature, &scratch_arena_,
                                                    &result_arena_, &layout));
    const loom_c_abi_object_t* object = &layout.arguments[0];
    const loom_c_abi_leaf_t* leaves =
        object->leaf_count ? layout.argument_leaves + object->leaf_offset
                           : nullptr;
    loom_x86_source_signature_classification_t classification;
    EXPECT_TRUE(loom_x86_source_signature_classify(
        &layout, object, leaves, maximum_vector_bytes, &classification));
    return classification;
  }

  void ExpectClasses(
      const loom_x86_source_signature_classification_t& actual,
      std::initializer_list<loom_x86_source_signature_class_t> expected) {
    ASSERT_FALSE(actual.is_memory);
    ASSERT_EQ(actual.eightbyte_count, expected.size());
    size_t i = 0;
    for (loom_x86_source_signature_class_t expected_class : expected) {
      EXPECT_EQ(actual.eightbytes[i++], expected_class);
    }
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  iree_arena_allocator_t result_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  std::vector<loom_attribute_t> types_;
};

TEST_F(X86SourceSignatureTest, ClassifiesCompleteScalarFamily) {
  const uint32_t boolean =
      AddType(/*flags=*/0, LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN, 1, 1, 1);
  const uint32_t signed_i8 = AddInteger(8);
  const uint32_t unsigned_i16 = AddInteger(16, true);
  const uint32_t signed_i32 = AddInteger(32);
  const uint32_t unsigned_i64 = AddInteger(64, true);
  const uint32_t f8 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F8E4M3FN, 1);
  const uint32_t f16 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F16, 2);
  const uint32_t bf16 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_BF16, 2);
  const uint32_t f32 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F32, 4);
  const uint32_t f64 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F64, 8);
  const uint32_t pointer = AddPointer();

  for (uint32_t root : {boolean, signed_i8, unsigned_i16, signed_i32,
                        unsigned_i64, f8, pointer}) {
    ExpectClasses(Classify(root, 32),
                  {LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER});
  }
  for (uint32_t root : {f16, bf16, f32, f64}) {
    ExpectClasses(Classify(root, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE});
  }
  EXPECT_EQ(Classify(pointer, 32).component_count, 2u);
}

TEST_F(X86SourceSignatureTest, MergesRecordFieldsWithinEachEightbyte) {
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t f32 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F32, 4);
  const uint32_t f64 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F64, 8);
  const uint32_t pair =
      AddRecord(8, 4, 2, {{"first", u32, 0}, {"second", u32, 4}});
  const uint32_t mixed =
      AddRecord(8, 4, 2, {{"value", f32, 0}, {"tag", u32, 4}});
  const uint32_t split =
      AddRecord(16, 8, 2, {{"value", f64, 0}, {"tag", u32, 8}});
  const uint32_t doubles =
      AddRecord(16, 8, 2, {{"first", f64, 0}, {"second", f64, 8}});

  ExpectClasses(Classify(pair, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER});
  ExpectClasses(Classify(mixed, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER});
  ExpectClasses(Classify(split, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                      LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER});
  ExpectClasses(Classify(doubles, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                        LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE});
}

TEST_F(X86SourceSignatureTest, DistinguishesNativeVectorsFromScalarArrays) {
  const uint32_t f32 = AddFloat(LOOM_FUNC_ABI_C_TYPE_FORMAT_F32, 4);
  const uint32_t vector4 = AddVector(f32, 4, 16, 16);
  const uint32_t vector8 = AddVector(f32, 8, 32, 32);
  const uint32_t vector16 = AddVector(f32, 16, 64, 64);
  const uint32_t wrapped_vector = AddRecord(32, 32, 1, {{"value", vector8, 0}});
  const uint32_t array4 = AddArray(f32, 4, 4, 4, 4);
  const uint32_t array8 = AddArray(f32, 8, 4, 4, 8);

  ExpectClasses(Classify(vector4, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                        LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP});
  ExpectClasses(Classify(vector8, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                        LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                                        LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                                        LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP});
  ExpectClasses(Classify(wrapped_vector, 32),
                {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP});
  ExpectClasses(Classify(vector16, 64),
                {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_SSEUP});
  ExpectClasses(Classify(array4, 32), {LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE,
                                       LOOM_X86_SOURCE_SIGNATURE_CLASS_SSE});
  EXPECT_TRUE(Classify(array8, 32).is_memory);
  EXPECT_TRUE(Classify(vector8, 16).is_memory);
  EXPECT_TRUE(Classify(vector16, 32).is_memory);
}

TEST_F(X86SourceSignatureTest, SendsLargeAndUnalignedRecordsThroughMemory) {
  const uint32_t u8 = AddInteger(8, true);
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t u64 = AddInteger(64, true);
  const uint32_t large = AddRecord(
      24, 8, 3, {{"first", u64, 0}, {"second", u64, 8}, {"third", u64, 16}});
  const uint32_t packed =
      AddRecord(5, 1, 2, {{"first", u8, 0}, {"second", u32, 1}});

  EXPECT_TRUE(Classify(large, 32).is_memory);
  EXPECT_TRUE(Classify(packed, 32).is_memory);
}

TEST_F(X86SourceSignatureTest, RetainsPaddingAsNoClass) {
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t padded = AddRecord(16, 4, 1, {{"value", u32, 0}});
  ExpectClasses(Classify(padded, 32),
                {LOOM_X86_SOURCE_SIGNATURE_CLASS_INTEGER,
                 LOOM_X86_SOURCE_SIGNATURE_CLASS_NO_CLASS});
}

TEST_F(X86SourceSignatureTest, RejectsEmptySourceObjects) {
  const uint32_t empty = AddRecord(1, 1, 0, {});
  const int64_t root_type = empty;
  const loom_func_c_abi_signature_t signature = {
      .arguments = loom_make_i64_array(&root_type, 1),
      .types = loom_make_parameterized_attr_array(types_.data(), types_.size()),
  };
  loom_c_abi_signature_layout_t layout;
  IREE_ASSERT_OK(loom_c_abi_signature_layout_build(&signature, &scratch_arena_,
                                                   &result_arena_, &layout));
  loom_x86_source_signature_classification_t classification;
  EXPECT_FALSE(loom_x86_source_signature_classify(&layout, &layout.arguments[0],
                                                  /*object_leaves=*/nullptr, 32,
                                                  &classification));
}

}  // namespace
}  // namespace loom
