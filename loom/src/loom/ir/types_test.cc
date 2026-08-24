// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"

namespace loom {
namespace {

class OwnedFunctionType {
 public:
  explicit OwnedFunctionType(loom_type_t type) : type_(type) {}
  OwnedFunctionType(const OwnedFunctionType&) = delete;
  OwnedFunctionType& operator=(const OwnedFunctionType&) = delete;
  OwnedFunctionType(OwnedFunctionType&& other) noexcept : type_(other.type_) {
    other.type_ = loom_type_none();
  }
  OwnedFunctionType& operator=(OwnedFunctionType&& other) noexcept {
    if (this == &other) return *this;
    iree_allocator_free(iree_allocator_system(),
                        (void*)loom_type_func_data(type_));
    type_ = other.type_;
    other.type_ = loom_type_none();
    return *this;
  }
  ~OwnedFunctionType() {
    iree_allocator_free(iree_allocator_system(),
                        (void*)loom_type_func_data(type_));
  }

  loom_type_t get() const { return type_; }

 private:
  loom_type_t type_;
};

static OwnedFunctionType BuildFunctionType(const loom_type_t* arg_types,
                                           uint16_t arg_count,
                                           const loom_type_t* result_types,
                                           uint16_t result_count) {
  loom_type_t type = {0};
  IREE_CHECK_OK(loom_type_function_build(arg_types, arg_count, result_types,
                                         result_count, iree_allocator_system(),
                                         &type));
  return OwnedFunctionType(type);
}

TEST(TypesTest, FunctionTypeEqualAndHashAreStructural) {
  loom_type_t i32 = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t f64 = loom_type_scalar(LOOM_SCALAR_TYPE_F64);

  OwnedFunctionType first = BuildFunctionType(&i32, 1, &f32, 1);
  OwnedFunctionType duplicate = BuildFunctionType(&i32, 1, &f32, 1);
  OwnedFunctionType different = BuildFunctionType(&i32, 1, &f64, 1);

  EXPECT_TRUE(loom_type_equal(first.get(), duplicate.get()));
  EXPECT_EQ(loom_type_hash(first.get()), loom_type_hash(duplicate.get()));
  EXPECT_FALSE(loom_type_equal(first.get(), different.get()));
}

TEST(TypesTest, DialectTypeEqualAndHashAreStructural) {
  loom_type_t params_a[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t params_b[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I32,
                          loom_dim_pack_static(4), 0),
  };
  loom_type_t params_c[] = {
      loom_type_scalar(LOOM_SCALAR_TYPE_F32),
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_I64,
                          loom_dim_pack_static(4), 0),
  };

  loom_type_t first = loom_type_dialect((loom_string_id_t)70000u,
                                        IREE_ARRAYSIZE(params_a), params_a);
  loom_type_t duplicate = loom_type_dialect((loom_string_id_t)70000u,
                                            IREE_ARRAYSIZE(params_b), params_b);
  loom_type_t different_name = loom_type_dialect(
      (loom_string_id_t)70001u, IREE_ARRAYSIZE(params_b), params_b);
  loom_type_t different_params = loom_type_dialect(
      (loom_string_id_t)70000u, IREE_ARRAYSIZE(params_c), params_c);

  EXPECT_EQ(loom_type_dialect_name_id(first), 70000u);
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_hash(first), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(first, different_name));
  EXPECT_FALSE(loom_type_equal(first, different_params));
}

TEST(TypesTest, RegisterTypeEqualAndHashAreStructural) {
  loom_type_t first = loom_type_register_payload(42, 4);
  loom_type_t duplicate = loom_type_register_payload(42, 4);
  loom_type_t different_payload0 = loom_type_register_payload(43, 4);
  loom_type_t different_payload1 = loom_type_register_payload(42, 8);

  EXPECT_TRUE(loom_type_is_register(first));
  EXPECT_TRUE(loom_type_has_inline_dims(first));
  EXPECT_FALSE(loom_type_register_has_value_type(first));
  EXPECT_EQ(loom_type_register_payload0(first), 42u);
  EXPECT_EQ(loom_type_register_payload1(first), 4u);
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_hash(first), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(first, different_payload0));
  EXPECT_FALSE(loom_type_equal(first, different_payload1));
}

TEST(TypesTest, ModuleIndependenceIsRecursiveAndAllowsSsaIdentity) {
  loom_type_t scalar = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t dynamic_view = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(7), 0);
  dynamic_view.encoding_id = 9;
  dynamic_view.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_type_t static_encoded_view = dynamic_view;
  static_encoded_view.encoding_id = 1;
  static_encoded_view.encoding_flags = 0;
  loom_type_t dialect = loom_type_dialect(/*name_id=*/3, 0, nullptr);

  EXPECT_TRUE(loom_type_is_module_independent(scalar));
  EXPECT_TRUE(loom_type_is_module_independent(dynamic_view));
  EXPECT_FALSE(loom_type_is_module_independent(static_encoded_view));
  EXPECT_FALSE(loom_type_is_module_independent(dialect));

  loom_register_type_data_t scalar_register_data = {42, 4, scalar};
  loom_register_type_data_t dialect_register_data = {42, 4, dialect};
  EXPECT_TRUE(loom_type_is_module_independent(
      loom_type_register_payload_with_value_type(&scalar_register_data)));
  EXPECT_FALSE(loom_type_is_module_independent(
      loom_type_register_payload_with_value_type(&dialect_register_data)));

  loom_type_t independent_types[] = {scalar, dynamic_view};
  loom_type_t dependent_types[] = {scalar, dialect};
  EXPECT_TRUE(loom_type_sequence_is_module_independent(
      independent_types, IREE_ARRAYSIZE(independent_types)));
  EXPECT_FALSE(loom_type_sequence_is_module_independent(
      dependent_types, IREE_ARRAYSIZE(dependent_types)));

  OwnedFunctionType independent_function = BuildFunctionType(
      independent_types, IREE_ARRAYSIZE(independent_types), &scalar, 1);
  OwnedFunctionType dependent_function = BuildFunctionType(
      dependent_types, IREE_ARRAYSIZE(dependent_types), &scalar, 1);
  EXPECT_TRUE(loom_type_is_module_independent(independent_function.get()));
  EXPECT_FALSE(loom_type_is_module_independent(dependent_function.get()));
}

typedef struct ValueRefCapture {
  loom_value_id_t values[2];
  iree_host_size_t count;
} ValueRefCapture;

static iree_status_t CaptureValueRef(loom_value_id_t value_id,
                                     void* user_data) {
  ValueRefCapture* capture = (ValueRefCapture*)user_data;
  capture->values[capture->count++] = value_id;
  return iree_ok_status();
}

TEST(TypesTest, MayReferenceValuesConservativelyClassifiesTypes) {
  loom_type_t scalar = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t static_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t dynamic_vector = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(7), 0);
  loom_type_t dynamic_view = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  dynamic_view.encoding_id = 9;
  dynamic_view.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  loom_type_t static_pool = loom_type_pool(loom_dim_pack_static(4096));
  loom_type_t dynamic_pool = loom_type_pool(loom_dim_pack_dynamic(11));
  loom_register_type_data_t register_data = {42, 4, scalar};
  loom_type_t function_type = {};
  function_type.header = loom_type_make_raw_header(LOOM_TYPE_FUNCTION, 0, 0,
                                                   LOOM_TYPE_FLAG_ALL_STATIC);
  loom_type_t dialect_type = {};
  dialect_type.header = loom_type_make_raw_header(LOOM_TYPE_DIALECT, 0, 0,
                                                  LOOM_TYPE_FLAG_ALL_STATIC);
  loom_type_t parameterized_type = {};
  parameterized_type.header = loom_type_make_raw_header(
      LOOM_TYPE_PARAMETERIZED, 0, 0, LOOM_TYPE_FLAG_ALL_STATIC);

  EXPECT_FALSE(loom_type_may_reference_values(loom_type_none()));
  EXPECT_FALSE(loom_type_may_reference_values(scalar));
  EXPECT_FALSE(loom_type_may_reference_values(static_vector));
  EXPECT_FALSE(loom_type_may_reference_values(static_pool));
  EXPECT_FALSE(
      loom_type_may_reference_values(loom_type_register_payload(42, 4)));
  EXPECT_TRUE(loom_type_may_reference_values(dynamic_vector));
  EXPECT_TRUE(loom_type_may_reference_values(dynamic_view));
  EXPECT_TRUE(loom_type_may_reference_values(dynamic_pool));
  EXPECT_TRUE(loom_type_may_reference_values(
      loom_type_register_payload_with_value_type(&register_data)));
  EXPECT_TRUE(loom_type_may_reference_values(function_type));
  EXPECT_TRUE(loom_type_may_reference_values(dialect_type));
  EXPECT_TRUE(loom_type_may_reference_values(parameterized_type));
}

class ModuleTypesTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(32 * 1024, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("types_test"),
                                        &block_pool_, /*hints=*/NULL,
                                        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
};

TEST_F(ModuleTypesTest, RegisterValueTypeParticipatesInStructuralLifecycle) {
  loom_type_t source_value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(7), 0);
  loom_type_t duplicate_value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(7), 0);
  loom_type_t target_value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(9), 0);
  loom_type_t different_value_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_dynamic(7), 0);

  loom_register_type_data_t source_data = {42, 4, source_value_type};
  loom_register_type_data_t duplicate_data = {42, 4, duplicate_value_type};
  loom_register_type_data_t target_data = {42, 4, target_value_type};
  loom_register_type_data_t different_data = {42, 4, different_value_type};
  loom_type_t source = loom_type_register_payload_with_value_type(&source_data);
  loom_type_t duplicate =
      loom_type_register_payload_with_value_type(&duplicate_data);
  loom_type_t target = loom_type_register_payload_with_value_type(&target_data);
  loom_type_t different =
      loom_type_register_payload_with_value_type(&different_data);

  EXPECT_TRUE(loom_type_register_has_value_type(source));
  EXPECT_FALSE(loom_type_has_inline_dims(source));
  ASSERT_NE(loom_type_register_value_type(source), nullptr);
  EXPECT_TRUE(loom_type_equal(*loom_type_register_value_type(source),
                              source_value_type));
  EXPECT_EQ(loom_type_register_payload0(source), 42u);
  EXPECT_EQ(loom_type_register_payload1(source), 4u);
  EXPECT_TRUE(loom_type_equal(source, duplicate));
  EXPECT_EQ(loom_type_hash(source), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(source, different));
  EXPECT_FALSE(loom_type_equal(source, loom_type_register_payload(42, 4)));

  EXPECT_TRUE(loom_type_references_value(module_, source, 7));
  EXPECT_FALSE(loom_type_references_value(module_, source, 9));
  ValueRefCapture capture = {};
  IREE_ASSERT_OK(
      loom_type_walk_value_refs(module_, source, CaptureValueRef, &capture));
  ASSERT_EQ(capture.count, 1u);
  EXPECT_EQ(capture.values[0], 7u);

  loom_register_type_data_t scalar_data = {
      42, 4, loom_type_scalar(LOOM_SCALAR_TYPE_I32)};
  loom_type_t scalar = loom_type_register_payload_with_value_type(&scalar_data);
  EXPECT_FALSE(loom_type_references_value(module_, scalar, 7));
  capture = {};
  IREE_ASSERT_OK(
      loom_type_walk_value_refs(module_, scalar, CaptureValueRef, &capture));
  EXPECT_EQ(capture.count, 0u);

  loom_type_t static_vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I32, loom_dim_pack_static(4), 0);
  loom_register_type_data_t static_vector_data = {42, 4, static_vector_type};
  loom_type_t static_vector =
      loom_type_register_payload_with_value_type(&static_vector_data);
  EXPECT_FALSE(loom_type_references_value(module_, static_vector, 7));
  capture = {};
  IREE_ASSERT_OK(loom_type_walk_value_refs(module_, static_vector,
                                           CaptureValueRef, &capture));
  EXPECT_EQ(capture.count, 0u);

  loom_value_id_t source_values[] = {7};
  loom_value_id_t target_values[] = {9};
  loom_type_value_remap_t remap = {
      source_values,
      target_values,
      IREE_ARRAYSIZE(source_values),
  };
  EXPECT_TRUE(
      loom_type_equal_after_value_remap(module_, source, target, &remap));
  EXPECT_FALSE(loom_type_equal(source, target));
}

TEST_F(ModuleTypesTest, ValueRemapComposesDiscontiguousSpans) {
  loom_type_t source = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(7),
      loom_dim_pack_dynamic(11), 0);
  loom_type_t target = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(9),
      loom_dim_pack_dynamic(13), 0);
  loom_value_id_t source_outer[] = {7};
  loom_value_id_t target_outer[] = {9};
  loom_value_id_t source_inner[] = {11};
  loom_value_id_t target_inner[] = {13};
  const loom_type_value_remap_t inner_remap = {
      /*.source_values=*/source_inner,
      /*.target_values=*/target_inner,
      /*.count=*/IREE_ARRAYSIZE(source_inner),
  };
  const loom_type_value_remap_t remap = {
      /*.source_values=*/source_outer,
      /*.target_values=*/target_outer,
      /*.count=*/IREE_ARRAYSIZE(source_outer),
      /*.next=*/&inner_remap,
  };

  EXPECT_TRUE(
      loom_type_equal_after_value_remap(module_, source, target, &remap));
  EXPECT_FALSE(
      loom_type_equal_after_value_remap(module_, source, target, &inner_remap));
}

TEST(TypesTest, RegisterClassNamesMustBeNamespaceQualified) {
  EXPECT_TRUE(loom_register_class_name_is_qualified(IREE_SV("amdgpu.vgpr")));
  EXPECT_TRUE(
      loom_register_class_name_is_qualified(IREE_SV("test.local.v128")));
  EXPECT_FALSE(loom_register_class_name_is_qualified(IREE_SV("vgpr")));
  EXPECT_FALSE(loom_register_class_name_is_qualified(IREE_SV(".vgpr")));
  EXPECT_FALSE(loom_register_class_name_is_qualified(IREE_SV("amdgpu.")));
}

TEST(TypesTest, VectorIsShapedButLayoutFree) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(16), 0);
  EXPECT_TRUE(loom_type_is_shaped(type));
  EXPECT_TRUE(loom_type_is_vector(type));
  EXPECT_FALSE(loom_type_can_have_encoding(type));
}

TEST(TypesTest, StaticZeroExtentImpliesZeroElements) {
  loom_type_t all_static =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(0), loom_dim_pack_static(16), 0);
  uint64_t element_count = UINT64_MAX;
  EXPECT_TRUE(loom_type_has_static_zero_extent(all_static));
  EXPECT_TRUE(loom_type_static_element_count(all_static, &element_count));
  EXPECT_EQ(element_count, 0u);

  loom_type_t mixed_dynamic = loom_type_shaped_2d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(0),
      loom_dim_pack_dynamic(42), 0);
  element_count = UINT64_MAX;
  EXPECT_TRUE(loom_type_has_static_zero_extent(mixed_dynamic));
  EXPECT_FALSE(loom_type_static_element_count(mixed_dynamic, &element_count));
  EXPECT_EQ(element_count, 0u);

  loom_type_t dynamic_unknown = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_dynamic(42), 0);
  EXPECT_FALSE(loom_type_has_static_zero_extent(dynamic_unknown));
}

TEST(TypesTest, ViewCanCarryLayoutAttachment) {
  loom_type_t type = loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                                         loom_dim_pack_static(256), 7);
  EXPECT_TRUE(loom_type_is_shaped(type));
  EXPECT_TRUE(loom_type_is_view(type));
  EXPECT_TRUE(loom_type_can_have_encoding(type));
  EXPECT_TRUE(loom_type_has_static_encoding(type));
}

TEST(TypesTest, BufferIsOpaqueStorageIdentity) {
  loom_type_t type = loom_type_buffer();
  EXPECT_TRUE(loom_type_is_buffer(type));
  EXPECT_FALSE(loom_type_is_shaped(type));
  EXPECT_FALSE(loom_type_has_encoding(type));
  EXPECT_TRUE(loom_type_is_all_static(type));
  EXPECT_TRUE(loom_type_equal(type, loom_type_buffer()));
  EXPECT_EQ(loom_type_hash(type), loom_type_hash(loom_type_buffer()));
  EXPECT_TRUE(loom_type_is_all_static(loom_type_buffer()));
}

TEST(TypesTest, EncodingRoleIsStructural) {
  loom_type_t unknown = loom_type_encoding();
  loom_type_t layout =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_type_t duplicate_layout =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_type_t schema =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_STORAGE_SCHEMA);

  EXPECT_TRUE(loom_type_is_encoding(layout));
  EXPECT_EQ(loom_type_encoding_role(unknown), LOOM_ENCODING_ROLE_UNKNOWN);
  EXPECT_EQ(loom_type_encoding_role(layout), LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  EXPECT_TRUE(loom_type_equal(layout, duplicate_layout));
  EXPECT_EQ(loom_type_hash(layout), loom_type_hash(duplicate_layout));
  EXPECT_FALSE(loom_type_equal(unknown, layout));
  EXPECT_FALSE(loom_type_equal(layout, schema));
}

TEST(TypesTest, UnassignedTypeKindIsInvalid) {
  EXPECT_FALSE(loom_type_kind_is_valid((loom_type_kind_t)4));
}

TEST(TypesTest, InvalidKindEqualityAndHashDoNotInterpretPayload) {
  loom_type_t first = {0};
  first.header =
      loom_type_make_header((loom_type_kind_t)99, (loom_scalar_type_t)0, 3, 0);
  first.dims[0] = 1;
  first.dims[1] = 2;
  loom_type_t duplicate = first;
  loom_type_t different = first;
  different.header =
      loom_type_make_header((loom_type_kind_t)100, (loom_scalar_type_t)0, 3, 0);
  loom_type_t different_payload = first;
  different_payload.dims[1] = 3;

  EXPECT_FALSE(loom_type_kind_is_valid(loom_type_kind(first)));
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_hash(first), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(first, different));
  EXPECT_FALSE(loom_type_equal(first, different_payload));
}

}  // namespace
}  // namespace loom
