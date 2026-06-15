// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ir/types.h"

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

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
  EXPECT_EQ(loom_type_register_payload0(first), 42u);
  EXPECT_EQ(loom_type_register_payload1(first), 4u);
  EXPECT_TRUE(loom_type_equal(first, duplicate));
  EXPECT_EQ(loom_type_hash(first), loom_type_hash(duplicate));
  EXPECT_FALSE(loom_type_equal(first, different_payload0));
  EXPECT_FALSE(loom_type_equal(first, different_payload1));
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
