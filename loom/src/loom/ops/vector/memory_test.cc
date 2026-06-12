// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/vector/memory.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"

namespace loom {
namespace {

class VectorMemoryTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t encoding_vtable_count = 0;
    const loom_op_vtable_t* const* encoding_vtables =
        loom_encoding_dialect_vtables(&encoding_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, encoding_vtables,
        (uint16_t)encoding_vtable_count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void BuildDenseLayout(loom_value_id_t* out_layout) {
    loom_op_t* layout = nullptr;
    IREE_ASSERT_OK(loom_encoding_layout_dense_build(
        &builder_, loom_type_encoding(), LOOM_LOCATION_UNKNOWN, &layout));
    *out_layout = loom_encoding_layout_dense_result(layout);
  }

  void BuildStridedLayout(const loom_value_id_t* dynamic_strides,
                          iree_host_size_t dynamic_stride_count,
                          const int64_t* static_strides,
                          iree_host_size_t static_stride_count,
                          loom_value_id_t* out_layout) {
    loom_op_t* layout = nullptr;
    IREE_ASSERT_OK(loom_encoding_layout_strided_build(
        &builder_, dynamic_strides, dynamic_stride_count, static_strides,
        static_stride_count, loom_type_encoding(), LOOM_LOCATION_UNKNOWN,
        &layout));
    *out_layout = loom_encoding_layout_strided_result(layout);
  }

  uint16_t AddEncoding(iree_string_view_t name,
                       const loom_named_attr_t* attributes,
                       uint8_t attribute_count) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    loom_encoding_t encoding = {
        /*.name_id=*/name_id,
        /*.alias_id=*/LOOM_STRING_ID_INVALID,
        /*.attribute_count=*/attribute_count,
        /*.reserved=*/{},
        /*.attributes=*/attributes,
    };
    uint16_t encoding_id = 0;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &encoding, &encoding_id));
    return encoding_id;
  }

  uint16_t AddGgmlQ4_0Schema() {
    loom_string_id_t block_elems_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t storage_bytes_name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(loom_module_intern_string(module_, IREE_SV("block_elems"),
                                            &block_elems_name));
    IREE_CHECK_OK(loom_module_intern_string(module_, IREE_SV("storage_bytes"),
                                            &storage_bytes_name));
    loom_named_attr_t attributes[] = {
        {
            /*.name_id=*/block_elems_name,
            /*.reserved=*/{},
            /*.value=*/loom_attr_i64(32),
        },
        {
            /*.name_id=*/storage_bytes_name,
            /*.reserved=*/{},
            /*.value=*/loom_attr_i64(18),
        },
    };
    return AddEncoding(IREE_SV("ggml_q4_0"), attributes,
                       (uint8_t)IREE_ARRAYSIZE(attributes));
  }

  uint16_t AddStaticStridedLayout(int64_t stride) {
    loom_string_id_t stride_name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("stride"), &stride_name));
    loom_named_attr_t attributes[] = {
        {
            /*.name_id=*/stride_name,
            /*.reserved=*/{},
            /*.value=*/loom_attr_i64(stride),
        },
    };
    return AddEncoding(IREE_SV("strided"), attributes,
                       (uint8_t)IREE_ARRAYSIZE(attributes));
  }

  uint16_t AddStaticPhysicalStorage(uint16_t layout, uint16_t schema) {
    loom_string_id_t layout_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t schema_name = LOOM_STRING_ID_INVALID;
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("layout"), &layout_name));
    IREE_CHECK_OK(
        loom_module_intern_string(module_, IREE_SV("schema"), &schema_name));
    loom_named_attr_t attributes[] = {
        {
            /*.name_id=*/layout_name,
            /*.reserved=*/{},
            /*.value=*/loom_attr_encoding(layout),
        },
        {
            /*.name_id=*/schema_name,
            /*.reserved=*/{},
            /*.value=*/loom_attr_encoding(schema),
        },
    };
    return AddEncoding(IREE_SV("physical_storage"), attributes,
                       (uint8_t)IREE_ARRAYSIZE(attributes));
  }

  void BuildGgmlQ4_0Schema(loom_value_id_t* out_schema) {
    uint16_t spec = AddGgmlQ4_0Schema();
    loom_op_t* schema = nullptr;
    IREE_ASSERT_OK(loom_encoding_define_build(
        &builder_, spec, /*params=*/nullptr, /*params_count=*/0,
        loom_type_encoding(), LOOM_LOCATION_UNKNOWN, &schema));
    *out_schema = loom_encoding_define_result(schema);
  }

  void BuildPhysicalStorage(loom_value_id_t layout, loom_value_id_t schema,
                            loom_value_id_t* out_storage) {
    uint16_t spec =
        AddEncoding(IREE_SV("physical_storage"), /*attributes=*/nullptr,
                    /*attribute_count=*/0);
    loom_string_id_t layout_name = LOOM_STRING_ID_INVALID;
    loom_string_id_t schema_name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("layout"), &layout_name));
    IREE_ASSERT_OK(
        loom_module_intern_string(module_, IREE_SV("schema"), &schema_name));
    loom_named_value_t params[] = {
        {
            /*.name_id=*/layout_name,
            /*.reserved=*/{},
            /*.value_id=*/layout,
        },
        {
            /*.name_id=*/schema_name,
            /*.reserved=*/{},
            /*.value_id=*/schema,
        },
    };
    loom_op_t* storage = nullptr;
    IREE_ASSERT_OK(loom_encoding_define_build(
        &builder_, spec, params, IREE_ARRAYSIZE(params), loom_type_encoding(),
        LOOM_LOCATION_UNKNOWN, &storage));
    *out_storage = loom_encoding_define_result(storage);
  }

  static loom_type_t ViewWithLayout(loom_type_t view_type,
                                    loom_value_id_t layout_id) {
    view_type.encoding_id = (uint16_t)layout_id;
    view_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
    return view_type;
  }

  static loom_type_t ViewWithStaticEncoding(loom_type_t view_type,
                                            uint16_t encoding_id) {
    view_type.encoding_id = encoding_id;
    view_type.encoding_flags = 0;
    return view_type;
  }

  bool Describe(loom_type_t view_type, loom_type_t vector_type,
                loom_vector_memory_access_t* out_access) {
    loom_value_fact_table_t facts;
    iree_status_t status = loom_value_fact_table_initialize(
        &facts, &module_->arena, module_->values.count);
    IREE_EXPECT_OK(status);
    if (!iree_status_is_ok(status)) {
      return false;
    }
    loom_func_like_t no_function = {};
    status = loom_value_fact_table_compute_region(
        &facts, module_, no_function, module_->body, /*parent_op=*/nullptr);
    IREE_EXPECT_OK(status);
    if (!iree_status_is_ok(status)) {
      return false;
    }
    return loom_vector_memory_access_describe(
        &facts.context, module_, view_type, vector_type, out_access);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(VectorMemoryTest, DenseLayoutComputesLaneOffsets) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_2d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(2), loom_dim_pack_static(4),
                          /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_DENSE);

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(row_stride, 16);
  EXPECT_EQ(column_stride, 1);

  int64_t static_indices[] = {3, 5};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {1, 3};
  int64_t element_offset = 0;
  int64_t byte_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_TRUE(loom_vector_memory_access_static_lane_byte_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &byte_offset));
  EXPECT_EQ(element_offset, 72);
  EXPECT_EQ(byte_offset, 288);
}

TEST_F(VectorMemoryTest, StridedLayoutComputesPaddedLaneOffsets) {
  int64_t static_strides[] = {64, 1};
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildStridedLayout(
      /*dynamic_strides=*/nullptr, /*dynamic_stride_count=*/0, static_strides,
      IREE_ARRAYSIZE(static_strides), &layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t row_extent = 0;
  int64_t column_extent = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_extent(&access, 0, &row_extent));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_extent(&access, 1, &column_extent));
  EXPECT_EQ(row_extent, 1);
  EXPECT_EQ(column_extent, 4);

  int64_t static_indices[] = {2, 3};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {2};
  int64_t element_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_EQ(element_offset, 133);
}

TEST_F(VectorMemoryTest, PhysicalStorageCompositionUsesAddressLayout) {
  int64_t static_strides[] = {64, 1};
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildStridedLayout(
      /*dynamic_strides=*/nullptr, /*dynamic_stride_count=*/0, static_strides,
      IREE_ARRAYSIZE(static_strides), &layout);
  loom_value_id_t schema = LOOM_VALUE_ID_INVALID;
  BuildGgmlQ4_0Schema(&schema);
  loom_value_id_t storage = LOOM_VALUE_ID_INVALID;
  BuildPhysicalStorage(layout, schema, &storage);

  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(8), loom_dim_pack_static(18),
                          /*encoding_id=*/0),
      storage);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(18), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t row_stride = 0;
  int64_t byte_stride = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &byte_stride));
  EXPECT_EQ(row_stride, 64);
  EXPECT_EQ(byte_stride, 1);
}

TEST_F(VectorMemoryTest, StaticStridedLayoutComputesLaneOffsets) {
  uint16_t layout = AddStaticStridedLayout(8);
  loom_type_t view_type = ViewWithStaticEncoding(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(16), /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t stride = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &stride));
  EXPECT_EQ(stride, 8);

  int64_t static_indices[] = {2};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {1};
  int64_t element_offset = 0;
  int64_t byte_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_TRUE(loom_vector_memory_access_static_lane_byte_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &byte_offset));
  EXPECT_EQ(element_offset, 24);
  EXPECT_EQ(byte_offset, 96);
}

TEST_F(VectorMemoryTest, AssumeSpecPhysicalStorageUsesStaticAddressLayout) {
  loom_value_id_t storage_arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_block_arg(&builder_, loom_module_block(module_),
                                    loom_type_encoding(), &storage_arg));
  uint16_t layout = AddStaticStridedLayout(64);
  uint16_t schema = AddGgmlQ4_0Schema();
  uint16_t storage_spec = AddStaticPhysicalStorage(layout, schema);
  loom_op_t* assume = nullptr;
  IREE_ASSERT_OK(loom_encoding_assume_spec_build(
      &builder_, storage_arg, storage_spec, loom_type_encoding(),
      LOOM_LOCATION_UNKNOWN, &assume));

  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(18), /*encoding_id=*/0),
      loom_encoding_assume_spec_result(assume));
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I8,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t stride = 0;
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &stride));
  EXPECT_EQ(stride, 64);
}

TEST_F(VectorMemoryTest, DenseLayoutReportsUnknownStaticSuffixStride) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_dynamic(42),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_FALSE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(column_stride, 1);
}

TEST_F(VectorMemoryTest, StridedLayoutSupportsDynamicStride) {
  int64_t static_strides[] = {INT64_MIN, 1};
  loom_value_id_t dynamic_stride = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_module_block(module_),
      loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &dynamic_stride));
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildStridedLayout(&dynamic_stride, /*dynamic_stride_count=*/1,
                     static_strides, IREE_ARRAYSIZE(static_strides), &layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_2d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(8), loom_dim_pack_static(16),
                          /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));
  EXPECT_EQ(access.layout_kind, LOOM_VECTOR_MEMORY_LAYOUT_STRIDED);

  int64_t row_stride = 0;
  int64_t column_stride = 0;
  EXPECT_FALSE(
      loom_vector_memory_access_static_axis_stride(&access, 0, &row_stride));
  EXPECT_TRUE(
      loom_vector_memory_access_static_axis_stride(&access, 1, &column_stride));
  EXPECT_EQ(column_stride, 1);

  int64_t static_indices[] = {1, 0};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {0};
  int64_t element_offset = 0;
  EXPECT_FALSE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));

  static_indices[0] = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_EQ(element_offset, 0);
}

TEST_F(VectorMemoryTest, ByteOffsetRejectsSubByteElementType) {
  loom_value_id_t layout = LOOM_VALUE_ID_INVALID;
  BuildDenseLayout(&layout);
  loom_type_t view_type = ViewWithLayout(
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_I1,
                          loom_dim_pack_static(8), /*encoding_id=*/0),
      layout);
  loom_type_t vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_I1,
                          loom_dim_pack_static(4), /*encoding_id=*/0);

  loom_vector_memory_access_t access;
  ASSERT_TRUE(Describe(view_type, vector_type, &access));

  int64_t static_indices[] = {2};
  loom_attribute_t static_index_attr = loom_attr_i64_array(
      static_indices, (uint16_t)IREE_ARRAYSIZE(static_indices));
  int64_t lane_indices[] = {1};
  int64_t element_offset = 0;
  int64_t byte_offset = 0;
  EXPECT_TRUE(loom_vector_memory_access_static_lane_element_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &element_offset));
  EXPECT_FALSE(loom_vector_memory_access_static_lane_byte_offset(
      &access, static_index_attr, lane_indices,
      (uint8_t)IREE_ARRAYSIZE(lane_indices), &byte_offset));
  EXPECT_EQ(element_offset, 3);
}

}  // namespace
}  // namespace loom
