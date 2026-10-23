// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/abi/c/signature_layout.h"

#include <initializer_list>
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

class CAbiSignatureLayoutTest : public ::testing::Test {
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
        &context_, IREE_SV("c_abi_signature_layout_test"), &block_pool_, NULL,
        iree_allocator_system(), &module_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&result_arena_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  uint32_t AddType(loom_func_abi_c_type_attr_build_flags_t flags,
                   loom_func_abi_c_type_kind_t kind, int64_t byte_length,
                   int64_t byte_alignment, int64_t component_count,
                   int64_t bitwidth = 0, int64_t element_type = 0,
                   int64_t element_count = 0, int64_t element_stride = 0,
                   loom_parameterized_attr_array_t fields =
                       loom_parameterized_attr_array_empty()) {
    loom_attribute_t type;
    IREE_CHECK_OK(loom_func_abi_c_type_attr_make(
        module_, flags, kind, byte_length, byte_alignment, component_count,
        bitwidth, (loom_func_abi_c_type_format_t)0, /*address_space=*/0,
        element_type, element_count, element_stride, LOOM_STRING_ID_INVALID,
        fields, &type));
    types_.push_back(type);
    return (uint32_t)types_.size() - 1u;
  }

  uint32_t AddInteger(uint32_t bitwidth, bool is_unsigned = false) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_BITWIDTH,
                   is_unsigned ? LOOM_FUNC_ABI_C_TYPE_KIND_UNSIGNED_INTEGER
                               : LOOM_FUNC_ABI_C_TYPE_KIND_SIGNED_INTEGER,
                   bitwidth / 8u, bitwidth / 8u, 1, bitwidth);
  }

  uint32_t AddPointer() {
    return AddType(/*flags=*/0, LOOM_FUNC_ABI_C_TYPE_KIND_POINTER,
                   /*byte_length=*/8, /*byte_alignment=*/8,
                   /*component_count=*/2);
  }

  uint32_t AddBoolean() {
    return AddType(/*flags=*/0, LOOM_FUNC_ABI_C_TYPE_KIND_BOOLEAN,
                   /*byte_length=*/1, /*byte_alignment=*/1,
                   /*component_count=*/1);
  }

  uint32_t AddVector(uint32_t element_type, uint32_t element_count,
                     uint32_t byte_length, uint32_t byte_alignment) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT,
                   LOOM_FUNC_ABI_C_TYPE_KIND_VECTOR, byte_length,
                   byte_alignment, /*component_count=*/1, /*bitwidth=*/0,
                   element_type, element_count);
  }

  uint32_t AddArray(uint32_t element_type, uint32_t element_count,
                    uint32_t element_stride, uint32_t byte_alignment,
                    uint32_t component_count) {
    return AddType(LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_COUNT |
                       LOOM_FUNC_ABI_C_TYPE_ATTR_BUILD_FLAG_HAS_ELEMENT_STRIDE,
                   LOOM_FUNC_ABI_C_TYPE_KIND_ARRAY,
                   element_count * element_stride, byte_alignment,
                   component_count, /*bitwidth=*/0, element_type, element_count,
                   element_stride);
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
                   /*element_type=*/0, /*element_count=*/0,
                   /*element_stride=*/0,
                   loom_make_parameterized_attr_array(field_attrs.data(),
                                                      field_attrs.size()));
  }

  loom_c_abi_signature_layout_t BuildLayout(
      std::initializer_list<int64_t> arguments,
      std::initializer_list<int64_t> results) {
    argument_roots_.assign(arguments.begin(), arguments.end());
    result_roots_.assign(results.begin(), results.end());
    const loom_func_c_abi_signature_t signature = {
        .arguments =
            loom_make_i64_array(argument_roots_.data(), argument_roots_.size()),
        .results =
            loom_make_i64_array(result_roots_.data(), result_roots_.size()),
        .types =
            loom_make_parameterized_attr_array(types_.data(), types_.size()),
    };
    loom_c_abi_signature_layout_t layout;
    IREE_EXPECT_OK(loom_c_abi_signature_layout_build(
        &signature, &scratch_arena_, &result_arena_, &layout));
    return layout;
  }

  iree_arena_block_pool_t block_pool_;
  iree_arena_allocator_t scratch_arena_;
  iree_arena_allocator_t result_arena_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  std::vector<loom_attribute_t> types_;
  std::vector<int64_t> argument_roots_;
  std::vector<int64_t> result_roots_;
};

TEST_F(CAbiSignatureLayoutTest, FlattensObjectsIntoTerminalLeaves) {
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t pointer = AddPointer();
  const uint32_t pair = AddArray(u32, 2, 4, 4, 2);
  const uint32_t record =
      AddRecord(16, 8, 4, {{"data", pointer, 0}, {"shape", pair, 8}});

  const loom_c_abi_signature_layout_t layout =
      BuildLayout({record, u32}, {pointer});
  ASSERT_EQ(layout.signature.arguments.count, 2u);
  ASSERT_EQ(layout.signature.results.count, 1u);
  EXPECT_EQ(layout.argument_logical_count, 5u);
  EXPECT_EQ(layout.argument_leaf_count, 4u);
  EXPECT_EQ(layout.result_logical_count, 2u);
  EXPECT_EQ(layout.result_leaf_count, 1u);

  const loom_c_abi_object_t& aggregate = layout.arguments[0];
  EXPECT_EQ(aggregate.type_index, record);
  EXPECT_EQ(aggregate.byte_length, 16u);
  EXPECT_EQ(aggregate.byte_alignment, 8u);
  EXPECT_EQ(aggregate.logical_offset, 0u);
  EXPECT_EQ(aggregate.logical_count, 4u);
  EXPECT_EQ(aggregate.leaf_offset, 0u);
  EXPECT_EQ(aggregate.leaf_count, 3u);
  EXPECT_EQ(layout.arguments[1].logical_offset, 4u);
  EXPECT_EQ(layout.arguments[1].leaf_offset, 3u);

  const loom_c_abi_leaf_t& data = layout.argument_leaves[0];
  EXPECT_EQ(data.type_index, pointer);
  EXPECT_EQ(data.byte_offset, 0u);
  EXPECT_EQ(data.byte_length, 8u);
  EXPECT_EQ(data.logical_offset, 0u);
  EXPECT_EQ(data.logical_count, 2u);
  EXPECT_EQ(data.flags, LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED |
                            LOOM_C_ABI_LEAF_FLAG_POINTER);
  EXPECT_EQ(layout.argument_leaves[1].byte_offset, 8u);
  EXPECT_EQ(layout.argument_leaves[1].logical_offset, 2u);
  EXPECT_EQ(layout.argument_leaves[2].byte_offset, 12u);
  EXPECT_EQ(layout.argument_leaves[2].logical_offset, 3u);
  EXPECT_EQ(layout.argument_leaves[3].byte_offset, 0u);
  EXPECT_EQ(layout.argument_leaves[3].logical_offset, 4u);

  EXPECT_EQ(layout.results[0].logical_offset, 0u);
  EXPECT_EQ(layout.result_leaves[0].logical_offset, 0u);
  EXPECT_EQ(layout.result_leaves[0].logical_count, 2u);
  EXPECT_EQ(
      layout.result_leaves[0].flags,
      LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED | LOOM_C_ABI_LEAF_FLAG_POINTER);
}

TEST_F(CAbiSignatureLayoutTest, RetainsNestedPackedLeafAlignment) {
  const uint32_t u8 = AddInteger(8, true);
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t packed =
      AddRecord(5, 1, 2, {{"tag", u8, 0}, {"value", u32, 1}});
  const uint32_t wrapper = AddRecord(8, 8, 2, {{"packed", packed, 0}});

  const loom_c_abi_signature_layout_t layout = BuildLayout({wrapper}, {});
  ASSERT_EQ(layout.argument_leaf_count, 2u);
  EXPECT_EQ(layout.argument_leaves[0].byte_offset, 0u);
  EXPECT_EQ(layout.argument_leaves[0].flags,
            LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED);
  EXPECT_EQ(layout.argument_leaves[1].byte_offset, 1u);
  EXPECT_EQ(layout.argument_leaves[1].flags, 0u);
}

TEST_F(CAbiSignatureLayoutTest, RetainsTerminalProjectionKinds) {
  const uint32_t u32 = AddInteger(32, true);
  const uint32_t boolean = AddBoolean();
  const uint32_t vector = AddVector(u32, 4, 16, 16);

  const loom_c_abi_signature_layout_t layout =
      BuildLayout({boolean, vector}, {});
  ASSERT_EQ(layout.argument_leaf_count, 2u);
  EXPECT_EQ(
      layout.argument_leaves[0].flags,
      LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED | LOOM_C_ABI_LEAF_FLAG_BOOLEAN);
  EXPECT_EQ(
      layout.argument_leaves[1].flags,
      LOOM_C_ABI_LEAF_FLAG_NATURALLY_ALIGNED | LOOM_C_ABI_LEAF_FLAG_VECTOR);
}

}  // namespace
}  // namespace loom
