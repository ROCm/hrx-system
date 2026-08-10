// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/ops/encoding/storage.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/matrix_operand.h"
#include "loom/ops/encoding/ops.h"

namespace loom {
namespace {

class EncodingStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t vtable_count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_encoding_dialect_vtables(&vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)vtable_count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_module_t* Parse(iree_string_view_t source) {
    loom_text_parse_options_t options = {};
    loom_module_t* module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(source, IREE_SV("storage_test.loom"),
                                   &context_, &block_pool_, &options, &module));
    return module;
  }

  uint16_t FirstSpecId(loom_module_t* module) {
    const loom_block_t* body = loom_module_block(module);
    const loom_op_t* op = loom_block_const_op(body, 0);
    EXPECT_TRUE(loom_encoding_define_isa(op));
    return loom_encoding_define_spec(op);
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

TEST_F(EncodingStorageTest, FixedNamedFp8Metadata) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define "
                    "#fp8_e4m3fn<rounding=finite_only> : encoding<schema>\n"));
  ASSERT_NE(module, nullptr);
  const uint16_t encoding_id = FirstSpecId(module);

  loom_encoding_record_geometry_t geometry;
  ASSERT_TRUE(loom_encoding_query_static_record_geometry(module, encoding_id,
                                                         &geometry));
  EXPECT_EQ(geometry.logical_element_count, 1u);
  EXPECT_EQ(geometry.storage_byte_count, 1u);
  EXPECT_EQ(geometry.required_alignment, 1u);

  loom_value_fact_storage_schema_t schema;
  ASSERT_TRUE(
      loom_encoding_query_static_storage_schema(module, encoding_id, &schema));
  EXPECT_EQ(schema.encoded_operand.element_format,
            LOOM_VALUE_FACT_NUMERIC_FORMAT_F8_E4M3FN);
  EXPECT_EQ(schema.encoded_operand.payload_packing,
            LOOM_VALUE_FACT_PAYLOAD_PACKING_DENSE_LANES);
  EXPECT_EQ(schema.encoded_operand.payload_element_count, 1u);
  EXPECT_EQ(schema.encoded_operand.rounding_policy,
            LOOM_VALUE_FACT_ROUNDING_POLICY_FINITE_ONLY);

  loom_module_free(module);
}

TEST_F(EncodingStorageTest, ParameterizedSchemaHasNoFixedGeometry) {
  loom_module_t* module =
      Parse(IREE_SV("%schema = encoding.define "
                    "#matrix_operand<element_format=f16, payload_elements=16, "
                    "payload_registers=8> : encoding<schema>\n"));
  ASSERT_NE(module, nullptr);

  loom_encoding_record_geometry_t geometry = {1, 1, 1};
  EXPECT_FALSE(loom_encoding_query_static_record_geometry(
      module, FirstSpecId(module), &geometry));
  EXPECT_EQ(geometry.logical_element_count, 0u);
  EXPECT_EQ(geometry.storage_byte_count, 0u);
  EXPECT_EQ(geometry.required_alignment, 0u);

  loom_module_free(module);
}

}  // namespace
}  // namespace loom
