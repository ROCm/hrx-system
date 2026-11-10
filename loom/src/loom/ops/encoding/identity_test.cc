// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ops/builder.h"
#include "loom/ops/encoding/facts.h"
#include "loom/ops/encoding/families.h"
#include "loom/ops/encoding/ops.h"
#include "loom/ops/encoding/summary.h"
#include "loom/ops/type_registry.h"

namespace loom {
namespace {

// A distinct encoding family may expose the same address map as a built-in.
static void SummarizeCustomLayout(const loom_encoding_family_summary_request_t*,
                                  loom_encoding_family_summary_t* summary) {
  summary->encoding.address_layout.kind = LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE;
}

static const loom_encoding_family_descriptor_t kCustomLayoutDescriptor = {
    .name = LOOM_BSTRING_REF(11, "test.layout"),
    .role = LOOM_ENCODING_ROLE_ADDRESS_LAYOUT,
};
static const loom_encoding_vtable_t kCustomLayoutVtable = {
    .descriptor = &kCustomLayoutDescriptor,
    .summarize = SummarizeCustomLayout,
};

class EncodingIdentityTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(), &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables =
        loom_encoding_dialect_vtables(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_ENCODING, vtables, (uint16_t)count));
    IREE_ASSERT_OK(loom_context_register_builtin_encoding_vtables(&context_));
    IREE_ASSERT_OK(
        loom_context_register_encoding_vtable(&context_, &kCustomLayoutVtable));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("identity"), &pool_,
                                        nullptr, iree_allocator_system(),
                                        &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(
        loom_value_fact_table_initialize(&facts_, &module_->arena, 0));
    loom_type_registry_configure_fact_context(&facts_.context);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  uint16_t AddSpec(iree_string_view_t name,
                   const loom_named_attr_t* attributes = nullptr,
                   uint8_t attribute_count = 0) {
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    const loom_encoding_t spec = {
        .name_id = name_id,
        .alias_id = LOOM_STRING_ID_INVALID,
        .attribute_count = attribute_count,
        .attributes = attributes,
    };
    uint16_t id;
    IREE_CHECK_OK(loom_module_add_encoding(module_, &spec, &id));
    return id;
  }

  loom_named_attr_t EncodingParameter(iree_string_view_t name, uint16_t id) {
    loom_string_id_t name_id;
    IREE_CHECK_OK(loom_module_intern_string(module_, name, &name_id));
    return {.name_id = name_id, .value = loom_attr_encoding(id)};
  }

  loom_op_t* Define(uint16_t id, loom_encoding_role_t role,
                    loom_value_facts_t* out_facts) {
    loom_op_t* op;
    IREE_CHECK_OK(loom_encoding_define_build(
        &builder_, id, nullptr, 0, loom_type_encoding_with_role(role), 0, &op));
    IREE_CHECK_OK(loom_encoding_define_facts(&facts_.context, module_, op,
                                             nullptr, out_facts));
    return op;
  }

  loom_value_facts_t Query(loom_value_id_t value, loom_value_facts_t facts,
                           uint16_t required) {
    loom_op_t* query;
    IREE_CHECK_OK(
        loom_encoding_isa_build(&builder_, required, value, 0, &query));
    loom_value_facts_t result;
    IREE_CHECK_OK(loom_encoding_isa_facts(&facts_.context, module_, query,
                                          &facts, &result));
    return result;
  }

  // Shared pool owns the module and its fact table.
  iree_arena_block_pool_t pool_;
  // Finalized registry includes a custom layout with dense addressing.
  loom_context_t context_;
  // Owns the static specifications and valid constructor/query operations.
  loom_module_t* module_ = nullptr;
  // Appends small API fixtures to the module body.
  loom_builder_t builder_;
  // Owns the producer summaries used by the query callbacks.
  loom_value_fact_table_t facts_;
};

TEST_F(EncodingIdentityTest, DenseAssumptionPreservesCustomIdentity) {
  const uint16_t custom = AddSpec(IREE_SV("test.layout"));
  const uint16_t dense = AddSpec(IREE_SV("encoding.layout.dense"));
  loom_value_facts_t original;
  loom_op_t* definition =
      Define(custom, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, &original);
  loom_op_t* assumption;
  IREE_ASSERT_OK(loom_encoding_layout_assume_dense_build(
      &builder_, loom_encoding_define_result(definition),
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), 0,
      &assumption));
  loom_value_facts_t refined;
  IREE_ASSERT_OK(loom_encoding_layout_assume_dense_facts(
      &facts_.context, module_, assumption, &original, &refined));
  const loom_value_id_t value =
      loom_encoding_layout_assume_dense_result(assumption);
  EXPECT_TRUE(loom_value_facts_equal(Query(value, refined, custom),
                                     loom_value_facts_exact_i64(1)));
  EXPECT_TRUE(loom_value_facts_equal(Query(value, refined, dense),
                                     loom_value_facts_exact_i64(0)));
}

TEST_F(EncodingIdentityTest, CommonAddressingDoesNotProveCommonFamily) {
  const uint16_t custom = AddSpec(IREE_SV("test.layout"));
  const uint16_t dense = AddSpec(IREE_SV("encoding.layout.dense"));
  loom_value_facts_t custom_facts;
  loom_value_facts_t dense_facts;
  loom_op_t* custom_value =
      Define(custom, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, &custom_facts);
  Define(dense, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, &dense_facts);
  loom_value_facts_t joined;
  IREE_ASSERT_OK(loom_value_fact_table_meet_for_type(
      &facts_, module_,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT), &facts_,
      custom_facts, &facts_, dense_facts, &joined));
  loom_value_fact_encoding_summary_t summary;
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&facts_.context, joined,
                                                      &summary));
  EXPECT_EQ(summary.address_layout.kind, LOOM_VALUE_FACT_ADDRESS_LAYOUT_DENSE);
  EXPECT_EQ(summary.family_id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_EQ(summary.address_layout.family_id, LOOM_ENCODING_FAMILY_ID_INVALID);
  EXPECT_TRUE(loom_value_facts_equal(
      Query(loom_encoding_define_result(custom_value), joined, dense),
      loom_value_facts_make(0, 1, 1)));
}

TEST_F(EncodingIdentityTest, CompositionRetainsCustomLayoutIdentity) {
  const uint16_t custom = AddSpec(IREE_SV("test.layout"));
  const uint16_t dense = AddSpec(IREE_SV("encoding.layout.dense"));
  const uint16_t schema = AddSpec(IREE_SV("ggml.q4_0"));
  const loom_named_attr_t schema_attribute =
      EncodingParameter(IREE_SV("schema"), schema);
  const uint16_t partial =
      AddSpec(IREE_SV("encoding.storage"), &schema_attribute, 1);
  loom_value_facts_t layout_facts;
  loom_op_t* layout =
      Define(custom, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, &layout_facts);
  const loom_named_attr_t layout_attribute =
      EncodingParameter(IREE_SV("layout"), custom);
  const loom_named_value_t binding = {
      .name_id = layout_attribute.name_id,
      .value_id = loom_encoding_define_result(layout),
  };
  loom_op_t* storage;
  IREE_ASSERT_OK(loom_encoding_define_build(
      &builder_, partial, &binding, 1,
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_PHYSICAL_STORAGE), 0,
      &storage));
  loom_value_facts_t storage_facts;
  IREE_ASSERT_OK(loom_encoding_define_facts(&facts_.context, module_, storage,
                                            &layout_facts, &storage_facts));
  loom_named_attr_t attributes[] = {layout_attribute, schema_attribute};
  const uint16_t exact = AddSpec(IREE_SV("encoding.storage"), attributes, 2);
  attributes[0].value = loom_attr_encoding(dense);
  const uint16_t different =
      AddSpec(IREE_SV("encoding.storage"), attributes, 2);
  const loom_value_id_t value = loom_encoding_define_result(storage);
  EXPECT_TRUE(loom_value_facts_equal(Query(value, storage_facts, exact),
                                     loom_value_facts_exact_i64(1)));
  EXPECT_TRUE(loom_value_facts_equal(Query(value, storage_facts, different),
                                     loom_value_facts_exact_i64(0)));
}

TEST_F(EncodingIdentityTest, IdentityFieldsSurviveInterningAndTransfer) {
  const uint16_t custom = AddSpec(IREE_SV("test.layout"));
  loom_value_facts_t original;
  Define(custom, LOOM_ENCODING_ROLE_ADDRESS_LAYOUT, &original);
  loom_value_fact_encoding_summary_t summary;
  ASSERT_TRUE(loom_value_facts_query_encoding_summary(&facts_.context, original,
                                                      &summary));
  loom_value_fact_table_t destination;
  IREE_ASSERT_OK(
      loom_value_fact_table_initialize(&destination, &module_->arena, 0));
  for (int field = 0; field < 4; ++field) {
    SCOPED_TRACE(field);
    auto partial = summary;
    switch (field) {
      case 0:
        partial.family_id = 0;
        break;
      case 1:
        partial.static_spec_encoding_id = 0;
        break;
      case 2:
        partial.address_layout.family_id = 0;
        break;
      case 3:
        partial.address_layout.static_spec_encoding_id = 0;
        break;
    }
    loom_value_facts_t weakened;
    IREE_ASSERT_OK(loom_value_facts_make_encoding_summary(&facts_.context,
                                                          partial, &weakened));
    EXPECT_NE(weakened.extension_id, original.extension_id);
    loom_value_facts_t cloned;
    IREE_ASSERT_OK(loom_value_fact_table_clone_fact(&destination, &facts_,
                                                    weakened, &cloned));
    EXPECT_TRUE(loom_value_fact_table_facts_equal(&facts_, weakened,
                                                  &destination, cloned));
    EXPECT_FALSE(loom_value_fact_table_facts_equal(&facts_, original,
                                                   &destination, cloned));
  }
}

}  // namespace
}  // namespace loom
