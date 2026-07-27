// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/sanitizer/site_collection.h"

#include <cstring>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/sanitizer/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

loom_sanitizer_site_payload_t MakePayload(
    loom_sanitizer_check_kind_t check_kind,
    loom_sanitizer_site_kind_t site_kind = LOOM_SANITIZER_SITE_KIND_VALUE) {
  loom_sanitizer_site_payload_t payload = {};
  payload.site_kind = site_kind;
  payload.check_kind = check_kind;
  payload.provenance_kind = LOOM_SANITIZER_PROVENANCE_KIND_ASSUME;
  payload.lane_policy = LOOM_SANITIZER_LANE_POLICY_SCALAR;
  payload.lineage_role = LOOM_SANITIZER_LINEAGE_ROLE_ORIGINAL;
  payload.flags = 0;
  payload.extension_data = iree_const_byte_span_empty();
  return payload;
}

loom_predicate_t MakeRangePredicate(loom_value_id_t value_id) {
  loom_predicate_t predicate = {};
  predicate.kind = LOOM_PREDICATE_RANGE;
  predicate.arg_count = 3;
  predicate.arg_tags[0] = LOOM_PRED_ARG_VALUE;
  predicate.arg_tags[1] = LOOM_PRED_ARG_CONST;
  predicate.arg_tags[2] = LOOM_PRED_ARG_CONST;
  predicate.args[0] = (int64_t)value_id;
  predicate.args[1] = 0;
  predicate.args[2] = 1024;
  return predicate;
}

void ExpectSamePayload(const loom_sanitizer_site_payload_t& expected,
                       const loom_sanitizer_site_payload_t& actual) {
  EXPECT_EQ(actual.site_kind, expected.site_kind);
  EXPECT_EQ(actual.check_kind, expected.check_kind);
  EXPECT_EQ(actual.provenance_kind, expected.provenance_kind);
  EXPECT_EQ(actual.lane_policy, expected.lane_policy);
  EXPECT_EQ(actual.lineage_role, expected.lineage_role);
  EXPECT_EQ(actual.flags, expected.flags);
  ASSERT_EQ(actual.extension_data.data_length,
            expected.extension_data.data_length);
  if (expected.extension_data.data_length > 0) {
    ASSERT_NE(actual.extension_data.data, nullptr);
    ASSERT_NE(expected.extension_data.data, nullptr);
    EXPECT_EQ(memcmp(actual.extension_data.data, expected.extension_data.data,
                     expected.extension_data.data_length),
              0);
  }
}

class SiteCollectionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));

    iree_host_size_t sanitizer_vtable_count = 0;
    const loom_op_vtable_t* const* sanitizer_vtables =
        loom_sanitizer_dialect_vtables(&sanitizer_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(
        &context_, LOOM_DIALECT_SANITIZER, sanitizer_vtables,
        (uint16_t)sanitizer_vtable_count));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));

    loom_builder_t module_builder;
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &module_builder);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_builder_intern_string(&module_builder,
                                              IREE_SV("test_fn"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    IREE_ASSERT_OK(loom_test_func_build(&module_builder, 0, 0, 0, callee, NULL,
                                        0, NULL, 0, NULL, 0, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &func_op_));
    func_like_ = loom_func_like_cast(module_, func_op_);
    body_ = loom_func_like_body(func_like_);
    loom_builder_initialize(module_, &module_->arena,
                            loom_region_entry_block(body_), &builder_);
    builder_.ip.parent_op = func_op_;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_op_t* BuildConstant(int64_t value) {
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(value),
        loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildViewValue(loom_value_id_t input_value_id) {
    loom_type_t view_type = loom_type_shaped_1d(
        LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 1);
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_test_convert_build(&builder_, input_value_id, view_type,
                                          LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  loom_op_t* BuildAssertAccess(loom_value_id_t view_id,
                               loom_location_id_t location) {
    const int64_t static_indices[] = {0};
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_sanitizer_assert_access_build(
        &builder_, LOOM_SANITIZER_ASSERT_ACCESS_KIND_READ, view_id, NULL, 0,
        static_indices, IREE_ARRAYSIZE(static_indices), NULL, 0, location,
        &op));
    return op;
  }

  loom_op_t* BuildRaceAccess(loom_value_id_t view_id,
                             loom_location_id_t location) {
    const int64_t static_indices[] = {0};
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_sanitizer_race_access_build(
        &builder_, 0, LOOM_SANITIZER_RACE_ACCESS_KIND_WRITE, view_id, NULL, 0,
        static_indices, IREE_ARRAYSIZE(static_indices), false, 0, 0, location,
        &op));
    return op;
  }

  loom_op_t* BuildAssertValue(loom_value_id_t value_id,
                              loom_location_id_t location) {
    loom_predicate_t predicate = MakeRangePredicate(value_id);
    loom_type_t result_type = loom_module_value_type(module_, value_id);
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_sanitizer_assert_value_build(&builder_, &value_id, 1,
                                                    &predicate, 1, &result_type,
                                                    1, location, &op));
    return op;
  }

  loom_op_t* BuildAssertOp(loom_value_id_t value_id,
                           loom_location_id_t location) {
    loom_predicate_t predicate = MakeRangePredicate(value_id);
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_sanitizer_assert_op_build(&builder_, &value_id, 1,
                                                 &predicate, 1, location, &op));
    return op;
  }

  loom_op_t* BuildAssertLayout(loom_value_id_t view_id,
                               loom_location_id_t location) {
    loom_type_t result_type = loom_module_value_type(module_, view_id);
    loom_op_t* op = NULL;
    IREE_CHECK_OK(loom_sanitizer_assert_layout_build(
        &builder_, view_id, result_type, location, &op));
    return op;
  }

  loom_location_id_t AddFileLocation(uint16_t start_line) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_CHECK_OK(loom_module_register_source(module_, IREE_SV("model.loom"),
                                              &source_id));
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module_,
        loom_location_file_range(source_id, start_line, 1, start_line, 12),
        &location_id));
    return location_id;
  }

  loom_location_id_t AddTaggedLocation(loom_location_tag_t tag,
                                       loom_location_id_t child,
                                       iree_const_byte_span_t data) {
    const uint8_t* stored_data = NULL;
    if (data.data_length > 0) {
      uint8_t* mutable_data = NULL;
      IREE_CHECK_OK(iree_arena_allocate(&module_->arena, data.data_length,
                                        (void**)&mutable_data));
      memcpy(mutable_data, data.data, data.data_length);
      stored_data = mutable_data;
    }
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_CHECK_OK(loom_module_add_location(
        module_,
        loom_location_tagged(tag, child, stored_data,
                             (uint32_t)data.data_length),
        &location_id));
    return location_id;
  }

  loom_location_id_t AddSanitizerSiteLocation(
      loom_location_id_t child, const loom_sanitizer_site_payload_t& payload) {
    uint8_t storage[LOOM_SANITIZER_SITE_PAYLOAD_HEADER_LENGTH] = {0};
    iree_host_size_t encoded_length = 0;
    IREE_CHECK_OK(loom_sanitizer_site_payload_encode(
        &payload, iree_make_byte_span(storage, sizeof(storage)),
        &encoded_length));
    return AddTaggedLocation(
        LOOM_LOCATION_TAG_SANITIZER_SITE, child,
        iree_make_const_byte_span(storage, encoded_length));
  }

  void FinalizeModule() { IREE_ASSERT_OK(loom_module_compute_uses(module_)); }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_op_t* func_op_ = nullptr;
  loom_func_like_t func_like_;
  loom_region_t* body_ = nullptr;
  loom_builder_t builder_;
};

TEST_F(SiteCollectionTest, AssignsDeterministicIdsInFunctionWalkOrder) {
  loom_op_t* constant = BuildConstant(7);
  loom_value_id_t value = loom_test_constant_result(constant);

  const loom_sanitizer_site_payload_t first_payload =
      MakePayload(LOOM_SANITIZER_CHECK_KIND_VALUE_RANGE);
  loom_location_id_t first_source_location = AddFileLocation(10);
  loom_location_id_t first_location =
      AddSanitizerSiteLocation(first_source_location, first_payload);
  loom_op_t* first_assert = BuildAssertValue(value, first_location);

  const loom_sanitizer_site_payload_t second_payload =
      MakePayload(LOOM_SANITIZER_CHECK_KIND_VALUE_DIVISIBILITY);
  loom_location_id_t second_source_location = AddFileLocation(20);
  loom_location_id_t second_location =
      AddSanitizerSiteLocation(second_source_location, second_payload);
  loom_op_t* second_assert = BuildAssertValue(value, second_location);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_ASSERT_OK(loom_sanitizer_site_collection_build_function(
      module_, func_like_, &arena, &collection));

  ASSERT_EQ(collection.row_count, 2u);
  ASSERT_NE(collection.rows, nullptr);
  EXPECT_EQ(collection.rows[0].site_id, 0u);
  EXPECT_EQ(collection.rows[0].op, first_assert);
  EXPECT_EQ(collection.rows[0].op_kind, LOOM_OP_SANITIZER_ASSERT_VALUE);
  EXPECT_EQ(collection.rows[0].location, first_location);
  EXPECT_EQ(collection.rows[0].payload_location, first_location);
  EXPECT_EQ(collection.rows[0].source_location, first_source_location);
  ASSERT_TRUE(loom_sanitizer_site_row_has_payload(&collection.rows[0]));
  ExpectSamePayload(first_payload, collection.rows[0].payload);

  EXPECT_EQ(collection.rows[1].site_id, 1u);
  EXPECT_EQ(collection.rows[1].op, second_assert);
  EXPECT_EQ(collection.rows[1].op_kind, LOOM_OP_SANITIZER_ASSERT_VALUE);
  EXPECT_EQ(collection.rows[1].location, second_location);
  EXPECT_EQ(collection.rows[1].payload_location, second_location);
  EXPECT_EQ(collection.rows[1].source_location, second_source_location);
  ASSERT_TRUE(loom_sanitizer_site_row_has_payload(&collection.rows[1]));
  ExpectSamePayload(second_payload, collection.rows[1].payload);
  iree_arena_deinitialize(&arena);
}

TEST_F(SiteCollectionTest, UnknownLocationsStillProduceDistinctRegionRows) {
  loom_op_t* constant = BuildConstant(8);
  loom_value_id_t value = loom_test_constant_result(constant);
  loom_op_t* first_assert = BuildAssertValue(value, LOOM_LOCATION_UNKNOWN);
  loom_op_t* second_assert = BuildAssertValue(value, LOOM_LOCATION_UNKNOWN);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_ASSERT_OK(loom_sanitizer_site_collection_build_region(
      module_, body_, &arena, &collection));

  ASSERT_EQ(collection.row_count, 2u);
  EXPECT_EQ(collection.rows[0].site_id, 0u);
  EXPECT_EQ(collection.rows[0].op, first_assert);
  EXPECT_EQ(collection.rows[0].location, LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(collection.rows[0].payload_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(collection.rows[0].source_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_FALSE(loom_sanitizer_site_row_has_payload(&collection.rows[0]));

  EXPECT_EQ(collection.rows[1].site_id, 1u);
  EXPECT_EQ(collection.rows[1].op, second_assert);
  EXPECT_EQ(collection.rows[1].location, LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(collection.rows[1].payload_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(collection.rows[1].source_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_FALSE(loom_sanitizer_site_row_has_payload(&collection.rows[1]));
  iree_arena_deinitialize(&arena);
}

TEST_F(SiteCollectionTest, CollectsEverySanitizerReportSiteKind) {
  loom_op_t* constant = BuildConstant(12);
  loom_value_id_t value = loom_test_constant_result(constant);
  loom_op_t* view = BuildViewValue(value);
  loom_value_id_t view_value = loom_test_convert_result(view);

  loom_op_t* access_assert =
      BuildAssertAccess(view_value, LOOM_LOCATION_UNKNOWN);
  loom_op_t* value_assert = BuildAssertValue(value, LOOM_LOCATION_UNKNOWN);
  loom_op_t* op_assert = BuildAssertOp(value, LOOM_LOCATION_UNKNOWN);
  loom_op_t* layout_assert =
      BuildAssertLayout(view_value, LOOM_LOCATION_UNKNOWN);
  loom_op_t* race_access = BuildRaceAccess(view_value, LOOM_LOCATION_UNKNOWN);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_ASSERT_OK(loom_sanitizer_site_collection_build_region(
      module_, body_, &arena, &collection));

  ASSERT_EQ(collection.row_count, 5u);
  EXPECT_EQ(collection.rows[0].site_id, 0u);
  EXPECT_EQ(collection.rows[0].op, access_assert);
  EXPECT_EQ(collection.rows[0].op_kind, LOOM_OP_SANITIZER_ASSERT_ACCESS);
  EXPECT_EQ(collection.rows[1].site_id, 1u);
  EXPECT_EQ(collection.rows[1].op, value_assert);
  EXPECT_EQ(collection.rows[1].op_kind, LOOM_OP_SANITIZER_ASSERT_VALUE);
  EXPECT_EQ(collection.rows[2].site_id, 2u);
  EXPECT_EQ(collection.rows[2].op, op_assert);
  EXPECT_EQ(collection.rows[2].op_kind, LOOM_OP_SANITIZER_ASSERT_OP);
  EXPECT_EQ(collection.rows[3].site_id, 3u);
  EXPECT_EQ(collection.rows[3].op, layout_assert);
  EXPECT_EQ(collection.rows[3].op_kind, LOOM_OP_SANITIZER_ASSERT_LAYOUT);
  EXPECT_EQ(collection.rows[4].site_id, 4u);
  EXPECT_EQ(collection.rows[4].op, race_access);
  EXPECT_EQ(collection.rows[4].op_kind, LOOM_OP_SANITIZER_RACE_ACCESS);
  iree_arena_deinitialize(&arena);
}

TEST_F(SiteCollectionTest, NonSanitizerTaggedLocationsDoNotCreatePayloads) {
  loom_op_t* constant = BuildConstant(9);
  loom_value_id_t value = loom_test_constant_result(constant);
  const uint8_t payload_bytes[] = {0x01, 0x02, 0x03};
  loom_location_id_t source_location = AddFileLocation(30);
  loom_location_id_t tagged_location = AddTaggedLocation(
      LOOM_LOCATION_TAG_TEMPLATE_INSTANTIATION, source_location,
      iree_make_const_byte_span(payload_bytes, sizeof(payload_bytes)));
  loom_op_t* assert_op = BuildAssertValue(value, tagged_location);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_ASSERT_OK(loom_sanitizer_site_collection_build_function(
      module_, func_like_, &arena, &collection));

  ASSERT_EQ(collection.row_count, 1u);
  EXPECT_EQ(collection.rows[0].site_id, 0u);
  EXPECT_EQ(collection.rows[0].op, assert_op);
  EXPECT_EQ(collection.rows[0].location, tagged_location);
  EXPECT_EQ(collection.rows[0].payload_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_EQ(collection.rows[0].source_location, LOOM_LOCATION_UNKNOWN);
  EXPECT_FALSE(loom_sanitizer_site_row_has_payload(&collection.rows[0]));
  iree_arena_deinitialize(&arena);
}

TEST_F(SiteCollectionTest, MalformedSanitizerPayloadFailsLoudly) {
  loom_op_t* constant = BuildConstant(10);
  loom_value_id_t value = loom_test_constant_result(constant);
  const uint8_t malformed_payload[] = {
      LOOM_SANITIZER_SITE_PAYLOAD_CURRENT_VERSION};
  loom_location_id_t source_location = AddFileLocation(40);
  loom_location_id_t tagged_location = AddTaggedLocation(
      LOOM_LOCATION_TAG_SANITIZER_SITE, source_location,
      iree_make_const_byte_span(malformed_payload, sizeof(malformed_payload)));
  BuildAssertValue(value, tagged_location);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_OUT_OF_RANGE,
                        loom_sanitizer_site_collection_build_function(
                            module_, func_like_, &arena, &collection));
  EXPECT_EQ(collection.row_count, 0u);
  EXPECT_EQ(collection.rows, nullptr);
  iree_arena_deinitialize(&arena);
}

TEST_F(SiteCollectionTest, InvalidSanitizerSourceLocationFailsLoudly) {
  loom_op_t* constant = BuildConstant(11);
  loom_value_id_t value = loom_test_constant_result(constant);
  const loom_sanitizer_site_payload_t payload =
      MakePayload(LOOM_SANITIZER_CHECK_KIND_VALUE_NOT_NAN);
  loom_location_id_t tagged_location =
      AddSanitizerSiteLocation((loom_location_id_t)1234, payload);
  BuildAssertValue(value, tagged_location);
  FinalizeModule();

  iree_arena_allocator_t arena;
  iree_arena_initialize(&block_pool_, &arena);
  loom_sanitizer_site_collection_t collection = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_sanitizer_site_collection_build_function(
                            module_, func_like_, &arena, &collection));
  EXPECT_EQ(collection.row_count, 0u);
  EXPECT_EQ(collection.rows, nullptr);
  iree_arena_deinitialize(&arena);
}

}  // namespace
}  // namespace loom
