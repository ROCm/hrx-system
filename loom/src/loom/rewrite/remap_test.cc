// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/rewrite/remap.h"

#include <cstring>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"
#include "loom/ops/test/types.h"

namespace loom {
namespace {

class RemapTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("source"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &source_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("target"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &target_));
    iree_arena_initialize(&block_pool_, &remap_arena_);
  }

  void TearDown() override {
    iree_arena_deinitialize(&remap_arena_);
    loom_module_free(target_);
    loom_module_free(source_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_value_id_t DefineValue(loom_module_t* module, loom_type_t type) {
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_CHECK_OK(loom_module_define_value(module, type, &value_id));
    return value_id;
  }

  loom_ir_remap_t InitializeRemap(
      const loom_ir_remap_options_t* options = nullptr) {
    loom_ir_remap_t remap = {};
    IREE_CHECK_OK(loom_ir_remap_initialize(source_, target_, &remap_arena_,
                                           options, &remap));
    return remap;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* source_ = nullptr;
  loom_module_t* target_ = nullptr;
  iree_arena_allocator_t remap_arena_;
};

TEST_F(RemapTest, RemapsDynamicDimsAndSsaEncodingRefs) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t layout_type =
      loom_type_encoding_with_role(LOOM_ENCODING_ROLE_ADDRESS_LAYOUT);
  loom_value_id_t source_dim = DefineValue(source_, index_type);
  loom_value_id_t source_layout = DefineValue(source_, layout_type);
  loom_value_id_t target_dim = DefineValue(target_, index_type);
  loom_value_id_t target_layout = DefineValue(target_, layout_type);

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_layout, target_layout));

  loom_type_t source_type =
      loom_type_shaped_1d(LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0);
  source_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;
  source_type.encoding_id = (uint16_t)source_layout;

  loom_type_t target_type = {};
  IREE_ASSERT_OK(loom_ir_remap_type(&remap, source_type, &target_type));

  ASSERT_TRUE(loom_type_is_view(target_type));
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(target_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(target_type, 0), target_dim);
  ASSERT_TRUE(loom_type_has_ssa_encoding(target_type));
  EXPECT_EQ(loom_type_encoding_value_id(target_type), target_layout);
}

TEST_F(RemapTest, RemapsRegisterValueTypesAcrossModules) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_dim = DefineValue(source_, index_type);
  loom_value_id_t target_dim = DefineValue(target_, index_type);
  loom_type_t source_value_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0);
  loom_type_t source_type = {};
  IREE_ASSERT_OK(loom_module_intern_register_type(
      source_, /*carrier_payload0=*/42, /*carrier_payload1=*/4,
      source_value_type, &source_type));

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_type_t target_type = {};
  IREE_ASSERT_OK(loom_ir_remap_type(&remap, source_type, &target_type));

  EXPECT_TRUE(loom_type_register_has_value_type(target_type));
  EXPECT_EQ(loom_type_register_payload0(target_type), 42u);
  EXPECT_EQ(loom_type_register_payload1(target_type), 4u);
  const loom_type_t* target_value_type =
      loom_type_register_value_type(target_type);
  ASSERT_NE(target_value_type, nullptr);
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(*target_value_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(*target_value_type, 0), target_dim);
  EXPECT_NE(loom_type_register_data(target_type),
            loom_type_register_data(source_type));
}

TEST_F(RemapTest, RemapsTypesNestedInParameterizedTypeSlots) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_dim = DefineValue(source_, index_type);
  loom_value_id_t target_dim = DefineValue(target_, index_type);
  loom_type_t source_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0);
  loom_type_id_t source_vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(source_, source_vector_type,
                                            &source_vector_type_id));
  loom_type_t source_array_type = {};
  IREE_ASSERT_OK(loom_test_array_type_make(
      source_, LOOM_TEST_ARRAY_TYPE_BUILD_FLAG_HAS_ALIGNMENT,
      source_vector_type_id, /*alignment=*/32, loom_named_attr_slice_empty(),
      &source_array_type));

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_type_t target_array_type = {};
  IREE_ASSERT_OK(
      loom_ir_remap_type(&remap, source_array_type, &target_array_type));

  ASSERT_TRUE(loom_test_array_type_isa(target_array_type));
  EXPECT_TRUE(loom_test_array_type_has_alignment(target_array_type));
  EXPECT_EQ(loom_test_array_type_alignment(target_array_type), 32);
  loom_type_id_t target_vector_type_id =
      loom_test_array_type_element_type(target_array_type);
  ASSERT_LT(target_vector_type_id, target_->types.count);
  loom_type_t target_vector_type =
      target_->types.entries[target_vector_type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(target_vector_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(target_vector_type, 0), target_dim);
}

TEST_F(RemapTest, RejectsSourceValuesDefinedAfterRemapInitialization) {
  loom_ir_remap_t remap = InitializeRemap();
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_value = DefineValue(source_, index_type);
  loom_value_id_t target_value = DefineValue(target_, index_type);

  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_ir_remap_map_value(&remap, source_value, target_value));
}

TEST_F(RemapTest, MapsSparseHighSourceValuesWithoutModuleSizedTable) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_values[128] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_values); ++i) {
    source_values[i] = DefineValue(source_, index_type);
  }
  loom_value_id_t first_target = DefineValue(target_, index_type);
  loom_value_id_t second_target = DefineValue(target_, index_type);

  loom_ir_remap_t remap = InitializeRemap();
  EXPECT_EQ(remap.source_value_snapshot_count, source_->values.count);
  EXPECT_EQ(remap.mapped_value_count, 0u);
  EXPECT_EQ(remap.value_map_entry_capacity, 0u);
  EXPECT_EQ(remap.value_map_entries, nullptr);

  loom_value_id_t source_value =
      source_values[IREE_ARRAYSIZE(source_values) - 1];
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, first_target));
  EXPECT_EQ(remap.mapped_value_count, 1u);
  EXPECT_LT(remap.value_map_entry_capacity, source_->values.count);

  loom_value_id_t resolved_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_ir_remap_resolve_value(&remap, source_value, &resolved_value));
  EXPECT_EQ(resolved_value, first_target);

  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, second_target));
  EXPECT_EQ(remap.mapped_value_count, 1u);
  IREE_ASSERT_OK(
      loom_ir_remap_resolve_value(&remap, source_value, &resolved_value));
  EXPECT_EQ(resolved_value, second_target);
}

TEST_F(RemapTest, MapsSourceIndexedValuesWithoutPopulatingSparseStorage) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_values[128] = {};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(source_values); ++i) {
    source_values[i] = DefineValue(source_, index_type);
  }
  loom_value_id_t first_target = DefineValue(target_, index_type);
  loom_value_id_t second_target = DefineValue(target_, index_type);

  const loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/{},
      /*.remap_symbol=*/{},
      /*.remap_same_module_symbols=*/{},
      /*.value_map_kind=*/LOOM_IR_REMAP_VALUE_MAP_SOURCE_INDEXED,
  };
  loom_ir_remap_t remap = InitializeRemap(&options);
  ASSERT_NE(remap.target_values_by_source, nullptr);
  EXPECT_EQ(remap.value_map_entries, nullptr);
  EXPECT_EQ(remap.value_map_entry_capacity, 0u);

  const loom_value_id_t source_value =
      source_values[IREE_ARRAYSIZE(source_values) - 1];
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, first_target));
  EXPECT_EQ(remap.mapped_value_count, 1u);
  loom_value_id_t resolved_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_ir_remap_resolve_value(&remap, source_value, &resolved_value));
  EXPECT_EQ(resolved_value, first_target);

  EXPECT_FALSE(loom_ir_remap_try_lookup_value(&remap, source_values[0],
                                              &resolved_value));
  EXPECT_EQ(resolved_value, LOOM_VALUE_ID_INVALID);
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, second_target));
  EXPECT_EQ(remap.mapped_value_count, 1u);
  IREE_ASSERT_OK(
      loom_ir_remap_resolve_value(&remap, source_value, &resolved_value));
  EXPECT_EQ(resolved_value, second_target);
  EXPECT_EQ(remap.value_map_entries, nullptr);
  EXPECT_EQ(remap.value_map_entry_capacity, 0u);
}

TEST_F(RemapTest, AllowsUnmappedValuesOnlyWithinSameModule) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_value = DefineValue(source_, index_type);
  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/true,
  };

  loom_ir_remap_t cross_module_remap = InitializeRemap(&options);
  loom_value_id_t target_value = LOOM_VALUE_ID_INVALID;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_NOT_FOUND,
                        loom_ir_remap_resolve_value(
                            &cross_module_remap, source_value, &target_value));

  loom_ir_remap_t same_module_remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(source_, source_, &remap_arena_,
                                          &options, &same_module_remap));
  IREE_ASSERT_OK(loom_ir_remap_resolve_value(&same_module_remap, source_value,
                                             &target_value));
  EXPECT_EQ(target_value, source_value);
}

TEST_F(RemapTest, RejectsUnknownTypeKind) {
  loom_ir_remap_t remap = InitializeRemap();
  loom_type_t malformed = {};
  malformed.header = UINT32_MAX;
  loom_type_t target_type = {};

  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_ir_remap_type(&remap, malformed, &target_type));
}

TEST_F(RemapTest, RemapsPredicateListsInsideDictAttributes) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_value = DefineValue(source_, index_type);
  loom_value_id_t target_value = DefineValue(target_, index_type);

  loom_string_id_t source_predicates_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("predicates"),
                                           &source_predicates_name));
  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_MUL,
      /*.arg_count=*/2,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_NONE},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)source_value, 16, 0},
  };
  loom_named_attr_t source_entries[] = {
      {
          /*.name_id=*/source_predicates_name,
          /*.reserved=*/0,
          /*.value=*/loom_attr_predicate_list(&predicate, 1),
      },
  };
  loom_attribute_t source_dict = {};
  IREE_ASSERT_OK(loom_module_make_canonical_attr_dict(
      source_,
      loom_make_named_attr_slice(source_entries,
                                 IREE_ARRAYSIZE(source_entries)),
      &source_dict));

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_value, target_value));
  loom_attribute_t target_dict = {};
  IREE_ASSERT_OK(loom_ir_remap_attribute(&remap, source_dict, &target_dict));

  ASSERT_EQ(target_dict.kind, LOOM_ATTR_DICT);
  ASSERT_EQ(target_dict.count, 1u);
  iree_string_view_t target_name =
      target_->strings.entries[target_dict.dict_entries[0].name_id];
  EXPECT_TRUE(iree_string_view_equal(target_name, IREE_SV("predicates")));
  loom_attribute_t target_predicates = target_dict.dict_entries[0].value;
  ASSERT_EQ(target_predicates.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_EQ(target_predicates.count, 1u);
  EXPECT_EQ(target_predicates.predicate_list[0].args[0], (int64_t)target_value);
  EXPECT_EQ(target_predicates.predicate_list[0].args[1], 16);
}

TEST_F(RemapTest, CopiesByteAttributePayloadAcrossModules) {
  const uint8_t source_bytes[] = {0x4C, 0x4F, 0x4F, 0x4D};
  loom_ir_remap_t remap = InitializeRemap();
  loom_attribute_t target_attr = {};

  IREE_ASSERT_OK(loom_ir_remap_attribute(
      &remap, loom_attr_bytes(source_bytes, sizeof(source_bytes)),
      &target_attr));

  ASSERT_EQ(target_attr.kind, LOOM_ATTR_BYTES);
  EXPECT_EQ(target_attr.reserved_1, sizeof(source_bytes));
  ASSERT_NE(target_attr.bytes, nullptr);
  EXPECT_NE(target_attr.bytes, source_bytes);
  EXPECT_EQ(memcmp(target_attr.bytes, source_bytes, sizeof(source_bytes)), 0);
}

TEST_F(RemapTest, RemapsCompactParameterizedAttributeSlotsByDescriptor) {
  loom_string_id_t source_label_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("wave"), &source_label_id));
  loom_attribute_t source_attr = {};
  IREE_ASSERT_OK(loom_test_compact_attr_make(
      source_, LOOM_TEST_COMPACT_ATTR_BUILD_FLAG_HAS_LABEL, source_label_id, 64,
      &source_attr));

  loom_ir_remap_t remap = InitializeRemap();
  loom_attribute_t target_attr = {};
  IREE_ASSERT_OK(loom_ir_remap_attribute(&remap, source_attr, &target_attr));

  ASSERT_TRUE(loom_test_compact_attr_isa(target_attr));
  EXPECT_EQ(loom_test_compact_attr_value(target_attr), 64);
  ASSERT_TRUE(loom_test_compact_attr_has_label(target_attr));
  loom_string_id_t target_label_id = loom_test_compact_attr_label(target_attr);
  ASSERT_LT(target_label_id, target_->strings.count);
  EXPECT_TRUE(iree_string_view_equal(target_->strings.entries[target_label_id],
                                     IREE_SV("wave")));
  EXPECT_NE(loom_attr_as_parameterized_slots(source_attr),
            loom_attr_as_parameterized_slots(target_attr));
}

TEST_F(RemapTest, RejectsMalformedPredicateListsWithoutReadingPastPayload) {
  loom_ir_remap_t remap = InitializeRemap();
  loom_predicate_t malformed = {
      /*.kind=*/LOOM_PREDICATE_EQ,
      /*.arg_count=*/4,
      /*.arg_tags=*/
      {LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{1, 1, 0},
  };
  loom_predicate_t* target_predicates = nullptr;
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_ir_remap_predicate_list(&remap, &malformed, 1, &target_predicates));
}

TEST_F(RemapTest, RemapsStaticEncodingDependenciesAcrossModules) {
  loom_string_id_t source_family_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t source_block_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("q4_test"),
                                           &source_family_id));
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("block"), &source_block_id));
  loom_named_attr_t source_attrs[] = {
      {
          /*.name_id=*/source_block_id,
          /*.reserved=*/0,
          /*.value=*/loom_attr_i64(32),
      },
  };
  loom_encoding_t source_encoding = {
      /*.name_id=*/source_family_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/IREE_ARRAYSIZE(source_attrs),
      /*.family=*/{},
      /*.attributes=*/source_attrs,
  };
  uint16_t source_encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(source_, &source_encoding, &source_encoding_id));

  loom_ir_remap_t remap = InitializeRemap();
  uint16_t target_encoding_id = 0;
  IREE_ASSERT_OK(loom_ir_remap_encoding_id(&remap, source_encoding_id,
                                           &target_encoding_id));

  const loom_encoding_t* target_encoding =
      loom_module_encoding(target_, target_encoding_id);
  ASSERT_NE(target_encoding, nullptr);
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[target_encoding->name_id], IREE_SV("q4_test")));
  ASSERT_EQ(target_encoding->attribute_count, 1u);
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[target_encoding->attributes[0].name_id],
      IREE_SV("block")));
  EXPECT_EQ(loom_attr_as_i64(target_encoding->attributes[0].value), 32);
}

TEST_F(RemapTest, RemapsOverflowDimsAndEncodingBeforeInterning) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_dim = DefineValue(source_, index_type);
  loom_value_id_t target_dim = DefineValue(target_, index_type);

  loom_string_id_t source_family_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("source_layout"),
                                           &source_family_id));
  loom_encoding_t source_encoding = {
      /*.name_id=*/source_family_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/0,
      /*.family=*/{},
      /*.attributes=*/NULL,
  };
  uint16_t source_encoding_id = 0;
  IREE_ASSERT_OK(
      loom_module_add_encoding(source_, &source_encoding, &source_encoding_id));

  loom_string_id_t target_dummy_family_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(
      target_, IREE_SV("preexisting_layout"), &target_dummy_family_id));
  loom_encoding_t target_dummy_encoding = {
      /*.name_id=*/target_dummy_family_id,
      /*.alias_id=*/LOOM_STRING_ID_INVALID,
      /*.attribute_count=*/0,
      /*.family=*/{},
      /*.attributes=*/NULL,
  };
  uint16_t target_dummy_encoding_id = 0;
  IREE_ASSERT_OK(loom_module_add_encoding(target_, &target_dummy_encoding,
                                          &target_dummy_encoding_id));

  loom_overflow_dim_t* source_dims = NULL;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &source_->arena, 3, sizeof(loom_overflow_dim_t), (void**)&source_dims));
  source_dims[0] = loom_dim_pack_dynamic(source_dim);
  source_dims[1] = loom_dim_pack_static(4);
  source_dims[2] = loom_dim_pack_static(8);
  loom_type_t source_type = {
      /*.header=*/
      loom_type_make_header(LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, 3, 0),
      /*.encoding_id=*/source_encoding_id,
      /*.encoding_flags=*/0,
      /*.dims=*/{(uint64_t)(uintptr_t)source_dims, 0},
  };

  loom_ir_remap_t remap = InitializeRemap();
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_type_t target_type = {};
  IREE_ASSERT_OK(loom_ir_remap_type(&remap, source_type, &target_type));

  ASSERT_EQ(target_->encodings.count, 2u);
  EXPECT_EQ(target_type.encoding_id, 2u);
  ASSERT_FALSE(loom_type_has_inline_dims(target_type));
  EXPECT_TRUE(loom_type_dim_is_dynamic_at(target_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(target_type, 0), target_dim);
  EXPECT_EQ(loom_type_dim_static_size_at(target_type, 1), 4);
  EXPECT_EQ(loom_type_dim_static_size_at(target_type, 2), 8);
  EXPECT_NE((const void*)(uintptr_t)target_type.dims[0],
            (const void*)source_dims);

  ASSERT_EQ(target_->types.count, 3u);
  EXPECT_TRUE(loom_type_equal(target_->types.entries[0], index_type));
  EXPECT_TRUE(loom_type_equal(target_->types.entries[1],
                              loom_type_scalar(LOOM_SCALAR_TYPE_F32)));
  EXPECT_TRUE(loom_type_equal(target_->types.entries[2], target_type));
}

TEST_F(RemapTest, RejectsDeepStaticEncodingNesting) {
  loom_string_id_t family_id = LOOM_STRING_ID_INVALID;
  loom_string_id_t next_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("nested"), &family_id));
  IREE_ASSERT_OK(loom_module_intern_string(source_, IREE_SV("next"), &next_id));

  uint16_t previous_encoding_id = 0;
  for (uint16_t i = 0; i < 18; ++i) {
    uint8_t attribute_count = previous_encoding_id == 0 ? 0 : 1;
    loom_named_attr_t attrs[] = {
        {
            /*.name_id=*/next_id,
            /*.reserved=*/0,
            /*.value=*/loom_attr_encoding(previous_encoding_id),
        },
    };
    loom_encoding_t encoding = {
        /*.name_id=*/family_id,
        /*.alias_id=*/LOOM_STRING_ID_INVALID,
        /*.attribute_count=*/attribute_count,
        /*.family=*/{},
        /*.attributes=*/attribute_count == 0 ? nullptr : attrs,
    };
    IREE_ASSERT_OK(
        loom_module_add_encoding(source_, &encoding, &previous_encoding_id));
  }

  loom_ir_remap_t remap = InitializeRemap();
  uint16_t target_encoding_id = 0;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_ir_remap_encoding_id(&remap, previous_encoding_id,
                                                  &target_encoding_id));
}

TEST_F(RemapTest, RemapsTypeAttributesAcrossModules) {
  loom_type_t source_type = loom_type_shaped_1d(
      LOOM_TYPE_TENSOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  loom_type_t interned_source_type = {};
  IREE_ASSERT_OK(
      loom_module_intern_type(source_, source_type, &interned_source_type));
  loom_type_id_t source_type_id = LOOM_TYPE_ID_INVALID;
  for (iree_host_size_t i = 0; i < source_->types.count; ++i) {
    if (loom_type_equal(source_->types.entries[i], interned_source_type)) {
      source_type_id = (loom_type_id_t)i;
      break;
    }
  }
  ASSERT_NE(source_type_id, LOOM_TYPE_ID_INVALID);

  loom_ir_remap_t remap = InitializeRemap();
  loom_attribute_t target_attr = {};
  IREE_ASSERT_OK(loom_ir_remap_attribute(&remap, loom_attr_type(source_type_id),
                                         &target_attr));

  ASSERT_EQ(target_attr.kind, LOOM_ATTR_TYPE);
  ASSERT_LT(target_attr.type_id, target_->types.count);
  EXPECT_TRUE(loom_type_equal(target_->types.entries[target_attr.type_id],
                              source_type));
}

TEST_F(RemapTest, RemapsLocationsAcrossModules) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(source_, IREE_SV("model.loom"), &source_id));
  loom_source_id_t target_preexisting_source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_register_source(target_,
                                             IREE_SV("target-preexisting.loom"),
                                             &target_preexisting_source_id));

  loom_location_entry_t file_entry =
      loom_location_file_range(source_id, 2, 3, 4, 5);
  loom_location_id_t file_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(
      loom_module_add_location(source_, file_entry, &file_location_id));
  loom_location_field_span_t field_span = {
      /*.kind=*/LOOM_LOCATION_FIELD_OPERAND,
      /*.index=*/0,
      /*.start_line=*/2,
      /*.start_col=*/3,
      /*.end_line=*/2,
      /*.end_col=*/7,
  };
  IREE_ASSERT_OK(loom_module_attach_location_field_spans(
      source_, file_location_id, &field_span, 1));

  loom_location_id_t* fused_children = nullptr;
  IREE_ASSERT_OK(iree_arena_allocate_array(
      &source_->arena, 1, sizeof(loom_location_id_t), (void**)&fused_children));
  fused_children[0] = file_location_id;
  loom_location_entry_t fused_entry = {};
  fused_entry.kind = LOOM_LOCATION_FUSED;
  fused_entry.flags = LOOM_LOCATION_FLAG_SYNTHETIC;
  fused_entry.fused.count = 1;
  fused_entry.fused.children = fused_children;
  loom_location_id_t fused_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(
      loom_module_add_location(source_, fused_entry, &fused_location_id));

  loom_ir_remap_t remap = InitializeRemap();
  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_ir_remap_location_id(&remap, fused_location_id,
                                           &target_location_id));

  ASSERT_LT(target_location_id, target_->locations.count);
  const loom_location_entry_t& target_fused =
      target_->locations.entries[target_location_id];
  ASSERT_EQ(target_fused.kind, LOOM_LOCATION_FUSED);
  ASSERT_EQ(target_fused.fused.count, 1u);
  ASSERT_NE(target_fused.fused.children, nullptr);

  loom_location_id_t target_child_id = target_fused.fused.children[0];
  ASSERT_LT(target_child_id, target_->locations.count);
  const loom_location_entry_t& target_child =
      target_->locations.entries[target_child_id];
  ASSERT_EQ(target_child.kind, LOOM_LOCATION_FILE);
  EXPECT_NE(target_child.file.source_id, source_id);
  EXPECT_NE(target_child.file.source_id, target_preexisting_source_id);
  ASSERT_LT(target_child.file.source_id, target_->sources.count);
  EXPECT_TRUE(iree_string_view_equal(
      target_->sources.entries[target_child.file.source_id],
      IREE_SV("model.loom")));
  EXPECT_EQ(target_child.file.start_line, 2u);
  ASSERT_EQ(target_child.file.field_span_count, 1u);
  ASSERT_NE(target_child.file.field_spans, nullptr);
  EXPECT_NE(target_child.file.field_spans, &field_span);
  EXPECT_EQ(target_child.file.field_spans[0].end_col, 7u);
}

TEST_F(RemapTest, RemapsTaggedLocationsAcrossModules) {
  loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_register_source(source_, IREE_SV("model.loom"), &source_id));

  loom_location_id_t child_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(
      source_, loom_location_file_range(source_id, 5, 6, 5, 6),
      &child_location_id));

  const uint8_t data[] = {0x01, 0x2A, 0xFF};
  loom_location_id_t tagged_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_module_add_location(
      source_,
      loom_location_tagged(LOOM_LOCATION_TAG_SANITIZER_SITE, child_location_id,
                           data, IREE_ARRAYSIZE(data)),
      &tagged_location_id));

  loom_ir_remap_t remap = InitializeRemap();
  loom_location_id_t target_location_id = LOOM_LOCATION_UNKNOWN;
  IREE_ASSERT_OK(loom_ir_remap_location_id(&remap, tagged_location_id,
                                           &target_location_id));

  ASSERT_LT(target_location_id, target_->locations.count);
  const loom_location_entry_t& target_tagged =
      target_->locations.entries[target_location_id];
  ASSERT_EQ(target_tagged.kind, LOOM_LOCATION_TAGGED);
  EXPECT_EQ(target_tagged.tagged.tag, LOOM_LOCATION_TAG_SANITIZER_SITE);
  ASSERT_NE(target_tagged.tagged.child, LOOM_LOCATION_UNKNOWN);
  ASSERT_LT(target_tagged.tagged.child, target_->locations.count);
  ASSERT_EQ(target_tagged.tagged.data_length, IREE_ARRAYSIZE(data));
  ASSERT_NE(target_tagged.tagged.data, nullptr);
  EXPECT_NE(target_tagged.tagged.data, data);
  EXPECT_EQ(std::memcmp(target_tagged.tagged.data, data, IREE_ARRAYSIZE(data)),
            0);

  const loom_location_entry_t& target_child =
      target_->locations.entries[target_tagged.tagged.child];
  ASSERT_EQ(target_child.kind, LOOM_LOCATION_FILE);
  ASSERT_LT(target_child.file.source_id, target_->sources.count);
  EXPECT_TRUE(iree_string_view_equal(
      target_->sources.entries[target_child.file.source_id],
      IREE_SV("model.loom")));
  EXPECT_EQ(target_child.file.start_line, 5u);
  EXPECT_EQ(target_child.file.start_col, 6u);
}

static iree_status_t RemapSymbolByName(void* user_data,
                                       const loom_module_t* source_module,
                                       loom_module_t* target_module,
                                       loom_symbol_ref_t source_ref,
                                       loom_symbol_ref_t* out_target_ref) {
  (void)user_data;
  const loom_symbol_t* source_symbol =
      &source_module->symbols.entries[source_ref.symbol_id];
  iree_string_view_t source_name =
      source_module->strings.entries[source_symbol->name_id];
  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_RETURN_IF_ERROR(
      loom_module_intern_string(target_module, source_name, &target_name_id));
  uint16_t target_symbol_id =
      loom_module_find_symbol(target_module, target_name_id);
  if (target_symbol_id == LOOM_SYMBOL_ID_INVALID) {
    IREE_RETURN_IF_ERROR(loom_module_add_symbol(target_module, target_name_id,
                                                &target_symbol_id));
  }
  *out_target_ref = {/*.module_id=*/0, /*.symbol_id=*/target_symbol_id};
  return iree_ok_status();
}

TEST_F(RemapTest, RemapsOrderedSymbolArraysAcrossModules) {
  loom_string_id_t alpha_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("alpha"), &alpha_name_id));
  uint16_t alpha_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, alpha_name_id, &alpha_symbol_id));
  loom_string_id_t beta_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("beta"), &beta_name_id));
  uint16_t beta_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, beta_name_id, &beta_symbol_id));

  const loom_symbol_ref_t source_refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
  };
  const loom_symbol_ref_t equal_refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
  };
  const loom_symbol_ref_t reordered_refs[] = {
      {/*.module_id=*/0, /*.symbol_id=*/alpha_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
      {/*.module_id=*/0, /*.symbol_id=*/beta_symbol_id},
  };
  loom_attribute_t source_attr =
      loom_attr_symbol_array(source_refs, IREE_ARRAYSIZE(source_refs));
  loom_attribute_t equal_attr =
      loom_attr_symbol_array(equal_refs, IREE_ARRAYSIZE(equal_refs));
  loom_attribute_t reordered_attr =
      loom_attr_symbol_array(reordered_refs, IREE_ARRAYSIZE(reordered_refs));
  EXPECT_TRUE(loom_attribute_equal(&source_attr, &equal_attr));
  EXPECT_EQ(loom_attribute_hash(&source_attr),
            loom_attribute_hash(&equal_attr));
  EXPECT_FALSE(loom_attribute_equal(&source_attr, &reordered_attr));

  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/{}, /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolByName, NULL),
  };
  loom_ir_remap_t remap = InitializeRemap(&options);
  loom_attribute_t target_attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_ir_remap_attribute(&remap, source_attr, &target_attr));

  loom_symbol_ref_array_t target_refs = loom_attr_as_symbol_array(target_attr);
  ASSERT_EQ(target_refs.count, IREE_ARRAYSIZE(source_refs));
  EXPECT_NE(target_refs.values, source_refs);
  EXPECT_EQ(target_refs.values[0].symbol_id, target_refs.values[2].symbol_id);
  EXPECT_NE(target_refs.values[0].symbol_id, target_refs.values[1].symbol_id);
  const loom_string_id_t target_beta_name_id =
      target_->symbols.entries[target_refs.values[0].symbol_id].name_id;
  const loom_string_id_t target_alpha_name_id =
      target_->symbols.entries[target_refs.values[1].symbol_id].name_id;
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[target_beta_name_id], IREE_SV("beta")));
  EXPECT_TRUE(iree_string_view_equal(
      target_->strings.entries[target_alpha_name_id], IREE_SV("alpha")));
}

TEST_F(RemapTest, RemapsParameterizedAttributeArraysAcrossModules) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t source_dim = DefineValue(source_, index_type);
  loom_value_id_t target_dim = DefineValue(target_, index_type);
  loom_type_t source_vector_type =
      loom_type_shaped_1d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_dynamic(source_dim), 0);
  loom_type_id_t source_vector_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(source_, source_vector_type,
                                            &source_vector_type_id));

  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("target"), &source_name_id));
  uint16_t source_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, source_name_id, &source_symbol_id));

  loom_attribute_t tile = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_tile_attr_make(source_, 8, &tile));
  loom_attribute_t options = loom_attr_absent();
  IREE_ASSERT_OK(loom_test_options_attr_make(
      source_,
      LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_ELEMENT_TYPE |
          LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TILE |
          LOOM_TEST_OPTIONS_ATTR_BUILD_FLAG_HAS_TARGET,
      LOOM_TEST_OPTIONS_MODE_FAST, loom_enum_array_empty(),
      source_vector_type_id, tile, (loom_symbol_ref_t){0, source_symbol_id},
      loom_parameterized_attr_array_empty(), &options));
  loom_attribute_t source_values[] = {tile, options, tile};
  loom_attribute_t source_array = loom_attr_absent();
  IREE_ASSERT_OK(loom_module_make_parameterized_attr_array(
      source_,
      loom_make_parameterized_attr_array(source_values,
                                         IREE_ARRAYSIZE(source_values)),
      &source_array));

  loom_ir_remap_options_t remap_options = {
      /*.allow_unmapped_values=*/{}, /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolByName, NULL),
  };
  loom_ir_remap_t remap = InitializeRemap(&remap_options);
  IREE_ASSERT_OK(loom_ir_remap_map_value(&remap, source_dim, target_dim));
  loom_attribute_t target_array = loom_attr_absent();
  IREE_ASSERT_OK(loom_ir_remap_attribute(&remap, source_array, &target_array));

  loom_parameterized_attr_array_t target_values =
      loom_attr_as_parameterized_array(target_array);
  ASSERT_EQ(target_values.count, 3u);
  EXPECT_TRUE(loom_test_tile_attr_isa(target_values.values[0]));
  EXPECT_TRUE(loom_test_options_attr_isa(target_values.values[1]));
  EXPECT_TRUE(loom_test_tile_attr_isa(target_values.values[2]));
  EXPECT_TRUE(
      loom_attribute_equal(&target_values.values[0], &target_values.values[2]));
  EXPECT_NE(target_values.values,
            loom_attr_as_parameterized_array(source_array).values);

  loom_attribute_t target_options = target_values.values[1];
  loom_type_id_t target_vector_type_id =
      loom_test_options_attr_element_type(target_options);
  ASSERT_LT(target_vector_type_id, target_->types.count);
  loom_type_t target_vector_type =
      target_->types.entries[target_vector_type_id];
  ASSERT_TRUE(loom_type_dim_is_dynamic_at(target_vector_type, 0));
  EXPECT_EQ(loom_type_dim_value_id_at(target_vector_type, 0), target_dim);
  loom_symbol_ref_t target_ref = loom_test_options_attr_target(target_options);
  ASSERT_LT(target_ref.symbol_id, target_->symbols.count);
  loom_string_id_t target_name_id =
      target_->symbols.entries[target_ref.symbol_id].name_id;
  EXPECT_TRUE(iree_string_view_equal(target_->strings.entries[target_name_id],
                                     IREE_SV("target")));
}

static iree_status_t RemapSymbolToMissingTarget(
    void* user_data, const loom_module_t* source_module,
    loom_module_t* target_module, loom_symbol_ref_t source_ref,
    loom_symbol_ref_t* out_target_ref) {
  (void)user_data;
  (void)source_module;
  (void)target_module;
  (void)source_ref;
  *out_target_ref = {/*.module_id=*/0, /*.symbol_id=*/42};
  return iree_ok_status();
}

typedef struct SameModuleSymbolRemap {
  // Source symbol replaced by the callback.
  loom_symbol_ref_t source_ref;

  // Target symbol returned by the callback.
  loom_symbol_ref_t target_ref;

  // Number of callback invocations observed.
  uint32_t invocation_count;
} SameModuleSymbolRemap;

static iree_status_t RemapSameModuleSymbol(void* user_data,
                                           const loom_module_t* source_module,
                                           loom_module_t* target_module,
                                           loom_symbol_ref_t source_ref,
                                           loom_symbol_ref_t* out_target_ref) {
  auto* state = static_cast<SameModuleSymbolRemap*>(user_data);
  EXPECT_EQ(source_module, target_module);
  ++state->invocation_count;
  *out_target_ref = source_ref.module_id == state->source_ref.module_id &&
                            source_ref.symbol_id == state->source_ref.symbol_id
                        ? state->target_ref
                        : source_ref;
  return iree_ok_status();
}

typedef struct SymbolPairRemap {
  loom_symbol_ref_t source_refs[2];
  loom_symbol_ref_t target_refs[2];
  uint32_t invocation_count;
} SymbolPairRemap;

static iree_status_t RemapSymbolPair(void* user_data,
                                     const loom_module_t* source_module,
                                     loom_module_t* target_module,
                                     loom_symbol_ref_t source_ref,
                                     loom_symbol_ref_t* out_target_ref) {
  auto* state = static_cast<SymbolPairRemap*>(user_data);
  EXPECT_EQ(source_module, target_module);
  ++state->invocation_count;
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(state->source_refs); ++i) {
    if (source_ref.module_id == state->source_refs[i].module_id &&
        source_ref.symbol_id == state->source_refs[i].symbol_id) {
      *out_target_ref = state->target_refs[i];
      return iree_ok_status();
    }
  }
  *out_target_ref = source_ref;
  return iree_ok_status();
}

TEST_F(RemapTest, SameModuleSymbolsRemapOnlyWhenEnabled) {
  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("source"), &source_name_id));
  uint16_t source_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, source_name_id, &source_symbol_id));
  const loom_symbol_ref_t source_ref = {/*.module_id=*/0,
                                        /*.symbol_id=*/source_symbol_id};

  loom_string_id_t target_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("target"), &target_name_id));
  uint16_t target_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, target_name_id, &target_symbol_id));
  const loom_symbol_ref_t target_ref = {/*.module_id=*/0,
                                        /*.symbol_id=*/target_symbol_id};

  SameModuleSymbolRemap state = {
      /*.source_ref=*/source_ref,
      /*.target_ref=*/target_ref,
      /*.invocation_count=*/0,
  };
  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/false,
      /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSameModuleSymbol, &state),
  };
  loom_ir_remap_t identity_remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(source_, source_, &remap_arena_,
                                          &options, &identity_remap));
  loom_attribute_t target_attr = {};
  IREE_ASSERT_OK(loom_ir_remap_attribute(
      &identity_remap, loom_attr_symbol(source_ref), &target_attr));
  EXPECT_EQ(target_attr.symbol.symbol_id, source_ref.symbol_id);
  EXPECT_EQ(state.invocation_count, 0u);

  options.remap_same_module_symbols = true;
  loom_ir_remap_t symbol_remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(source_, source_, &remap_arena_,
                                          &options, &symbol_remap));
  IREE_ASSERT_OK(loom_ir_remap_attribute(
      &symbol_remap, loom_attr_symbol(source_ref), &target_attr));
  EXPECT_EQ(target_attr.symbol.symbol_id, target_ref.symbol_id);
  EXPECT_EQ(state.invocation_count, 1u);
}

TEST_F(RemapTest, SymbolSetRemapCanonicalizesByTargetSymbolName) {
  loom_symbol_ref_t refs[4] = {};
  const iree_string_view_t names[] = {
      IREE_SV("alpha"),
      IREE_SV("zeta"),
      IREE_SV("omega"),
      IREE_SV("beta"),
  };
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(source_, names[i], &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(source_, name_id, &symbol_id));
    refs[i] = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  SymbolPairRemap state = {
      /*.source_refs=*/{refs[0], refs[1]},
      /*.target_refs=*/{refs[2], refs[3]},
      /*.invocation_count=*/0,
  };
  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/false,
      /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolPair, &state),
      /*.remap_same_module_symbols=*/true,
  };
  loom_ir_remap_t remap = {};
  IREE_ASSERT_OK(loom_ir_remap_initialize(source_, source_, &remap_arena_,
                                          &options, &remap));
  loom_attribute_t target_attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_ir_remap_attribute(
      &remap, loom_attr_symbol_set(refs, /*count=*/2), &target_attr));
  ASSERT_EQ(target_attr.kind, LOOM_ATTR_SYMBOL_SET);
  loom_symbol_ref_array_t target_set = loom_attr_as_symbol_set(target_attr);
  ASSERT_EQ(target_set.count, 2u);
  EXPECT_EQ(target_set.values[0].symbol_id, refs[3].symbol_id);
  EXPECT_EQ(target_set.values[1].symbol_id, refs[2].symbol_id);
  EXPECT_EQ(state.invocation_count, 2u);

  state.target_refs[0] = refs[3];
  state.target_refs[1] = refs[3];
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_ir_remap_attribute(&remap, loom_attr_symbol_set(refs, /*count=*/2),
                              &target_attr));
}

TEST_F(RemapTest, CrossModuleSymbolRefsRequirePolicy) {
  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("callee"), &source_name_id));
  uint16_t source_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, source_name_id, &source_symbol_id));
  loom_symbol_ref_t source_ref = {/*.module_id=*/0,
                                  /*.symbol_id=*/source_symbol_id};

  loom_ir_remap_t strict_remap = InitializeRemap();
  loom_attribute_t target_attr = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_ir_remap_attribute(&strict_remap, loom_attr_symbol(source_ref),
                              &target_attr));

  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/{}, /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolByName, NULL),
  };
  loom_ir_remap_t policy_remap = InitializeRemap(&options);
  IREE_ASSERT_OK(loom_ir_remap_attribute(
      &policy_remap, loom_attr_symbol(source_ref), &target_attr));

  ASSERT_EQ(target_attr.kind, LOOM_ATTR_SYMBOL);
  ASSERT_TRUE(loom_symbol_ref_is_valid(target_attr.symbol));
  ASSERT_LT(target_attr.symbol.symbol_id, target_->symbols.count);
  loom_string_id_t target_name_id =
      target_->symbols.entries[target_attr.symbol.symbol_id].name_id;
  EXPECT_TRUE(iree_string_view_equal(target_->strings.entries[target_name_id],
                                     IREE_SV("callee")));
}

TEST_F(RemapTest, RemapsSymbolsNestedInParameterizedTypes) {
  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("target"), &source_name_id));
  uint16_t source_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, source_name_id, &source_symbol_id));

  loom_type_id_t bf16_type_id = LOOM_TYPE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_type_id(
      source_, loom_type_scalar(LOOM_SCALAR_TYPE_BF16), &bf16_type_id));
  loom_type_t source_type = {};
  IREE_ASSERT_OK(loom_test_matrix_type_make(
      source_, LOOM_TEST_MATRIX_TYPE_BUILD_FLAG_HAS_TARGET, bf16_type_id,
      LOOM_TEST_MATRIX_TYPE_SCOPE_SUBGROUP, 16,
      (loom_symbol_ref_t){0, source_symbol_id}, &source_type));

  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/{}, /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolByName, NULL),
  };
  loom_ir_remap_t remap = InitializeRemap(&options);
  loom_type_t target_type = {};
  IREE_ASSERT_OK(loom_ir_remap_type(&remap, source_type, &target_type));

  ASSERT_TRUE(loom_test_matrix_type_isa(target_type));
  ASSERT_TRUE(loom_test_matrix_type_has_target(target_type));
  loom_symbol_ref_t target_ref = loom_test_matrix_type_target(target_type);
  ASSERT_LT(target_ref.symbol_id, target_->symbols.count);
  loom_string_id_t target_name_id =
      target_->symbols.entries[target_ref.symbol_id].name_id;
  EXPECT_TRUE(iree_string_view_equal(target_->strings.entries[target_name_id],
                                     IREE_SV("target")));
}

TEST_F(RemapTest, CrossModuleSymbolPolicyMustReturnTargetSymbol) {
  loom_string_id_t source_name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_intern_string(source_, IREE_SV("callee"), &source_name_id));
  uint16_t source_symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(
      loom_module_add_symbol(source_, source_name_id, &source_symbol_id));
  loom_symbol_ref_t source_ref = {/*.module_id=*/0,
                                  /*.symbol_id=*/source_symbol_id};

  loom_ir_remap_options_t options = {
      /*.allow_unmapped_values=*/{}, /*.remap_symbol=*/
      loom_ir_remap_symbol_callback_make(RemapSymbolToMissingTarget, NULL),
  };
  loom_ir_remap_t remap = InitializeRemap(&options);
  loom_attribute_t target_attr = {};
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_ir_remap_attribute(&remap, loom_attr_symbol(source_ref),
                              &target_attr));
}

}  // namespace
}  // namespace loom
