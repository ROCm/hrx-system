// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdio>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/global/ops.h"
#include "loom/ops/test/ops.h"

namespace loom {
namespace {

class BuilderStorageTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterDialect(LOOM_DIALECT_TEST, loom_test_dialect_vtables);
    RegisterDialect(LOOM_DIALECT_GLOBAL, loom_global_dialect_vtables);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    loom_op_t* constant = nullptr;
    IREE_ASSERT_OK(loom_test_constant_build(
        &builder_, loom_attr_i64(42), loom_type_scalar(LOOM_SCALAR_TYPE_INDEX),
        LOOM_LOCATION_UNKNOWN, &constant));
    input_ = loom_test_constant_result(constant);
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  void RegisterDialect(uint8_t dialect_id,
                       decltype(&loom_test_dialect_vtables) dialect) {
    iree_host_size_t count = 0;
    const loom_op_vtable_t* const* vtables = dialect(&count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, dialect_id, vtables,
                                                 (uint16_t)count));
  }

  // Storage backing the module's arena.
  iree_arena_block_pool_t block_pool_;
  // Registry containing the generated test and global dialects.
  loom_context_t context_;
  // Module owning built operations and their attribute payloads.
  loom_module_t* module_ = nullptr;
  // Builder inserting into the module body.
  loom_builder_t builder_;
  // Index value used by operand tables and predicate references.
  loom_value_id_t input_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(BuilderStorageTest,
       OperandDictionariesKeepNameValuePairsThroughSorting) {
  const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_op_t* second_constant = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(99), type,
                                          LOOM_LOCATION_UNKNOWN,
                                          &second_constant));
  const loom_value_id_t values[] = {input_,
                                    loom_test_constant_result(second_constant)};
  for (iree_host_size_t count : {0u, 1u, 64u, 65u, 1024u, UINT16_MAX - 1u}) {
    SCOPED_TRACE(count);
    std::vector<loom_named_value_t> parameters(count);
    std::vector<loom_string_id_t> names(count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      const iree_host_size_t ordinal = count - i - 1;
      char name[32];
      std::snprintf(name, sizeof(name), "parameter_%05zu", (size_t)ordinal);
      IREE_ASSERT_OK(loom_builder_intern_string(
          &builder_, iree_make_cstring_view(name), &names[ordinal]));
      parameters[i] = {names[ordinal], 0, values[ordinal % 2]};
    }

    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_operand_dict_build(&builder_, input_,
                                                parameters.data(), count, type,
                                                LOOM_LOCATION_UNKNOWN, &op));
    const auto operands = loom_test_operand_dict_params(op);
    const auto dictionary = loom_test_operand_dict_param_names(op);
    ASSERT_EQ(operands.count, count);
    ASSERT_EQ(dictionary.count, count);
    for (iree_host_size_t i = 0; i < count; ++i) {
      EXPECT_EQ(parameters[i].name_id, names[count - i - 1]);
      EXPECT_EQ(parameters[i].value_id, values[(count - i - 1) % 2]);
      parameters[i] = {};
      EXPECT_EQ(operands.values[i], values[i % 2]);
      EXPECT_EQ(dictionary.entries[i].name_id, names[i]);
      EXPECT_EQ(dictionary.entries[i].value.i64, (int64_t)i);
    }
    EXPECT_EQ(loom_test_operand_dict_has_param_names(op), count != 0);
  }
}

TEST_F(BuilderStorageTest, OperandDictionaryRetainsOnlyItsOwnedNameEntries) {
  constexpr iree_host_size_t kCount = 1024;
  std::vector<loom_named_value_t> parameters(kCount);
  std::vector<loom_value_id_t> operands(kCount);
  for (iree_host_size_t i = 0; i < kCount; ++i) {
    char name[32];
    std::snprintf(name, sizeof(name), "parameter_%05zu", (size_t)(kCount - i));
    IREE_ASSERT_OK(loom_builder_intern_string(
        &builder_, iree_make_cstring_view(name), &parameters[i].name_id));
    parameters[i].value_id = input_;
  }
  const auto before = module_->arena.used_allocation_size;
  loom_attribute_t dictionary = {};
  IREE_ASSERT_OK(loom_builder_set_operand_dict(
      &builder_, loom_make_named_value_slice(parameters.data(), kCount),
      operands.data(), &dictionary));
  EXPECT_EQ(module_->arena.used_allocation_size - before,
            kCount * sizeof(loom_named_attr_t));
  IREE_ASSERT_OK(loom_module_verify_canonical_attr_dict(module_, dictionary));
}

TEST_F(BuilderStorageTest, OperandDictionaryRejectsSeparatedDuplicateKeys) {
  loom_string_id_t alpha = LOOM_STRING_ID_INVALID;
  loom_string_id_t beta = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("alpha"), &alpha));
  IREE_ASSERT_OK(loom_builder_intern_string(&builder_, IREE_SV("beta"), &beta));
  const loom_named_value_t parameters[] = {
      {beta, 0, input_}, {alpha, 0, input_}, {beta, 0, input_}};
  loom_value_id_t operands[] = {LOOM_VALUE_ID_INVALID, LOOM_VALUE_ID_INVALID,
                                LOOM_VALUE_ID_INVALID};
  loom_attribute_t dictionary = loom_attr_i64(42);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INVALID_ARGUMENT,
      loom_builder_set_operand_dict(&builder_,
                                    loom_make_named_value_slice(parameters, 3),
                                    operands, &dictionary));
  EXPECT_TRUE(loom_attr_is_absent(dictionary));
  for (auto operand : operands) {
    EXPECT_EQ(operand, LOOM_VALUE_ID_INVALID);
  }
}

TEST_F(BuilderStorageTest,
       OperandDictionaryValidatesBeforeAllocatingOrWriting) {
  loom_string_id_t name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(&builder_, IREE_SV("name"), &name));
  const loom_named_value_t invalid_entries[] = {
      {name, 1, input_},
      {LOOM_STRING_ID_INVALID, 0, input_},
      {(loom_string_id_t)module_->strings.count, 0, input_},
      {name, 0, LOOM_VALUE_ID_INVALID},
      {name, 0, (loom_value_id_t)module_->values.count},
  };
  for (const auto& entry : invalid_entries) {
    const auto before = module_->arena.used_allocation_size;
    loom_value_id_t operand = LOOM_VALUE_ID_INVALID;
    loom_attribute_t dictionary = loom_attr_i64(42);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                          loom_builder_set_operand_dict(
                              &builder_, loom_make_named_value_slice(&entry, 1),
                              &operand, &dictionary));
    EXPECT_TRUE(loom_attr_is_absent(dictionary));
    EXPECT_EQ(operand, LOOM_VALUE_ID_INVALID);
    EXPECT_EQ(module_->arena.used_allocation_size, before);
  }
}

TEST_F(BuilderStorageTest, SeparateResultTypesFollowDeclaredFields) {
  const loom_type_t first_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  const loom_type_t second_type = loom_type_scalar(LOOM_SCALAR_TYPE_BF16);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_result_pair_build(&builder_, second_type, first_type,
                                             LOOM_LOCATION_UNKNOWN, &op));

  const loom_value_id_t first = loom_test_result_pair_first(op);
  const loom_value_id_t second = loom_test_result_pair_second(op);
  ASSERT_NE(first, LOOM_VALUE_ID_INVALID);
  ASSERT_NE(second, LOOM_VALUE_ID_INVALID);
  EXPECT_NE(first, second);
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, first), first_type));
  EXPECT_TRUE(
      loom_type_equal(loom_module_value_type(module_, second), second_type));
}

TEST_F(BuilderStorageTest, IntegerArraysOutliveCallerStorage) {
  for (iree_host_size_t count : {0u, 2u}) {
    int64_t keys[] = {-7, 42};
    const loom_value_id_t values[] = {input_, input_, input_};
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_attr_table_build(
        &builder_, input_, count ? keys : nullptr, count, values, count + 1,
        &type, 1, nullptr, 0, LOOM_LOCATION_UNKNOWN, &op));

    keys[0] = 100;
    keys[1] = 200;
    const loom_attribute_t attribute = loom_test_attr_table_case_keys(op);
    ASSERT_EQ(attribute.kind, LOOM_ATTR_I64_ARRAY);
    ASSERT_EQ(attribute.count, count);
    if (count) {
      EXPECT_EQ(attribute.i64_array[0], -7);
      EXPECT_EQ(attribute.i64_array[1], 42);
    }
  }
}

TEST_F(BuilderStorageTest, PredicatesOutliveCallerStorage) {
  for (iree_host_size_t count : {0u, 1u}) {
    loom_predicate_t predicate = {
        LOOM_PREDICATE_LT,
        2,
        {LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
        {},
        {(int64_t)input_, 64},
    };
    const loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_assume_build(
        &builder_, &input_, 1, count ? &predicate : nullptr, count, &type, 1,
        LOOM_LOCATION_UNKNOWN, &op));

    predicate = {};
    const loom_attribute_t attribute = loom_test_assume_predicates(op);
    ASSERT_EQ(attribute.kind, LOOM_ATTR_PREDICATE_LIST);
    ASSERT_EQ(attribute.count, count);
    if (count) {
      const loom_predicate_t& stored = attribute.predicate_list[0];
      EXPECT_EQ(stored.kind, LOOM_PREDICATE_LT);
      EXPECT_EQ(stored.arg_count, 2);
      EXPECT_EQ(stored.arg_tags[0], LOOM_PRED_ARG_VALUE);
      EXPECT_EQ(stored.arg_tags[1], LOOM_PRED_ARG_CONST);
      EXPECT_EQ(stored.args[0], input_);
      EXPECT_EQ(stored.args[1], 64);
      EXPECT_TRUE(
          loom_value_has_attribute_uses(loom_module_value(module_, input_)));
    }
  }
}

TEST_F(BuilderStorageTest, BytesOutliveCallerStorage) {
  for (iree_host_size_t count : {0u, 3u}) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, count ? IREE_SV("payload") : IREE_SV("empty"), &name));
    loom_symbol_id_t symbol = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name, &symbol));
    uint8_t bytes[] = {0x00, 0x7F, 0xFF};
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_global_rodata_def_build(
        &builder_, 0, {0, symbol}, 0,
        iree_make_const_byte_span(count ? bytes : nullptr, count),
        LOOM_LOCATION_UNKNOWN, &op));

    memset(bytes, 0xAB, sizeof(bytes));
    const iree_const_byte_span_t stored = loom_global_rodata_def_contents(op);
    EXPECT_EQ(loom_global_rodata_def_contents_attr(op).kind, LOOM_ATTR_BYTES);
    ASSERT_EQ(stored.data_length, count);
    if (count) {
      EXPECT_EQ(stored.data[0], 0x00);
      EXPECT_EQ(stored.data[1], 0x7F);
      EXPECT_EQ(stored.data[2], 0xFF);
    }
  }
}

TEST_F(BuilderStorageTest, EnumArraysPreserveOrderAndOptionalPresence) {
  for (bool present : {false, true}) {
    for (iree_host_size_t count : {0u, 3u}) {
      uint8_t required[] = {255, 1, 255};
      uint8_t optional[] = {7, 42, 1};
      loom_op_t* op = nullptr;
      IREE_ASSERT_OK(loom_test_enum_array_attrs_build(
          &builder_,
          present ? LOOM_TEST_ENUM_ARRAY_ATTRS_BUILD_FLAG_HAS_OPTIONAL_VALUES
                  : 0,
          loom_make_enum_array(required, IREE_ARRAYSIZE(required)),
          loom_make_enum_array(optional, count), loom_named_attr_slice_empty(),
          LOOM_LOCATION_UNKNOWN, &op));

      memset(required, 0, sizeof(required));
      memset(optional, 0, sizeof(optional));
      const loom_enum_array_t stored =
          loom_test_enum_array_attrs_required_values(op);
      ASSERT_EQ(stored.count, 3u);
      EXPECT_EQ(stored.values[0], 255);
      EXPECT_EQ(stored.values[1], 1);
      EXPECT_EQ(stored.values[2], 255);
      EXPECT_EQ(loom_test_enum_array_attrs_has_optional_values(op), present);
      const loom_enum_array_t stored_optional =
          loom_test_enum_array_attrs_optional_values(op);
      ASSERT_EQ(stored_optional.count, present ? count : 0);
      if (present && count) {
        EXPECT_EQ(stored_optional.values[0], 7);
        EXPECT_EQ(stored_optional.values[1], 42);
        EXPECT_EQ(stored_optional.values[2], 1);
      }
    }
  }
}

TEST_F(BuilderStorageTest, SignedEnumSetsCopyBothPolarities) {
  for (bool present : {false, true}) {
    for (iree_host_size_t count : {0u, 4u}) {
      uint64_t words[] = {UINT64_C(1) << 1, 0, 0, UINT64_C(1) << 63,
                          UINT64_C(1) << 7, 0, 0, 0};
      loom_op_t* op = nullptr;
      IREE_ASSERT_OK(loom_test_signed_enum_set_attrs_build(
          &builder_,
          present
              ? LOOM_TEST_SIGNED_ENUM_SET_ATTRS_BUILD_FLAG_HAS_OPTIONAL_FEATURES
              : 0,
          loom_make_signed_enum_set(words, 4),
          loom_make_signed_enum_set(words, count),
          loom_named_attr_slice_empty(), LOOM_LOCATION_UNKNOWN, &op));

      memset(words, 0, sizeof(words));
      const loom_signed_enum_set_t stored =
          loom_test_signed_enum_set_attrs_required_features(op);
      EXPECT_TRUE(loom_signed_enum_set_contains_positive(stored, 1));
      EXPECT_TRUE(loom_signed_enum_set_contains_positive(stored, 255));
      EXPECT_TRUE(loom_signed_enum_set_contains_negative(stored, 7));
      EXPECT_FALSE(loom_signed_enum_set_contains_negative(stored, 1));
      EXPECT_FALSE(loom_signed_enum_set_contains_positive(stored, 7));
      EXPECT_EQ(loom_test_signed_enum_set_attrs_has_optional_features(op),
                present);
      const loom_signed_enum_set_t stored_optional =
          loom_test_signed_enum_set_attrs_optional_features(op);
      ASSERT_EQ(stored_optional.word_count, present ? count : 0);
      if (present && count) {
        EXPECT_TRUE(loom_signed_enum_set_contains_positive(stored_optional, 1));
        EXPECT_TRUE(
            loom_signed_enum_set_contains_positive(stored_optional, 255));
        EXPECT_TRUE(loom_signed_enum_set_contains_negative(stored_optional, 7));
      }
    }
  }
}

TEST_F(BuilderStorageTest, SymbolArraysPreserveOrderWhileSetsCanonicalize) {
  loom_symbol_ref_t symbols[2];
  const iree_string_view_t names[] = {IREE_SV("b"), IREE_SV("a")};
  for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(names); ++i) {
    loom_string_id_t name = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(module_, names[i], &name));
    symbols[i] = {};
    IREE_ASSERT_OK(
        loom_module_add_symbol(module_, name, &symbols[i].symbol_id));
  }
  for (bool present : {false, true}) {
    for (iree_host_size_t count : {0u, 3u}) {
      loom_symbol_ref_t values[] = {symbols[0], symbols[1], symbols[0]};
      loom_op_t* array_op = nullptr;
      IREE_ASSERT_OK(loom_test_symbol_array_attrs_build(
          &builder_,
          present ? LOOM_TEST_SYMBOL_ARRAY_ATTRS_BUILD_FLAG_HAS_AVAILABLE : 0,
          loom_make_symbol_ref_array(values, IREE_ARRAYSIZE(values)),
          loom_make_symbol_ref_array(values, count), LOOM_LOCATION_UNKNOWN,
          &array_op));
      loom_op_t* set_op = nullptr;
      if (count) {
        const uint32_t op_count = loom_module_block(module_)->op_count;
        IREE_EXPECT_STATUS_IS(
            IREE_STATUS_INVALID_ARGUMENT,
            loom_test_symbol_set_attrs_build(
                &builder_, loom_make_symbol_ref_array(values, count),
                LOOM_LOCATION_UNKNOWN, &set_op));
        EXPECT_EQ(set_op, nullptr);
        EXPECT_EQ(loom_module_block(module_)->op_count, op_count);
      }
      IREE_ASSERT_OK(loom_test_symbol_set_attrs_build(
          &builder_, loom_make_symbol_ref_array(values, count ? 2 : 0),
          LOOM_LOCATION_UNKNOWN, &set_op));

      memset(values, 0, sizeof(values));
      const loom_symbol_ref_array_t stored =
          loom_test_symbol_array_attrs_dependencies(array_op);
      ASSERT_EQ(stored.count, 3u);
      EXPECT_EQ(stored.values[0].symbol_id, symbols[0].symbol_id);
      EXPECT_EQ(stored.values[1].symbol_id, symbols[1].symbol_id);
      EXPECT_EQ(stored.values[2].symbol_id, symbols[0].symbol_id);
      EXPECT_EQ(loom_test_symbol_array_attrs_has_available(array_op), present);
      const loom_symbol_ref_array_t available =
          loom_test_symbol_array_attrs_available(array_op);
      ASSERT_EQ(available.count, present ? count : 0);
      if (present && count) {
        EXPECT_EQ(available.values[0].symbol_id, symbols[0].symbol_id);
        EXPECT_EQ(available.values[1].symbol_id, symbols[1].symbol_id);
        EXPECT_EQ(available.values[2].symbol_id, symbols[0].symbol_id);
      }
      const loom_symbol_ref_array_t set =
          loom_test_symbol_set_attrs_symbols(set_op);
      ASSERT_EQ(set.count, count ? 2u : 0u);
      if (count) {
        EXPECT_EQ(set.values[0].symbol_id, symbols[1].symbol_id);
        EXPECT_EQ(set.values[1].symbol_id, symbols[0].symbol_id);
      }
    }
  }
}

}  // namespace
}  // namespace loom
