// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/attribute.h"

#include <cstdio>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

static iree_status_t AcceptDiagnostic(void* user_data,
                                      const loom_diagnostic_t* diagnostic) {
  (void)user_data;
  (void)diagnostic;
  return iree_ok_status();
}

class BytecodeAttributeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    iree_arena_initialize(&block_pool_, &scratch_arena_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, iree_string_view_empty(),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("attribute_test.loombc"), &error_count_, &decoder_);
    module_view_.strings.values = strings_;
    module_view_.strings.count = IREE_ARRAYSIZE(strings_);
    for (iree_host_size_t i = 0; i < IREE_ARRAYSIZE(strings_); ++i) {
      loom_string_id_t name = LOOM_STRING_ID_INVALID;
      IREE_ASSERT_OK(loom_module_intern_string(module_, strings_[i], &name));
      ASSERT_EQ(name, i);
    }
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_deinitialize(&scratch_arena_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  loom_bytecode_reader_cursor_t MakeCursor(const uint8_t* data,
                                           iree_host_size_t length) {
    loom_bytecode_reader_cursor_t cursor;
    loom_bytecode_reader_cursor_initialize(data, length, 4096, IREE_SV("TEST"),
                                           &cursor);
    return cursor;
  }

  loom_bytecode_attribute_validator_t MakeValidator() {
    return loom_bytecode_attribute_validator_t{
        .decoder = &decoder_,
        .context = &context_,
        .module_view = &module_view_,
    };
  }

  loom_bytecode_attribute_materializer_t MakeMaterializer() {
    return loom_bytecode_attribute_materializer_t{
        .decoder = &decoder_,
        .context = &context_,
        .module_view = &module_view_,
        .scratch_arena = &scratch_arena_,
        .output_module = module_,
    };
  }

  // Borrowed table entries used by named predicate payloads.
  iree_string_view_t strings_[2] = {IREE_SV(""), IREE_SV("dimension")};
  // Minimal validated module tables exposed to the attribute reader.
  loom_bytecode_reader_module_view_t module_view_ = {};
  // Bounded wire decoder sharing this fixture's diagnostic count.
  loom_bytecode_reader_decoder_t decoder_ = {};
  // Number of accepted malformed-input diagnostics.
  uint32_t error_count_ = 0;
  // Block source shared by scratch and output-module arenas.
  iree_arena_block_pool_t block_pool_;
  // Resettable storage used for aggregate attribute construction.
  iree_arena_allocator_t scratch_arena_;
  // Finalized empty registry sufficient for scalar predicate attributes.
  loom_context_t context_;
  // Output module owning materialized predicate arrays.
  loom_module_t* module_ = nullptr;
};

TEST_F(BytecodeAttributeTest, NamedPredicatesValidateAndMaterialize) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MULTIPLE_OF, 0x02, LOOM_PRED_ARG_VALUE,
      0x01, LOOM_PRED_ARG_CONST,        0x20,
  };
  loom_bytecode_attribute_validator_t validator = MakeValidator();
  loom_bytecode_reader_cursor_t validation_cursor =
      MakeCursor(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_attribute_validate_named(
      &validator, &validation_cursor, /*descriptor=*/nullptr,
      LOOM_BYTECODE_ATTR_PREDICATE_LIST, /*available_type_count=*/0));
  EXPECT_EQ(validation_cursor.cursor.position, sizeof(data));

  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  loom_bytecode_reader_cursor_t materialization_cursor =
      MakeCursor(data, sizeof(data));
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_named(
      &materializer, &materialization_cursor, /*descriptor=*/nullptr,
      LOOM_BYTECODE_ATTR_PREDICATE_LIST, &attr,
      /*available_type_count=*/0));

  ASSERT_EQ(attr.kind, LOOM_ATTR_PREDICATE_LIST);
  ASSERT_EQ(attr.count, 1u);
  EXPECT_EQ(attr.predicate_list[0].kind, LOOM_PREDICATE_MULTIPLE_OF);
  EXPECT_EQ(attr.predicate_list[0].arg_count, 2u);
  EXPECT_EQ(attr.predicate_list[0].arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ(attr.predicate_list[0].args[0], 1);
  EXPECT_EQ(attr.predicate_list[0].arg_tags[1], LOOM_PRED_ARG_CONST);
  EXPECT_EQ(attr.predicate_list[0].args[1], 16);
  EXPECT_EQ(materialization_cursor.cursor.position, sizeof(data));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, GlobalTypeParametersAcceptConstantPredicates) {
  const uint8_t data[] = {
      1, LOOM_PREDICATE_EQ, 2, LOOM_PRED_ARG_CONST, 8, LOOM_PRED_ARG_CONST, 8,
  };
  auto validator = MakeValidator();
  auto cursor = MakeCursor(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_attribute_validate_type_parameter(
      &validator, &cursor, nullptr, LOOM_BYTECODE_ATTR_PREDICATE_LIST, 0));
  EXPECT_EQ(cursor.cursor.position, sizeof(data));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest,
       GlobalTypeParametersRequireScopeForValuePredicates) {
  const uint8_t data[] = {
      1, LOOM_PREDICATE_EQ, 2, LOOM_PRED_ARG_VALUE, 0, LOOM_PRED_ARG_CONST, 8,
  };
  auto validator = MakeValidator();
  auto cursor = MakeCursor(data, sizeof(data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_attribute_validate_type_parameter(
          &validator, &cursor, nullptr, LOOM_BYTECODE_ATTR_PREDICATE_LIST, 0));
  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeAttributeTest, SsaPredicatesResolveThroughConcreteValueMap) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MULTIPLE_OF, 0x02, LOOM_PRED_ARG_VALUE,
      0x00, LOOM_PRED_ARG_CONST,        0x20,
  };
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
  const loom_value_id_t values[] = {value_id};
  const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
      .symbol_name = IREE_SV("function"),
      .values = values,
      .value_count = IREE_ARRAYSIZE(values),
  };
  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  loom_bytecode_reader_cursor_t cursor = MakeCursor(data, sizeof(data));
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_ssa(
      &materializer, &cursor, /*descriptor=*/nullptr,
      LOOM_BYTECODE_ATTR_PREDICATE_LIST, &attr,
      /*available_type_count=*/0, &scope));

  ASSERT_EQ(attr.count, 1u);
  EXPECT_EQ(attr.predicate_list[0].args[0], value_id);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, CompleteBindingsSurviveAttributeScratchReset) {
  loom_value_id_t width = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &width));
  iree_arena_allocator_t scope_arena;
  iree_arena_initialize(&block_pool_, &scope_arena);
  loom_bytecode_type_bindings_t bindings = {.arena = &scope_arena};
  const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
      .symbol_name = IREE_SV("function"),
      .values = &width,
      .value_count = 1,
      .bindings = &bindings,
  };
  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  // A complete group record, then a later attribute reusing its scoped node.
  const uint8_t first[] = {1, 6, 1, LOOM_BYTECODE_TYPE_GROUP, 1, 1, 1, 1};
  const uint8_t second[] = {1, 2, 0, 1};
  loom_bytecode_reader_cursor_t first_cursor = MakeCursor(first, sizeof(first));
  loom_attribute_t first_attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_ssa(
      &materializer, &first_cursor, nullptr, LOOM_BYTECODE_ATTR_TYPE,
      &first_attr, 0, &scope));
  EXPECT_EQ(scratch_arena_.used_allocation_size, 0u);
  ASSERT_EQ(bindings.count, 1u);
  EXPECT_EQ(bindings.entries[0].type, first_attr.type_id);

  iree_arena_reset(&scratch_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  loom_bytecode_reader_cursor_t second_cursor =
      MakeCursor(second, sizeof(second));
  loom_attribute_t second_attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_ssa(
      &materializer, &second_cursor, nullptr, LOOM_BYTECODE_ATTR_TYPE,
      &second_attr, 0, &scope));
  EXPECT_EQ(first_attr.type_id, second_attr.type_id);
  EXPECT_EQ(bindings.count, 1u);
  iree_arena_deinitialize(&scope_arena);
  iree_arena_block_pool_trim(&block_pool_);
  const loom_type_t type =
      loom_type_table_get(&module_->types, second_attr.type_id);
  EXPECT_EQ(loom_type_kind(type), LOOM_TYPE_GROUP);
  EXPECT_EQ(loom_dim_value_id(loom_type_dim(type, 0)), width);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, ScopedRegisterRetainsCarrierAndBoundValueType) {
  loom_value_id_t width = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &width));
  iree_arena_allocator_t scope_arena;
  iree_arena_initialize(&block_pool_, &scope_arena);
  loom_bytecode_type_bindings_t bindings = {.arena = &scope_arena};
  const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
      .symbol_name = IREE_SV("function"),
      .values = &width,
      .value_count = 1,
      .bindings = &bindings,
  };
  // The vector is completed before its register parent; the parent references
  // that final child rather than an unbound native type template.
  const uint8_t data[] = {
      1,
      15,
      2,
      LOOM_BYTECODE_TYPE_VECTOR,
      LOOM_SCALAR_TYPE_F32,
      1,
      0,
      0,
      1,
      1,
      LOOM_BYTECODE_TYPE_REGISTER,
      1,
      0x80,
      0x80,
      0x04,
      1,
      2,
  };
  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  loom_bytecode_reader_cursor_t cursor = MakeCursor(data, sizeof(data));
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_ssa(
      &materializer, &cursor, nullptr, LOOM_BYTECODE_ATTR_TYPE, &attr, 0,
      &scope));
  EXPECT_EQ(cursor.cursor.position, sizeof(data));
  EXPECT_EQ(bindings.count, 2u);

  iree_arena_deinitialize(&scope_arena);
  iree_arena_reset(&scratch_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  const loom_type_t type = loom_type_table_get(&module_->types, attr.type_id);
  ASSERT_EQ(loom_type_kind(type), LOOM_TYPE_REGISTER);
  EXPECT_EQ(loom_type_register_payload0(type), 1u);
  EXPECT_EQ(loom_type_register_payload1(type), UINT64_C(1) << 16);
  const loom_type_t* value_type = loom_type_register_value_type(type);
  ASSERT_NE(value_type, nullptr);
  EXPECT_EQ(loom_type_kind(*value_type), LOOM_TYPE_VECTOR);
  EXPECT_EQ(loom_dim_value_id(loom_type_dim(*value_type, 0)), width);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, ScopedViewAlignmentValidatesBeforeNarrowing) {
  loom_value_id_t width = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &width));
  for (uint16_t alignment : {0, 1, 2, 4, 8, 3, 16, 128, 255, 256}) {
    SCOPED_TRACE(alignment);
    iree_arena_allocator_t scope_arena;
    iree_arena_initialize(&block_pool_, &scope_arena);
    loom_bytecode_type_bindings_t bindings = {.arena = &scope_arena};
    const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
        .symbol_name = IREE_SV("function"),
        .values = &width,
        .value_count = 1,
        .bindings = &bindings,
    };
    // One scoped view node with a dynamic dimension and no encoding. All
    // fields after the kind use varints, unlike the global type table.
    std::vector<uint8_t> data = {
        1, 0, 1, LOOM_BYTECODE_TYPE_VIEW, LOOM_SCALAR_TYPE_I64, 1, 0, 0};
    if (alignment < 128) {
      data.push_back((uint8_t)alignment);
    } else {
      data.push_back((uint8_t)(alignment | 0x80));
      data.push_back((uint8_t)(alignment >> 7));
    }
    data.insert(data.end(), {1, 1, 1});  // Dynamic dimension, binder, root.
    data[1] = (uint8_t)(data.size() - 2);
    auto materializer = MakeMaterializer();
    auto cursor = MakeCursor(data.data(), data.size());
    loom_attribute_t attr = loom_attr_absent();
    error_count_ = 0;
    iree_status_t status = loom_bytecode_attribute_materialize_ssa(
        &materializer, &cursor, nullptr, LOOM_BYTECODE_ATTR_TYPE, &attr, 0,
        &scope);
    if (alignment == 0 ||
        loom_type_view_alignment_is_valid(LOOM_SCALAR_TYPE_I64, alignment)) {
      IREE_EXPECT_OK(status);
      EXPECT_EQ(error_count_, 0u);
      const loom_type_t type =
          loom_type_table_get(&module_->types, attr.type_id);
      EXPECT_EQ(loom_type_view_alignment(type), alignment ? alignment : 8);
      EXPECT_EQ(loom_dim_value_id(loom_type_dim(type, 0)), width);
    } else {
      IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED, status);
      EXPECT_EQ(error_count_, 1u);
    }
    iree_arena_deinitialize(&scope_arena);
  }
}

TEST_F(BytecodeAttributeTest, ScopedDialectNameUsesFullStringOrdinal) {
  // Full reads have an ordered, validated string table. Populate the actual
  // output table so the first 17-bit name ID exercises that production
  // contract.
  std::vector<iree_string_view_t> strings(strings_,
                                          strings_ + IREE_ARRAYSIZE(strings_));
  for (uint32_t i = IREE_ARRAYSIZE(strings_); i <= UINT16_MAX; ++i) {
    char buffer[32];
    const int length = std::snprintf(buffer, sizeof(buffer), "name_%u", i);
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(loom_module_intern_string(
        module_, iree_make_string_view(buffer, length), &name_id));
    ASSERT_EQ(name_id, i);
    strings.push_back(loom_string_table_get(&module_->strings, name_id));
  }
  loom_string_id_t family_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_module_intern_string(module_, IREE_SV("test.wide_name"),
                                           &family_name));
  ASSERT_EQ(family_name, UINT32_C(1) << 16);
  strings.push_back(loom_string_table_get(&module_->strings, family_name));
  module_view_.strings.values = strings.data();
  module_view_.strings.count = strings.size();

  loom_value_id_t width = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &width));
  iree_arena_allocator_t scope_arena;
  iree_arena_initialize(&block_pool_, &scope_arena);
  loom_bytecode_type_bindings_t bindings = {.arena = &scope_arena};
  const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
      .symbol_name = IREE_SV("function"),
      .values = &width,
      .value_count = 1,
      .bindings = &bindings,
  };
  const uint8_t data[] = {
      1,    12,   2,    LOOM_BYTECODE_TYPE_GROUP,
      1,    1,    1,    LOOM_BYTECODE_TYPE_DIALECT,
      0x80, 0x80, 0x04, 1,
      1,    2,
  };
  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  loom_bytecode_reader_cursor_t cursor = MakeCursor(data, sizeof(data));
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_ssa(
      &materializer, &cursor, nullptr, LOOM_BYTECODE_ATTR_TYPE, &attr, 0,
      &scope));
  iree_arena_deinitialize(&scope_arena);
  iree_arena_reset(&scratch_arena_);
  iree_arena_block_pool_trim(&block_pool_);
  const loom_type_t type = loom_type_table_get(&module_->types, attr.type_id);
  ASSERT_EQ(loom_type_kind(type), LOOM_TYPE_DIALECT);
  EXPECT_EQ(loom_type_dialect_name_id(type), family_name);
  ASSERT_EQ(loom_type_dialect_param_count(type), 1u);
  EXPECT_EQ(
      loom_dim_value_id(loom_type_dim(loom_type_dialect_params(type)[0], 0)),
      width);
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, SsaValidationRejectsOutOfRangeValue) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MULTIPLE_OF, 0x02, LOOM_PRED_ARG_VALUE,
      0x01, LOOM_PRED_ARG_CONST,        0x20,
  };
  const loom_bytecode_attribute_ssa_validation_scope_t scope = {
      .symbol_name = IREE_SV("function"),
      .value_count = 1,
  };
  loom_bytecode_attribute_validator_t validator = MakeValidator();
  loom_bytecode_reader_cursor_t cursor = MakeCursor(data, sizeof(data));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_DEFERRED,
                        loom_bytecode_attribute_validate_ssa(
                            &validator, &cursor, /*descriptor=*/nullptr,
                            LOOM_BYTECODE_ATTR_PREDICATE_LIST,
                            /*available_type_count=*/0, &scope));

  EXPECT_EQ(error_count_, 1u);
}

TEST_F(BytecodeAttributeTest, PredicateArityMustMatchKind) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MULTIPLE_OF, 0x01, LOOM_PRED_ARG_VALUE, 0x00,
  };
  loom_bytecode_attribute_validator_t validator = MakeValidator();
  loom_bytecode_reader_cursor_t cursor = MakeCursor(data, sizeof(data));
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_DEFERRED,
      loom_bytecode_attribute_validate_named(
          &validator, &cursor, /*descriptor=*/nullptr,
          LOOM_BYTECODE_ATTR_PREDICATE_LIST, /*available_type_count=*/0));

  EXPECT_EQ(error_count_, 1u);
}

}  // namespace
}  // namespace loom
