// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/bytecode/reader/attribute.h"

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
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("attribute_test"),
                                        &block_pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_bytecode_reader_decoder_initialize(
        loom_diagnostic_sink_t{AcceptDiagnostic, nullptr},
        IREE_SV("attribute_test.loombc"), &error_count_, &decoder_);
    module_view_.strings.values = strings_;
    module_view_.strings.count = IREE_ARRAYSIZE(strings_);
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
        /*.decoder=*/&decoder_,
        /*.context=*/&context_,
        /*.module_view=*/&module_view_,
    };
  }

  loom_bytecode_attribute_materializer_t MakeMaterializer() {
    return loom_bytecode_attribute_materializer_t{
        /*.decoder=*/&decoder_,
        /*.context=*/&context_,
        /*.module_view=*/&module_view_,
        /*.scratch_arena=*/&scratch_arena_,
        /*.output_module=*/module_,
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

TEST_F(BytecodeAttributeTest, U64MaxValidatesAndMaterializesExactly) {
  const uint8_t data[] = {
      0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01,
  };
  loom_bytecode_attribute_validator_t validator = MakeValidator();
  loom_bytecode_reader_cursor_t validation_cursor =
      MakeCursor(data, sizeof(data));
  IREE_ASSERT_OK(loom_bytecode_attribute_validate_named(
      &validator, &validation_cursor, /*descriptor=*/nullptr,
      LOOM_BYTECODE_ATTR_U64, /*available_type_count=*/0));
  EXPECT_EQ(validation_cursor.cursor.position, sizeof(data));

  loom_bytecode_attribute_materializer_t materializer = MakeMaterializer();
  loom_bytecode_reader_cursor_t materialization_cursor =
      MakeCursor(data, sizeof(data));
  loom_attribute_t attr = loom_attr_absent();
  IREE_ASSERT_OK(loom_bytecode_attribute_materialize_named(
      &materializer, &materialization_cursor, /*descriptor=*/nullptr,
      LOOM_BYTECODE_ATTR_U64, &attr, /*available_type_count=*/0));
  EXPECT_EQ(materialization_cursor.cursor.position, sizeof(data));
  EXPECT_EQ(attr.kind, LOOM_ATTR_U64);
  EXPECT_EQ(loom_attr_as_u64(attr), UINT64_MAX);
}

TEST_F(BytecodeAttributeTest, NamedPredicatesValidateAndMaterialize) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MUL,  0x02, LOOM_PRED_ARG_VALUE,
      0x01, LOOM_PRED_ARG_CONST, 0x20,
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
  EXPECT_EQ(attr.predicate_list[0].kind, LOOM_PREDICATE_MUL);
  EXPECT_EQ(attr.predicate_list[0].arg_count, 2u);
  EXPECT_EQ(attr.predicate_list[0].arg_tags[0], LOOM_PRED_ARG_VALUE);
  EXPECT_EQ(attr.predicate_list[0].args[0], 1);
  EXPECT_EQ(attr.predicate_list[0].arg_tags[1], LOOM_PRED_ARG_CONST);
  EXPECT_EQ(attr.predicate_list[0].args[1], 16);
  EXPECT_EQ(materialization_cursor.cursor.position, sizeof(data));
  EXPECT_EQ(error_count_, 0u);
}

TEST_F(BytecodeAttributeTest, SsaPredicatesResolveThroughConcreteValueMap) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MUL,  0x02, LOOM_PRED_ARG_VALUE,
      0x00, LOOM_PRED_ARG_CONST, 0x20,
  };
  loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_module_define_value(
      module_, loom_type_scalar(LOOM_SCALAR_TYPE_INDEX), &value_id));
  const loom_value_id_t values[] = {value_id};
  const loom_bytecode_attribute_ssa_materialization_scope_t scope = {
      /*.symbol_name=*/IREE_SV("function"),
      /*.values=*/values,
      /*.value_count=*/IREE_ARRAYSIZE(values),
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

TEST_F(BytecodeAttributeTest, SsaValidationRejectsOutOfRangeValue) {
  const uint8_t data[] = {
      0x01, LOOM_PREDICATE_MUL,  0x02, LOOM_PRED_ARG_VALUE,
      0x01, LOOM_PRED_ARG_CONST, 0x20,
  };
  const loom_bytecode_attribute_ssa_validation_scope_t scope = {
      /*.symbol_name=*/IREE_SV("function"),
      /*.value_count=*/1,
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
      0x01, LOOM_PREDICATE_MUL, 0x01, LOOM_PRED_ARG_VALUE, 0x00,
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
