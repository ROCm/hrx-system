// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/format/text/printer/format_tables.h"

#include <cstdio>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/format/text/printer/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/test/ops.h"
#include "loom/ops/test/registry.h"

namespace loom {
namespace {

// The public printer can receive unverified IR during diagnostic rendering.
// These API cases exercise ordinal permutations and malformed dictionaries that
// the text parser cannot construct.
class OperandDictionaryPrintTest : public ::testing::Test {
 protected:
  static iree_status_t Allocate(void* self, iree_allocator_command_t command,
                                const void* parameters, void** pointer) {
    auto* test = static_cast<OperandDictionaryPrintTest*>(self);
    if (test->fail_allocations_ && command != IREE_ALLOCATOR_COMMAND_FREE) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "injected allocation failure");
    }
    const bool releases = command == IREE_ALLOCATOR_COMMAND_FREE && *pointer;
    const bool acquires =
        command != IREE_ALLOCATOR_COMMAND_FREE &&
        (command != IREE_ALLOCATOR_COMMAND_REALLOC || !*pointer);
    const iree_allocator_t allocator = iree_allocator_system();
    iree_status_t status =
        allocator.ctl(allocator.self, command, parameters, pointer);
    if (iree_status_is_ok(status)) {
      if (releases) {
        --test->live_allocations_;
      }
      if (acquires && *pointer) {
        ++test->live_allocations_;
      }
    }
    return status;
  }

  void SetUp() override {
    iree_arena_block_pool_initialize(32768, {this, Allocate}, &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_test_dialect_register(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("dictionary"),
                                        &pool_, nullptr,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);
    IREE_ASSERT_OK(loom_module_define_value(
        module_, loom_type_scalar(LOOM_SCALAR_TYPE_F32), &input_));
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
    EXPECT_EQ(live_allocations_, 0u);
  }

  loom_op_t* Dictionary(uint16_t count) {
    std::vector<loom_named_value_t> parameters(count);
    for (uint16_t i = 0; i < count; ++i) {
      char name[32];
      snprintf(name, sizeof(name), "parameter_%05u", static_cast<unsigned>(i));
      IREE_CHECK_OK(loom_builder_intern_string(
          &builder_, iree_make_cstring_view(name), &parameters[i].name_id));
      IREE_CHECK_OK(loom_module_define_value(
          module_,
          loom_type_scalar(i % 2 ? LOOM_SCALAR_TYPE_I32 : LOOM_SCALAR_TYPE_F32),
          &parameters[i].value_id));
    }
    loom_op_t* op = nullptr;
    IREE_CHECK_OK(loom_test_operand_dict_build(
        &builder_, input_, parameters.data(), count,
        loom_type_scalar(LOOM_SCALAR_TYPE_F32), LOOM_LOCATION_UNKNOWN, &op));
    return op;
  }

  std::vector<loom_named_attr_t> Names(loom_op_t* op) {
    auto names = loom_test_operand_dict_param_names(op);
    return {names.entries, names.entries + names.count};
  }

  iree_status_t PrintStatus(loom_op_t* op) {
    iree_arena_block_pool_trim(&pool_);
    const auto allocations_before = live_allocations_;
    const auto used_before = module_->arena.used_allocation_size;
    loom_output_stream_t stream;
    loom_output_stream_null(&stream);
    iree_status_t status = loom_text_print_operation(module_, op, &stream,
                                                     LOOM_TEXT_PRINT_DEFAULT);
    iree_arena_block_pool_trim(&pool_);
    EXPECT_EQ(live_allocations_, allocations_before);
    EXPECT_EQ(module_->arena.used_allocation_size, used_before);
    return status;
  }

  // Block allocation failures are enabled only after IR construction.
  bool fail_allocations_ = false;
  // Live system allocations backing module and printer arenas.
  size_t live_allocations_ = 0;
  // Shared blocks whose returned scratch is observable by trimming the pool.
  iree_arena_block_pool_t pool_ = {};
  // Production format metadata for the test dialect.
  loom_context_t context_ = {};
  // Module owning the printed operands and keys.
  loom_module_t* module_ = nullptr;
  // Builder establishing normal field storage and operand segmentation.
  loom_builder_t builder_ = {};
  // Fixed leading operand preceding the dictionary's ordinal domain.
  loom_value_id_t input_ = LOOM_VALUE_ID_INVALID;
};

TEST_F(OperandDictionaryPrintTest, PermutationsPreserveTextAndFieldRanges) {
  for (uint16_t count : {1, 63, 64, 65, 128, 129}) {
    SCOPED_TRACE(count);
    loom_op_t* op = Dictionary(count);
    auto names = Names(op);
    auto parameters = loom_test_operand_dict_params(op);
    std::string expected = "%" +
                           std::to_string(loom_test_operand_dict_result(op)) +
                           " = test.operand_dict %0 {";
    for (uint16_t i = 0; i < count; ++i) {
      const uint16_t ordinal = count - i - 1;
      names[i].value = loom_attr_i64(ordinal);
      if (i) {
        expected += ", ";
      }
      char name[32];
      snprintf(name, sizeof(name), "parameter_%05u", static_cast<unsigned>(i));
      expected += std::string(name) + " = %" +
                  std::to_string(parameters.values[ordinal]) +
                  (ordinal % 2 ? " : i32" : " : f32");
    }
    expected += "} : f32\n";
    IREE_ASSERT_OK(loom_test_operand_dict_set_param_names(
        module_, op, loom_make_canonical_attr_dict(names.data(), count)));
    struct Field {
      // Operand or attribute described by this output range.
      loom_print_field_ref_t reference;
      // First output byte belonging to the field.
      iree_host_size_t start;
      // First byte after the field.
      iree_host_size_t end;
    };
    std::vector<Field> fields;
    loom_print_field_callback_t callback = {
        [](void* user_data, loom_print_field_ref_t field,
           iree_host_size_t start, iree_host_size_t end) {
          static_cast<std::vector<Field>*>(user_data)->push_back(
              {field, start, end});
        },
        &fields,
    };
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    IREE_ASSERT_OK(loom_text_print_operation_with_field_callback(
        module_, op, &output, LOOM_TEXT_PRINT_DEFAULT, callback));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    EXPECT_EQ(text, expected);
    size_t operand_field_count = 0;
    for (const auto& field : fields) {
      if (field.reference.kind != LOOM_PRINT_FIELD_OPERAND) {
        continue;
      }
      ASSERT_LT(field.reference.index, op->operand_count);
      EXPECT_EQ(text.substr(field.start, field.end - field.start),
                "%" + std::to_string(
                          loom_op_const_operands(op)[field.reference.index]));
      ++operand_field_count;
    }
    EXPECT_EQ(operand_field_count, count + 1u);
    iree_string_builder_deinitialize(&output);
  }
}

TEST_F(OperandDictionaryPrintTest, RejectsInvalidOrdinalsAcrossWordBoundaries) {
  for (uint16_t count : {2, 64, 65, 128, 129}) {
    SCOPED_TRACE(count);
    loom_op_t* op = Dictionary(count);
    auto names = Names(op);
    IREE_ASSERT_OK(loom_test_operand_dict_set_param_names(
        module_, op, loom_make_canonical_attr_dict(names.data(), count)));
    for (int64_t ordinal :
         {int64_t(0), int64_t(count - 2), int64_t(-1), int64_t(count)}) {
      SCOPED_TRACE(ordinal);
      names.back().value = loom_attr_i64(ordinal);
      IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, PrintStatus(op));
    }
    names.back().value = loom_attr_bool(true);
    IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, PrintStatus(op));
    names.back().value = loom_attr_i64(count - 1);
    IREE_ASSERT_OK(PrintStatus(op));
  }
}

TEST_F(OperandDictionaryPrintTest, SmallDictionariesNeedNoAllocation) {
  for (uint16_t count : {0, 1, 64}) {
    loom_op_t* op = Dictionary(count);
    fail_allocations_ = true;
    IREE_EXPECT_OK(PrintStatus(op));
    fail_allocations_ = false;
  }
}

TEST_F(OperandDictionaryPrintTest, RejectsInvalidKeysAndReleasesScratch) {
  loom_op_t* op = Dictionary(65);
  auto names = Names(op);
  IREE_ASSERT_OK(loom_test_operand_dict_set_param_names(
      module_, op, loom_make_canonical_attr_dict(names.data(), names.size())));
  const loom_string_id_t last_name = names.back().name_id;
  names.back().name_id = names.front().name_id;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, PrintStatus(op));
  names.back().name_id = LOOM_STRING_ID_INVALID;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, PrintStatus(op));
  names.back().name_id = last_name;
  IREE_ASSERT_OK(PrintStatus(op));
}

TEST_F(OperandDictionaryPrintTest, AllocationFailurePropagatesAndCanBeRetried) {
  loom_op_t* op = Dictionary(65);
  fail_allocations_ = true;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED, PrintStatus(op));
  fail_allocations_ = false;
  IREE_ASSERT_OK(PrintStatus(op));
}

TEST_F(OperandDictionaryPrintTest, OutputFailureReleasesScratch) {
  loom_op_t* op = Dictionary(129);
  iree_arena_block_pool_trim(&pool_);
  const auto allocations_before = live_allocations_;
  const auto used_before = module_->arena.used_allocation_size;
  loom_output_stream_t stream = {
      [](void*, iree_string_view_t text) -> iree_status_t {
        if (iree_string_view_equal(text, IREE_SV("parameter_00064"))) {
          return iree_make_status(IREE_STATUS_ABORTED,
                                  "injected output failure");
        }
        return iree_ok_status();
      },
      nullptr,
      0,
  };
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_ABORTED,
      loom_text_print_operation(module_, op, &stream, LOOM_TEXT_PRINT_DEFAULT));
  iree_arena_block_pool_trim(&pool_);
  EXPECT_EQ(live_allocations_, allocations_before);
  EXPECT_EQ(module_->arena.used_allocation_size, used_before);
  IREE_ASSERT_OK(PrintStatus(op));
}

TEST_F(OperandDictionaryPrintTest, MaximumDictionaryReleasesScratchEachTime) {
  loom_op_t* op = Dictionary(UINT16_MAX - 1);
  for (int i = 0; i < 3; ++i) {
    IREE_ASSERT_OK(PrintStatus(op));
  }
}

}  // namespace
}  // namespace loom
