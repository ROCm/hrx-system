// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests for the structured IR generator.
//
// These tests verify the generator's own contract:
//   - Determinism: same seed always produces the same module.
//   - Validity: every generated module passes the verifier.
//   - Edge cases: zero ops, max nesting, empty hooks, etc.
//   - Presets: all built-in presets produce valid modules.
//
// Testing downstream consumers (printer, parser, passes, bytecode)
// against generated IR is the job of per-component fuzz targets,
// not this test.

#include "loom/testing/gen.h"

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/format/text/printer.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/op_defs.h"
#include "loom/ops/test/ops.h"
#include "loom/util/stream.h"
#include "loom/verify/verify.h"

namespace loom {
namespace {

class GenTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);

    // Core generator presets use only the synthetic test dialect.
    iree_host_size_t test_vtable_count = 0;
    const loom_op_vtable_t* const* test_vtables =
        loom_test_dialect_vtables(&test_vtable_count);
    IREE_ASSERT_OK(loom_context_register_dialect(&context_, LOOM_DIALECT_TEST,
                                                 test_vtables,
                                                 (uint16_t)test_vtable_count));

    IREE_ASSERT_OK(loom_context_finalize(&context_));
  }

  void TearDown() override {
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Generates a module and verifies it. Returns the module for
  // additional inspection if needed.
  iree_status_t generate_and_verify(uint64_t seed,
                                    const loom_test_gen_module_config_t& config,
                                    loom_module_t** out_module) {
    loom_test_gen_t gen;
    loom_test_gen_initialize_seeded(seed, &gen);
    IREE_RETURN_IF_ERROR(loom_test_gen_module(&gen, &config, &context_,
                                              &block_pool_, out_module));
    loom_verify_options_t options = {};
    options.sink = {loom_diagnostic_stderr_sink, NULL};
    loom_verify_result_t result = {};
    iree_status_t status = loom_verify_module(*out_module, &options, &result);
    if (!iree_status_is_ok(status)) {
      return status;
    }
    if (result.error_count > 0) {
      return iree_make_status(
          IREE_STATUS_INTERNAL,
          "generated module failed verification with %d errors",
          result.error_count);
    }
    return iree_ok_status();
  }

  // Prints a module to a string.
  std::string print_module(loom_module_t* module) {
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    iree_status_t status =
        loom_text_print_module_to_builder(module, &builder, 0);
    std::string result;
    if (iree_status_is_ok(status)) {
      result.assign(iree_string_builder_buffer(&builder),
                    iree_string_builder_size(&builder));
    }
    iree_string_builder_deinitialize(&builder);
    IREE_EXPECT_OK(status);
    return result;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
};

//===----------------------------------------------------------------------===//
// Determinism
//===----------------------------------------------------------------------===//

TEST_F(GenTest, SameSeedProducesSameModule) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);

  loom_module_t* module_a = nullptr;
  IREE_ASSERT_OK(generate_and_verify(42, config, &module_a));
  std::string text_a = print_module(module_a);

  loom_module_t* module_b = nullptr;
  IREE_ASSERT_OK(generate_and_verify(42, config, &module_b));
  std::string text_b = print_module(module_b);

  EXPECT_EQ(text_a, text_b)
      << "Same seed must produce identical IR across runs";

  loom_module_free(module_a);
  loom_module_free(module_b);
}

TEST_F(GenTest, DifferentSeedsProduceDifferentModules) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);

  loom_module_t* module_a = nullptr;
  IREE_ASSERT_OK(generate_and_verify(1, config, &module_a));
  std::string text_a = print_module(module_a);

  loom_module_t* module_b = nullptr;
  IREE_ASSERT_OK(generate_and_verify(2, config, &module_b));
  std::string text_b = print_module(module_b);

  EXPECT_NE(text_a, text_b) << "Different seeds should produce different IR";

  loom_module_free(module_a);
  loom_module_free(module_b);
}

//===----------------------------------------------------------------------===//
// Verification across presets
//===----------------------------------------------------------------------===//

TEST_F(GenTest, RepresentativePresetVerifies) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(2);
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(100, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, CSEStressPresetVerifies) {
  loom_test_gen_body_config_t body = loom_test_gen_body_config_cse_stress(2);
  loom_test_gen_module_config_t config = {0};
  config.function_count = 2;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config = body;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(200, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, DCEStressPresetVerifies) {
  loom_test_gen_body_config_t body = loom_test_gen_body_config_dce_stress(2);
  loom_test_gen_module_config_t config = {0};
  config.function_count = 2;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config = body;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(300, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, NestingStressPresetVerifies) {
  loom_test_gen_body_config_t body =
      loom_test_gen_body_config_nesting_stress(1);
  loom_test_gen_module_config_t config = {0};
  config.function_count = 2;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config = body;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(400, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, FormatStressPresetVerifies) {
  loom_test_gen_body_config_t body = loom_test_gen_body_config_format_stress(1);
  loom_test_gen_module_config_t config = {0};
  config.function_count = 2;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config = body;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(500, config, &module));
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST_F(GenTest, ZeroOpCount) {
  loom_test_gen_body_config_t body =
      loom_test_gen_body_config_representative(1);
  body.op_count = 0;
  loom_test_gen_module_config_t config = {0};
  config.function_count = 1;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config = body;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(600, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, SingleFunction) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);
  config.function_count = 1;
  config.declaration_count = 0;
  config.calls_per_function = 0;
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(700, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, DeclarationsOnly) {
  loom_test_gen_module_config_t config = {0};
  config.function_count = 0;
  config.declaration_count = 5;
  config.calls_per_function = 0;
  config.body_config = loom_test_gen_body_config_representative(1);
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(800, config, &module));
  loom_module_free(module);
}

TEST_F(GenTest, ManyFunctionsWithCalls) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(3);
  config.body_config.block_arg_count = 0;

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(generate_and_verify(900, config, &module));
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Seed sweep: run many seeds through verification
//===----------------------------------------------------------------------===//

TEST_F(GenTest, SeedSweepRepresentative) {
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);
  for (uint64_t seed = 0; seed < 50; ++seed) {
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(generate_and_verify(seed, config, &module))
        << "Failed at seed=" << seed;
    loom_module_free(module);
  }
}

//===----------------------------------------------------------------------===//
// Fuzz mode: arbitrary bytes must produce valid IR
//===----------------------------------------------------------------------===//

TEST_F(GenTest, FuzzModeEmptyInput) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_fuzz(nullptr, 0, &gen);
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);
  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_test_gen_module(&gen, &config, &context_, &block_pool_, &module));

  loom_verify_options_t options = {};
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
  EXPECT_EQ(result.error_count, 0);
  loom_module_free(module);
}

TEST_F(GenTest, FuzzModeArbitraryBytes) {
  // A few representative byte sequences.
  static const uint8_t inputs[][16] = {
      {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
      {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF},
      {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF},
      {0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE, 0xBA, 0xBE},
      {0x42, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04,
       0x05, 0x06, 0x07, 0x08},
  };
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_representative(1);

  for (size_t i = 0; i < sizeof(inputs) / sizeof(inputs[0]); ++i) {
    loom_test_gen_t gen;
    loom_test_gen_initialize_fuzz(inputs[i], sizeof(inputs[i]), &gen);
    loom_module_t* module = nullptr;
    IREE_ASSERT_OK(
        loom_test_gen_module(&gen, &config, &context_, &block_pool_, &module))
        << "Fuzz input " << i << " failed to generate";

    loom_verify_options_t options = {};
    loom_verify_result_t result = {};
    IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
    EXPECT_EQ(result.error_count, 0) << "Fuzz input " << i << " failed verify";
    loom_module_free(module);
  }
}

TEST_F(GenTest, FuzzPresetRegressionBytes) {
  static const uint8_t input[] = {0xFF, 0x2B, 0x0A};
  loom_test_gen_t gen;
  loom_test_gen_initialize_fuzz(input + 2, IREE_ARRAYSIZE(input) - 2, &gen);
  loom_test_gen_module_config_t config =
      loom_test_gen_module_config_fuzz_preset(input[0], (input[1] % 5) + 1);

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(
      loom_test_gen_module(&gen, &config, &context_, &block_pool_, &module));

  loom_verify_options_t options = {};
  options.sink = {loom_diagnostic_stderr_sink, NULL};
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &options, &result));
  EXPECT_EQ(result.error_count, 0);
  loom_module_free(module);
}

//===----------------------------------------------------------------------===//
// Value set unit tests
//===----------------------------------------------------------------------===//

TEST_F(GenTest, ValueSetPickAnyEmptyReturnsInvalid) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  EXPECT_EQ(loom_test_gen_values_pick_any(&gen, &values),
            LOOM_VALUE_ID_INVALID);
}

TEST_F(GenTest, ValueSetPickTypedEmptyReturnsInvalid) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  EXPECT_EQ(
      loom_test_gen_values_pick_typed(&gen, &values, LOOM_SCALAR_TYPE_I32),
      LOOM_VALUE_ID_INVALID);
}

TEST_F(GenTest, ValueSetReportsOmittedValuesAfterCapacity) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);

  loom_type_t type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  for (iree_host_size_t i = 0; i < LOOM_TEST_GEN_VALUES_MAX_CAPACITY + 3; ++i) {
    loom_test_gen_values_add(&values, (loom_value_id_t)i, type);
  }

  EXPECT_EQ(values.count, LOOM_TEST_GEN_VALUES_MAX_CAPACITY);
  EXPECT_EQ(values.total_count, LOOM_TEST_GEN_VALUES_MAX_CAPACITY + 3);
}

TEST_F(GenTest, ValueSetPickTypedFindsCorrectType) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);

  // Add an i32 and an f32.
  loom_test_gen_values_add(&values, 10, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_test_gen_values_add(&values, 20, loom_type_scalar(LOOM_SCALAR_TYPE_F32));

  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);

  // Picking i32 should always return 10.
  loom_value_id_t picked =
      loom_test_gen_values_pick_typed(&gen, &values, LOOM_SCALAR_TYPE_I32);
  EXPECT_EQ(picked, 10u);

  // Picking f32 should always return 20.
  picked = loom_test_gen_values_pick_typed(&gen, &values, LOOM_SCALAR_TYPE_F32);
  EXPECT_EQ(picked, 20u);

  // Picking i64 should return invalid (none exist).
  picked = loom_test_gen_values_pick_typed(&gen, &values, LOOM_SCALAR_TYPE_I64);
  EXPECT_EQ(picked, LOOM_VALUE_ID_INVALID);
}

TEST_F(GenTest, ValueSetPickIntegerIgnoresFloats) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);

  // Only float values.
  loom_test_gen_values_add(&values, 10, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  loom_test_gen_values_add(&values, 20, loom_type_scalar(LOOM_SCALAR_TYPE_F64));

  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  EXPECT_EQ(loom_test_gen_values_pick_integer(&gen, &values),
            LOOM_VALUE_ID_INVALID);
}

TEST_F(GenTest, ValueSetPickFloatIgnoresIntegers) {
  loom_test_gen_values_t values;
  loom_test_gen_values_initialize(&values);

  // Only integer values.
  loom_test_gen_values_add(&values, 10, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  loom_test_gen_values_add(&values, 20, loom_type_scalar(LOOM_SCALAR_TYPE_I64));

  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  EXPECT_EQ(loom_test_gen_values_pick_float(&gen, &values),
            LOOM_VALUE_ID_INVALID);
}

//===----------------------------------------------------------------------===//
// Type palette
//===----------------------------------------------------------------------===//

TEST_F(GenTest, DefaultPaletteCoversExpectedTypes) {
  loom_test_gen_type_palette_t palette;
  loom_test_gen_type_palette_default(&palette);
  EXPECT_EQ(palette.count, 9);
  EXPECT_GT(palette.total_weight, 0);
}

TEST_F(GenTest, PalettePickProducesValidTypes) {
  loom_test_gen_type_palette_t palette;
  loom_test_gen_type_palette_default(&palette);
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);

  for (int i = 0; i < 100; ++i) {
    loom_type_t type = loom_test_gen_type_palette_pick(&gen, &palette);
    EXPECT_TRUE(loom_type_is_scalar(type))
        << "Palette should only produce scalar types";
  }
}

//===----------------------------------------------------------------------===//
// Randomness source
//===----------------------------------------------------------------------===//

TEST_F(GenTest, RangeUpperBoundRespected) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(12345, &gen);
  for (int i = 0; i < 1000; ++i) {
    uint32_t value = loom_test_gen_next_range(&gen, 10);
    EXPECT_LT(value, 10u);
  }
}

TEST_F(GenTest, RangeOneAlwaysZero) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(loom_test_gen_next_range(&gen, 1), 0u);
  }
}

TEST_F(GenTest, ProbabilityZeroAlwaysFalse) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(loom_test_gen_next_probability(&gen, 0));
  }
}

TEST_F(GenTest, ProbabilityHundredAlwaysTrue) {
  loom_test_gen_t gen;
  loom_test_gen_initialize_seeded(0, &gen);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(loom_test_gen_next_probability(&gen, 100));
  }
}

}  // namespace
}  // namespace loom
