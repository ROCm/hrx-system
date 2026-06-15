// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/error/renderer.h"

#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"

namespace loom {
namespace {

// Helper: render a message template with params and return as std::string.
std::string RenderMessage(const loom_error_def_t* error,
                          const loom_diagnostic_param_t* params,
                          iree_host_size_t param_count,
                          loom_type_formatter_t type_formatter = {nullptr,
                                                                  nullptr}) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_diagnostic_render_message(error, params, param_count,
                                               type_formatter, &stream));
  std::string result(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return result;
}

std::string RenderFixHint(const loom_error_def_t* error,
                          const loom_diagnostic_param_t* params,
                          iree_host_size_t param_count,
                          loom_type_formatter_t type_formatter = {nullptr,
                                                                  nullptr}) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_diagnostic_render_fix_hint(error, params, param_count,
                                                type_formatter, &stream));
  std::string result(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return result;
}

std::string FormatMinimalType(loom_type_t type) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_CHECK_OK(loom_type_format_minimal(type, nullptr, &stream));
  std::string result(iree_string_builder_buffer(&builder),
                     iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  return result;
}

//===----------------------------------------------------------------------===//
// SameType (ERR_TYPE_001): 4 params, STRING + TYPE + STRING + TYPE
//===----------------------------------------------------------------------===//

TEST(Renderer, SameTypeError) {
  // ERR_TYPE_001: "'{field_a}' type {type_a} does not match '{field_b}' type
  // {type_b}"
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);

  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("lhs")),
      loom_param_type(i32_type),
      loom_param_string(IREE_SV("rhs")),
      loom_param_type(f32_type),
  };

  std::string message =
      RenderMessage(loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1), params, 4,
                    {loom_type_format_minimal, nullptr});
  EXPECT_EQ(message, "'lhs' type i32 does not match 'rhs' type f32");
}

TEST(Renderer, SameTypeFixHint) {
  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("lhs")),
      loom_param_type(loom_type_scalar(LOOM_SCALAR_TYPE_I32)),
      loom_param_string(IREE_SV("rhs")),
      loom_param_type(loom_type_scalar(LOOM_SCALAR_TYPE_F32)),
  };

  std::string hint =
      RenderFixHint(loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1), params, 4,
                    {loom_type_format_minimal, nullptr});
  EXPECT_EQ(hint, "Ensure 'lhs' and 'rhs' have the same type");
}

//===----------------------------------------------------------------------===//
// Structure errors: STRING + U32 + U32
//===----------------------------------------------------------------------===//

TEST(Renderer, StructureOperandCount) {
  // ERR_STRUCTURE_001: "'{op_name}' has {actual_count} operands, expected
  // {expected_count}"
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_u32(3),
      loom_param_u32(2),
  };

  std::string message = RenderMessage(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1), params, 3);
  EXPECT_EQ(message, "'test.addi' has 3 operands, expected 2");
}

//===----------------------------------------------------------------------===//
// Bytecode errors: U64 offsets and lengths
//===----------------------------------------------------------------------===//

TEST(Renderer, BytecodeRangeUsesU64Params) {
  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("module[0]")),
      loom_param_u64(UINT64_C(4294967296)),
      loom_param_u64(64),
      loom_param_u64(UINT64_C(4294967297)),
  };

  std::string message = RenderMessage(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_BYTECODE, 7), params, 4);
  EXPECT_EQ(message,
            "invalid range 'module[0]' at offset 4294967296 with length 64; "
            "container length is 4294967297");
}

//===----------------------------------------------------------------------===//
// STRING_LIST params
//===----------------------------------------------------------------------===//

TEST(Renderer, StringListParam) {
  iree_string_view_t contributors[] = {
      IREE_SV("%acc0"),
      IREE_SV("%acc1"),
      IREE_SV("%a_frag0"),
  };
  loom_diagnostic_param_t params[10] = {
      loom_param_string(IREE_SV("amdgpu.gfx950")),
      loom_param_string(IREE_SV("matmul_q4")),
      loom_param_string(IREE_SV("wg128")),
      loom_param_string(IREE_SV("matmul_q4_low")),
      loom_param_string(IREE_SV("vgpr")),
      loom_param_u32(96),
      loom_param_u32(132),
      loom_param_string(IREE_SV("^k_loop")),
      loom_param_string(IREE_SV("%acc8")),
      loom_param_string_list(contributors, IREE_ARRAYSIZE(contributors)),
  };

  std::string message =
      RenderMessage(loom_error_def_lookup(LOOM_ERROR_DOMAIN_BACKEND, 3), params,
                    IREE_ARRAYSIZE(params));
  EXPECT_NE(message.find("contributors: [%acc0, %acc1, %a_frag0]"),
            std::string::npos);
}

//===----------------------------------------------------------------------===//
// Dominance errors: STRING param
//===----------------------------------------------------------------------===//

TEST(Renderer, UndefinedValue) {
  // ERR_DOMINANCE_001: "use of undefined value '{value_name}'"
  loom_diagnostic_param_t params[1] = {
      loom_param_string(IREE_SV("%y")),
  };

  std::string message = RenderMessage(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 1), params, 1);
  EXPECT_EQ(message, "use of undefined value '%y'");
}

//===----------------------------------------------------------------------===//
// I64 params
//===----------------------------------------------------------------------===//

TEST(Renderer, ShapeRankMismatch) {
  // ERR_SHAPE_001: "'{operand_a}' rank ({rank_a}) does not match
  // '{operand_b}' rank ({rank_b})"
  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("input")),
      loom_param_i64(2),
      loom_param_string(IREE_SV("output")),
      loom_param_i64(3),
  };

  std::string message = RenderMessage(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 1), params, 4);
  EXPECT_EQ(message, "'input' rank (2) does not match 'output' rank (3)");
}

//===----------------------------------------------------------------------===//
// TYPE param without formatter (fallback to "<type>")
//===----------------------------------------------------------------------===//

TEST(Renderer, TypeWithoutFormatter) {
  loom_diagnostic_param_t params[4] = {
      loom_param_string(IREE_SV("lhs")),
      loom_param_type(loom_type_scalar(LOOM_SCALAR_TYPE_I32)),
      loom_param_string(IREE_SV("rhs")),
      loom_param_type(loom_type_scalar(LOOM_SCALAR_TYPE_F32)),
  };

  // NULL type formatter → TYPE renders as "<type>".
  std::string message = RenderMessage(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1), params, 4);
  EXPECT_EQ(message, "'lhs' type <type> does not match 'rhs' type <type>");
}

//===----------------------------------------------------------------------===//
// Minimal type formatter
//===----------------------------------------------------------------------===//

TEST(Renderer, MinimalTypeFormatterScalar) {
  loom_type_t f16_type = loom_type_scalar(LOOM_SCALAR_TYPE_F16);
  EXPECT_EQ(FormatMinimalType(f16_type), "f16");
}

TEST(Renderer, MinimalTypeFormatterTile) {
  loom_type_t tile_type = loom_type_shaped_1d(
      LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 0);
  EXPECT_EQ(FormatMinimalType(tile_type), "tile<...>");
}

TEST(Renderer, MinimalTypeFormatterVector) {
  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  EXPECT_EQ(FormatMinimalType(vector_type), "vector<...>");
}

TEST(Renderer, MinimalTypeFormatterView) {
  loom_type_t view_type = loom_type_shaped_1d(
      LOOM_TYPE_VIEW, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(16), 0);
  EXPECT_EQ(FormatMinimalType(view_type), "view<...>");
}

TEST(Renderer, MinimalTypeFormatterUnknownKind) {
  loom_type_t unknown_type = {0};
  unknown_type.header =
      loom_type_make_header((loom_type_kind_t)99, (loom_scalar_type_t)0, 0, 0);
  EXPECT_EQ(FormatMinimalType(unknown_type), "<type:99>");
}

//===----------------------------------------------------------------------===//
// Edge cases
//===----------------------------------------------------------------------===//

TEST(Renderer, NullErrorDef) {
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_render_message(nullptr, nullptr, 0,
                                                {nullptr, nullptr}, &stream));
  EXPECT_EQ(iree_string_builder_size(&builder), 0u);
  iree_string_builder_deinitialize(&builder);
}

TEST(Renderer, NoFixHint) {
  // ERR_STRUCTURE_001 has fix_hint_template = NULL.
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_u32(3),
      loom_param_u32(2),
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_ASSERT_OK(loom_diagnostic_render_fix_hint(
      loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1), params, 3,
      {nullptr, nullptr}, &stream));
  EXPECT_EQ(iree_string_builder_size(&builder), 0u);
  iree_string_builder_deinitialize(&builder);
}

TEST(Renderer, NullParamsWithNonZeroCountReturnsError) {
  // param_count > 0 with NULL params is a programming error.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_diagnostic_render_message(
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1),
                            nullptr, 4, {nullptr, nullptr}, &stream));
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        loom_diagnostic_render_fix_hint(
                            loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1),
                            nullptr, 4, {nullptr, nullptr}, &stream));
  iree_string_builder_deinitialize(&builder);
}

TEST(Renderer, ParamKindMismatchReturnsError) {
  // Pass a STRING param where the schema expects U32.
  loom_diagnostic_param_t params[3] = {
      loom_param_string(IREE_SV("test.addi")),
      loom_param_string(IREE_SV("wrong")),  // Schema expects U32.
      loom_param_u32(2),
  };
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_INTERNAL,
      loom_diagnostic_render_message(
          loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1), params, 3,
          {nullptr, nullptr}, &stream));
  iree_string_builder_deinitialize(&builder);
}

TEST(Renderer, BoolParam) {
  // Construct a synthetic error def to test BOOL rendering.
  static const loom_error_param_def_t bool_param_defs[] = {
      {"flag", LOOM_PARAM_BOOL},
  };
  static const loom_error_def_t bool_error = {
      /*.error_id=*/{},
      /*.domain=*/LOOM_ERROR_DOMAIN_STRUCTURE,
      /*.severity=*/LOOM_DIAGNOSTIC_ERROR,
      /*.code=*/99,
      /*.summary=*/"Test",
      /*.message_template=*/"flag is {flag}",
      /*.fix_hint_template=*/nullptr,
      /*.param_defs=*/bool_param_defs,
      /*.param_count=*/1,
  };

  loom_diagnostic_param_t params_true[1] = {
      loom_param_bool(true),
  };
  EXPECT_EQ(RenderMessage(&bool_error, params_true, 1), "flag is true");

  loom_diagnostic_param_t params_false[1] = {
      loom_param_bool(false),
  };
  EXPECT_EQ(RenderMessage(&bool_error, params_false, 1), "flag is false");
}

}  // namespace
}  // namespace loom
