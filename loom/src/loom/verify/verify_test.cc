// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/verify/verify.h"

#include <regex>
#include <string>
#include <vector>

#include "iree/base/internal/arena.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/error/diagnostic.h"
#include "loom/error/error_defs.h"
#include "loom/error/json_sink.h"
#include "loom/error/renderer.h"
#include "loom/format/text/parser.h"
#include "loom/ir/context.h"
#include "loom/ir/module.h"
#include "loom/ops/low/ops.h"
#include "loom/ops/test/ops.h"
#include "loom/testing/diagnostic_matchers.h"
#include "loom/util/stream.h"

namespace loom {
namespace {

//===----------------------------------------------------------------------===//
// Test infrastructure
//===----------------------------------------------------------------------===//

// Collects diagnostics into a vector for assertions.
struct DiagnosticCollector {
  std::vector<std::string> errors;
  std::vector<std::string> warnings;
  std::vector<std::string> remarks;
};

static iree_status_t CollectDiagnostic(void* user_data,
                                       const loom_diagnostic_t* diagnostic) {
  auto* collector = static_cast<DiagnosticCollector*>(user_data);
  // Render message from error def + params.
  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_allocator_system(), &builder);
  loom_output_stream_t stream;
  loom_output_stream_for_builder(&builder, &stream);
  loom_type_formatter_t type_formatter = {loom_type_format_minimal, NULL};
  IREE_RETURN_IF_ERROR(loom_diagnostic_render_message(
      diagnostic->error, diagnostic->params, diagnostic->param_count,
      type_formatter, &stream));
  std::string message(iree_string_builder_buffer(&builder),
                      iree_string_builder_size(&builder));
  iree_string_builder_deinitialize(&builder);
  switch (diagnostic->severity) {
    case LOOM_DIAGNOSTIC_ERROR:
      collector->errors.push_back(std::move(message));
      break;
    case LOOM_DIAGNOSTIC_WARNING:
      collector->warnings.push_back(std::move(message));
      break;
    case LOOM_DIAGNOSTIC_REMARK:
      collector->remarks.push_back(std::move(message));
      break;
    default:
      break;
  }
  return iree_ok_status();
}

using ::loom::testing::CapturedDiagnostic;
using ::loom::testing::DiagnosticCapture;
using ::loom::testing::ExpectFieldRefParam;
using ::loom::testing::ExpectHighlightFieldRef;
using ::loom::testing::ExpectI64Param;
using ::loom::testing::ExpectNoFieldRefParam;
using ::loom::testing::ExpectRelatedHighlightFieldRef;
using ::loom::testing::ExpectTypeParam;
using ::loom::testing::ExpectU32Param;
using ::loom::testing::FindDiagnostic;
using ::loom::testing::GetStringParam;

// Registers test dialect vtables on the context.
static void RegisterTestDialect(loom_context_t* context) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, LOOM_DIALECT_TEST,
                                               vtables, (uint16_t)count));
}

static void RegisterLowDialect(loom_context_t* context) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_low_dialect_vtables(&count);
  IREE_ASSERT_OK(loom_context_register_dialect(context, LOOM_DIALECT_LOW,
                                               vtables, (uint16_t)count));
}

static const loom_op_vtable_t* TestVtable(loom_op_kind_t kind) {
  iree_host_size_t count = 0;
  const loom_op_vtable_t* const* vtables = loom_test_dialect_vtables(&count);
  uint8_t index = loom_op_dialect_index(kind);
  if (loom_op_dialect_id(kind) != LOOM_DIALECT_TEST || index >= count) {
    return nullptr;
  }
  return vtables[index];
}

class VerifyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(4096, iree_allocator_system(),
                                     &block_pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    RegisterTestDialect(&context_);
    RegisterLowDialect(&context_);
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    IREE_ASSERT_OK(loom_module_allocate(&context_, IREE_SV("verify_test"),
                                        &block_pool_, NULL,
                                        iree_allocator_system(), &module_));
    loom_builder_initialize(module_, &module_->arena,
                            loom_module_block(module_), &builder_);

    collector_ = {};
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = CollectDiagnostic;
    options_.sink.user_data = &collector_;
    options_.max_errors = 100;
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&block_pool_);
  }

  // Creates a value in the module's value table.
  loom_value_id_t DefineValue(loom_scalar_type_t scalar_type) {
    loom_type_t type = loom_type_scalar(scalar_type);
    loom_value_id_t value_id = LOOM_VALUE_ID_INVALID;
    IREE_EXPECT_OK(loom_builder_define_value(&builder_, type, &value_id));
    return value_id;
  }

  // Creates a test.func at module level and switches the builder to its
  // body block. Returns |arg_count| value IDs (the function's block
  // arguments) via |out_args|. Subsequent builder ops go inside the
  // function body. Call TerminateFunc() before Verify() to add a
  // terminator to the body block.
  void EnterTestFunc(const loom_type_t* arg_types, iree_host_size_t arg_count,
                     loom_value_id_t* out_args) {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_ASSERT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("test"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
    loom_op_t* func_op = nullptr;
    IREE_ASSERT_OK(loom_test_func_build(
        &builder_, 0, 0, 0, callee, arg_types, arg_count, nullptr, 0, nullptr,
        0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
    loom_region_t* body = loom_test_func_body(func_op);
    loom_builder_set_block(&builder_, loom_region_entry_block(body));
    for (iree_host_size_t i = 0; i < arg_count; ++i) {
      out_args[i] = loom_region_entry_arg_id(body, (uint16_t)i);
    }
  }

  // Adds a test.yield terminator to the current block. Call after
  // building all ops inside the test.func body.
  void TerminateFunc() {
    loom_op_t* yield_op = nullptr;
    IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                         LOOM_LOCATION_UNKNOWN, &yield_op));
  }

  loom_symbol_ref_t DefineTestCallee() {
    loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
    IREE_EXPECT_OK(
        loom_builder_intern_string(&builder_, IREE_SV("callee"), &name_id));
    uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
    IREE_EXPECT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
    return (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  }

  loom_verify_result_t Verify() {
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    return result;
  }

  loom_verify_result_t VerifyStructured(DiagnosticCapture* capture) {
    options_.sink = capture->sink();
    return Verify();
  }

  loom_module_t* ParseSourceModule(const char* source, const char* filename) {
    DiagnosticCapture parse_capture;
    loom_text_parse_options_t parse_options = {};
    parse_options.diagnostic_sink = parse_capture.sink();
    parse_options.max_errors = 20;
    loom_module_t* parsed_module = nullptr;
    IREE_EXPECT_OK(loom_text_parse(
        iree_make_cstring_view(source), iree_make_cstring_view(filename),
        &context_, &block_pool_, &parse_options, &parsed_module));
    if (!parse_capture.diagnostics.empty()) {
      ADD_FAILURE() << "Expected parser success, got "
                    << parse_capture.diagnostics.size() << " diagnostics";
      if (parsed_module) loom_module_free(parsed_module);
      return nullptr;
    }
    EXPECT_NE(parsed_module, nullptr);
    return parsed_module;
  }

  loom_source_id_t FindModuleSourceId(const loom_module_t* module,
                                      const char* filename) const {
    iree_string_view_t source_name = iree_make_cstring_view(filename);
    for (iree_host_size_t i = 0; i < module->sources.count; ++i) {
      if (iree_string_view_equal(module->sources.entries[i], source_name)) {
        return (loom_source_id_t)i;
      }
    }
    ADD_FAILURE() << "Expected module source table to contain " << filename;
    return LOOM_SOURCE_ID_INVALID;
  }

  loom_location_id_t AddFileLocation(const char* filename, uint16_t start_line,
                                     uint16_t start_col, uint16_t end_line,
                                     uint16_t end_col) {
    loom_source_id_t source_id = LOOM_SOURCE_ID_INVALID;
    IREE_EXPECT_OK(loom_module_register_source(
        module_, iree_make_cstring_view(filename), &source_id));
    EXPECT_NE(source_id, LOOM_SOURCE_ID_INVALID);
    loom_location_id_t location_id = LOOM_LOCATION_UNKNOWN;
    IREE_EXPECT_OK(loom_module_add_location(
        module_,
        loom_location_file_range(source_id, start_line, start_col, end_line,
                                 end_col),
        &location_id));
    EXPECT_NE(location_id, LOOM_LOCATION_UNKNOWN);
    return location_id;
  }

  loom_verify_result_t VerifyParsedSourceModuleStructured(
      loom_module_t* parsed_module, const char* source, const char* filename,
      DiagnosticCapture* capture) {
    loom_source_entry_t source_entries[] = {{
        /*.source_id=*/FindModuleSourceId(parsed_module, filename),
        /*.source=*/iree_make_cstring_view(source),
        /*.filename=*/iree_make_cstring_view(filename),
    }};
    EXPECT_NE(source_entries[0].source_id, LOOM_SOURCE_ID_INVALID);
    loom_source_table_resolver_t resolver_data = {
        /*.entries=*/source_entries,
        /*.count=*/IREE_ARRAYSIZE(source_entries),
    };

    loom_verify_options_t options = {};
    options.sink = capture->sink();
    options.max_errors = 20;
    options.source_resolver = {loom_source_table_resolve, &resolver_data};

    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(parsed_module, &options, &result));
    return result;
  }

  // Runs verification with the caret formatter and returns the output.
  std::string VerifyAndFormatCaret() {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    auto sink = [](void* user_data,
                   const loom_diagnostic_t* d) -> iree_status_t {
      auto* builder = static_cast<iree_string_builder_t*>(user_data);
      loom_output_stream_t stream;
      loom_output_stream_for_builder(builder, &stream);
      return loom_diagnostic_format(d, &stream);
    };
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = sink;
    options_.sink.user_data = &output;
    options_.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return text;
  }

  // Runs verification with the JSON sink and returns the output.
  std::string VerifyAndFormatJson() {
    iree_string_builder_t output;
    iree_string_builder_initialize(iree_allocator_system(), &output);
    loom_output_stream_t json_stream;
    loom_output_stream_for_builder(&output, &json_stream);
    loom_json_sink_options_t json_options;
    memset(&json_options, 0, sizeof(json_options));
    json_options.stream = &json_stream;
    json_options.type_formatter = {loom_type_format_minimal, nullptr};
    memset(&options_, 0, sizeof(options_));
    options_.sink.fn = loom_diagnostic_json_sink;
    options_.sink.user_data = &json_options;
    options_.max_errors = 20;
    loom_verify_result_t result = {};
    IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
    std::string text(iree_string_builder_buffer(&output),
                     iree_string_builder_size(&output));
    iree_string_builder_deinitialize(&output);
    return text;
  }

  iree_arena_block_pool_t block_pool_;
  loom_context_t context_;
  loom_module_t* module_ = nullptr;
  loom_builder_t builder_;
  DiagnosticCollector collector_;
  loom_verify_options_t options_;
};

static loom_trait_flags_t BadHintPureEffectiveTraits(const loom_op_t* op) {
  (void)op;
  return LOOM_TRAIT_HINT | LOOM_TRAIT_PURE;
}

static void ExpectBadTraitDiagnostic(const loom_op_vtable_t* bad_vtable,
                                     const char* op_name, const char* trait_a,
                                     const char* trait_b) {
  iree_arena_block_pool_t block_pool;
  iree_arena_block_pool_initialize(4096, iree_allocator_system(), &block_pool);

  loom_context_t context;
  loom_context_initialize(iree_allocator_system(), &context);
  RegisterTestDialect(&context);

  const loom_op_vtable_t* const kBadDialectVtables[] = {
      bad_vtable,
  };
  constexpr uint8_t kBadDialectId = LOOM_DIALECT_BUILTIN_COUNT_ - 1;
  IREE_ASSERT_OK(
      loom_context_register_dialect(&context, kBadDialectId, kBadDialectVtables,
                                    IREE_ARRAYSIZE(kBadDialectVtables)));
  IREE_ASSERT_OK(loom_context_finalize(&context));

  loom_module_t* module = nullptr;
  IREE_ASSERT_OK(loom_module_allocate(&context, IREE_SV("bad_traits"),
                                      &block_pool, nullptr,
                                      iree_allocator_system(), &module));
  loom_builder_t builder;
  loom_builder_initialize(module, &module->arena, loom_module_block(module),
                          &builder);

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder, IREE_SV("test"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&builder, 0, 0, 0, callee, nullptr, 0,
                                      nullptr, 0, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));
  loom_region_t* body = loom_test_func_body(func_op);
  loom_builder_set_block(&builder, loom_region_entry_block(body));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder,
                                          LOOM_OP_KIND(kBadDialectId, 0), 0, 0,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  IREE_ASSERT_OK(loom_builder_finalize_op(&builder, op));
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder, nullptr, 0,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));

  DiagnosticCapture capture;
  loom_verify_options_t options = {};
  options.sink = capture.sink();
  options.max_errors = 20;
  loom_verify_result_t result = {};
  IREE_ASSERT_OK(loom_verify_module(module, &options, &result));

  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 16));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/016 incompatible-traits error";
  EXPECT_EQ(GetStringParam(*entry, 0), op_name);
  EXPECT_EQ(GetStringParam(*entry, 1), trait_a);
  EXPECT_EQ(GetStringParam(*entry, 2), trait_b);

  loom_module_free(module);
  loom_context_deinitialize(&context);
  iree_arena_block_pool_deinitialize(&block_pool);
}

TEST(VerifyTraitConsistencyTest, RejectsDeclaredIncompatibleHintTraits) {
  static const uint8_t kBadHintName[] = {
      8, 3, 'b', 'a', 'd', '.', 'h', 'i', 'n', 't', '\0',
  };
  static const loom_op_vtable_t kBadHintVtable = {
      /*.traits=*/LOOM_TRAIT_HINT | LOOM_TRAIT_PURE,
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kBadHintName,
  };
  ExpectBadTraitDiagnostic(&kBadHintVtable, "bad.hint", "HINT", "PURE");
}

TEST(VerifyTraitConsistencyTest, RejectsEffectiveIncompatibleHintTraits) {
  static const uint8_t kBadHintName[] = {
      8, 3, 'b', 'a', 'd', '.', 'h', 'i', 'n', 't', '\0',
  };
  static const loom_op_vtable_t kBadHintVtable = {
      /*.traits=*/LOOM_TRAIT_HINT,
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/BadHintPureEffectiveTraits,
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kBadHintName,
  };
  ExpectBadTraitDiagnostic(&kBadHintVtable, "bad.hint", "HINT", "PURE");
}

TEST(VerifyTraitConsistencyTest, RejectsDeclaredIncompatibleSpeculationTraits) {
  static const uint8_t kBadSpecName[] = {
      8, 3, 'b', 'a', 'd', '.', 's', 'p', 'e', 'c', '\0',
  };
  static const loom_op_vtable_t kBadSpecVtable = {
      /*.traits=*/LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_UNKNOWN_EFFECTS,
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kBadSpecName,
  };
  ExpectBadTraitDiagnostic(&kBadSpecVtable, "bad.spec", "SAFE_TO_SPECULATE",
                           "UNKNOWN_EFFECTS");
}

TEST(VerifyTraitConsistencyTest, RejectsDeclaredSpeculatableConvergentTraits) {
  static const uint8_t kBadConvergentName[] = {
      8, 3, 'b', 'a', 'd', '.', 'c', 'o', 'n', 'v', '\0',
  };
  static const loom_op_vtable_t kBadConvergentVtable = {
      /*.traits=*/LOOM_TRAIT_SAFE_TO_SPECULATE | LOOM_TRAIT_CONVERGENT,
      /*.fixed_operand_count=*/{},
      /*.fixed_result_count=*/{},
      /*.attribute_count=*/{},
      /*.region_count=*/{},
      /*.vtable_flags=*/{},
      /*.symbol_kind=*/{},
      /*.constraint_count=*/{},
      /*.operand_descriptor_count=*/{},
      /*.control_flow_flags=*/{},
      /*.control_flow_reserved=*/{},
      /*.successor_selector_operand_index=*/{},
      /*.canonicalize=*/{},
      /*.infer_facts=*/{},
      /*.effective_traits=*/{},
      /*.attr_descriptors=*/{},
      /*.operand_descriptors=*/{},
      /*.type_transfer=*/{},
      /*.result_descriptors=*/{},
      /*.region_descriptors=*/{},
      /*.constraints=*/{},
      /*.verify=*/{},
      /*.name=*/kBadConvergentName,
  };
  ExpectBadTraitDiagnostic(&kBadConvergentVtable, "bad.conv",
                           "SAFE_TO_SPECULATE", "CONVERGENT");
}

//===----------------------------------------------------------------------===//
// Structural checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, EmptyModulePasses) {
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, ValidAddiPasses) {
  // Build a valid test.addi: %r = test.addi %a, %b : i32
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[0], args[1], i32_type,
                                      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, RejectsPredicateArityMismatch) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t argument = LOOM_VALUE_ID_INVALID;
  EnterTestFunc(&index_type, 1, &argument);

  loom_predicate_t predicate = {
      /*.kind=*/LOOM_PREDICATE_POW2,
      /*.arg_count=*/2,
      /*.arg_tags=*/{LOOM_PRED_ARG_VALUE, LOOM_PRED_ARG_CONST},
      /*.reserved=*/{},
      /*.args=*/{(int64_t)argument, 16},
  };
  loom_op_t* assume_op = nullptr;
  IREE_ASSERT_OK(loom_test_assume_build(&builder_, &argument, 1, &predicate, 1,
                                        &index_type, 1, LOOM_LOCATION_UNKNOWN,
                                        &assume_op));

  TerminateFunc();
  DiagnosticCapture capture;
  auto result = VerifyStructured(&capture);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      capture, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 21));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/021 predicate-arity diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "predicates");
  ExpectU32Param(*entry, 1, 0u);
  EXPECT_EQ(GetStringParam(*entry, 2), "pow2");
  ExpectU32Param(*entry, 3, 1u);
  ExpectU32Param(*entry, 4, 2u);
}

TEST_F(VerifyTest, LoopBodyMissingMaterializedTerminatorFails) {
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_type_t arg_types[] = {index_type, index_type, index_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_op_t* loop_op = nullptr;
  IREE_ASSERT_OK(loom_test_loop_build(&builder_, args[0], args[1], args[2],
                                      nullptr, 0, nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &loop_op));
  ASSERT_NE(loop_op, nullptr);
  loom_region_t* body = loom_test_loop_body(loop_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(loom_region_entry_block(body)->op_count, 0u);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 5));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/005 missing-terminator diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.loop");
  ExpectU32Param(*entry, 1, 0);
}

TEST_F(VerifyTest, WrongOperandCountDetected) {
  // Manually create an op with wrong operand count.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI,
                                          /*operand_count=*/1,  // Should be 2.
                                          /*result_count=*/1,
                                          /*region_count=*/0,
                                          /*tied_result_count=*/0,
                                          /*attribute_count=*/0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1));
  ASSERT_NE(entry, nullptr) << "Expected STRUCTURE/001 operand-count error";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.addi");
  ExpectU32Param(*entry, 1, 1);
  ExpectU32Param(*entry, 2, 2);
}

TEST_F(VerifyTest, OperandDictOperandsRequireNamesAttribute) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t argument_types[] = {f32_type, i32_type};
  loom_value_id_t arguments[2];
  EnterTestFunc(argument_types, IREE_ARRAYSIZE(argument_types), arguments);

  loom_string_id_t alpha_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("alpha"), &alpha_name));
  loom_named_value_t parameters[] = {
      {/*.name_id=*/alpha_name, /*.reserved=*/0, /*.value_id=*/arguments[1]},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_operand_dict_build(
      &builder_, arguments[0], parameters, IREE_ARRAYSIZE(parameters), f32_type,
      LOOM_LOCATION_UNKNOWN, &op));
  loom_op_attrs(op)[0] = loom_attr_absent();

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 13));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/013 operand-dictionary count diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "param_names");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, 0);
  ExpectU32Param(*entry, 1, 0);
  EXPECT_EQ(GetStringParam(*entry, 2), "operand dictionary operands");
  ExpectU32Param(*entry, 3, 1);
}

TEST_F(VerifyTest, OperandDictOrdinalsMustStayInOperandRange) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t argument_types[] = {f32_type, i32_type};
  loom_value_id_t arguments[2];
  EnterTestFunc(argument_types, IREE_ARRAYSIZE(argument_types), arguments);

  loom_string_id_t alpha_name = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("alpha"), &alpha_name));
  loom_named_value_t parameters[] = {
      {/*.name_id=*/alpha_name, /*.reserved=*/0, /*.value_id=*/arguments[1]},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_operand_dict_build(
      &builder_, arguments[0], parameters, IREE_ARRAYSIZE(parameters), f32_type,
      LOOM_LOCATION_UNKNOWN, &op));
  loom_named_attr_t names[] = {
      {/*.name_id=*/alpha_name, /*.reserved=*/0, /*.value=*/loom_attr_i64(1)},
  };
  loom_op_attrs(op)[0] =
      loom_make_canonical_attr_dict(names, IREE_ARRAYSIZE(names));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 14));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/014 operand-dictionary ordinal diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "alpha");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, 0);
  ExpectI64Param(*entry, 1, 1);
  EXPECT_EQ(GetStringParam(*entry, 2), "operand ordinal in range");
}

TEST_F(VerifyTest, MissingSuccessorTargetDetected) {
  EnterTestFunc(nullptr, 0, nullptr);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op_with_successors(
      &builder_, LOOM_OP_TEST_BR, /*operand_count=*/0, /*result_count=*/0,
      /*successor_count=*/1, /*region_count=*/0, /*tied_result_count=*/0,
      /*attribute_count=*/0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_successors(op)[0] = nullptr;

  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 23));
  ASSERT_NE(entry, nullptr) << "Expected STRUCTURE/023 successor target error";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.br");
  ExpectU32Param(*entry, 1, 0);
  ExpectFieldRefParam(*entry, 1, LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 0);
}

TEST_F(VerifyTest, SuccessorOutsideParentRegionDetected) {
  EnterTestFunc(nullptr, 0, nullptr);

  loom_region_t* foreign_region = nullptr;
  IREE_ASSERT_OK(loom_module_allocate_region(module_, 1, &foreign_region));
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_br_build(&builder_,
                                    loom_region_entry_block(foreign_region),
                                    LOOM_LOCATION_UNKNOWN, &op));

  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 24));
  ASSERT_NE(entry, nullptr) << "Expected STRUCTURE/024 successor-region error";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.br");
  ExpectU32Param(*entry, 1, 0);
  ExpectFieldRefParam(*entry, 1, LOOM_DIAGNOSTIC_FIELD_SUCCESSOR, 0);
}

TEST_F(VerifyTest, OpAfterTerminatorDetected) {
  EnterTestFunc(nullptr, 0, nullptr);
  TerminateFunc();

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_op_t* constant_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(&builder_, loom_attr_i64(0), i32_type,
                                          LOOM_LOCATION_UNKNOWN, &constant_op));

  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 12));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/012 op-after-terminator error";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.constant");
}

TEST_F(VerifyTest, OpAfterTerminatorReportsTerminatorLocation) {
  static const char kSource[] =
      "test.func @main() {\n"
      "  test.yield\n"
      "  %result = test.constant 0 : i32\n"
      "}\n";
  const char* filename = "op_after_terminator.loom";

  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("main"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(
      &builder_, 0, 0, 0,
      (loom_symbol_ref_t){/*.module_id=*/0, /*.symbol_id=*/symbol_id}, nullptr,
      0, nullptr, 0, nullptr, 0, nullptr, 0, LOOM_LOCATION_UNKNOWN, &func_op));
  loom_builder_set_block(&builder_,
                         loom_region_entry_block(loom_test_func_body(func_op)));

  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, nullptr, 0,
                                       AddFileLocation(filename, 2, 3, 2, 13),
                                       &yield_op));

  loom_op_t* constant_op = nullptr;
  IREE_ASSERT_OK(loom_test_constant_build(
      &builder_, loom_attr_i64(0), loom_type_scalar(LOOM_SCALAR_TYPE_I32),
      AddFileLocation(filename, 3, 3, 3, 34), &constant_op));

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      module_, kSource, filename, &structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* structure_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 12));
  ASSERT_NE(structure_error, nullptr)
      << "Expected STRUCTURE/012 op-after-terminator diagnostic";
  EXPECT_EQ(GetStringParam(*structure_error, 0), "test.constant");
  ASSERT_EQ(structure_error->related_locations.size(), 1u);
  const auto& terminator_note = structure_error->related_locations[0];
  EXPECT_EQ(terminator_note.label, "terminator here");
  EXPECT_TRUE(terminator_note.has_source_range);
  EXPECT_EQ(terminator_note.source_location.provenance,
            LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(terminator_note.source_location.start_line, 2u);
  EXPECT_EQ(terminator_note.source_location.start_column, 3u);
}

//===----------------------------------------------------------------------===//
// Type constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, TypeConstraintViolationDetected) {
  // test.addi expects INTEGER operands. Give it f32 values.
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type, f32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 3));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/003 operand constraint error";
  EXPECT_EQ(GetStringParam(*entry, 0), "lhs");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*entry, 1, f32_type);
  ExpectNoFieldRefParam(*entry, 1);
  EXPECT_EQ(GetStringParam(*entry, 2), "integer");
  ExpectNoFieldRefParam(*entry, 2);
}

//===----------------------------------------------------------------------===//
// Semantic constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, SameTypeConstraintPasses) {
  // test.addi has SameType(lhs, rhs, result). All i32 → passes.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[0], args[1], i32_type,
                                      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, SameTypeConstraintViolation) {
  // Build addi with mismatched types: i32 operands but f32 result.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);

  // Manually build to set a different result type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/001 SameType diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "lhs");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*entry, 1, i32_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "result");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  ExpectTypeParam(*entry, 3, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  ASSERT_EQ(entry->highlights.size(), 2u);
  ExpectHighlightFieldRef(*entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, 2);
  ExpectHighlightFieldRef(*entry, 1, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, 0);
}

//===----------------------------------------------------------------------===//
// SSA dominance checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, UndefinedOperandDetected) {
  // Create a test.func with no args, then build an op inside it that
  // references value IDs that exist in the value table but were never
  // defined by a prior op or as a block arg.
  EnterTestFunc(nullptr, 0, nullptr);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_value_id_t lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_operands(op)[0] = lhs;
  loom_op_operands(op)[1] = rhs;
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_results(op)[0] = result_val;

  // The verifier should catch the undefined operands.
  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 1));
  ASSERT_NE(entry, nullptr) << "Expected DOMINANCE/001 undefined-value error";
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
}

//===----------------------------------------------------------------------===//
// Constraint table structure
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, ConstraintTableSize) {
  static_assert(sizeof(loom_constraint_t) == 10,
                "loom_constraint_t must be 10 bytes");
}

TEST_F(VerifyTest, AddiHasConstraints) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_ADDI);
  EXPECT_EQ(vtable->constraint_count, 1);
  EXPECT_NE(vtable->constraints, nullptr);
  EXPECT_EQ(vtable->constraints[0].relation, LOOM_RELATION_PAIRWISE_EQ);
  EXPECT_EQ(vtable->constraints[0].property, LOOM_PROPERTY_TYPE);
  EXPECT_EQ(vtable->constraints[0].arg_count, 3);
  EXPECT_EQ(vtable->constraints[0].error_ref,
            LOOM_ERROR_REF(LOOM_ERROR_DOMAIN_TYPE, 1));
}

TEST_F(VerifyTest, MapHasRegionConstraints) {
  // test.map has 5 constraints including region-value relationships.
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_MAP);
  EXPECT_EQ(vtable->constraint_count, 5);
  EXPECT_NE(vtable->constraints, nullptr);
  EXPECT_EQ(vtable->constraints[0].relation, LOOM_RELATION_ALL_SAME);
  EXPECT_EQ(vtable->constraints[0].property, LOOM_PROPERTY_SHAPE);
  EXPECT_EQ(vtable->constraints[1].relation, LOOM_RELATION_REGION_ARG_COUNT);
}

TEST_F(VerifyTest, LowFuncHasRegisterBlockArgConstraint) {
  iree_host_size_t low_count = 0;
  const loom_op_vtable_t* const* low_vtables =
      loom_low_dialect_vtables(&low_count);
  ASSERT_GT(low_count, loom_op_dialect_index(LOOM_OP_LOW_FUNC_DEF));
  const loom_op_vtable_t* vtable =
      low_vtables[loom_op_dialect_index(LOOM_OP_LOW_FUNC_DEF)];
  EXPECT_EQ(vtable->constraints[0].relation, LOOM_RELATION_REGION_ARGS_SATISFY);
  EXPECT_EQ(vtable->constraints[0].property, LOOM_TYPE_CONSTRAINT_REGISTER);
}

TEST_F(VerifyTest, LowFuncRejectsNonRegisterBlockArg) {
  const char kSource[] =
      "test.target<low_core> @gfx1100\n"
      "low.func.def target(@gfx1100) @bad(%input: i32) {\n"
      "  low.return\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_low_non_register_arg.loom");
  ASSERT_NE(parsed_module, nullptr);

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      parsed_module, kSource, "parsed_low_non_register_arg.loom", &structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 14));
  ASSERT_NE(entry, nullptr)
      << "Expected TYPE/014 block argument constraint diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "region 0.args[0]");
  ExpectTypeParam(*entry, 1, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  EXPECT_EQ(GetStringParam(*entry, 2), "register");

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, ConstantHasNoConstraints) {
  const loom_op_vtable_t* vtable = TestVtable(LOOM_OP_TEST_CONSTANT);
  EXPECT_EQ(vtable->constraint_count, 0);
  EXPECT_EQ(vtable->constraints, nullptr);
}

//===----------------------------------------------------------------------===//
// Structured diagnostic output (print-op fallback)
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, StructuredDiagnosticHasErrorDef) {
  // Build an addi with wrong operand count to trigger a structured error.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 1, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_results(op)[0] = result_val;

  // Use the structured collector instead of the default one.
  DiagnosticCapture structured;

  TerminateFunc();
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  ASSERT_GT(structured.diagnostics.size(), 0u);

  // The first error should be a structured STRUCTURE_001 diagnostic.
  const auto& entry = structured.diagnostics[0];
  EXPECT_EQ(entry.severity, LOOM_DIAGNOSTIC_ERROR);
  EXPECT_EQ(entry.emitter, LOOM_EMITTER_VERIFIER);
  ASSERT_NE(entry.error, nullptr);
  EXPECT_EQ(entry.error->domain, LOOM_ERROR_DOMAIN_STRUCTURE);
  EXPECT_EQ(entry.error->code, 1);
  EXPECT_GT(entry.params.size(), 0);
}

TEST_F(VerifyTest, PrintOpFallbackProvidesSourceRange) {
  // Build a valid op so we can trigger a SameType constraint error.
  // addi with i32 operands but f32 result → SameType violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  // No source resolver configured → the verifier must fall back to
  // printing the op for the source range.
  DiagnosticCapture structured;

  TerminateFunc();
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);

  // Find the SameType error (there may also be type constraint errors).
  bool found_with_source = false;
  for (const auto& entry : structured.diagnostics) {
    if (entry.has_source_range) {
      found_with_source = true;
      // The source text should be the printed op (contains "test.addi").
      EXPECT_NE(entry.source_text.find("test.addi"), std::string::npos)
          << "Printed op source: " << entry.source_text;
      EXPECT_EQ(entry.origin.provenance,
                LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK);
      EXPECT_EQ(entry.source_location.provenance,
                LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK);
      // The filename should be the verifier pseudo-filename.
      EXPECT_EQ(entry.filename, "<verifier>");
      if (!entry.highlights.empty()) {
        ExpectHighlightFieldRef(entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, 0);
      }
      break;
    }
  }
  EXPECT_TRUE(found_with_source)
      << "Expected at least one diagnostic with a print-op source range";
}

TEST_F(VerifyTest, ParsedSourceResolverHighlightsExactResultAndOperandTokens) {
  static const char kSource[] =
      "%lhs = test.constant 1 : i32\n"
      "%rhs = test.constant 2 : i32\n"
      "%sum = test.addi %lhs, %rhs : f32\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_verify_test.loom");
  ASSERT_NE(parsed_module, nullptr);

  loom_source_entry_t source_entries[] = {
      {
          /*.source_id=*/
          FindModuleSourceId(parsed_module, "parsed_verify_test.loom"),
          /*.source=*/iree_make_cstring_view(kSource),
          /*.filename=*/IREE_SV("parsed_verify_test.loom"),
      },
  };
  ASSERT_NE(source_entries[0].source_id, LOOM_SOURCE_ID_INVALID);
  loom_source_table_resolver_t resolver_data = {
      /*.entries=*/source_entries,
      /*.count=*/IREE_ARRAYSIZE(source_entries),
  };

  DiagnosticCapture structured;
  loom_verify_options_t options = {};
  options.sink = structured.sink();
  options.max_errors = 20;
  options.source_resolver = {loom_source_table_resolve, &resolver_data};

  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(parsed_module, &options, &result));
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* type_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(type_error, nullptr) << "Expected TYPE/001 SameType diagnostic";
  EXPECT_EQ(type_error->origin.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(type_error->source_location.provenance,
            LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(type_error->filename, "parsed_verify_test.loom");
  EXPECT_EQ(type_error->origin_line, 3u);
  EXPECT_EQ(type_error->origin_column, 1u);
  EXPECT_EQ(type_error->source_text, kSource);
  ASSERT_EQ(type_error->highlights.size(), 2u);
  ExpectHighlightFieldRef(*type_error, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, 2);
  EXPECT_EQ(type_error->source_text.substr(type_error->highlights[0].start,
                                           type_error->highlights[0].end -
                                               type_error->highlights[0].start),
            "%sum");
  ExpectHighlightFieldRef(*type_error, 1, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, 0);
  EXPECT_EQ(type_error->source_text.substr(type_error->highlights[1].start,
                                           type_error->highlights[1].end -
                                               type_error->highlights[1].start),
            "%lhs");

  const CapturedDiagnostic* result_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 4));
  ASSERT_NE(result_error, nullptr)
      << "Expected TYPE/004 result constraint diagnostic";
  ASSERT_EQ(result_error->highlights.size(), 1u);
  ExpectHighlightFieldRef(*result_error, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0, 0);
  EXPECT_EQ(
      result_error->source_text.substr(
          result_error->highlights[0].start,
          result_error->highlights[0].end - result_error->highlights[0].start),
      "%sum");

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, ParsedUseAfterConsumeReportsRelatedConsumeLocation) {
  static const char kSource[] =
      "test.decl @callee(%arg: f32) -> (%arg as f32)\n"
      "test.func @main(%arg: f32) -> (f32) {\n"
      "  %next = test.invoke @callee(%arg) : (f32) -> (%arg as f32)\n"
      "  test.use %arg : f32\n"
      "  test.yield %next : f32\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_use_after_consume.loom");
  ASSERT_NE(parsed_module, nullptr);

  loom_source_entry_t source_entries[] = {{
      /*.source_id=*/
      FindModuleSourceId(parsed_module, "parsed_use_after_consume.loom"),
      /*.source=*/iree_make_cstring_view(kSource),
      /*.filename=*/IREE_SV("parsed_use_after_consume.loom"),
  }};
  ASSERT_NE(source_entries[0].source_id, LOOM_SOURCE_ID_INVALID);
  loom_source_table_resolver_t resolver_data = {
      /*.entries=*/source_entries,
      /*.count=*/IREE_ARRAYSIZE(source_entries),
  };

  DiagnosticCapture structured;
  loom_verify_options_t options = {};
  options.sink = structured.sink();
  options.max_errors = 20;
  options.source_resolver = {loom_source_table_resolve, &resolver_data};

  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(parsed_module, &options, &result));
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* dominance_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 2));
  ASSERT_NE(dominance_error, nullptr)
      << "Expected DOMINANCE/002 consumed-value diagnostic";
  EXPECT_EQ(GetStringParam(*dominance_error, 0), "arg");
  ExpectFieldRefParam(*dominance_error, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  EXPECT_EQ(GetStringParam(*dominance_error, 1), "test.invoke");
  EXPECT_EQ(dominance_error->origin.provenance,
            LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(dominance_error->origin_line, 4u);
  EXPECT_EQ(dominance_error->origin_column, 3u);
  ASSERT_EQ(dominance_error->highlights.size(), 1u);
  ExpectHighlightFieldRef(*dominance_error, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                          0);
  EXPECT_EQ(dominance_error->source_text.substr(
                dominance_error->highlights[0].start,
                dominance_error->highlights[0].end -
                    dominance_error->highlights[0].start),
            "%arg");

  ASSERT_EQ(dominance_error->related_locations.size(), 1u);
  const auto& consume_note = dominance_error->related_locations[0];
  EXPECT_EQ(consume_note.label, "consumed here");
  EXPECT_TRUE(consume_note.has_source_range);
  EXPECT_EQ(consume_note.source_location.provenance,
            LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(std::string(consume_note.source_location.filename.data,
                        consume_note.source_location.filename.size),
            "parsed_use_after_consume.loom");
  EXPECT_EQ(consume_note.source_location.start_line, 3u);
  EXPECT_EQ(consume_note.source_location.start_column, 3u);
  EXPECT_NE(std::string(consume_note.source_location.source.data +
                            consume_note.source_location.start,
                        consume_note.source_location.end -
                            consume_note.source_location.start)
                .find("test.invoke @callee(%arg)"),
            std::string::npos);

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, ParsedInvokeOperandCountMismatchReportsCalleeLocation) {
  static const char kSource[] =
      "test.decl @callee(%arg: f32) -> (%arg as f32)\n"
      "test.func @main(%arg: f32, %extra: f32) -> (f32) {\n"
      "  %result = test.invoke @callee(%arg, %extra) : (f32, f32) -> (%arg as "
      "f32)\n"
      "  test.yield %result : f32\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_invoke_operand_count.loom");
  ASSERT_NE(parsed_module, nullptr);

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      parsed_module, kSource, "parsed_invoke_operand_count.loom", &structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* invoke_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 1));
  ASSERT_NE(invoke_error, nullptr)
      << "Expected STRUCTURE/001 invoke operand-count diagnostic";
  EXPECT_EQ(GetStringParam(*invoke_error, 0), "test.invoke");
  ExpectU32Param(*invoke_error, 1, 2);
  ExpectU32Param(*invoke_error, 2, 1);
  ASSERT_EQ(invoke_error->related_locations.size(), 1u);
  const auto& definition_note = invoke_error->related_locations[0];
  EXPECT_EQ(definition_note.label, "defined here");
  EXPECT_TRUE(definition_note.has_source_range);
  EXPECT_EQ(definition_note.source_location.start_line, 1u);
  EXPECT_EQ(definition_note.source_location.start_column, 1u);

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, ParsedInvokeOperandTypeMismatchReportsCalleeLocation) {
  static const char kSource[] =
      "test.decl @callee(%arg: f32) -> (f32)\n"
      "test.func @main(%arg: i32) -> (f32) {\n"
      "  %result = test.invoke @callee(%arg) : (i32) -> (f32)\n"
      "  test.yield %result : f32\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_invoke_operand_type.loom");
  ASSERT_NE(parsed_module, nullptr);

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      parsed_module, kSource, "parsed_invoke_operand_type.loom", &structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* invoke_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(invoke_error, nullptr)
      << "Expected TYPE/001 invoke operand-type diagnostic";
  EXPECT_EQ(GetStringParam(*invoke_error, 0), "operand 0");
  ExpectFieldRefParam(*invoke_error, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*invoke_error, 1, loom_type_scalar(LOOM_SCALAR_TYPE_I32));
  EXPECT_EQ(GetStringParam(*invoke_error, 2), "callee argument 0");
  ExpectNoFieldRefParam(*invoke_error, 2);
  ExpectTypeParam(*invoke_error, 3, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
  EXPECT_EQ(invoke_error->origin.provenance,
            LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  EXPECT_EQ(invoke_error->origin_line, 3u);
  EXPECT_EQ(invoke_error->origin_column, 3u);
  ASSERT_EQ(invoke_error->highlights.size(), 1u);
  ExpectHighlightFieldRef(*invoke_error, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                          0);
  EXPECT_EQ(
      invoke_error->source_text.substr(
          invoke_error->highlights[0].start,
          invoke_error->highlights[0].end - invoke_error->highlights[0].start),
      "%arg");
  ASSERT_EQ(invoke_error->related_locations.size(), 1u);
  const auto& definition_note = invoke_error->related_locations[0];
  EXPECT_EQ(definition_note.label, "defined here");
  EXPECT_TRUE(definition_note.has_source_range);
  EXPECT_EQ(definition_note.source_location.start_line, 1u);
  EXPECT_EQ(definition_note.source_location.start_column, 1u);

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, ParsedInvokeResultCountMismatchReportsCalleeLocation) {
  static const char kSource[] =
      "test.decl @callee(%arg: f32) -> (%arg as f32, f32)\n"
      "test.func @main(%arg: f32) -> (f32) {\n"
      "  %result = test.invoke @callee(%arg) : (f32) -> (%arg as f32)\n"
      "  test.yield %result : f32\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_invoke_result_count.loom");
  ASSERT_NE(parsed_module, nullptr);

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      parsed_module, kSource, "parsed_invoke_result_count.loom", &structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* invoke_error = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 2));
  ASSERT_NE(invoke_error, nullptr)
      << "Expected STRUCTURE/002 invoke result-count diagnostic";
  EXPECT_EQ(GetStringParam(*invoke_error, 0), "test.invoke");
  ExpectU32Param(*invoke_error, 1, 1);
  ExpectU32Param(*invoke_error, 2, 2);
  ASSERT_EQ(invoke_error->related_locations.size(), 1u);
  const auto& definition_note = invoke_error->related_locations[0];
  EXPECT_EQ(definition_note.label, "defined here");
  EXPECT_TRUE(definition_note.has_source_range);
  EXPECT_EQ(definition_note.source_location.start_line, 1u);
  EXPECT_EQ(definition_note.source_location.start_column, 1u);

  loom_module_free(parsed_module);
}

//===----------------------------------------------------------------------===//
// Golden output tests
//===----------------------------------------------------------------------===//
//
// Exact-match tests for diagnostic output. One per scenario. When
// formatting changes, the diff shows exactly what changed.

// Scenario: no source resolver, op has no location (post-pass).
// The verifier prints the op and uses it as the source line.
TEST_F(VerifyTest, GoldenCaretNoSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  std::string output = VerifyAndFormatCaret();
  EXPECT_EQ(output,
            "<verifier>:1:1: error [TYPE/004]: result 'result' has type f32,"
            " expected integer\n"
            " 1 | %2 = test.addi %0, %1 : f32\n"
            "   | ^^                         \n"
            "   = help: 'result' must satisfy type constraint 'integer'\n"
            "<verifier>:1:1: error [TYPE/001]: 'lhs' type i32 does not"
            " match 'result' type f32\n"
            " 1 | %2 = test.addi %0, %1 : f32\n"
            "   | ^^             ^^          \n"
            "   = help: Ensure 'lhs' and 'result' have the same"
            " type\n");
}

TEST_F(VerifyTest, GoldenJsonNoSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  std::string output = VerifyAndFormatJson();
  EXPECT_EQ(output,
            "{\"severity\":\"error\",\"error_id\":\"ERR_TYPE_004\","
            "\"domain\":\"TYPE\",\"code\":4,"
            "\"summary\":\"Result type constraint violated.\","
            "\"emitter\":\"verifier\","
            "\"origin\":{\"provenance\":\"printed_ir_fallback\","
            "\"filename\":\"<verifier>\",\"start_line\":1,"
            "\"start_column\":1,\"end_line\":1,\"end_column\":29,"
            "\"start_byte\":0,\"end_byte\":28,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":27,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"%2 = test.addi %0, %1 : f32\"}},"
            "\"source_location\":{\"provenance\":\"printed_ir_fallback\","
            "\"filename\":\"<verifier>\","
            "\"start_line\":1,\"start_column\":1,\"end_line\":1,"
            "\"end_column\":29,\"start_byte\":0,\"end_byte\":28,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":27,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"%2 = test.addi %0, %1 : f32\"}},"
            "\"highlights\":[{\"start_byte\":0,\"end_byte\":2,"
            "\"field\":{\"kind\":\"result\",\"index\":0,\"occurrence\":0},"
            "\"param\":\"result_name\"}],"
            "\"message\":\"result 'result' has type f32, expected integer\","
            "\"fix_hint\":\"'result' must satisfy type constraint 'integer'\","
            "\"params\":{\"result_name\":\"result\","
            "\"actual_type\":\"f32\",\"expected_constraint\":\"integer\"},"
            "\"param_fields\":{\"result_name\":{\"kind\":\"result\","
            "\"index\":0,\"occurrence\":0}}}\n"
            "{\"severity\":\"error\",\"error_id\":\"ERR_TYPE_001\","
            "\"domain\":\"TYPE\",\"code\":1,"
            "\"summary\":\"SameType constraint violated.\","
            "\"emitter\":\"verifier\","
            "\"origin\":{\"provenance\":\"printed_ir_fallback\","
            "\"filename\":\"<verifier>\",\"start_line\":1,"
            "\"start_column\":1,\"end_line\":1,\"end_column\":29,"
            "\"start_byte\":0,\"end_byte\":28,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":27,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"%2 = test.addi %0, %1 : f32\"}},"
            "\"source_location\":{\"provenance\":\"printed_ir_fallback\","
            "\"filename\":\"<verifier>\","
            "\"start_line\":1,\"start_column\":1,\"end_line\":1,"
            "\"end_column\":29,\"start_byte\":0,\"end_byte\":28,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":27,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"%2 = test.addi %0, %1 : f32\"}},"
            "\"highlights\":[{\"start_byte\":0,\"end_byte\":2,"
            "\"field\":{\"kind\":\"result\",\"index\":0,\"occurrence\":0},"
            "\"param\":\"field_b\"},"
            "{\"start_byte\":15,\"end_byte\":17,"
            "\"field\":{\"kind\":\"operand\",\"index\":0,\"occurrence\":0},"
            "\"param\":\"field_a\"}],"
            "\"message\":\"'lhs' type i32 does not match"
            " 'result' type f32\","
            "\"fix_hint\":\"Ensure 'lhs' and 'result'"
            " have the same type\","
            "\"params\":{\"field_a\":\"lhs\",\"type_a\":\"i32\","
            "\"field_b\":\"result\",\"type_b\":\"f32\"},"
            "\"param_fields\":{\"field_a\":{\"kind\":\"operand\","
            "\"index\":0,\"occurrence\":0},\"field_b\":{\"kind\":\"result\","
            "\"index\":0,\"occurrence\":0}}}\n");
}

// Scenario: source resolver provides original source text.
// The diagnostic uses the original source for carets, not the printed op.
static bool FakeSourceResolver(void* user_data, const loom_module_t* module,
                               loom_location_id_t location,
                               loom_source_range_t* out_range) {
  // Pretend every op comes from this source line.
  static const char source[] = "  %result = test.addi %input_a, %input_b : i32";
  (void)module;
  if (location == LOOM_LOCATION_UNKNOWN) return false;
  out_range->provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
  out_range->filename = IREE_SV("model.loom");
  out_range->source = iree_make_cstring_view(source);
  out_range->start = 2;
  out_range->end = 46;  // strlen(source) = 46.
  out_range->start_line = 42;
  out_range->start_column = 3;
  out_range->end_line = 42;
  out_range->end_column = 47;
  return true;
}

TEST_F(VerifyTest, GoldenCaretWithSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0,
                                          /*location=*/1,  // Non-zero so
                                                           // resolver fires.
                                          &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  // Wire up the fake resolver.
  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  auto sink = [](void* user_data, const loom_diagnostic_t* d) -> iree_status_t {
    auto* builder = static_cast<iree_string_builder_t*>(user_data);
    loom_output_stream_t stream;
    loom_output_stream_for_builder(builder, &stream);
    return loom_diagnostic_format(d, &stream);
  };
  memset(&options_, 0, sizeof(options_));
  options_.sink.fn = sink;
  options_.sink.user_data = &output;
  options_.max_errors = 20;
  options_.source_resolver.fn = FakeSourceResolver;

  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
  std::string text(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
  iree_string_builder_deinitialize(&output);

  EXPECT_EQ(
      text,
      "model.loom:42:3: error [TYPE/004]: result 'result' has type f32,"
      " expected integer\n"
      " 42 |   %result = test.addi %input_a, %input_b : i32\n"
      "    |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"  // 44 carets
      "    = help: 'result' must satisfy type constraint 'integer'\n"
      "model.loom:42:3: error [TYPE/001]: 'lhs' type i32 does not"
      " match 'result' type f32\n"
      " 42 |   %result = test.addi %input_a, %input_b : i32\n"
      "    |   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n"  // 44 carets
      "    = help: Ensure 'lhs' and 'result' have the same type\n");
}

TEST_F(VerifyTest, GoldenJsonWithSource) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_F32);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, /*location=*/1, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = result_val;

  iree_string_builder_t output;
  iree_string_builder_initialize(iree_allocator_system(), &output);
  loom_output_stream_t json_stream;
  loom_output_stream_for_builder(&output, &json_stream);
  loom_json_sink_options_t json_options;
  memset(&json_options, 0, sizeof(json_options));
  json_options.stream = &json_stream;
  json_options.type_formatter = {loom_type_format_minimal, nullptr};
  memset(&options_, 0, sizeof(options_));
  options_.sink.fn = loom_diagnostic_json_sink;
  options_.sink.user_data = &json_options;
  options_.max_errors = 20;
  options_.source_resolver.fn = FakeSourceResolver;

  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
  std::string text(iree_string_builder_buffer(&output),
                   iree_string_builder_size(&output));
  iree_string_builder_deinitialize(&output);

  // The JSON sink now serializes the resolved source range, so verifier
  // diagnostics preserve the original file location for IDE consumers.
  EXPECT_EQ(text,
            "{\"severity\":\"error\",\"error_id\":\"ERR_TYPE_004\","
            "\"domain\":\"TYPE\",\"code\":4,"
            "\"summary\":\"Result type constraint violated.\","
            "\"emitter\":\"verifier\","
            "\"origin\":{\"provenance\":\"exact_source\","
            "\"filename\":\"model.loom\",\"start_line\":42,"
            "\"start_column\":3,\"end_line\":42,\"end_column\":47,"
            "\"start_byte\":2,\"end_byte\":46,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":46,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"  %result = test.addi %input_a, %input_b : i32\"}},"
            "\"source_location\":{\"provenance\":\"exact_source\","
            "\"filename\":\"model.loom\","
            "\"start_line\":42,\"start_column\":3,\"end_line\":42,"
            "\"end_column\":47,\"start_byte\":2,\"end_byte\":46,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":46,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"  %result = test.addi %input_a, %input_b : i32\"}},"
            "\"message\":\"result 'result' has type f32, expected integer\","
            "\"fix_hint\":\"'result' must satisfy type constraint 'integer'\","
            "\"params\":{\"result_name\":\"result\","
            "\"actual_type\":\"f32\",\"expected_constraint\":\"integer\"},"
            "\"param_fields\":{\"result_name\":{\"kind\":\"result\","
            "\"index\":0,\"occurrence\":0}}}\n"
            "{\"severity\":\"error\",\"error_id\":\"ERR_TYPE_001\","
            "\"domain\":\"TYPE\",\"code\":1,"
            "\"summary\":\"SameType constraint violated.\","
            "\"emitter\":\"verifier\","
            "\"origin\":{\"provenance\":\"exact_source\","
            "\"filename\":\"model.loom\",\"start_line\":42,"
            "\"start_column\":3,\"end_line\":42,\"end_column\":47,"
            "\"start_byte\":2,\"end_byte\":46,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":46,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"  %result = test.addi %input_a, %input_b : i32\"}},"
            "\"source_location\":{\"provenance\":\"exact_source\","
            "\"filename\":\"model.loom\","
            "\"start_line\":42,\"start_column\":3,\"end_line\":42,"
            "\"end_column\":47,\"start_byte\":2,\"end_byte\":46,"
            "\"excerpt\":{\"start_byte\":0,\"end_byte\":46,"
            "\"truncated_prefix\":false,\"truncated_suffix\":false,"
            "\"text\":\"  %result = test.addi %input_a, %input_b : i32\"}},"
            "\"message\":\"'lhs' type i32 does not match"
            " 'result' type f32\","
            "\"fix_hint\":\"Ensure 'lhs' and 'result'"
            " have the same type\","
            "\"params\":{\"field_a\":\"lhs\",\"type_a\":\"i32\","
            "\"field_b\":\"result\",\"type_b\":\"f32\"},"
            "\"param_fields\":{\"field_a\":{\"kind\":\"operand\","
            "\"index\":0,\"occurrence\":0},\"field_b\":{\"kind\":\"result\","
            "\"index\":0,\"occurrence\":0}}}\n");
}

TEST_F(VerifyTest, StructuredDominanceError) {
  // Create an op referencing undefined values to trigger dominance errors.
  EnterTestFunc(nullptr, 0, nullptr);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_ADDI, 2, 1, 0,
                                          0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_value_id_t lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_operands(op)[0] = lhs;
  loom_op_operands(op)[1] = rhs;
  loom_op_results(op)[0] = result_val;

  DiagnosticCapture structured;

  TerminateFunc();
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);

  // Find a DOMINANCE error.
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 1));
  ASSERT_NE(entry, nullptr) << "Expected a DOMINANCE/001 undefined-value error";
  EXPECT_TRUE(entry->has_source_range)
      << "Dominance error should have print-op fallback source";
  EXPECT_EQ(entry->origin.provenance,
            LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK);
  EXPECT_EQ(entry->source_location.provenance,
            LOOM_SOURCE_PROVENANCE_PRINTED_IR_FALLBACK);
  EXPECT_EQ(entry->filename, "<verifier>");
  EXPECT_NE(entry->source_text.find("test.addi"), std::string::npos)
      << "Printed op source: " << entry->source_text;
}

//===----------------------------------------------------------------------===//
// Tied result validation
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, DuplicateTiedResultIndexDetected) {
  static const char kSource[] =
      "%first, %second = test.invoke @callee(%arg0, %arg1) : (f32, f32) -> "
      "(%arg0 as f32, %arg1 as f32)";
  const char* filename = "duplicate_tied_result.loom";

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type, f32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_location_id_t location =
      AddFileLocation(filename, 1, 1, 1, (uint16_t)sizeof(kSource));
  loom_location_field_span_t field_spans[] = {
      {
          /*.kind=*/LOOM_LOCATION_FIELD_RESULT,
          /*.index=*/0,
          /*.start_line=*/1,
          /*.start_col=*/1,
          /*.end_line=*/1,
          /*.end_col=*/7,
      },
      {
          /*.kind=*/LOOM_LOCATION_FIELD_RESULT,
          /*.index=*/0,
          /*.start_line=*/1,
          /*.start_col=*/9,
          /*.end_line=*/1,
          /*.end_col=*/16,
      },
  };
  IREE_ASSERT_OK(loom_module_attach_location_field_spans(
      module_, location, field_spans, IREE_ARRAYSIZE(field_spans)));

  loom_type_t result_types[] = {f32_type, f32_type};
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
      {/*.result_index=*/0, /*.operand_index=*/1, /*.has_type_change=*/false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), args, 2, result_types, 2, tied_results,
      IREE_ARRAYSIZE(tied_results), location, &op));

  loom_source_entry_t source_entries[] = {{
      /*.source_id=*/FindModuleSourceId(module_, filename),
      /*.source=*/iree_make_cstring_view(kSource),
      /*.filename=*/IREE_SV("duplicate_tied_result.loom"),
  }};
  loom_source_table_resolver_t resolver_data = {
      /*.entries=*/source_entries,
      /*.count=*/IREE_ARRAYSIZE(source_entries),
  };
  options_.source_resolver = {loom_source_table_resolve, &resolver_data};

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 6));
  ASSERT_NE(entry, nullptr)
      << "Expected DOMINANCE/006 duplicate tied-result error";
  ExpectU32Param(*entry, 0, 0);
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0,
                      /*expected_occurrence=*/1);
  EXPECT_EQ(GetStringParam(*entry, 1), "test.invoke");
  EXPECT_EQ(entry->origin.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  ASSERT_EQ(entry->highlights.size(), 1u);
  ExpectHighlightFieldRef(*entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0,
                          /*expected_param_index=*/0,
                          /*expected_occurrence=*/1);
  EXPECT_EQ(entry->source_text.substr(
                entry->highlights[0].start,
                entry->highlights[0].end - entry->highlights[0].start),
            "%second");

  ASSERT_EQ(entry->related_locations.size(), 1u);
  const auto& first_claim_note = entry->related_locations[0];
  EXPECT_EQ(first_claim_note.label, "previously tied here");
  EXPECT_TRUE(first_claim_note.has_source_range);
  ASSERT_EQ(first_claim_note.highlights.size(), 1u);
  ExpectRelatedHighlightFieldRef(*entry, 0, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0,
                                 /*expected_occurrence=*/0);
  EXPECT_EQ(std::string(first_claim_note.source_location.source.data +
                            first_claim_note.highlights[0].start,
                        first_claim_note.highlights[0].end -
                            first_claim_note.highlights[0].start),
            "%first");
}

TEST_F(VerifyTest, DuplicateTiedOperandIndexDetected) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_type_t result_types[] = {f32_type, f32_type};
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
      {/*.result_index=*/1, /*.operand_index=*/0, /*.has_type_change=*/false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), args, 1, result_types, 2, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 7));
  ASSERT_NE(entry, nullptr)
      << "Expected DOMINANCE/007 duplicate tied-operand error";
  ExpectU32Param(*entry, 0, 0);
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                      /*expected_occurrence=*/2);
  EXPECT_EQ(GetStringParam(*entry, 1), "test.invoke");
}

TEST_F(VerifyTest, ParsedDuplicateTiedOperandIndexReportsPreviousTiedLocation) {
  static const char kSource[] =
      "test.decl @callee(%arg: f32) -> (f32, f32)\n"
      "test.func @main(%arg: f32) -> (f32, f32) {\n"
      "  %first, %second = test.invoke @callee(%arg) : (f32) -> "
      "(%arg as f32, %arg as f32)\n"
      "  test.yield %first, %second : f32, f32\n"
      "}\n";
  loom_module_t* parsed_module =
      ParseSourceModule(kSource, "parsed_duplicate_tied_operand.loom");
  ASSERT_NE(parsed_module, nullptr);

  DiagnosticCapture structured;
  loom_verify_result_t result = VerifyParsedSourceModuleStructured(
      parsed_module, kSource, "parsed_duplicate_tied_operand.loom",
      &structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 7));
  ASSERT_NE(entry, nullptr)
      << "Expected DOMINANCE/007 duplicate tied-operand error";
  ExpectU32Param(*entry, 0, 0);
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                      /*expected_occurrence=*/2);
  EXPECT_EQ(GetStringParam(*entry, 1), "test.invoke");
  EXPECT_EQ(entry->origin.provenance, LOOM_SOURCE_PROVENANCE_EXACT_SOURCE);
  ASSERT_EQ(entry->highlights.size(), 1u);
  ExpectHighlightFieldRef(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                          /*expected_param_index=*/0,
                          /*expected_occurrence=*/2);
  EXPECT_EQ(entry->source_text.substr(
                entry->highlights[0].start,
                entry->highlights[0].end - entry->highlights[0].start),
            "%arg");

  ASSERT_EQ(entry->related_locations.size(), 1u);
  const auto& first_claim_note = entry->related_locations[0];
  EXPECT_EQ(first_claim_note.label, "previously tied here");
  EXPECT_TRUE(first_claim_note.has_source_range);
  ASSERT_EQ(first_claim_note.highlights.size(), 1u);
  ExpectRelatedHighlightFieldRef(*entry, 0, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0,
                                 /*expected_occurrence=*/1);
  EXPECT_LT(first_claim_note.highlights[0].start, entry->highlights[0].start);
  EXPECT_EQ(std::string(first_claim_note.source_location.source.data +
                            first_claim_note.highlights[0].start,
                        first_claim_note.highlights[0].end -
                            first_claim_note.highlights[0].start),
            "%arg");

  loom_module_free(parsed_module);
}

TEST_F(VerifyTest, AmbiguousRepeatedOperandValueDetected) {
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {f32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_value_id_t operands[] = {args[0], args[0]};
  loom_type_t result_types[] = {f32_type};
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/1, /*.has_type_change=*/false},
  };
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_invoke_build(
      &builder_, DefineTestCallee(), operands, IREE_ARRAYSIZE(operands),
      result_types, 1, tied_results, IREE_ARRAYSIZE(tied_results),
      LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_DOMINANCE, 8));
  ASSERT_NE(entry, nullptr)
      << "Expected DOMINANCE/008 ambiguous tied operand-value error";
  ExpectU32Param(*entry, 0, 1);
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 1);
  EXPECT_EQ(GetStringParam(*entry, 1), "test.invoke");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_OPERAND, 1);
}

TEST_F(VerifyTest, FuncDefTiedResultUsesEntryBlockArgsWithoutConsumingThem) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("identity"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
  };
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_func_build(&builder_, 0, 0, 0, callee, &f32_type, 1,
                                      &f32_type, 1, tied_results,
                                      IREE_ARRAYSIZE(tied_results), nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &func_op));

  loom_region_t* body = loom_test_func_body(func_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(body->block_count, 1u);
  ASSERT_EQ(loom_region_entry_arg_count(body), 1u);
  loom_builder_set_block(&builder_, loom_region_entry_block(body));

  loom_value_id_t arg_id = loom_region_entry_arg_id(body, 0);
  loom_op_t* return_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &arg_id, 1,
                                       LOOM_LOCATION_UNKNOWN, &return_op));

  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "" : collector_.errors[0]);
}

TEST_F(VerifyTest, FuncDeclTiedResultUsesSignatureOperandsWithoutDominance) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_intern_string(
      &builder_, IREE_SV("extern_identity"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_tied_result_t tied_results[] = {
      {/*.result_index=*/0, /*.operand_index=*/0, /*.has_type_change=*/false},
  };
  loom_op_t* func_op = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(
      &builder_, 0, 0, 0, callee, &f32_type, 1, &f32_type, 1, tied_results,
      IREE_ARRAYSIZE(tied_results), LOOM_LOCATION_UNKNOWN, &func_op));

  EXPECT_EQ(func_op->operand_count, 1u);
  EXPECT_EQ(func_op->tied_result_count, 1u);

  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "" : collector_.errors[0]);
}

TEST_F(VerifyTest, RejectsNonLocalSymbolRef) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("foreign"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/1, /*.symbol_id=*/symbol_id};

  loom_op_t* decl_op = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &decl_op));

  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 4));
  ASSERT_NE(entry, nullptr) << "Expected SYMBOL/004 non-local symbol ref error";
  ExpectU32Param(*entry, 0, 1);
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, 0);
}

TEST_F(VerifyTest, RejectsDuplicateSymbolDefinition) {
  loom_string_id_t name_id = LOOM_STRING_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_intern_string(&builder_, IREE_SV("duplicate"), &name_id));
  uint16_t symbol_id = LOOM_SYMBOL_ID_INVALID;
  IREE_ASSERT_OK(loom_module_add_symbol(module_, name_id, &symbol_id));
  loom_symbol_ref_t callee = {/*.module_id=*/0, /*.symbol_id=*/symbol_id};

  loom_op_t* first_decl = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &first_decl));
  EXPECT_EQ(module_->symbols.entries[symbol_id].defining_op, first_decl);

  loom_op_t* second_decl = nullptr;
  IREE_ASSERT_OK(loom_test_decl_build(&builder_, 0, 0, 0, callee, nullptr, 0,
                                      nullptr, 0, nullptr, 0,
                                      LOOM_LOCATION_UNKNOWN, &second_decl));
  EXPECT_EQ(module_->symbols.entries[symbol_id].defining_op, first_decl);

  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SYMBOL, 5));
  ASSERT_NE(entry, nullptr)
      << "Expected SYMBOL/005 duplicate symbol definition error";
  EXPECT_EQ(GetStringParam(*entry, 0), "duplicate");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_ATTRIBUTE, 0);
}

//===----------------------------------------------------------------------===//
// SSA encoding reference validation
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, ValidSsaEncodingPasses) {
  // Create a test.func with an encoding arg. The encoding value ID is
  // needed to construct the tile type, so we pass just the encoding
  // type to EnterTestFunc and add the tile arg manually afterwards.
  loom_type_t encoding_type = loom_type_encoding();
  loom_type_t arg_types[] = {encoding_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t enc_value = args[0];

  // Create a tile type that references the encoding via SSA.
  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = enc_value;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Add a second block arg with the encoded tile type.
  loom_value_id_t input = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(&builder_, builder_.ip.block,
                                               tile_type, &input));

  // Build test.convert (accepts ANY types) with the encoded tile.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, input, tile_type,
                                         LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, SsaEncodingOutOfRange) {
  // Create a tile type with an SSA encoding value_id that is way out of range.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = 9999;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build test.convert with the bad encoding type on the result.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[0], tile_type,
                                         LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 3));
  ASSERT_NE(entry, nullptr) << "Expected ENCODING/003 out-of-range error";
  EXPECT_EQ(GetStringParam(*entry, 0), "result");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  ExpectU32Param(*entry, 1, 9999);
}

TEST_F(VerifyTest, SsaEncodingNotDefined) {
  // Create a test.func with an i32 arg.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  // Create an encoding value but do NOT add it as a block arg --
  // it won't be in scope when used.
  loom_type_t encoding_type = loom_type_encoding();
  loom_value_id_t enc_value = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(
      loom_builder_define_value(&builder_, encoding_type, &enc_value));

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = enc_value;
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build a test.convert with a normal input but the bad operand type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[0], i32_type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  // Inject the undefined encoding ref into the operand's type.
  IREE_ASSERT_OK(loom_module_set_value_type(module_, args[0], tile_type));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 4));
  ASSERT_NE(entry, nullptr) << "Expected ENCODING/004 not-defined error";
  EXPECT_EQ(entry->origin.provenance,
            LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(entry->source_location.provenance,
            LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(GetStringParam(*entry, 0), "block arg 0");
  ExpectNoFieldRefParam(*entry, 0);
}

TEST_F(VerifyTest, SsaEncodingWrongType) {
  // Create two i32 args: one to use as a wrong-type encoding reference,
  // one as the input to convert.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_type_t tile_type =
      loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32, 256, 0);
  tile_type.encoding_id = args[0];
  tile_type.encoding_flags = LOOM_ENCODING_FLAG_SSA;

  // Build a test.convert with a normal input but inject the bad type.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[1], i32_type,
                                         LOOM_LOCATION_UNKNOWN, &op));
  // Inject the wrong-type encoding ref into the operand's type.
  IREE_ASSERT_OK(loom_module_set_value_type(module_, args[1], tile_type));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_ENCODING, 5));
  ASSERT_NE(entry, nullptr) << "Expected ENCODING/005 wrong-type error";
  EXPECT_EQ(entry->origin.provenance,
            LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(entry->source_location.provenance,
            LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE);
  EXPECT_EQ(GetStringParam(*entry, 0), "block arg 1");
  ExpectNoFieldRefParam(*entry, 0);
  ExpectTypeParam(*entry, 2, i32_type);
}

TEST_F(VerifyTest, RejectsRankZeroVectorType) {
  loom_type_t vector_type =
      loom_type_shaped_0d(LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, 0);
  loom_type_t arg_types[] = {vector_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 10));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/010 malformed vector error";
  EXPECT_EQ(GetStringParam(*entry, 0), "block arg 0");
  ExpectNoFieldRefParam(*entry, 0);
  ExpectTypeParam(*entry, 1, vector_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "vector_rank_zero");
}

TEST_F(VerifyTest, RejectsOutOfRangeTypeKind) {
  loom_type_t invalid_type = {0};
  invalid_type.header =
      loom_type_make_header((loom_type_kind_t)99, (loom_scalar_type_t)0, 0, 0);
  loom_type_t arg_types[] = {invalid_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 10));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/010 invalid type kind error";
  EXPECT_EQ(GetStringParam(*entry, 0), "block arg 0");
  ExpectNoFieldRefParam(*entry, 0);
  ExpectTypeParam(*entry, 1, invalid_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "type_kind_out_of_range");
}

TEST_F(VerifyTest, RejectsVectorEncodingAttachment) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_type_t vector_type = loom_type_shaped_1d(
      LOOM_TYPE_VECTOR, LOOM_SCALAR_TYPE_F32, loom_dim_pack_static(4), 7);
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_convert_build(&builder_, args[0], vector_type,
                                         LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 10));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/010 vector encoding error";
  EXPECT_EQ(GetStringParam(*entry, 0), "result");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  ExpectTypeParam(*entry, 1, vector_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "vector_encoding_attachment");
}

//===----------------------------------------------------------------------===//
// Variadic field constraint checks
//===----------------------------------------------------------------------===//

TEST_F(VerifyTest, VariadicSameTypeConstraintPasses) {
  // test.reduce has SameType("inputs", "result") where inputs is variadic.
  // All i32 → passes.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type, i32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, args, 3, i32_type,
                                        LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, VariadicSameTypeMismatchInVariadic) {
  // test.reduce with inputs [i32, f32, i32] and result i32.
  // The second input has type f32 → constraint violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {i32_type, f32_type, i32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  // Manually build to control operand types precisely.
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_REDUCE, 3, 1,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_operands(op)[2] = args[2];
  loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
  loom_op_results(op)[0] = result_val;

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(entry, nullptr)
      << "Expected TYPE/001 mismatch within variadic inputs";
  EXPECT_EQ(GetStringParam(*entry, 0), "inputs[0]");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*entry, 1, i32_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "inputs[1]");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_OPERAND, 1);
  ExpectTypeParam(*entry, 3, f32_type);
}

TEST_F(VerifyTest, SegmentedSameTypeConstraintPasses) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type, i32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_value_id_t lhs[] = {args[1]};
  loom_value_id_t rhs[] = {args[2]};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, args[0], LOOM_VALUE_ID_INVALID, lhs, IREE_ARRAYSIZE(lhs),
      rhs, IREE_ARRAYSIZE(rhs), i32_type, LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u)
      << (collector_.errors.empty() ? "" : collector_.errors[0]);
}

TEST_F(VerifyTest, SegmentedSameTypeMismatchNamesSegmentElement) {
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t arg_types[] = {i32_type, i32_type, f32_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);

  loom_value_id_t lhs[] = {args[1]};
  loom_value_id_t rhs[] = {args[2]};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_segmented_build(
      &builder_, 0, args[0], LOOM_VALUE_ID_INVALID, lhs, IREE_ARRAYSIZE(lhs),
      rhs, IREE_ARRAYSIZE(rhs), i32_type, LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/001 mismatch in segmented span";
  EXPECT_EQ(GetStringParam(*entry, 0), "root");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*entry, 1, i32_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "rhs[0]");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_OPERAND, 3);
  ExpectTypeParam(*entry, 3, f32_type);
}

TEST_F(VerifyTest, VariadicSameTypeMismatchHighlightsWideOperandIndex) {
  constexpr uint16_t kInputCount = 65;
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  std::vector<loom_type_t> arg_types(kInputCount, i32_type);
  arg_types.back() = f32_type;
  std::vector<loom_value_id_t> args(kInputCount);
  EnterTestFunc(arg_types.data(), arg_types.size(), args.data());

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_REDUCE,
                                          kInputCount, 1, 0, 0, 0,
                                          LOOM_LOCATION_UNKNOWN, &op));
  for (uint16_t i = 0; i < kInputCount; ++i) {
    loom_op_operands(op)[i] = args[i];
  }
  loom_op_results(op)[0] = DefineValue(LOOM_SCALAR_TYPE_I32);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(entry, nullptr)
      << "Expected TYPE/001 mismatch for the 65th variadic operand";
  EXPECT_EQ(GetStringParam(*entry, 0), "inputs[0]");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  EXPECT_EQ(GetStringParam(*entry, 2), "inputs[64]");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_OPERAND, 64);
  ASSERT_EQ(entry->highlights.size(), 2u);
  ExpectHighlightFieldRef(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0, 0);
  ExpectHighlightFieldRef(*entry, 1, LOOM_DIAGNOSTIC_FIELD_OPERAND, 64, 2);
  EXPECT_LT(entry->highlights[1].start, entry->highlights[1].end);

  std::string output = VerifyAndFormatJson();
  static const std::regex kWideOperandHighlightRegex(
      "\"start_byte\":[0-9]+,\"end_byte\":[0-9]+,"
      "\"field\":\\{\"kind\":\"operand\",\"index\":64,"
      "\"occurrence\":0\\},"
      "\"param\":\"field_b\"");
  EXPECT_TRUE(std::regex_search(output, kWideOperandHighlightRegex))
      << "Expected JSON highlight for operand index 64, got: " << output;
}

TEST_F(VerifyTest, VariadicSameTypeMismatchAgainstResult) {
  // test.reduce with inputs [i32, i32] and result f32.
  // All inputs match each other but not the result → constraint violation.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type, i32_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_REDUCE, 2, 1,
                                          0, 0, 0, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = args[0];
  loom_op_operands(op)[1] = args[1];
  loom_op_results(op)[0] = DefineValue(LOOM_SCALAR_TYPE_F32);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 1));
  ASSERT_NE(entry, nullptr)
      << "Expected TYPE/001 mismatch between variadic input and result";
  EXPECT_EQ(GetStringParam(*entry, 0), "inputs");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectTypeParam(*entry, 1, i32_type);
  EXPECT_EQ(GetStringParam(*entry, 2), "result");
  ExpectFieldRefParam(*entry, 2, LOOM_DIAGNOSTIC_FIELD_RESULT, 0);
  ExpectTypeParam(*entry, 3, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
}

TEST_F(VerifyTest, VariadicSameTypeSingleInput) {
  // test.reduce with a single input [i32] and result i32.
  // Should pass: one variadic element matching the result.
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_type_t arg_types[] = {i32_type};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_reduce_build(&builder_, args, 1, i32_type,
                                        LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

//===----------------------------------------------------------------------===//
// Adversarial input tests
//===----------------------------------------------------------------------===//
// These tests exercise the verifier on pathological IR that a malicious
// or buggy producer could construct. They verify that the verifier
// handles resource limits gracefully (returns RESOURCE_EXHAUSTED)
// and that arena allocation/growth works on large inputs.

TEST_F(VerifyTest, LargeModuleManyValues) {
  // Create a module with many values and ops to exercise the arena
  // allocation path and defined_stack dynamic growth. With 4000 values,
  // the initial defined_stack capacity (value_count/4 = 1000) will be
  // exceeded, forcing iree_arena_grow_array to fire.
  constexpr int kValueCount = 4000;
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  std::vector<loom_type_t> arg_types(kValueCount, i32_type);
  std::vector<loom_value_id_t> args(kValueCount);
  EnterTestFunc(arg_types.data(), kValueCount, args.data());

  // Build ops that use pairs of these values.
  for (int i = 0; i + 1 < kValueCount; i += 2) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[i], args[i + 1],
                                        i32_type, LOOM_LOCATION_UNKNOWN, &op));
  }

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, DeepNestingAtLimit) {
  // Build IR nested to exactly LOOM_VERIFY_MAX_SCOPE_DEPTH (32) levels.
  // The module body is scope 0, the test.func body is scope 1, so we
  // can nest 30 additional isolated_region ops (scopes 2..31).
  constexpr int kNestingDepth = 30;

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  EnterTestFunc(nullptr, 0, nullptr);

  // Build nested isolated_region ops. Each has one result and one body
  // region. We use save/restore to build inside each region.
  loom_builder_ip_t saved_ips[kNestingDepth];
  loom_op_t* ops[kNestingDepth];

  for (int depth = 0; depth < kNestingDepth; ++depth) {
    loom_type_t result_types[] = {i32_type};
    IREE_ASSERT_OK(
        loom_test_isolated_region_build(&builder_, result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &ops[depth]));
    // Switch to building inside the region body.
    loom_region_t* body = loom_test_isolated_region_body(ops[depth]);
    ASSERT_NE(body, nullptr);
    ASSERT_GT(body->block_count, 0);
    saved_ips[depth] = loom_builder_enter_region(&builder_, ops[depth], body);
  }

  // Restore all the way back out.
  for (int depth = kNestingDepth - 1; depth >= 0; --depth) {
    loom_builder_restore(&builder_, saved_ips[depth]);
  }

  // The nested isolated_region ops don't have proper terminators, so
  // the verifier reports structural errors. The point of this test is
  // that at the LOOM_VERIFY_MAX_SCOPE_DEPTH limit the verifier
  // completes the walk without RESOURCE_EXHAUSTED.
  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_OK(loom_verify_module(module_, &options_, &result));
}

TEST_F(VerifyTest, DeepNestingExceedsLimit) {
  // Build IR nested to LOOM_VERIFY_MAX_SCOPE_DEPTH + 1 levels.
  // The verifier should return RESOURCE_EXHAUSTED (not crash, not
  // silently succeed). Module body is scope 0, func body is scope 1,
  // then 31 isolated_region regions push scopes 2..32 (exceeding 32).
  constexpr int kNestingDepth = 31;

  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  EnterTestFunc(nullptr, 0, nullptr);

  loom_builder_ip_t saved_ips[kNestingDepth];
  loom_op_t* ops[kNestingDepth];

  for (int depth = 0; depth < kNestingDepth; ++depth) {
    loom_type_t result_types[] = {i32_type};
    IREE_ASSERT_OK(
        loom_test_isolated_region_build(&builder_, result_types, 1, NULL, 0,
                                        LOOM_LOCATION_UNKNOWN, &ops[depth]));
    loom_region_t* body = loom_test_isolated_region_body(ops[depth]);
    ASSERT_NE(body, nullptr);
    ASSERT_GT(body->block_count, 0);
    saved_ips[depth] = loom_builder_enter_region(&builder_, ops[depth], body);
  }

  for (int depth = kNestingDepth - 1; depth >= 0; --depth) {
    loom_builder_restore(&builder_, saved_ips[depth]);
  }

  TerminateFunc();
  loom_verify_result_t result = {};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_RESOURCE_EXHAUSTED,
                        loom_verify_module(module_, &options_, &result));
}

TEST_F(VerifyTest, DefinedStackGrowsPastInitialCapacity) {
  // The defined_stack starts at value_count/4 capacity. If we create
  // a module with N values but define them all in a single scope (no
  // nesting), the stack must grow to hold all N. This exercises
  // iree_arena_grow_array.
  //
  // 80 block args → value_count starts at ~80, initial capacity =
  // 80/4 = 20. We define 80 values in one scope → growth required.
  constexpr int kArgCount = 80;
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  std::vector<loom_type_t> arg_types(kArgCount, i32_type);
  std::vector<loom_value_id_t> args(kArgCount);
  EnterTestFunc(arg_types.data(), kArgCount, args.data());

  // Build some ops so the results also go onto the defined stack.
  for (int i = 0; i + 1 < kArgCount; i += 2) {
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_test_addi_build(&builder_, args[i], args[i + 1],
                                        i32_type, LOOM_LOCATION_UNKNOWN, &op));
  }

  TerminateFunc();
  auto result = Verify();
  // Should succeed -- the stack grew dynamically.
  EXPECT_EQ(result.error_count, 0u);
}

TEST_F(VerifyTest, MaxErrorsStopsWalk) {
  // Set max_errors to 2, build IR with many errors, verify that
  // the verifier stops after 2 and doesn't waste time on the rest.
  options_.max_errors = 2;
  EnterTestFunc(nullptr, 0, nullptr);

  // Build 10 ops with undefined operands (each generates a dominance error).
  for (int i = 0; i < 10; ++i) {
    loom_value_id_t fake_lhs = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_value_id_t fake_rhs = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_value_id_t result_val = DefineValue(LOOM_SCALAR_TYPE_I32);
    loom_op_t* op = nullptr;
    IREE_ASSERT_OK(loom_builder_allocate_op(
        &builder_, LOOM_OP_TEST_ADDI, 2, 1, /*region_count=*/0,
        /*tied_result_count=*/0, /*attribute_count=*/0, LOOM_LOCATION_UNKNOWN,
        &op));
    loom_op_operands(op)[0] = fake_lhs;
    loom_op_operands(op)[1] = fake_rhs;
    loom_op_results(op)[0] = result_val;
  }

  TerminateFunc();
  auto result = Verify();
  // Should have exactly 2 errors (stopped at limit).
  EXPECT_EQ(result.error_count, 2u);
}

//===----------------------------------------------------------------------===//
// Constraint relation coverage: all 8 relations
//===----------------------------------------------------------------------===//
// These tests exercise each constraint relation end-to-end (positive
// and/or negative paths) to ensure the verifier catches violations.

// --- ALL_SAME (relation 1): input tiles must have identical shapes ---

TEST_F(VerifyTest, AllSameViolationDifferentShapes) {
  // test.map has AllShapesMatch("inputs"). Build with two inputs of
  // different tile shapes → violation.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);
  loom_type_t tile8 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile4, tile8};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SHAPE, 3));
  ASSERT_NE(entry, nullptr) << "Expected SHAPE/003 AllSame diagnostic";
  ExpectTypeParam(*entry, 0, tile4);
  ExpectU32Param(*entry, 1, 1);
  ExpectTypeParam(*entry, 2, tile8);
}

TEST_F(VerifyTest, AllSamePassesIdenticalShapes) {
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4, tile4};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield the first block arg (the builder auto-creates f32 block args
  // for the tile inputs).
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- REGION_ARG_COUNT (relation 4): block args count must match inputs ---

TEST_F(VerifyTest, RegionArgCountViolation) {
  // Build test.map with 2 inputs, then manually change the region to
  // have 3 block args instead of 2.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4, tile4};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0], args[1]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 2, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Add a third block arg to create a count mismatch.
  loom_region_t* body = loom_test_map_body(op);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t extra_arg = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_block_arg(
      &builder_, loom_region_entry_block(body), scalar_f32, &extra_arg));

  // Add a yield using the first block arg so we don't trip unrelated checks.
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 7));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/007 region-arg-count diagnostic";
  ExpectU32Param(*entry, 0, 3);
  ExpectU32Param(*entry, 1, 2);
}

// --- REGION_ARG_MATCH (relation 5): block arg types must match element types
// ---

TEST_F(VerifyTest, RegionArgMatchViolation) {
  // Build test.map with f32 tile input. The builder auto-creates block
  // args with the element type (f32). We then manually change the first
  // block arg's type to i32.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Change the block arg type to i32.
  loom_region_t* body = loom_test_map_body(op);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  IREE_ASSERT_OK(loom_module_set_value_type(
      module_, loom_region_entry_arg_id(body, 0), i32_type));

  // Yield the (now i32) block arg. This yields an i32 for a tile<4xf32>
  // result, which will also trigger a yield type mismatch, but the specific
  // error we care about is the block arg type mismatch.
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 8));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/008 region-arg-match diagnostic";
  ExpectU32Param(*entry, 0, 0);
  ExpectTypeParam(*entry, 1, i32_type);
  ExpectTypeParam(*entry, 2, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
}

// --- YIELD_COUNT (relation 6): yield operands count must match results ---

TEST_F(VerifyTest, YieldCountViolation) {
  // Build test.map with 1 result, but yield 2 values.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield 2 values when result count is 1.
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_type_t scalar_f32 = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_value_id_t val_a = LOOM_VALUE_ID_INVALID;
  loom_value_id_t val_b = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, scalar_f32, &val_a));
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, scalar_f32, &val_b));
  loom_value_id_t yield_vals[] = {val_a, val_b};
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, yield_vals, 2,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 8));
  ASSERT_NE(entry, nullptr) << "Expected STRUCTURE/008 yield-count diagnostic";
  ExpectU32Param(*entry, 0, 2);
  ExpectU32Param(*entry, 1, 1);
}

TEST_F(VerifyTest, YieldCountSkipsMissingMaterializedTerminator) {
  // test.map has one result but the body is empty. Structural verification
  // rejects the missing materialized terminator before yield-count relations
  // inspect terminator operands.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* map_op = nullptr;
  IREE_ASSERT_OK(loom_test_map_build(&builder_, args, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &map_op));
  ASSERT_NE(map_op, nullptr);
  loom_region_t* body = loom_test_map_body(map_op);
  ASSERT_NE(body, nullptr);
  ASSERT_EQ(loom_region_entry_block(body)->op_count, 0u);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_STRUCTURE, 5));
  ASSERT_NE(entry, nullptr)
      << "Expected STRUCTURE/005 missing-terminator diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "test.map");
  ExpectU32Param(*entry, 1, 0);
  EXPECT_EQ(FindDiagnostic(structured, loom_error_def_lookup(
                                           LOOM_ERROR_DOMAIN_STRUCTURE, 8)),
            nullptr);
}

// --- YIELD_MATCH (relation 7): yield types must match result types ---

TEST_F(VerifyTest, YieldMatchViolation) {
  // Build test.map with result tile<4xf32>, but yield an i32 value.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield an i32 when result is tile<f32>.
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);
  loom_value_id_t yield_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, i32_type, &yield_val));
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 9));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/009 yield-match diagnostic";
  ExpectTypeParam(*entry, 0, i32_type);
  ExpectTypeParam(*entry, 1, loom_type_scalar(LOOM_SCALAR_TYPE_F32));
}

TEST_F(VerifyTest, YieldMatchViolationReportsResultElementTypeParam) {
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);
  loom_type_t f32_type = loom_type_scalar(LOOM_SCALAR_TYPE_F32);
  loom_type_t i32_type = loom_type_scalar(LOOM_SCALAR_TYPE_I32);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, i32_type, &yield_val));
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  DiagnosticCapture structured;

  TerminateFunc();
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);

  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_TYPE, 9));
  ASSERT_NE(entry, nullptr) << "Expected TYPE/009 yield-match diagnostic";
  ExpectTypeParam(*entry, 0, i32_type);
  ExpectTypeParam(*entry, 1, f32_type);
}

// --- COUNT_MATCHES_RANK (relation 2): offset count must match source rank ---

TEST_F(VerifyTest, CountMatchesRankViolation) {
  // test.slice with rank-2 source but only 1 dynamic offset → violation.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_type_t arg_types[] = {tile_2d, index_type};
  loom_value_id_t args[2];
  EnterTestFunc(arg_types, 2, args);
  loom_value_id_t source = args[0];
  loom_value_id_t offset = args[1];

  // Provide 1 dynamic offset for a rank-2 source. static_offsets has 2
  // entries so the builder creates a valid op structurally.
  int64_t static_offsets[] = {0, 0};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, &offset, 1,
                                       static_offsets, 2, tile_2d, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 1));
  ASSERT_NE(entry, nullptr)
      << "Expected SUBRANGE/001 count-matches-rank diagnostic";
  EXPECT_EQ(GetStringParam(*entry, 0), "source");
  ExpectFieldRefParam(*entry, 0, LOOM_DIAGNOSTIC_FIELD_OPERAND, 0);
  ExpectU32Param(*entry, 1, 1);
  ExpectI64Param(*entry, 2, 2);
}

TEST_F(VerifyTest, CountMatchesRankPasses) {
  // test.slice with rank-2 source and 2 dynamic offsets → passes.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);

  loom_type_t arg_types[] = {tile_2d, index_type, index_type};
  loom_value_id_t args[3];
  EnterTestFunc(arg_types, 3, args);
  loom_value_id_t source = args[0];
  loom_value_id_t offset_a = args[1];
  loom_value_id_t offset_b = args[2];

  int64_t static_offsets[] = {0, 0};
  loom_value_id_t offsets[] = {offset_a, offset_b};
  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_test_slice_build(&builder_, source, offsets, 2,
                                       static_offsets, 2, tile_2d, NULL, 0,
                                       LOOM_LOCATION_UNKNOWN, &op));

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- ATTR_IN_RANGE_RANK (relation 3): dim index must be in [0, rank) ---

TEST_F(VerifyTest, AttrInRangeRankViolation) {
  // test.dim with rank-1 source but dim_index=5 → out of bounds.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(5);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 2));
  ASSERT_NE(entry, nullptr)
      << "Expected SUBRANGE/002 dim-index-out-of-bounds diagnostic";
  ExpectI64Param(*entry, 0, 5);
  ExpectI64Param(*entry, 1, 1);
}

TEST_F(VerifyTest, AttrInRangeRankNegativeIndex) {
  // test.dim with rank-2 source but dim_index=-1 → out of bounds.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile_2d};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(-1);

  TerminateFunc();
  DiagnosticCapture structured;
  auto result = VerifyStructured(&structured);
  EXPECT_GT(result.error_count, 0u);
  const CapturedDiagnostic* entry = FindDiagnostic(
      structured, loom_error_def_lookup(LOOM_ERROR_DOMAIN_SUBRANGE, 2));
  ASSERT_NE(entry, nullptr)
      << "Expected SUBRANGE/002 negative dim-index diagnostic";
  ExpectI64Param(*entry, 0, -1);
  ExpectI64Param(*entry, 1, 2);
}

TEST_F(VerifyTest, AttrInRangeRankPasses) {
  // test.dim with rank-2 source and dim_index=1 → valid.
  loom_type_t tile_2d =
      loom_type_shaped_2d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                          loom_dim_pack_static(4), loom_dim_pack_static(8), 0);

  loom_type_t arg_types[] = {tile_2d};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);
  loom_value_id_t source = args[0];
  loom_type_t index_type = loom_type_scalar(LOOM_SCALAR_TYPE_INDEX);
  loom_value_id_t result_val = LOOM_VALUE_ID_INVALID;
  IREE_ASSERT_OK(loom_builder_define_value(&builder_, index_type, &result_val));

  loom_op_t* op = nullptr;
  IREE_ASSERT_OK(loom_builder_allocate_op(&builder_, LOOM_OP_TEST_DIM, 1, 1, 0,
                                          0, 1, LOOM_LOCATION_UNKNOWN, &op));
  loom_op_operands(op)[0] = source;
  loom_op_results(op)[0] = result_val;
  loom_op_attrs(op)[0] = loom_attr_i64(1);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

// --- Pairwise EQ positive: test.map with correct types ---

TEST_F(VerifyTest, PairwiseEqPassesTestMap) {
  // Build a complete, valid test.map with yield.
  loom_type_t tile4 = loom_type_shaped_1d(LOOM_TYPE_TILE, LOOM_SCALAR_TYPE_F32,
                                          loom_dim_pack_static(4), 0);

  loom_type_t arg_types[] = {tile4};
  loom_value_id_t args[1];
  EnterTestFunc(arg_types, 1, args);

  loom_op_t* op = nullptr;
  loom_value_id_t inputs[] = {args[0]};
  IREE_ASSERT_OK(loom_test_map_build(&builder_, inputs, 1, tile4, NULL, 0,
                                     LOOM_LOCATION_UNKNOWN, &op));

  // Yield the body's block arg (f32 element type matching the tile result).
  loom_region_t* body = loom_test_map_body(op);
  loom_builder_ip_t saved = loom_builder_enter_region(&builder_, op, body);
  loom_value_id_t yield_val = loom_region_entry_arg_id(body, 0);
  loom_op_t* yield_op = nullptr;
  IREE_ASSERT_OK(loom_test_yield_build(&builder_, &yield_val, 1,
                                       LOOM_LOCATION_UNKNOWN, &yield_op));
  loom_builder_restore(&builder_, saved);

  TerminateFunc();
  auto result = Verify();
  EXPECT_EQ(result.error_count, 0u);
}

}  // namespace
}  // namespace loom
