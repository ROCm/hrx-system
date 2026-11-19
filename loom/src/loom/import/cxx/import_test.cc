// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/import.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/printer.h"
#include "loom/import/cxx/source/catalog.h"
#include "loom/import/cxx/unit.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/test/descriptors.h"
#include "loom/testing/diagnostic_matchers.h"

namespace {

class ImportTest : public ::testing::Test {
 protected:
  void SetUp() override {
    iree_arena_block_pool_initialize(64 * 1024, iree_allocator_system(),
                                     &pool_);
    loom_context_initialize(iree_allocator_system(), &context_);
    IREE_ASSERT_OK(loom_op_registry_register_all_dialects(&context_));
    IREE_ASSERT_OK(loom_context_finalize(&context_));
    loom_cxx_import_options_initialize(&options_);
    options_.diagnostic_sink = {CaptureDiagnostic, this};
  }

  void TearDown() override {
    loom_module_free(module_);
    loom_cxx_preamble_release(preamble_);
    loom_context_deinitialize(&context_);
    iree_arena_block_pool_deinitialize(&pool_);
  }

  iree_status_t Import(iree_string_view_t source) {
    loom_module_free(module_);
    module_ = nullptr;
    return loom_cxx_import(source, IREE_SV("/app/source.cpp"), &context_,
                           &pool_, &options_, iree_allocator_system(),
                           &module_);
  }

  std::string Print() {
    const loom_text_print_options_t options = {
        .flags = LOOM_TEXT_PRINT_DEFAULT,
        .low_asm_environment = options_.low_asm_environment,
    };
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder_with_options(
        module_, &builder, &options));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  static iree_status_t CaptureDiagnostic(void* user_data,
                                         const loom_diagnostic_t* diagnostic) {
    auto& self = *static_cast<ImportTest*>(user_data);
    ++self.diagnostic_count_;
    if (diagnostic->severity == LOOM_DIAGNOSTIC_ERROR) ++self.error_count_;
    self.diagnostic_ref_ = loom_error_def_ref(diagnostic->error);
    auto filename = diagnostic->source_location.filename;
    self.diagnostic_filename_.assign(filename.data ? filename.data : "",
                                     filename.size);
    auto source = diagnostic->source_location.source;
    self.diagnostic_source_.assign(source.data ? source.data : "", source.size);
    return loom_diagnostic_stderr_sink(nullptr, diagnostic);
  }

  static iree_status_t ProvideSource(void* user_data, iree_string_view_t path,
                                     bool* out_found,
                                     iree_string_view_t* out_source) {
    auto& self = *static_cast<ImportTest*>(user_data);
    auto key = std::string(path.data, path.size);
    ++self.source_requests_[key];
    auto found = self.headers_.find(key);
    *out_found = found != self.headers_.end();
    if (*out_found) {
      *out_source =
          iree_make_string_view(found->second.data(), found->second.size());
    }
    return iree_ok_status();
  }

  // Shared native dialect context, finalized before import.
  loom_context_t context_ = {};
  // Arena storage outliving every imported module.
  iree_arena_block_pool_t pool_ = {};
  // Owned import output released before its context and pool.
  loom_module_t* module_ = nullptr;
  // Per-invocation source configuration.
  loom_cxx_import_options_t options_ = {};
  // Optional immutable preamble released after every import output.
  loom_cxx_preamble_t* preamble_ = nullptr;
  // Number of callbacks delivered by the source diagnostic boundary.
  int diagnostic_count_ = 0;
  // Number of those callbacks with error severity.
  int error_count_ = 0;
  // Structured identity of the most recently delivered diagnostic.
  loom_error_ref_t diagnostic_ref_ = LOOM_ERROR_REF_NONE;
  // Copied source identity retained after the frontend is destroyed.
  std::string diagnostic_filename_;
  // Copied source bytes retained after the frontend is destroyed.
  std::string diagnostic_source_;
  // Provider-owned header text, independent of frontend storage.
  std::map<std::string, std::string> headers_;
  // Lookup counts witness per-invocation source reuse.
  std::map<std::string, int> source_requests_;
};

TEST_F(ImportTest, DerivedAssumptionsPublishExpressionFacts) {
  IREE_ASSERT_OK(
      Import(IREE_SV("[[loom::assume]] void assume(bool); "
                     "void entry(unsigned value, unsigned limit) { "
                     "assume(value < 256u && value < limit + 1u); }")));
  ASSERT_NE(module_, nullptr);
  const auto text = Print();
  EXPECT_NE(text.find("scalar.addi"), std::string::npos);
  EXPECT_NE(text.find("scalar.assume"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, FunctionsAndRootsOutliveSource) {
  std::string source =
      "static int helper(int x) { return x + 1; }\n"
      "int first(int x) { return helper(x); }\n"
      "int second(int x) { return x * 2; }\n";
  iree_string_view_t root = IREE_SV("first");
  options_.roots = &root;
  options_.root_count = 1;
  IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
  ASSERT_NE(module_, nullptr);
  source.assign(source.size(), '?');
  auto text = Print();
  EXPECT_NE(text.find("func.def public @first"), std::string::npos);
  EXPECT_NE(text.find("func.def @helper"), std::string::npos);
  EXPECT_EQ(text.find("@second"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, ExplicitSymbolNamesOutliveSource) {
  std::string source =
      "[[loom::symbol(\"library.helper\")]] static int helper(int x);\n"
      "static int helper(int x) { return x + 1; }\n"
      "[[loom::symbol(\"library.entry\")]] int entry(int x) { return "
      "helper(x); }\n";
  const iree_string_view_t root = IREE_SV("entry");
  options_.roots = &root;
  options_.root_count = 1;
  IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
  ASSERT_NE(module_, nullptr);
  source.assign(source.size(), '?');
  auto text = Print();
  EXPECT_NE(text.find("func.def public @library.entry"), std::string::npos);
  EXPECT_NE(text.find("func.def @library.helper"), std::string::npos);
  EXPECT_NE(text.find("func.call pure @library.helper"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, RecordComponentsOutliveSourceAndFunctionBindingStorage) {
  std::string source =
      "struct Result { unsigned value; unsigned* next; bool valid; };\n"
      "Result advance(unsigned* pointer, unsigned value) {\n"
      "  Result original{value, pointer + 1u, true};\n"
      "  Result copy = original; copy.next += value; return copy;\n"
      "}\n"
      "unsigned consume(unsigned* pointer, unsigned value) {\n"
      "  Result result = advance(pointer, value); return result.value;\n"
      "}\n";
  IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
  ASSERT_NE(module_, nullptr);
  source.assign(source.size(), '?');
  auto text = Print();
  EXPECT_NE(text.find("-> (i32, buffer, index, i1)"), std::string::npos);
  EXPECT_NE(text.find("func.call pure @advance"), std::string::npos);
  EXPECT_NE(text.find("func.def public @consume"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, ViewFacadePreservesItsObjectContractAcrossDataModels) {
  const auto source = IREE_SV(
      "#include <loomcxx/view.h>\n"
      "namespace loomt = loom::type;\n"
      "using Layout = loomt::encoding<loom::encoding::role::layout, 2>;\n"
      "using Rows = loomt::view<loomt::shape<loomt::dynamic, 32>, "
      "const float, Layout>;\n"
      "static_assert(__is_trivially_copyable(Layout));\n"
      "static_assert(__is_trivially_copyable(Rows));\n"
      "static_assert(sizeof(Layout) == 2 * sizeof(loomt::size_type));\n"
      "static_assert(sizeof(Rows) == sizeof(void*) + "
      "4 * sizeof(loomt::size_type));\n"
      "Rows identity(Rows value) { return value; }\n");
  for (auto data_model : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                          LOOM_CXX_DATA_MODEL_ILP32}) {
    SCOPED_TRACE(static_cast<int>(data_model));
    options_.data_model = data_model;
    IREE_ASSERT_OK(Import(source));
    ASSERT_NE(module_, nullptr);
    auto text = Print();
    EXPECT_NE(text.find("func.def public @identity"), std::string::npos);
    EXPECT_NE(text.find("view<[%value_dim0]x32xf32, %value_layout>"),
              std::string::npos);
  }
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, PlainViewsPreserveFunctionSignaturesAndDenseSubviews) {
  const auto source = IREE_SV(
      "#include <loomcxx/view.h>\n"
      "namespace loomt = loom::type;\n"
      "using Matrix = loomt::view<loomt::shape<4, 64>, const float>;\n"
      "using Row = loomt::view<loomt::shape<1, 64>, const float>;\n"
      "[[loom::symbol(\"library.row\")]] Row row(Matrix input) {\n"
      "  return loom::view::subview<1, 64>(input, {2, 0}, {});\n"
      "}\n"
      "[[loom::symbol(\"library.first\")]] float first(Row input) {\n"
      "  return loom::view::load(input, 0, 0);\n"
      "}\n");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("func.def public @library.row(%input: view<4x64xf32>) "
                      "-> (view<1x64xf32>)"),
            std::string::npos);
  EXPECT_NE(text.find(" = view.subview %input["), std::string::npos);
  EXPECT_NE(text.find(": view<4x64xf32> -> view<1x64xf32>"), std::string::npos);
  EXPECT_NE(text.find("func.def public @library.first(%input: "
                      "view<1x64xf32>) -> (f32)"),
            std::string::npos);
  EXPECT_EQ(text.find("encoding<layout>"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, MutableViewReferencesReturnIndependentDependentTuples) {
  const auto source = IREE_SV(
      "#include <loomcxx/view.h>\n"
      "namespace loomt = loom::type;\n"
      "using Layout = loomt::encoding<loom::encoding::role::layout, 2>;\n"
      "using Rows = loomt::view<loomt::shape<loomt::dynamic, 4>, float, "
      "Layout>;\n"
      "static void replace(Rows& rows, Rows replacement) {\n"
      "  rows = replacement;\n"
      "}\n"
      "Rows entry(Rows rows, Rows replacement) {\n"
      "  replace(rows, replacement);\n"
      "  return rows;\n"
      "}\n");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);

  auto* entry = loom_block_op(loom_module_block(module_), 0);
  ASSERT_TRUE(loom_func_def_isa(entry));
  auto* call = loom_region_entry_block(loom_func_def_body(entry))->first_op;
  ASSERT_TRUE(loom_func_call_isa(call));
  EXPECT_EQ(call->tied_result_count, 0u);
  ASSERT_EQ(call->result_count, 3u);
  const auto* call_results = loom_op_results(call);
  auto returned_view = loom_module_value_type(module_, call_results[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(returned_view, 0), call_results[0]);
  EXPECT_TRUE(loom_type_has_ssa_encoding(returned_view));
  EXPECT_EQ(loom_type_encoding_value_id(returned_view), call_results[1]);

  auto* replace = loom_block_op(loom_module_block(module_), 1);
  ASSERT_TRUE(loom_func_def_isa(replace));
  EXPECT_EQ(replace->tied_result_count, 0u);
  ASSERT_EQ(replace->result_count, 3u);
  const auto* signature_results = loom_op_results(replace);
  auto signature_view = loom_module_value_type(module_, signature_results[2]);
  EXPECT_EQ(loom_type_dim_value_id_at(signature_view, 0), signature_results[0]);
  EXPECT_TRUE(loom_type_has_ssa_encoding(signature_view));
  EXPECT_EQ(loom_type_encoding_value_id(signature_view), signature_results[1]);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, RuntimeDefaultMembersPreserveSourceDataModels) {
  const auto source = IREE_SV(
      "struct Result { long value; long next = value + 1; };\n"
      "Result entry(long value) { return Result{value}; }\n");
  for (auto data_model : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                          LOOM_CXX_DATA_MODEL_ILP32}) {
    SCOPED_TRACE(static_cast<int>(data_model));
    options_.data_model = data_model;
    IREE_ASSERT_OK(Import(source));
    ASSERT_NE(module_, nullptr);
    auto text = Print();
    auto type = data_model == LOOM_CXX_DATA_MODEL_LP64 ? "i64" : "i32";
    auto signature = std::string("func.def public @entry(%value: ") + type +
                     ") -> (" + type + ", " + type + ")";
    EXPECT_NE(text.find(signature), std::string::npos);
    EXPECT_NE(text.find("scalar.addi %value"), std::string::npos);
  }
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, HeaderProviderUsesNormalIncludeSearch) {
  const auto overrides =
      std::filesystem::path("/overrides/").make_preferred().string();
  const auto facade =
      std::filesystem::path("/facade/").make_preferred().string();
  const iree_string_view_t paths[] = {iree_make_cstring_view(overrides.c_str()),
                                      iree_make_cstring_view(facade.c_str())};
  options_.include_paths = paths;
  options_.include_path_count = IREE_ARRAYSIZE(paths);
  options_.source_provider = {ProvideSource, this};
  headers_["/overrides/value.h"] = "#pragma once\n#include_next <value.h>\n";
  headers_["/facade/value.h"] =
      "#pragma once\n#include \"factor.h\"\n"
      "template<int N> int scale(int x) { return x * N; }\n";
  headers_["/facade/factor.h"] = "#define FACTOR 3\n";
  IREE_ASSERT_OK(Import(
      IREE_SV("#include <value.h>\n#include <value.h>\n"
              "#if !__has_include(<value.h>) || __has_include(<absent.h>)\n"
              "#error include lookup failed\n#endif\n"
              "int entry(int x) { return scale<FACTOR>(x); }\n")));
  ASSERT_NE(module_, nullptr);
  headers_.clear();
  EXPECT_NE(Print().find("@scale_3"), std::string::npos);
  EXPECT_EQ(source_requests_["/overrides/value.h"], 1);
  EXPECT_EQ(source_requests_["/facade/value.h"], 1);
  EXPECT_EQ(source_requests_["/facade/factor.h"], 1);
}

TEST_F(ImportTest,
       ConflictingKernelContractsIdentifyBothHeadersInIncludeOrder) {
  headers_["/app/declare.h"] =
      "template<class T> [[loom::kernel, loom::workgroup_size(32, 1, 1)]] void "
      "tile();\n";
  headers_["/app/define.h"] =
      "template<class T> [[loom::workgroup_size(64, 1, 1)]] void tile() {}\n";
  options_.source_provider = {ProvideSource, this};
  loom::testing::DiagnosticCapture capture;
  options_.diagnostic_sink = capture.sink();
  // The declaration between includes leaves only the first header in the
  // preamble.
  const auto declaration_first_source = IREE_SV(
      "#include \"declare.h\"\n"
      "void consumer();\n"
      "#include \"define.h\"\n");
  const auto definition_first_source = IREE_SV(
      "#include \"define.h\"\n"
      "void consumer();\n"
      "#include \"declare.h\"\n");
  for (bool definition_first : {false, true}) {
    const auto text =
        definition_first ? definition_first_source : declaration_first_source;
    IREE_ASSERT_OK(loom_cxx_preamble_create(text, IREE_SV("/app/source.cpp"),
                                            &options_, iree_allocator_system(),
                                            &preamble_));
    ASSERT_NE(preamble_, nullptr);
    for (bool restore : {false, true}) {
      SCOPED_TRACE(::testing::Message()
                   << "definition_first=" << definition_first
                   << ", restore=" << restore);
      options_.preamble = restore ? preamble_ : nullptr;
      source_requests_.clear();
      capture.Reset();
      IREE_ASSERT_OK(Import(text));
      EXPECT_EQ(module_, nullptr);
      EXPECT_EQ(source_requests_[definition_first ? "/app/define.h"
                                                  : "/app/declare.h"],
                restore ? 0 : 1);
      ASSERT_EQ(capture.diagnostics.size(), 1u);
      const auto& error = capture.diagnostics.front();
      EXPECT_EQ(error.error, LOOM_ERR_LOWERING_059);
      EXPECT_EQ(error.filename,
                definition_first ? "/app/declare.h" : "/app/define.h");
      ASSERT_EQ(error.related_locations.size(), 1u);
      const auto& previous = error.related_locations.front();
      EXPECT_EQ(previous.label, "previous launch contract");
      EXPECT_TRUE(iree_string_view_equal(
          previous.source_location.filename,
          iree_make_cstring_view(definition_first ? "/app/define.h"
                                                  : "/app/declare.h")));
    }
    options_.preamble = nullptr;
    loom_cxx_preamble_release(preamble_);
    preamble_ = nullptr;
  }
}

TEST_F(ImportTest, PreambleRestoresHeaderIRAndExactSourceWithoutProviderCalls) {
  const std::string header =
      "#include <loomcxx/low.h>\n"
      "struct [[loom::representation(\"test.low.core\")]] Contract {};\n"
      "int header_entry(int value) {\n"
      "  return loom::low::assembly<Contract, int>(R\"loom(\n"
      "      (%value: reg<test.i32>) -> (reg<test.i32>) {\n"
      "        return %value\n"
      "      }\n"
      "  )loom\", value);\n"
      "}\n";
  const auto source =
      IREE_SV("#include \"library.h\"\nvoid consumer_marker();\n");
  headers_["/app/library.h"] = header;
  options_.source_provider = {ProvideSource, this};
  const loom_low_descriptor_set_provider_t descriptor_set_providers[] = {
      loom_test_low_core_descriptor_set,
  };
  const loom_low_descriptor_registry_t descriptor_registry = {
      .descriptor_set_providers = descriptor_set_providers,
      .descriptor_set_provider_count = IREE_ARRAYSIZE(descriptor_set_providers),
  };
  loom_low_descriptor_text_asm_environment_initialize(
      &descriptor_registry, &options_.low_asm_environment);

  struct Capture {
    // Exact sources observed by physical filename.
    std::map<std::string, std::string> sources;
  } capture;
  options_.source_observer = {
      [](void* user_data, loom_source_id_t, iree_string_view_t filename,
         iree_string_view_t source) {
        auto& capture = *static_cast<Capture*>(user_data);
        capture.sources.emplace(std::string(filename.data, filename.size),
                                std::string(source.data, source.size));
        return iree_ok_status();
      },
      &capture};

  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  const std::string cold = Print();
  EXPECT_NE(cold.find("low.invoke pure inline"), std::string::npos);
  EXPECT_NE(cold.find("low.func.def target<test.low.core>"), std::string::npos);
  IREE_ASSERT_OK(loom_cxx_preamble_create(source, IREE_SV("/app/source.cpp"),
                                          &options_, iree_allocator_system(),
                                          &preamble_));
  ASSERT_NE(preamble_, nullptr);

  source_requests_.clear();
  headers_.clear();
  capture.sources.clear();
  options_.preamble = preamble_;
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(Print(), cold);
  EXPECT_TRUE(source_requests_.empty());
  ASSERT_TRUE(capture.sources.contains("/app/library.h"));
  EXPECT_EQ(capture.sources.at("/app/library.h"), header);
}

TEST_F(ImportTest, PreambleConfigurationMismatchUsesColdParsing) {
  const auto source =
      IREE_SV("#include \"library.h\"\nvoid consumer_marker();\n");
  headers_["/app/library.h"] =
      "int header_entry(int value) { return value + 1; }\n";
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(loom_cxx_preamble_create(source, IREE_SV("/app/source.cpp"),
                                          &options_, iree_allocator_system(),
                                          &preamble_));
  ASSERT_NE(preamble_, nullptr);

  source_requests_.clear();
  options_.preamble = preamble_;
  options_.data_model = LOOM_CXX_DATA_MODEL_LLP64;
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(source_requests_["/app/library.h"], 1);
}

TEST_F(ImportTest, PreambleSourceMismatchUsesColdParsing) {
  const auto producer_source =
      IREE_SV("#include \"library.h\"\nvoid consumer_marker();\n");
  const auto consumer_source =
      IREE_SV("#include \"other.h\"\nvoid consumer_marker();\n");
  headers_["/app/library.h"] = "int library_entry();\n";
  headers_["/app/other.h"] = "int other_entry();\n";
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(loom_cxx_preamble_create(producer_source,
                                          IREE_SV("/app/source.cpp"), &options_,
                                          iree_allocator_system(), &preamble_));
  ASSERT_NE(preamble_, nullptr);

  source_requests_.clear();
  options_.preamble = preamble_;
  IREE_ASSERT_OK(Import(consumer_source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(source_requests_["/app/library.h"], 0);
  EXPECT_EQ(source_requests_["/app/other.h"], 1);
}

TEST_F(ImportTest, PreambleNewDirectiveUsesColdParsing) {
  const auto producer_source =
      IREE_SV("#include \"library.h\"\nvoid consumer_marker();\n");
  const auto consumer_source = IREE_SV(
      "#include \"library.h\"\n#define NEW_DIRECTIVE 1\n"
      "void consumer_marker();\n");
  headers_["/app/library.h"] =
      "int header_entry(int value) { return value + 1; }\n";
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(loom_cxx_preamble_create(producer_source,
                                          IREE_SV("/app/source.cpp"), &options_,
                                          iree_allocator_system(), &preamble_));
  ASSERT_NE(preamble_, nullptr);

  source_requests_.clear();
  options_.preamble = preamble_;
  IREE_ASSERT_OK(Import(consumer_source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(source_requests_["/app/library.h"], 1);
}

TEST_F(ImportTest, RestoredHeaderDiagnosticRetainsExactSource) {
  const std::string header =
      "bool before(int* lhs, int* rhs) { return lhs < rhs; }\n";
  const auto source =
      IREE_SV("#include \"invalid.h\"\nvoid consumer_marker();\n");
  headers_["/app/invalid.h"] = header;
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(loom_cxx_preamble_create(source, IREE_SV("/app/source.cpp"),
                                          &options_, iree_allocator_system(),
                                          &preamble_));
  ASSERT_NE(preamble_, nullptr);

  headers_.clear();
  options_.preamble = preamble_;
  IREE_ASSERT_OK(Import(source));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
  EXPECT_EQ(diagnostic_filename_, "/app/invalid.h");
  EXPECT_EQ(diagnostic_source_, header);
}

TEST_F(ImportTest, PreambleCreationRequiresLeadingIncludes) {
  IREE_EXPECT_STATUS_IS(
      IREE_STATUS_FAILED_PRECONDITION,
      loom_cxx_preamble_create(IREE_SV("int entry() { return 1; }\n"),
                               IREE_SV("/app/source.cpp"), &options_,
                               iree_allocator_system(), &preamble_));
  EXPECT_EQ(preamble_, nullptr);
}

TEST_F(ImportTest, PreambleCreationReportsRejectedHeaderSource) {
  const std::string header = "int broken( {\n";
  headers_["/app/invalid.h"] = header;
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(loom_cxx_preamble_create(
      IREE_SV("#include \"invalid.h\"\nvoid consumer_marker();\n"),
      IREE_SV("/app/source.cpp"), &options_, iree_allocator_system(),
      &preamble_));
  EXPECT_EQ(preamble_, nullptr);
  EXPECT_GT(diagnostic_count_, 0);
  EXPECT_EQ(diagnostic_filename_, "/app/invalid.h");
  EXPECT_EQ(diagnostic_source_, header);
}

TEST_F(ImportTest, RejectedSourceDoesNotPoisonNextImport) {
  IREE_ASSERT_OK(Import(IREE_SV("int broken( {")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_GT(diagnostic_count_, 0);
  EXPECT_EQ(diagnostic_filename_, "/app/source.cpp");
  EXPECT_EQ(diagnostic_source_, "int broken( {");
  IREE_ASSERT_OK(Import(IREE_SV("int repaired() { return 7; }")));
  ASSERT_NE(module_, nullptr);
  EXPECT_NE(Print().find("@repaired"), std::string::npos);
}

TEST_F(ImportTest, UnsupportedSwitchShapeHasDiagnosticInsteadOfInvalidModule) {
  IREE_ASSERT_OK(
      Import(IREE_SV("void entry(int x) { switch (x) case 1: return; }")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
  EXPECT_EQ(diagnostic_filename_, "/app/source.cpp");
}

TEST_F(ImportTest, VerificationRejectsSourceWithoutPublishingPartialModule) {
  IREE_ASSERT_OK(Import(IREE_SV(
      "#include <loomcxx/kernel.h>\n"
      "static loom::kernel::configuration configure(unsigned* count) {\n"
      "  *count = 1;\n"
      "  return {{*count, 1, 1}, {1, 1, 1}};\n"
      "}\n"
      "[[loom::kernel(configure)]] void entry() {}\n")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
  EXPECT_EQ(diagnostic_ref_, LOOM_ERR_STRUCTURE_058_REF);
  EXPECT_EQ(diagnostic_filename_, "/app/source.cpp");
  EXPECT_NE(diagnostic_source_.find("*count = 1"), std::string::npos);
}

TEST_F(ImportTest, SinkFailurePropagates) {
  options_.diagnostic_sink = {[](void*, const loom_diagnostic_t*) {
                                return iree_make_status(
                                    IREE_STATUS_CANCELLED,
                                    "diagnostic consumer stopped");
                              },
                              nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED,
                        Import(IREE_SV("int broken( {")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, ProviderFailurePropagates) {
  options_.source_provider = {
      [](void*, iree_string_view_t, bool*, iree_string_view_t*) {
        return iree_make_status(IREE_STATUS_UNAVAILABLE,
                                "source store unavailable");
      },
      nullptr};
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE,
                        Import(IREE_SV("#include \"value.h\"\n")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, RejectsUnknownLanguageStandard) {
  options_.standard = IREE_SV("unknown");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(IREE_SV("")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, EmbeddedFacadeAndExternalProviderAgree) {
  const auto source = IREE_SV(
      "#include <hip/hip_runtime.h>\n#include <hip/hip_vector_types.h>\n"
      "#include <hip/hip_fp16.h>\n"
      "#include <loomcxx/atomic.h>\n"
      "__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(2, "
      "1, 1)]] "
      "void entry(const float* input, float* output) { "
      "unsigned index = blockIdx.x * blockDim.x + threadIdx.x; "
      "__builtin_assume(index < 128u); "
      "output[index] = fmaxf(__shfl_xor(__expf(input[index]), 1), 0.0f); }"
      "__device__ float convert(half value) { return __half2float(value); }"
      "unsigned claim(unsigned* pointer) { return loom::view::atomic::rmw<"
      "loom::atomic::kind::addi, loom::atomic::ordering::relaxed, "
      "loom::atomic::scope::device>(1u, pointer); }");
  IREE_ASSERT_OK(Import(source));
  if (loom::cxx_import::builtin_include_root().empty()) {
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, 0);
    return;
  }
  ASSERT_NE(module_, nullptr);
  auto embedded = Print();
  EXPECT_NE(embedded.find("kernel.def @entry"), std::string::npos);
  EXPECT_NE(embedded.find("scalar.expf<afn>"), std::string::npos);
  EXPECT_NE(embedded.find("kernel.subgroup.shuffle"), std::string::npos);
  EXPECT_NE(
      embedded.find("view.atomic.rmw<addi, root_in_bounds|type_disjoint>"),
      std::string::npos);

  // The provider borrows the same immutable source bytes under an external
  // include root; preprocessing and binding resolution remain identical.
  for (const char* name :
       {"type_traits", "hip/hip_runtime.h", "hip/hip_vector_types.h",
        "hip/hip_fp16.h", "loomcxx/kernel.h", "loomcxx/atomic.h",
        "loomcxx/buffer.h", "loomcxx/cache.h", "loomcxx/math.h",
        "loomcxx/predicate.h", "loomcxx/scalar.h", "loomcxx/view.h",
        "loomcxx/encoding_type.h", "loomcxx/index.h", "loomcxx/shape.h",
        "loomcxx/target.h", "loomcxx/target/amdgpu.h", "loomcxx/template.h"}) {
    auto contents = loom::cxx_import::builtin_include(name);
    ASSERT_TRUE(contents.has_value());
    headers_[std::string("/edited/") + name] = *contents;
  }
  options_.flags |= LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  IREE_ASSERT_OK(Import(source));
  EXPECT_EQ(module_, nullptr);
  const iree_string_view_t path = IREE_SV("/edited");
  options_.include_paths = &path;
  options_.include_path_count = 1;
  options_.source_provider = {ProvideSource, this};
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  headers_.clear();
  EXPECT_EQ(Print(), embedded);
}

TEST_F(ImportTest, StandardFloatFacadeNeedsOnlyOneHeader) {
  auto contents = loom::cxx_import::builtin_include("stdfloat");
  if (!contents) {
    GTEST_SKIP() << "Embedded includes are disabled";
  }
  const auto source = IREE_SV(
      "#include <stdfloat>\n#include <stdfloat>\n"
      "static_assert(__STDCPP_FLOAT16_T__ && __STDCPP_BFLOAT16_T__);\n"
      "std::bfloat16_t entry(std::float16_t value) { "
      "return (std::bfloat16_t)value; }\n");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  auto embedded = Print();

  headers_["/numeric/stdfloat"] = *contents;
  const iree_string_view_t path = IREE_SV("/numeric");
  options_.include_paths = &path;
  options_.include_path_count = 1;
  options_.source_provider = {ProvideSource, this};
  options_.flags |= LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(Print(), embedded);
  EXPECT_EQ(source_requests_.size(), 1u);
  EXPECT_EQ(source_requests_.at("/numeric/stdfloat"), 1);
}

TEST_F(ImportTest, TemplateFacadeNeedsOnlyOneHeader) {
  auto contents = loom::cxx_import::builtin_include("loomcxx/template.h");
  if (!contents) {
    GTEST_SKIP() << "Embedded includes are disabled";
  }
  const auto source = IREE_SV(
      "#include <loomcxx/template.h>\n#include <loomcxx/template.h>\n"
      "LOOM_TEMPLATE_DECL(\"test.transform\") unsigned transform(unsigned);\n"
      "unsigned entry(unsigned value) { "
      "return loom::template_apply<transform>(value); }\n");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  auto embedded = Print();

  headers_["/templates/loomcxx/template.h"] = *contents;
  const iree_string_view_t path = IREE_SV("/templates");
  options_.include_paths = &path;
  options_.include_path_count = 1;
  options_.source_provider = {ProvideSource, this};
  options_.flags |= LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES;
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(Print(), embedded);
  EXPECT_EQ(source_requests_.size(), 1u);
  EXPECT_EQ(source_requests_.at("/templates/loomcxx/template.h"), 1);
}

// Host code names kernels through offload::launch::kernel_ref<&K>, a variable
// template the test declares as a driver would. Each requested
// specialization is a kernel root, exported under K's Itanium symbol, with
// launch facts reported under the specialization's Itanium symbol.
// Body has a base, an array and two pointers; Other has one pointer; fixed
// has an exact workgroup size.
constexpr char kKernelRefSource[] = R"cxx(
#include <loomcxx/kernel.h>

// The driver's launch reference: the importer reads only the kernel argument.
namespace offload::launch {
struct info;
template <auto Kernel>
[[loom::kernel_ref]] extern const info kernel_ref;
}  // namespace offload::launch

static loom::kernel::configuration launch_config(unsigned groups) {
  return {{groups, 1, 1}, {64, 1, 1}};
}

template <class T>
[[loom::kernel(launch_config)]]
void run(T tag) { tag(); }

[[loom::kernel, loom::workgroup_size(64, 1, 1), loom::workgroup_count(1, 1, 1)]]
void fixed(float* y) { y[0] = 1.0f; }

struct Base { float scale; };
struct Body : Base {
  int pad[2];
  const float* x;
  float* y;
  [[loom::inline]] void operator()() const {
    unsigned i = static_cast<unsigned>(loom::kernel::workitem::id.x);
    y[i] = scale * x[i] + static_cast<float>(pad[1]);
  }
};
struct Other {
  float* y;
  [[loom::inline]] void operator()() const { y[0] = 1.0f; }
};

template <class T>
const offload::launch::info* describe() {
  return &offload::launch::kernel_ref<&run<T>>;
}

const offload::launch::info* host(bool other) {
  if (other) return describe<Other>();
  (void)&offload::launch::kernel_ref<&fixed>;
  return describe<Body>();
}
)cxx";

constexpr char kBodyReference[] =
    "_ZN7offload6launch10kernel_refIXadL_Z3runI4BodyEvT_EEEE";
constexpr char kOtherReference[] =
    "_ZN7offload6launch10kernel_refIXadL_Z3runI5OtherEvT_EEEE";
constexpr char kFixedReference[] =
    "_ZN7offload6launch10kernel_refIXadL_Z5fixedPfEEEE";

class KernelRefImportTest : public ImportTest {
 protected:
  void SetUp() override {
    ImportTest::SetUp();
    UseKernelRefs({kBodyReference, kOtherReference, kFixedReference});
  }

  void TearDown() override {
    loom_cxx_unit_free(unit_);
    ImportTest::TearDown();
  }

  // Exports the kernels of the kernel_ref specializations |names|, as
  // A driver names the specializations its host code references.
  void UseKernelRefs(std::vector<std::string> names) {
    kernel_ref_names_ = std::move(names);
    kernel_ref_views_.clear();
    for (const auto& name : kernel_ref_names_) {
      kernel_ref_views_.push_back(
          iree_make_string_view(name.data(), name.size()));
    }
    options_.kernel_refs = kernel_ref_views_.data();
    options_.kernel_ref_count = kernel_ref_views_.size();
  }

  // Parses |source| as a hosted unit and imports it, as a driver does, so the
  // unit reports the host launch facts of its kernel_ref specializations.
  iree_status_t ImportUnit(iree_string_view_t source) {
    loom_module_free(module_);
    module_ = nullptr;
    loom_cxx_unit_free(unit_);
    unit_ = nullptr;
    options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
    options_.triple = IREE_SV("x86_64-linux-gnu");
    IREE_RETURN_IF_ERROR(loom_cxx_unit_parse(source, IREE_SV("/app/source.cpp"),
                                             &options_, iree_allocator_system(),
                                             &unit_));
    if (!unit_) {
      return iree_ok_status();
    }
    return loom_cxx_import_unit(unit_, &context_, &pool_, &options_,
                                iree_allocator_system(), &module_);
  }

  // The launch facts of the kernel_ref specialization |symbol|, or null.
  const loom::cxx_import::KernelLaunch* Launch(
      const std::string& symbol) const {
    for (const auto& launch : loom::cxx_import::unit_kernel_launches(unit_)) {
      if (launch.symbol == symbol) {
        return &launch;
      }
    }
    return nullptr;
  }

  std::vector<std::string> kernel_ref_names_;
  std::vector<iree_string_view_t> kernel_ref_views_;
  // Unit of the most recent ImportUnit().
  loom_cxx_unit_t* unit_ = nullptr;
};

TEST_F(KernelRefImportTest, KernelRefRootsExportReferencedKernels) {
  IREE_ASSERT_OK(Import(iree_make_cstring_view(kKernelRefSource)));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
  // Requested kernels are exported under their own Itanium symbols; the
  // referencing host functions are not lowered.
  const std::string printed = Print();
  EXPECT_NE(printed.find("@_Z3runI4BodyEvT_"), std::string::npos) << printed;
  EXPECT_NE(printed.find("@_Z3runI5OtherEvT_"), std::string::npos) << printed;
  EXPECT_NE(printed.find("@_Z5fixedPf"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("@host"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("describe"), std::string::npos) << printed;
}

TEST_F(KernelRefImportTest, KernelRefRootsExportOnlyRequestedKernels) {
  // Host code also references Other and fixed, but only Body is requested:
  // the import reads no host body to find more.
  UseKernelRefs({kBodyReference});
  IREE_ASSERT_OK(ImportUnit(iree_make_cstring_view(kKernelRefSource)));
  ASSERT_NE(module_, nullptr);
  const std::string printed = Print();
  EXPECT_NE(printed.find("@_Z3runI4BodyEvT_"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("_Z3runI5OtherEvT_"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("_Z5fixedPf"), std::string::npos) << printed;
  EXPECT_EQ(loom::cxx_import::unit_kernel_launches(unit_).size(), 1u);
}

TEST_F(KernelRefImportTest, UnknownKernelRefRootIsRejected) {
  UseKernelRefs(
      {kBodyReference, "_ZN7offload6launch10kernel_refIXadL_Z4nopeEEE"});
  IREE_ASSERT_OK(ImportUnit(iree_make_cstring_view(kKernelRefSource)));
  EXPECT_EQ(module_, nullptr);
  EXPECT_GT(error_count_, 0);
}

TEST_F(KernelRefImportTest, KernelRefRootsCombineWithFunctionRoots) {
  // Source roots add to the kernel_ref roots. A source root that names a
  // kernel_ref kernel exports it once, under its kernel_ref export name.
  const std::string source = std::string(kKernelRefSource) +
                             "int twice(int value) { return 2 * value; }\n";
  UseKernelRefs({kBodyReference, kFixedReference});
  const iree_string_view_t roots[] = {IREE_SV("twice"), IREE_SV("fixed")};
  options_.roots = roots;
  options_.root_count = IREE_ARRAYSIZE(roots);
  IREE_ASSERT_OK(
      ImportUnit(iree_make_string_view(source.data(), source.size())));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
  const std::string printed = Print();
  EXPECT_NE(printed.find("@twice"), std::string::npos) << printed;
  EXPECT_NE(printed.find("@_Z3runI4BodyEvT_"), std::string::npos) << printed;
  EXPECT_NE(printed.find("@_Z5fixedPf"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("@fixed"), std::string::npos) << printed;
  EXPECT_EQ(printed.find("@_Z3runI5OtherEvT_"), std::string::npos) << printed;
  // Only kernel_ref roots have launch facts.
  EXPECT_EQ(loom::cxx_import::unit_kernel_launches(unit_).size(), 2u);
}

TEST_F(KernelRefImportTest, KernelLaunchesNameKernelRefSpecializations) {
  IREE_ASSERT_OK(ImportUnit(iree_make_cstring_view(kKernelRefSource)));
  ASSERT_NE(module_, nullptr);
  // One launch per specialization, under the Itanium symbol a host C++
  // compiler references for kernel_ref<&K>.
  EXPECT_EQ(loom::cxx_import::unit_kernel_launches(unit_).size(), 3u);
  for (const char* symbol :
       {kBodyReference, kOtherReference, kFixedReference}) {
    EXPECT_NE(Launch(symbol), nullptr) << symbol;
  }
}

// Closures as kernel_ref template arguments: a lambda in a function template
// and one in an inline member function have Itanium closure names; one in an
// ordinary function has internal linkage and a $_N name.
constexpr char kLambdaKernelRefSource[] = R"cxx(
#include <loomcxx/kernel.h>

// The driver's launch reference: the importer reads only the kernel argument.
namespace offload::launch {
struct info;
template <auto Kernel>
[[loom::kernel_ref]] extern const info kernel_ref;
}  // namespace offload::launch

static loom::kernel::configuration launch_config(unsigned groups) {
  return {{groups, 1, 1}, {64, 1, 1}};
}

template <class T>
[[loom::kernel(launch_config)]]
void run(T tag) { tag(); }

template <class F>
const offload::launch::info* launch(F f) {
  return &offload::launch::kernel_ref<&run<F>>;
}

template <class T>
const offload::launch::info* in_template(T* y) {
  return launch([=] { y[0] = T(1); });
}

struct Owner {
  const offload::launch::info* member(float* y) const {
    return launch([=] { y[0] = 3.0f; });
  }
};

const offload::launch::info* host(float* y) {
  (void)launch([=] { y[0] = 2.0f; });
  (void)Owner{}.member(y);
  return in_template(y);
}
)cxx";

TEST_F(KernelRefImportTest, KernelRefRootsResolveClosureSpecializations) {
  // The names cxx's host code generation gives these specializations.
  const std::vector<std::string> references = {
      "_ZN7offload6launch10kernel_refIXadL_Z3runIZ11in_templateIfEPKNS0_"
      "4infoEPT_EUlvE_EvS7_EEEE",
      "_ZN7offload6launch10kernel_refIXadL_Z3runIZ4hostPfE3$_0EvT_EEEE",
      "_ZN7offload6launch10kernel_refIXadL_Z3runIZNK5Owner6memberEPfEUlvE_"
      "EvT_EEEE",
  };
  UseKernelRefs(references);
  IREE_ASSERT_OK(ImportUnit(iree_make_cstring_view(kLambdaKernelRefSource)));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
  EXPECT_EQ(loom::cxx_import::unit_kernel_launches(unit_).size(), 3u);
  for (const auto& symbol : references) {
    EXPECT_NE(Launch(symbol), nullptr) << symbol;
  }
}

TEST_F(KernelRefImportTest, KernelLaunchesFollowKernelArgumentFlattening) {
  IREE_ASSERT_OK(ImportUnit(iree_make_cstring_view(kKernelRefSource)));
  ASSERT_NE(module_, nullptr);
  // (kind, source offset, size) of one leaf per kernel argument, in argument
  // order: the base's field, each array element, then one buffer leaf per
  // pointer. Source offsets are the host object's (LP64): Base at 0, pad at
  // 4, x at 16, y at 24.
  auto expect_leaves = [](const loom::cxx_import::KernelLaunch& launch,
                          std::vector<std::array<uint64_t, 3>> expected) {
    ASSERT_EQ(launch.leaves.size(), expected.size()) << launch.symbol;
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(launch.leaves[i].parameter, 0u) << i;
      EXPECT_EQ(launch.leaves[i].kind, expected[i][0]) << i;
      EXPECT_EQ(launch.leaves[i].source_offset, expected[i][1]) << i;
      EXPECT_EQ(launch.leaves[i].size, expected[i][2]) << i;
    }
  };
  const auto* body = Launch(kBodyReference);
  ASSERT_NE(body, nullptr);
  EXPECT_TRUE(body->available);
  EXPECT_EQ(body->export_name, "_Z3runI4BodyEvT_");
  // The configuration computes the workgroup size.
  EXPECT_EQ(body->workgroup_size, (std::array<uint32_t, 3>{0, 0, 0}));
  EXPECT_EQ(body->parameter_count, 1u);
  expect_leaves(*body,
                {{0, 0, 4}, {0, 4, 4}, {0, 8, 4}, {1, 16, 8}, {1, 24, 8}});
  const auto* other = Launch(kOtherReference);
  ASSERT_NE(other, nullptr);
  expect_leaves(*other, {{1, 0, 8}});
  // An exact workgroup_size annotation fixes the launch's size.
  const auto* fixed = Launch(kFixedReference);
  ASSERT_NE(fixed, nullptr);
  EXPECT_EQ(fixed->workgroup_size, (std::array<uint32_t, 3>{64, 1, 1}));
  EXPECT_EQ(fixed->export_name, "_Z5fixedPf");
  expect_leaves(*fixed, {{1, 0, 8}});
}

// A unit is parsed once and imported from its existing translation unit, as
// a single-source driver does after generating host code from it.
TEST_F(KernelRefImportTest, ParsedUnitImportsWithoutReparsing) {
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  const auto before = loom::cxx_import::frontend_statistics();
  loom_cxx_unit_t* unit = nullptr;
  IREE_ASSERT_OK(loom_cxx_unit_parse(iree_make_cstring_view(kKernelRefSource),
                                     IREE_SV("/app/source.cpp"), &options_,
                                     iree_allocator_system(), &unit));
  ASSERT_NE(unit, nullptr);
  const auto parsed = loom::cxx_import::frontend_statistics();
  EXPECT_EQ(parsed.preprocess_count - before.preprocess_count, 1u);
  EXPECT_EQ(parsed.parse_count - before.parse_count, 1u);

  // Consumers see the unit's frontend state and hosted profile.
  cxx::TranslationUnit& translation_unit =
      loom::cxx_import::unit_translation_unit(unit);
  EXPECT_NE(translation_unit.ast(), nullptr);
  // The hosted profile parses as a host compile without exceptions, so host
  // code generation from the unit emits no unwinding.
  EXPECT_FALSE(loom::cxx_import::unit_toolchain(unit).exceptionsEnabled());
  EXPECT_FALSE(translation_unit.config().exceptionsEnabled);
  EXPECT_FALSE(loom::cxx_import::unit_has_error(unit));

  IREE_ASSERT_OK(loom_cxx_import_unit(unit, &context_, &pool_, &options_,
                                      iree_allocator_system(), &module_));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
  EXPECT_NE(Print().find("@_Z3runI4BodyEvT_"), std::string::npos);
  auto body_available = [&] {
    for (const auto& launch : loom::cxx_import::unit_kernel_launches(unit)) {
      if (launch.symbol == kBodyReference) {
        return launch.available;
      }
    }
    return false;
  };
  EXPECT_TRUE(body_available());

  // Importing again, for example without the root of a kernel the device
  // compiler rejected, reuses the same translation unit.
  loom_module_free(module_);
  module_ = nullptr;
  UseKernelRefs({kOtherReference, kFixedReference});
  IREE_ASSERT_OK(loom_cxx_import_unit(unit, &context_, &pool_, &options_,
                                      iree_allocator_system(), &module_));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(Print().find("@_Z3runI4BodyEvT_"), std::string::npos);
  EXPECT_FALSE(body_available());
  EXPECT_EQ(loom::cxx_import::unit_statistics(unit).import_count, 2u);
  const auto imported = loom::cxx_import::frontend_statistics();
  EXPECT_EQ(imported.preprocess_count, parsed.preprocess_count);
  EXPECT_EQ(imported.parse_count, parsed.parse_count);
  loom_cxx_unit_free(unit);
}

// Sub-word scalars mixed with pointers. The importer reports each kernel
// argument's host object offset and size; it does not decide where the
// argument lives in the native kernel argument segment.
constexpr char kMixedReference[] =
    "_ZN7offload6launch10kernel_refIXadL_Z3runI5MixedEvT_EEEE";

TEST_F(KernelRefImportTest, KernelLaunchesCarryHostLeavesOfSubWordScalars) {
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  std::string source = std::string(kKernelRefSource) + R"cxx(
struct Mixed {
  char c;
  short s;
  bool b;
  float* y;
  char c2;
  double d;
  int* z;
  bool b2;
  unsigned char u;
  int i;
  [[loom::inline]] void operator()() const {
    y[0] = static_cast<float>(c) + static_cast<float>(s) +
           static_cast<float>(c2) + static_cast<float>(d) + (b ? 1.0f : 0.0f);
    z[0] = static_cast<int>(u) + i + (b2 ? 1 : 0);
  }
};
const offload::launch::info* host_mixed() { return describe<Mixed>(); }
)cxx";
  UseKernelRefs(
      {kBodyReference, kOtherReference, kFixedReference, kMixedReference});
  loom_cxx_unit_t* unit = nullptr;
  IREE_ASSERT_OK(loom_cxx_unit_parse(
      iree_make_string_view(source.data(), source.size()),
      IREE_SV("/app/source.cpp"), &options_, iree_allocator_system(), &unit));
  ASSERT_NE(unit, nullptr);
  IREE_ASSERT_OK(loom_cxx_import_unit(unit, &context_, &pool_, &options_,
                                      iree_allocator_system(), &module_));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);

  const loom::cxx_import::KernelLaunch* mixed = nullptr;
  for (const auto& launch : loom::cxx_import::unit_kernel_launches(unit)) {
    if (launch.symbol == kMixedReference) {
      mixed = &launch;
    }
  }
  ASSERT_NE(mixed, nullptr);
  EXPECT_TRUE(mixed->available);
  EXPECT_EQ(mixed->export_name, "_Z3runI5MixedEvT_");
  EXPECT_EQ(mixed->parameter_count, 1u);
  // (kind, source offset, size) in kernel argument order; LP64 host layout.
  const std::vector<std::array<uint64_t, 3>> expected = {
      {0, 0, 1},  {0, 2, 2},  {0, 4, 1},  {1, 8, 8},  {0, 16, 1},
      {0, 24, 8}, {1, 32, 8}, {0, 40, 1}, {0, 41, 1}, {0, 44, 4},
  };
  ASSERT_EQ(mixed->leaves.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    EXPECT_EQ(mixed->leaves[i].parameter, 0u) << i;
    EXPECT_EQ(mixed->leaves[i].kind, expected[i][0]) << i;
    EXPECT_EQ(mixed->leaves[i].source_offset, expected[i][1]) << i;
    EXPECT_EQ(mixed->leaves[i].size, expected[i][2]) << i;
  }

  loom_cxx_unit_free(unit);
}

TEST_F(ImportTest, ParsedUnitRejectsSourceErrorsOnce) {
  loom_cxx_unit_t* unit = nullptr;
  IREE_ASSERT_OK(loom_cxx_unit_parse(IREE_SV("int entry() { return x; }\n"),
                                     IREE_SV("/app/source.cpp"), &options_,
                                     iree_allocator_system(), &unit));
  EXPECT_EQ(unit, nullptr);
  EXPECT_GT(error_count_, 0);
}

TEST_F(ImportTest, ParsedUnitDiagnosticsFollowTheImportSink) {
  // The parse sink belongs to the parse; an import reports through its own.
  loom_cxx_unit_t* unit = nullptr;
  IREE_ASSERT_OK(loom_cxx_unit_parse(IREE_SV("int counter = 0;\n"),
                                     IREE_SV("/app/source.cpp"), &options_,
                                     iree_allocator_system(), &unit));
  ASSERT_NE(unit, nullptr);
  EXPECT_EQ(diagnostic_count_, 0);
  loom_cxx_unit_set_diagnostic_sink(unit, {});
  // Mutable global storage is rejected when the unit is imported.
  IREE_ASSERT_OK(loom_cxx_import_unit(unit, &context_, &pool_, &options_,
                                      iree_allocator_system(), &module_));
  EXPECT_EQ(module_, nullptr);
  EXPECT_GT(error_count_, 0);
  loom_cxx_unit_free(unit);
}

TEST_F(ImportTest, HostedFlagSelectsTheHostedProfile) {
  const auto source = IREE_SV(
      "#ifndef __linux__\n#error not hosted\n#endif\n"
      "int entry(int value) { return value; }\n");
  IREE_ASSERT_OK(Import(source));
  EXPECT_EQ(module_, nullptr);
  EXPECT_GT(error_count_, 0);

  error_count_ = 0;
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
}

TEST_F(ImportTest, HostedProfileDisablesExceptions) {
  const auto source = IREE_SV(
      "#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)\n"
      "#error exceptions enabled\n#endif\n"
      "int entry(int value) { return value; }\n");
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
}

TEST_F(ImportTest, ExceptionsFlagEnablesHostedExceptions) {
  // A driver whose host compile uses exceptions parses the unit with them.
  const auto source = IREE_SV(
      "#if !defined(__cpp_exceptions) || !defined(__EXCEPTIONS)\n"
      "#error exceptions disabled\n#endif\n"
      "int entry(int value) { return value; }\n");
  options_.flags |=
      LOOM_CXX_IMPORT_FLAG_HOSTED | LOOM_CXX_IMPORT_FLAG_EXCEPTIONS;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
}

TEST_F(ImportTest, ExceptionsFlagRequiresHosted) {
  options_.flags |= LOOM_CXX_IMPORT_FLAG_EXCEPTIONS;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Import(IREE_SV("int entry() { return 0; }\n")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, HostedProfileDefaultsToCxx20) {
  // Hosted import reads the platform's C++ library headers, which parse at
  // C++20; an explicit standard still applies.
  const auto source = IREE_SV(
      "static_assert(__cplusplus == 202002L);\n"
      "int entry(int value) { return value; }\n");
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  options_.standard = iree_string_view_empty();
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
  options_.standard = IREE_SV("c++26");
  IREE_ASSERT_OK(Import(source));
  EXPECT_EQ(module_, nullptr);
  EXPECT_GT(error_count_, 0);
}

TEST_F(ImportTest, HostedProfileRequiresALinuxTriple) {
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Import(IREE_SV("int entry() { return 0; }\n")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, HostedProfileDoesNotMarkTheDevicePass) {
  // Hosted source sees exactly the host compile's macros, so it cannot branch
  // on the pass with the preprocessor.
  const auto source = IREE_SV(
      "#ifdef __loom__\n#error device pass is marked\n#endif\n"
      "int entry(int value) { return value; }\n");
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
}

TEST_F(ImportTest, HostedProfileUsesTheHostLayout) {
  // x86-64 Linux: 16-byte long double with a 64-bit significand, in both the
  // layout and the macros, as in a host compile for the same triple.
  const auto source = IREE_SV(
      "static_assert(sizeof(long double) == 16);\n"
      "static_assert(__SIZEOF_LONG_DOUBLE__ == 16);\n"
      "static_assert(__LDBL_MANT_DIG__ == 64);\n"
      "int entry(int value) { return value; }\n");
  options_.flags |= LOOM_CXX_IMPORT_FLAG_HOSTED;
  options_.triple = IREE_SV("x86_64-linux-gnu");
  IREE_ASSERT_OK(Import(source));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(error_count_, 0);
}

TEST_F(ImportTest, RejectsUnknownImportFlags) {
  options_.flags = LOOM_CXX_IMPORT_FLAG_ALL + 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Import(IREE_SV("int entry() { return 0; }\n")));
  EXPECT_EQ(module_, nullptr);
}

TEST_F(ImportTest, RejectsKernelRefCountWithoutStorage) {
  options_.kernel_ref_count = 1;
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT,
                        Import(IREE_SV("int entry() { return 0; }\n")));
  EXPECT_EQ(module_, nullptr);
}

}  // namespace
