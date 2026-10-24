// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/import.h"

#include <filesystem>
#include <map>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/codegen/low/text_asm.h"
#include "loom/error/error_catalog.h"
#include "loom/format/text/printer.h"
#include "loom/import/cxx/source/catalog.h"
#include "loom/ops/func/ops.h"
#include "loom/ops/op_registry.h"
#include "loom/target/test/descriptors.h"

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
  EXPECT_NE(text.find("-> (i32, buffer, offset, i1)"), std::string::npos);
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
      "const float>;\n"
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

TEST_F(ImportTest, MutableViewReferencesRetainTypeChangingTies) {
  const auto source = IREE_SV(
      "#include <loomcxx/view.h>\n"
      "namespace loomt = loom::type;\n"
      "using Rows = loomt::view<loomt::shape<loomt::dynamic, 4>, float>;\n"
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
  ASSERT_EQ(call->tied_result_count, 3u);
  EXPECT_FALSE(loom_op_tied_results(call)[0].has_type_change);
  EXPECT_FALSE(loom_op_tied_results(call)[1].has_type_change);
  EXPECT_TRUE(loom_op_tied_results(call)[2].has_type_change);

  auto* replace = loom_block_op(loom_module_block(module_), 1);
  ASSERT_TRUE(loom_func_def_isa(replace));
  ASSERT_EQ(replace->tied_result_count, 3u);
  EXPECT_FALSE(loom_op_tied_results(replace)[0].has_type_change);
  EXPECT_FALSE(loom_op_tied_results(replace)[1].has_type_change);
  EXPECT_TRUE(loom_op_tied_results(replace)[2].has_type_change);
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
      "long distance(int* lhs, int* rhs) { return lhs - rhs; }\n";
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
      Import(IREE_SV("int entry(int x) { switch (x) case 1: return x; }")));
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
      "#include <hip/hip_runtime.h>\n#include <hip/hip_fp16.h>\n"
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
  EXPECT_NE(embedded.find("view.atomic.rmw<addi>"), std::string::npos);

  // The provider borrows the same immutable source bytes under an external
  // include root; preprocessing and binding resolution remain identical.
  for (const char* name :
       {"hip/hip_runtime.h", "hip/hip_fp16.h", "loomcxx/kernel.h",
        "loomcxx/atomic.h", "loomcxx/buffer.h", "loomcxx/cache.h",
        "loomcxx/math.h", "loomcxx/scalar.h", "loomcxx/view.h",
        "loomcxx/encoding_type.h", "loomcxx/index.h", "loomcxx/shape.h",
        "loomcxx/target.h", "loomcxx/target/amdgpu.h"}) {
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

}  // namespace
