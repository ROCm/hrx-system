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
#include "loom/format/text/printer.h"
#include "loom/import/cxx/source/catalog.h"
#include "loom/ops/op_registry.h"

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
    iree_string_builder_t builder;
    iree_string_builder_initialize(iree_allocator_system(), &builder);
    IREE_EXPECT_OK(loom_text_print_module_to_builder(module_, &builder, 0));
    std::string result(iree_string_builder_buffer(&builder),
                       iree_string_builder_size(&builder));
    iree_string_builder_deinitialize(&builder);
    return result;
  }

  static iree_status_t CaptureDiagnostic(void* user_data,
                                         const loom_diagnostic_t* diagnostic) {
    auto& self = *static_cast<ImportTest*>(user_data);
    ++self.diagnostic_count_;
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
  // Number of callbacks delivered by the source diagnostic boundary.
  int diagnostic_count_ = 0;
  // Copied source identity retained after the frontend is destroyed.
  std::string diagnostic_filename_;
  // Copied source bytes retained after the frontend is destroyed.
  std::string diagnostic_source_;
  // Provider-owned header text, independent of frontend storage.
  std::map<std::string, std::string> headers_;
  // Lookup counts witness per-invocation source reuse.
  std::map<std::string, int> source_requests_;
};

TEST_F(ImportTest, RejectedAssumptionsPublishNoPartialModule) {
  IREE_ASSERT_OK(
      Import(IREE_SV("[[loom::assume]] void assume(bool); "
                     "void entry(unsigned value, unsigned limit) { "
                     "assume(value < 256u && value < limit); }")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
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
  EXPECT_NE(text.find("func.call @library.helper"), std::string::npos);
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

TEST_F(ImportTest, UnsupportedSourceHasDiagnosticInsteadOfInvalidModule) {
  IREE_ASSERT_OK(
      Import(IREE_SV("int entry(int x) { while (x) { break; } return x; }")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
  EXPECT_EQ(diagnostic_filename_, "/app/source.cpp");
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
      "__global__ [[loom::workgroup_size(64, 1, 1), loom::workgroup_count(2, "
      "1, 1)]] "
      "void entry(const float* input, float* output) { "
      "unsigned index = blockIdx.x * blockDim.x + threadIdx.x; "
      "__builtin_assume(index < 128u); "
      "output[index] = fmaxf(__shfl_xor(__expf(input[index]), 1), 0.0f); }"
      "__device__ float convert(half value) { return __half2float(value); }");
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

  // The provider borrows the same immutable source bytes under an external
  // include root; preprocessing and binding resolution remain identical.
  for (const char* name :
       {"hip/hip_runtime.h", "hip/hip_fp16.h", "loomcxx/kernel.h",
        "loomcxx/math.h", "loomcxx/scalar.h"}) {
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

}  // namespace
