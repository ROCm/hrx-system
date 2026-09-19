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

TEST_F(ImportTest, RejectsMutationOfConstObjectsAndBindings) {
  for (auto source : {
           IREE_SV("void store(const int* p) { p[0u] = 1; }"),
           IREE_SV(
               "int entry() { const int value = 1; value = 2; return value; }"),
           IREE_SV("int entry() { const int value = 1; value += 2; return "
                   "value; }"),
           IREE_SV("void entry(int* const p, int* q) { p = q; }"),
           IREE_SV("void entry(int* const p) { p += 1; }"),
           IREE_SV(
               "int entry() { const int value = 1; ++value; return value; }"),
       }) {
    IREE_ASSERT_OK(Import(source));
    EXPECT_EQ(module_, nullptr);
  }
  EXPECT_GE(diagnostic_count_, 6);
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    int entry(int* const output, const int* input) {
      const int value = input[0u];
      output[0u] = value;
      return value;
    }
  )")));
  ASSERT_NE(module_, nullptr);
}

TEST_F(ImportTest, RejectedAssumptionsPublishNoPartialModule) {
  for (auto source : {
           IREE_SV("[[loom::assume]] void assume(bool); "
                   "void entry(unsigned value, unsigned limit) { "
                   "assume(value < 256u && value < limit); }"),
           IREE_SV("[[loom::assume]] void assume(bool); unsigned side(); "
                   "void entry(unsigned value) { "
                   "assume(value < 256u && value < (side(), 16u)); }"),
           IREE_SV("[[loom::assume]] void assume(bool); struct Bound {}; "
                   "bool operator&&(bool, Bound); "
                   "void entry(unsigned value, Bound bound) { "
                   "assume((value < 256u) && bound); }"),
       }) {
    IREE_ASSERT_OK(Import(source));
    EXPECT_EQ(module_, nullptr);
  }
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    [[loom::assume]] void assume(bool);
    unsigned entry(unsigned value, unsigned other) {
      assume(value < (1u << 8) && other < 256u);
      return value + other;
    }
  )")));
  ASSERT_NE(module_, nullptr);
}

TEST_F(ImportTest, ExplicitVectorsRetainMasksAndStructuredValueTransport) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    typedef unsigned u32x16 __attribute__((vector_size(64)));
    typedef int i32x16 __attribute__((vector_size(64)));
    static u32x16 update(u32x16 value, u32x16 depth) {
      return (value & 65535u) | (depth & 0xffff0000u);
    }
    void entry(const u32x16* input, u32x16* output, i32x16* masks,
               unsigned count) {
      u32x16 value = input[0];
      for (unsigned i = 0; i < count; ++i) {
        value = update(value, input[i]);
      }
      if (count > 1u) { value += 1u; }
      output[0] = value;
      masks[0] = value > input[0];
      masks[1] = !value;
    }
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("vector<16xi32>"), std::string::npos);
  EXPECT_NE(text.find("vector.load"), std::string::npos);
  EXPECT_NE(text.find("vector.store"), std::string::npos);
  EXPECT_NE(text.find("vector.splat"), std::string::npos);
  EXPECT_NE(text.find("vector.cmpi ugt"), std::string::npos);
  EXPECT_NE(text.find("vector.extsi"), std::string::npos);
  EXPECT_NE(text.find("scf.for"), std::string::npos);
}

TEST_F(ImportTest, VectorLanesAndBitCastsKeepSourceRepresentation) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    typedef unsigned U __attribute__((vector_size(16)));
    typedef float F __attribute__((vector_size(16)));
    unsigned lane(unsigned* input, unsigned index) {
      U lanes = {1u, 2u};
      const U* source = reinterpret_cast<const U*>(input);
      U value = *source + lanes;
      F bits = (F)value;
      U recovered = __builtin_bit_cast(U, bits);
      return recovered[index];
    }
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("vector.from_elements"), std::string::npos);
  EXPECT_NE(text.find("vector.extract"), std::string::npos);
  EXPECT_NE(text.find("vector.bitcast"), std::string::npos);
}

TEST_F(ImportTest, UnsupportedVectorFormsDiagnoseAtSourceAdmission) {
  for (auto source : {
           "typedef bool V __attribute__((ext_vector_type(16))); V f(V x) { "
           "return x; }",
           "typedef float V __attribute__((ext_vector_type(3))); V f(V x) { "
           "return x; }",
           "typedef int V __attribute__((vector_size(16))); void f(V x) { x[0] "
           "= 1; }",
           "typedef int V __attribute__((vector_size(16))); V f(V x, V y) { "
           "return x && y; }",
           "typedef int V __attribute__((vector_size(16))); V f(V x, V y) { "
           "return x ? x : y; }",
       }) {
    SCOPED_TRACE(source);
    auto before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_cstring_view(source)));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
}

TEST_F(ImportTest, BracedVectorConstructionRejectsInvalidInitializers) {
  for (auto source : {
           "typedef int V __attribute__((vector_size(16))); V f() { "
           "return V{1, 2, 3, 4, 5}; }",
           "typedef unsigned V __attribute__((ext_vector_type(4))); V f() { "
           "return V{1u, 2u, 3u, 4u, 5u}; }",
           "typedef unsigned char V __attribute__((vector_size(16))); V f() { "
           "return V{256}; }",
           "typedef int V __attribute__((vector_size(16))); V f(float x) { "
           "return V{x, 0, 0, 0}; }",
           "struct Value { int field; }; int f(int x) { "
           "return Value{x}.field; }",
       }) {
    SCOPED_TRACE(source);
    auto before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_cstring_view(source)));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
}

TEST_F(ImportTest, ShapedIntrinsicsRetainHeterogeneousSignaturesAndKinds) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    typedef unsigned Table __attribute__((vector_size(64)));
    typedef unsigned Indices __attribute__((vector_size(16)));
    typedef signed char Coordinates __attribute__((vector_size(16)));
    typedef unsigned char Weights __attribute__((vector_size(16)));
    typedef int Result __attribute__((vector_size(16)));
    [[loom::op("vector.table.lookup")]] Indices lookup(Table, Indices);
    [[loom::op("vector.dot4i", "s8u8")]] Result dot(Coordinates, Weights, Result);
    Indices lookup_row(Table table, Indices indices) {
      return lookup(table, indices);
    }
    Result transform(Coordinates points, Weights rows, Result translation) {
      return dot(points, rows, translation);
    }
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("vector.table.lookup"), std::string::npos);
  EXPECT_NE(text.find("vector<16xi32>, vector<4xi32> -> vector<4xi32>"),
            std::string::npos);
  EXPECT_NE(text.find("vector.dot4i<s8u8>"), std::string::npos);
  EXPECT_NE(text.find("vector<16xi8>, vector<16xi8>, vector<4xi32>"),
            std::string::npos);
}

TEST_F(ImportTest, PointerOriginsCrossCallsBranchesAndLoops) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    static int* advance(int* pointer, long long count) {
      if (count == 0) return pointer;
      return pointer + count;
    }
    [[loom::kernel, loom::workgroup_size(1, 1, 1),
      loom::workgroup_count(1, 1, 1)]]
    void entry(int* input, int* output, long long start, unsigned count) {
      int* row = advance(input, start);
      int* selected = count ? row : row + 1;
      if (count > 3) { selected += 2; }
      for (unsigned index = 0; index < count; ++index) {
        selected = advance(selected, 1);
      }
      while (count > 0) { selected -= 1; --count; }
      do { --selected; } while (count > 0);
      output[0] = selected[-1];
      (*output) = *advance(&2[row], -1);
    }
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("kernel.def @entry"), std::string::npos);
  EXPECT_NE(text.find("-> (buffer, offset)"), std::string::npos);
  EXPECT_NE(text.find("%pointer_byte_offset: offset"), std::string::npos);
  EXPECT_NE(text.find("scf.while"), std::string::npos);
  EXPECT_NE(text.find("scf.for"), std::string::npos);
}

TEST_F(ImportTest, PointerSubscriptsPreserveSourceIntegerWidths) {
  for (auto layout : {LOOM_CXX_DATA_MODEL_LP64, LOOM_CXX_DATA_MODEL_LLP64,
                      LOOM_CXX_DATA_MODEL_ILP32}) {
    options_.data_model = layout;
    IREE_ASSERT_OK(Import(IREE_SV(R"(
      int read(const int* input, long index, unsigned large, unsigned long long wide) {
        return input[index] + input[large] + input[wide];
      }
    )")));
    ASSERT_NE(module_, nullptr);
    auto text = Print();
    EXPECT_NE(text.find("scalar.extui"), std::string::npos);
    EXPECT_EQ(text.find("to index"), std::string::npos);
  }
}

TEST_F(ImportTest, PointerUnaryPlusPreservesRootAndOrigin) {
  IREE_ASSERT_OK(Import(IREE_SV("int* identity(int* p) { return +p; }")));
  ASSERT_NE(module_, nullptr);
  EXPECT_NE(Print().find("func.return %p, %p_byte_offset : buffer, offset"),
            std::string::npos);
}

TEST_F(ImportTest, UnsupportedPointerOperationsProduceSourceDiagnostics) {
  for (auto source : {IREE_SV("long diff(int* a, int* b) { return a - b; }"),
                      IREE_SV("bool equal(int* a, int* b) { return a == b; }"),
                      IREE_SV("bool truth(int* a) { return a; }"),
                      IREE_SV("int* local() { int x = 1; return &x; }")}) {
    IREE_EXPECT_OK(Import(source));
    EXPECT_EQ(module_, nullptr);
  }
  EXPECT_GE(diagnostic_count_, 4);
}

TEST_F(ImportTest, PointerIncrementResultsPreserveTheSourceABI) {
  IREE_ASSERT_OK(
      Import(IREE_SV("int* next(int* p) { return ++p; }\n"
                     "int* previous(int* p) { return p++; }\n"
                     "int* retreat(int* p) { return --p; }\n"
                     "int* old_retreat(int* p) { return p--; }\n")));
  ASSERT_NE(module_, nullptr);
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

TEST_F(ImportTest, MultipleKernelsAndSymbolicLaunch) {
  IREE_ASSERT_OK(
      Import(IREE_SV("[[loom::kernel, loom::workgroup_size(64, 1, 1), "
                     "loom::workgroup_count(2, 1, 1)]] void first() {}\n"
                     "[[loom::kernel]] void second() {}\n")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("kernel.def @first"), std::string::npos);
  EXPECT_NE(text.find("kernel.def @second"), std::string::npos);
  EXPECT_NE(text.find("config.decl @second.workgroup_size.x"),
            std::string::npos);
  EXPECT_EQ(text.find("config.decl @first"), std::string::npos);
}

TEST_F(ImportTest, LaunchBoundsFromDeclarationsBecomeConfigPredicates) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    [[loom::kernel, loom::workgroup_count_range(1, 4, 1, 1, 1, 1)]]
    void bounded();
    [[using loom: workgroup_size_range(32, 128, 1, 1, 1, 1)]]
    void bounded() {}
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("config.decl @bounded.workgroup_count.x"),
            std::string::npos);
  EXPECT_NE(text.find("config.decl @bounded.workgroup_size.x"),
            std::string::npos);
  EXPECT_NE(text.find(", 1, 4)]"), std::string::npos);
  EXPECT_NE(text.find(", 32, 128)]"), std::string::npos);
  EXPECT_NE(text.find("kernel.def @bounded"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, OverloadedKernelsHaveDistinctConfigurationSymbols) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    namespace kernels {
    [[loom::kernel]] void entry(float* output) {}
    [[loom::kernel]] void entry(int* output) {}
    }
  )")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("config.decl @kernels__entry.workgroup_count.x"),
            std::string::npos);
  EXPECT_NE(text.find("config.decl @kernels__entry_1.workgroup_count.x"),
            std::string::npos);
}

TEST_F(ImportTest, RejectsInvalidOrContradictoryLaunchContracts) {
  for (const char* attributes : {
           "loom::workgroup_size(64.5, 1, 1)",
           "loom::workgroup_size(0, 1, 1)",
           "loom::workgroup_size(2147483648u, 1, 1)",
           "loom::workgroup_size(64, 1)",
           "loom::workgroup_size",
           "loom::workgroup_size(64, 1, 1), loom::workgroup_size(64, 1, 1)",
           "loom::workgroup_size(64, 1, 1), "
           "loom::workgroup_size_range(32, 128, 1, 1, 1, 1)",
           "loom::workgroup_count_range(4, 1, 1, 1, 1, 1)",
           "loom::workgroup_count_range(1, 4, 1, 1, 1)",
           "loom::workgroup_count_range(1, 4, 1, 1, 1, 1, 1)",
       }) {
    SCOPED_TRACE(attributes);
    auto source =
        std::string("[[loom::kernel, ") + attributes + "]] void entry() {}";
    int before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    [[loom::kernel, loom::workgroup_size(32, 1, 1)]] void entry();
    [[loom::workgroup_size(64, 1, 1)]] void entry() {}
  )")));
  EXPECT_EQ(module_, nullptr);
  IREE_ASSERT_OK(Import(IREE_SV(
      "[[loom::workgroup_size(64, 1, 1)]] void ordinary_function() {}")));
  EXPECT_EQ(module_, nullptr);
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

TEST_F(ImportTest, PreAndPostTestLoopsPreserveScalarRecurrences) {
  IREE_ASSERT_OK(Import(IREE_SV(R"cpp(
    unsigned long long entry(unsigned count) {
      unsigned char narrow = 254;
      unsigned long long wide = 0;
      while (wide < count) {
        do {
          ++narrow;
          ++wide;
        } while (narrow < 2);
      }
      for (; wide < count + 2;) {
        ++wide;
      }
      return wide + narrow;
    }
  )cpp")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("scf.while"), std::string::npos);
  EXPECT_NE(text.find("i8"), std::string::npos);
  EXPECT_NE(text.find("i64"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, TemplateLoopSchedulesBecomeExplicitSSAOperands) {
  IREE_ASSERT_OK(Import(IREE_SV(R"cpp(
    template <unsigned Factor, unsigned Depth>
    static int sum(unsigned count) {
      int total = 0;
      [[loom::unroll(Factor), loom::pipeline(Depth + 1),
        loom::schedule("interleaved")]]
      for (unsigned index = 0; index < count; ++index) {
        total += (int)index;
      }
      return total;
    }
    int entry(unsigned count) { return sum<3, 1>(count); }
  )cpp")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("pipeline(%"), std::string::npos);
  EXPECT_NE(text.find("unroll(%"), std::string::npos);
  EXPECT_NE(text.find("schedule(interleaved)"), std::string::npos);
  EXPECT_EQ(text.find("scf.while"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, BareUnrollAndConstantExpressionSchedules) {
  IREE_ASSERT_OK(Import(IREE_SV(R"cpp(
    int entry() {
      constexpr unsigned depth = 1;
      int total = 0;
      [[using loom: unroll, pipeline(depth), schedule("recurrence")]]
      for (unsigned index = 0; index < 4u; ++index) {
        total += (int)index;
      }
      return total;
    }
  )cpp")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("unroll schedule(recurrence)"), std::string::npos);
  EXPECT_NE(text.find("pipeline(%"), std::string::npos);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, InvalidLoopSchedulesAreNotSilentlyDiscarded) {
  for (const char* attributes :
       {"loom::unroll(0)", "loom::unroll(-1)", "loom::unroll(2.5)",
        "loom::unroll(2147483648u)", "loom::unroll()", "loom::unroll(2, 3)",
        "loom::unroll, loom::unroll(2)", "loom::pipeline",
        "loom::pipeline(count)", "loom::pipeline(1), loom::pipeline(2)",
        "loom::schedule(3)", "loom::schedule(\"linear\")",
        "loom::unroll(2), loom::schedule(\"unknown\")",
        "loom::unroll(2), loom::schedule(\"linear\"), "
        "loom::schedule(\"linear\")",
        "loom::surprise(2)"}) {
    SCOPED_TRACE(attributes);
    auto source = std::string("int entry(unsigned count) { int total = 0; [[") +
                  attributes +
                  "]] for (unsigned i = 0; i < count; ++i) { total += (int)i; "
                  "} return total; }";
    int previous = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
    EXPECT_EQ(module_, nullptr);
    EXPECT_EQ(diagnostic_count_, previous + 1);
  }
  for (const char* loop :
       {"[[loom::unroll(2)]] while (count) { --count; }",
        "[[loom::pipeline(2)]] do { --count; } while (count);",
        "[[loom::unroll]] for (int i = 0; i < (int)count; ++i) {}"}) {
    SCOPED_TRACE(loop);
    auto source = std::string("unsigned entry(unsigned count) { ") + loop +
                  " return count; }";
    int previous = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_string_view(source.data(), source.size())));
    EXPECT_EQ(module_, nullptr);
    EXPECT_EQ(diagnostic_count_, previous + 1);
  }
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

TEST_F(ImportTest, StandardSelectionUsesFrontendMacros) {
  options_.standard = IREE_SV("c23");
  IREE_ASSERT_OK(Import(
      IREE_SV("#if __STDC_VERSION__ != 202311L\n#error wrong standard\n#endif\n"
              "int entry(void) { return 1; }\n")));
  ASSERT_NE(module_, nullptr);
  options_.standard = IREE_SV("c++17");
  IREE_ASSERT_OK(Import(
      IREE_SV("#if __cplusplus != 201703L\n#error wrong standard\n#endif\n"
              "int entry() { return 1; }\n")));
  ASSERT_NE(module_, nullptr);
  options_.standard = IREE_SV("unknown");
  IREE_EXPECT_STATUS_IS(IREE_STATUS_INVALID_ARGUMENT, Import(IREE_SV("")));
}

TEST_F(ImportTest, PromotionsAndAssignmentConversionsProduceVerifiedIR) {
  IREE_ASSERT_OK(Import(IREE_SV(
      "unsigned long long shift(unsigned long long x, unsigned count) { return "
      "x << count; }\n"
      "unsigned char narrow(unsigned char x) { x += 300; ++x; return x; }\n"
      "unsigned char retreat(unsigned char x) { x--; return x; }\n"
      "short signed_retreat(short x) { --x; return x; }\n"
      "unsigned long long increment(unsigned long long x) { ++x; return x; }\n"
      "bool truth(double x) { return bool(x); }\n"
      "double widen(bool x) { return double(x); }\n")));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 0);
}

TEST_F(ImportTest, SourceDataModelControlsLongWidth) {
  options_.data_model = LOOM_CXX_DATA_MODEL_LP64;
  IREE_ASSERT_OK(Import(IREE_SV("long identity(long x) { return x; }")));
  ASSERT_NE(module_, nullptr);
  EXPECT_NE(Print().find("%x: i64"), std::string::npos);
  options_.data_model = LOOM_CXX_DATA_MODEL_LLP64;
  IREE_ASSERT_OK(Import(IREE_SV("long identity(long x) { return x; }")));
  ASSERT_NE(module_, nullptr);
  EXPECT_NE(Print().find("%x: i32"), std::string::npos);
}

TEST_F(ImportTest, IncrementRejectsUnprojectedLvaluesAndSourceTypes) {
  for (const char* source : {
           "unsigned entry() { const unsigned x = 1; return x++; }",
           "unsigned entry() { volatile unsigned x = 1; return ++x; }",
           "unsigned entry(unsigned* pointer) { return (*pointer)++; }",
           "unsigned entry(unsigned* pointer) { return ++pointer[0u]; }",
           "float entry(float value) { return value++; }",
           "extern unsigned value; unsigned entry() { return ++value; }",
           "using V = unsigned __attribute__((vector_size(16))); "
           "V entry(V value) { return value++; }",
       }) {
    SCOPED_TRACE(source);
    auto before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_cstring_view(source)));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
}

TEST_F(ImportTest, ShortCircuitOperandsUseContextualBooleanConversions) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    bool conjunction(int left, float right) { return left && right; }
    bool disjunction(double left, unsigned long long right) {
      return left || right;
    }
    bool nested(int a, int b, int c) { return a && (b || !c); }
    bool guarded(const int* input, unsigned index, unsigned length) {
      return index < length && input[index];
    }
    unsigned scan(const int* input, unsigned length) {
      unsigned index = 0;
      while (index < length && input[index]) { ++index; }
      return index;
    }
  )")));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 0);
  auto text = Print();
  EXPECT_NE(text.find("scf.if"), std::string::npos);
  EXPECT_NE(text.find("scalar.cmpf une"), std::string::npos);
}

TEST_F(ImportTest, ConditionalReturnsKeepOneStructuredFunctionExit) {
  IREE_ASSERT_OK(Import(IREE_SV(R"(
    int guarded(const int* input, unsigned length) {
      if (length == 0u) { return 3; }
      return input[0u];
    }
    int nested(int x, int y) {
      if (x) { if (y) return 2; return 3; }
      else { return 4; }
    }
    int partial(int x, int y) {
      if (x) { if (y) return 2; }
      else { return 4; }
      return 3;
    }
    int state(int x, int y) {
      if (x < 0) { y += 10; return y; }
      else { y += 4; }
      { if (x == 0) return y; y += 2; }
      return y;
    }
    int convert_result(bool choose, float value) {
      if (choose) return value;
      return 7.5;
    }
    void store(int* output) { output[0u] = 7; }
    void effect(int* output, bool choose) {
      if (choose) return store(output);
      output[0u] = 9;
    }
    void unreachable(int* output) { return; output[0u] = 10; }
  )")));
  ASSERT_NE(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 0);
  auto text = Print();
  EXPECT_NE(text.find("scf.if"), std::string::npos);
  EXPECT_EQ(text.find("cfg."), std::string::npos);
  EXPECT_EQ(text.find("scalar.poison"), std::string::npos);
}

TEST_F(ImportTest, RejectsIncompleteReturnsAndUnprojectedExitJoins) {
  for (const char* source : {
           "int entry(int x) { if (x) return 1; }",
           "int entry() { return; }",
           "void entry() { return 1; }",
           "int entry(int x, int y) { if (x) { if (y) return 1; x = 2; } "
           "return x; }",
           "int entry(int x) { while (x) { return x; } return 0; }",
           "int entry(int x) { do { return x; } while (x); }",
           "int entry() { for (unsigned i=0; i<4u; ++i) { return 1; } "
           "return 0; }",
       }) {
    SCOPED_TRACE(source);
    auto before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_cstring_view(source)));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
}

TEST_F(ImportTest, VoidHelpersAndVisibility) {
  IREE_ASSERT_OK(
      Import(IREE_SV("[[gnu::visibility(\"hidden\")]] void helper(float* p) { "
                     "p[0u] = 1.0f; }\n"
                     "void entry(float* p) { helper(p); }\n")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("func.def public @entry"), std::string::npos);
  EXPECT_NE(text.find("func.def @helper"), std::string::npos);
}

TEST_F(ImportTest, GlobalConstantsAndUnsupportedStorage) {
  IREE_ASSERT_OK(Import(IREE_SV(
      "constexpr int factor = 3; int entry(int x) { return x * factor; }\n"
      "constexpr unsigned mask = 0xffffffffu; unsigned bits() { return mask; "
      "}\n"
      "constexpr unsigned char byte = 255; unsigned char small() { return "
      "byte; }\n")));
  ASSERT_NE(module_, nullptr);
  IREE_ASSERT_OK(Import(IREE_SV("int global = 3; int entry() { return 1; }")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
}

TEST_F(ImportTest, ExternalStorageCannotBecomeAnAutomaticBinding) {
  IREE_ASSERT_OK(
      Import(IREE_SV("extern int counter; void entry() { counter = 3; }")));
  EXPECT_EQ(module_, nullptr);
  EXPECT_EQ(diagnostic_count_, 1);
}

TEST_F(ImportTest, WideLoopComparisonRetainsInductionWraparound) {
  IREE_ASSERT_OK(Import(IREE_SV(
      "unsigned entry(unsigned long long upper) { unsigned sum = 0; "
      "for (unsigned i = 0; i < upper; ++i) { sum += i; } return sum; }")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("scf.while"), std::string::npos);
  EXPECT_EQ(text.find("scf.for"), std::string::npos);
}

TEST_F(ImportTest, GeneratedIntrinsicSignaturesAreChecked) {
  for (const char* source : {
           "[[loom::op(\"scalar.expf\", 42)]] float broken(float); float "
           "entry(float x) { return broken(x); }",
           "[[loom::op(\"scalar.expf\"), loom::op(\"scalar.logf\")]] float "
           "broken(float); float entry(float x) { return broken(x); }",
           "[[loom::op(\"scalar.expf\")]] float broken(float x) { return x; }",
           "[[loom::op(\"scalar.expf\")]] float broken(int); float entry(int "
           "x) { return broken(x); }",
           "[[loom::op(\"scalar.expf\")]] int broken(int); int entry(int x) { "
           "return broken(x); }",
           "[[loom::op(\"scalar.expf\")]] float broken(float, float); float "
           "entry(float x) { return broken(x, x); }",
           "[[loom::op(\"scalar.expf\", \"surprise\")]] float broken(float); "
           "float entry(float x) { return broken(x); }",
           "[[loom::op(\"scf.for\")]] float broken(float); float entry(float "
           "x) { return broken(x); }",
       }) {
    SCOPED_TRACE(source);
    auto before = diagnostic_count_;
    IREE_ASSERT_OK(Import(iree_make_cstring_view(source)));
    EXPECT_EQ(module_, nullptr);
    EXPECT_GT(diagnostic_count_, before);
  }
}

TEST_F(ImportTest, GeneratedIntrinsicFlagsAndTypes) {
  IREE_ASSERT_OK(Import(
      IREE_SV("[[loom::op(\"scalar.expf\", \"afn\")]] float approximate(float);"
              "[[loom::op(\"scalar.maxnumf\")]] double maximum(double, double);"
              "[[loom::op(\"scalar.sqrtf\")]] _Float16 root(_Float16);"
              "float entry(float x) { return approximate(x); }"
              "double wide(double x) { return maximum(x, 1.0); }"
              "_Float16 narrow(_Float16 x) { return root(x); }")));
  ASSERT_NE(module_, nullptr);
  auto text = Print();
  EXPECT_NE(text.find("scalar.expf<afn>"), std::string::npos);
  EXPECT_NE(text.find("scalar.maxnumf"), std::string::npos);
  EXPECT_NE(text.find("scalar.sqrtf"), std::string::npos);
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
