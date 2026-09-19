// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/source.h"

#include <cxx/ast.h>
#include <cxx/control.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>

#include <filesystem>
#include <map>
#include <optional>
#include <string>

#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {
namespace {

TEST(SourceTest, LayoutAndMutableSemanticStateBelongToEachSource) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source first(IREE_SV("static_assert(sizeof(long) == 8); int value = 1;"),
               IREE_SV("first.cpp"), options);
  options.data_model = LOOM_CXX_DATA_MODEL_LLP64;
  Source second(IREE_SV("static_assert(sizeof(long) == 4); int value = 2;"),
                IREE_SV("second.cpp"), options);
  options.data_model = LOOM_CXX_DATA_MODEL_ILP32;
  Source third(IREE_SV("static_assert(sizeof(void*) == 4); int value = 3;"),
               IREE_SV("third.cpp"), options);
  EXPECT_EQ(first.unit().control()->memoryLayout()->sizeOfLong(), 8);
  EXPECT_EQ(second.unit().control()->memoryLayout()->sizeOfLong(), 4);
  EXPECT_EQ(third.unit().control()->memoryLayout()->sizeOfPointer(), 4);
  EXPECT_NE(first.unit().ast(), second.unit().ast());
  EXPECT_NE(first.unit().globalScope(), second.unit().globalScope());
}

TEST(SourceTest, ProviderBytesAreCopiedBeforeTheNextCallback) {
  struct Store {
    // Mutable provider buffer invalidated by every subsequent lookup.
    std::string scratch;
    // Candidate request counts, including missing paths.
    std::map<std::string, unsigned> requests;
  } store;
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.source_provider = {
      [](void* user_data, iree_string_view_t path, bool* out_found,
         iree_string_view_t* out_source) {
        auto& store = *static_cast<Store*>(user_data);
        std::string filename(path.data, path.size);
        ++store.requests[filename];
        store.scratch.assign(256, '?');
        if (filename == "/include/first.h") {
          store.scratch =
              "#pragma once\n#include <second.h>\n"
              "constexpr int first = second + 1;\n";
        } else if (filename == "/include/second.h") {
          store.scratch = "#pragma once\nconstexpr int second = 41;\n";
        } else {
          *out_found = false;
          *out_source = iree_string_view_empty();
          return iree_ok_status();
        }
        *out_found = true;
        *out_source = view(store.scratch);
        return iree_ok_status();
      },
      &store};
  iree_string_view_t directory = IREE_SV("/include");
  options.include_paths = &directory;
  options.include_path_count = 1;
  Source source(
      IREE_SV("#include <first.h>\n#include <first.h>\n"
              "#if __has_include(<missing.h>)\n#error missing\n#endif\n"
              "#if __has_include(<missing.h>)\n#error missing\n#endif\n"
              "static_assert(first == 42);\n"),
      IREE_SV("/app/source.cpp"), options);
  store.scratch.clear();
  ASSERT_NE(source.unit().ast(), nullptr);
  EXPECT_EQ(store.requests.at("/include/first.h"), 1u);
  EXPECT_EQ(store.requests.at("/include/second.h"), 1u);
  EXPECT_EQ(store.requests.at("/include/missing.h"), 1u);
}

TEST(SourceTest, RejectedSourceLeavesTheNextInvocationIndependent) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  EXPECT_THROW(Source(IREE_SV("int invalid = ;"), IREE_SV("bad.cpp"), options),
               SourceRejected);
  Source source(IREE_SV("int valid = 7;"), IREE_SV("good.cpp"), options);
  EXPECT_FALSE(source.diagnostics().has_error());
  ASSERT_NE(source.unit().ast(), nullptr);
}

TEST(SourceTest, ComparisonsDoNotSpeculateOnNonTemplateArguments) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  constexpr bool bounded(unsigned a, unsigned b, unsigned c,
                                         unsigned d, unsigned e, unsigned f,
                                         unsigned g) {
                    return a < 256u && b < 256u && c < 256u && d < 256u &&
                           e < 256u && f < 256u && g < 256u;
                  }
                  static_assert(bounded(0, 1, 2, 3, 4, 5, 255));
                  static_assert(!bounded(0, 1, 2, 3, 4, 5, 256));

                  namespace bounds {
                  constexpr unsigned first = 1;
                  constexpr unsigned second = 2;
                  }  // namespace bounds
                  static_assert(bounds::first < 2u && bounds::second < 3u);

                  struct Bounds {
                    unsigned first;
                    unsigned second;
                  };
                  constexpr Bounds pair{1, 2};
                  static_assert(pair.first < 2u && pair.second < 3u);

                  template <unsigned Limit>
                  constexpr bool below(unsigned value) {
                    return value < Limit && Limit < 256u;
                  }
                  static_assert(below<16>(15));
                  static_assert(!below<16>(16));
                )cpp"),
                IREE_SV("comparisons.cpp"), options);
  EXPECT_FALSE(source.diagnostics().has_error());
}

TEST(SourceTest, TemplateLookaheadRetainsOverloadsCastsAndDependentNames) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  Source source(IREE_SV(R"cpp(
                  constexpr int increment(int value) {
                    return value + 10;
                  }
                  template <class T>
                  constexpr T increment(T value) {
                    return T(value) + T{1};
                  }
                  static_assert(increment(2) == 12);
                  static_assert(increment<unsigned>(2) == 3);

                  template <class T>
                  struct Box {
                    using type = T;
                    template <class U>
                    struct Rebind {
                      using type = U;
                    };
                    template <class U>
                    constexpr U get() const {
                      return U{3};
                    }
                  };
                  template <class T, class U>
                  constexpr U extract(const Box<T>& box) {
                    return box.template get<U>();
                  }
                  constexpr Box<float> box;
                  static_assert(extract<float, int>(box) == 3);

                  template <class T>
                  struct Alias {
                    using type = typename T::template Rebind<int>::type;
                  };
                  static_assert(sizeof(Alias<Box<int>>::type) == sizeof(int));
                )cpp"),
                IREE_SV("template_names.cpp"), options);
  EXPECT_FALSE(source.diagnostics().has_error());
}

TEST(SourceTest, IncludeDirectoryRetainsFilesystemRoot) {
  const auto root = std::filesystem::current_path().root_path();
  const auto directory = root.string();
  const auto include_path = view(directory);
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.include_paths = &include_path;
  options.include_path_count = 1;
  Source source(IREE_SV("int value = 1;"), IREE_SV("source.cpp"), options);
  const auto& paths = source.unit().preprocessor()->userIncludePaths();
  ASSERT_EQ(paths.size(), 1u);
  EXPECT_EQ(paths.front(), root.generic_string());
}

TEST(SourceTest, DiagnosticSinkFailureCrossesTheParserSafeBoundary) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  options.diagnostic_sink = {[](void*, const loom_diagnostic_t*) {
                               return iree_make_status(IREE_STATUS_CANCELLED,
                                                       "consumer stopped");
                             },
                             nullptr};
  try {
    Source source(IREE_SV("int invalid = ;"), IREE_SV("bad.cpp"), options);
    FAIL() << "Expected the retained diagnostic sink failure";
  } catch (StatusError& error) {
    IREE_EXPECT_STATUS_IS(IREE_STATUS_CANCELLED, error.release());
  }
}

TEST(SourceTest, CopiedProviderFailureRetainsOriginalStatus) {
  loom_cxx_import_options_t options;
  loom_cxx_import_options_initialize(&options);
  iree_status_t provided_status = nullptr;
  options.source_provider = {
      [](void* user_data, iree_string_view_t, bool*, iree_string_view_t*) {
        auto status =
            iree_make_status(IREE_STATUS_UNAVAILABLE, "provider failed");
        // Retain only the identity; the source boundary owns the status.
        *static_cast<iree_status_t*>(user_data) = status;
        return status;
      },
      &provided_status};
  std::optional<StatusError> retained_error;
  try {
    Source source(IREE_SV("#include \"missing.h\"\n"),
                  IREE_SV("/app/source.cpp"), options);
    FAIL() << "Expected the provider failure";
  } catch (const StatusError& error) {
    retained_error.emplace(error);
  }
  ASSERT_TRUE(retained_error.has_value());
  auto status = retained_error->release();
  EXPECT_EQ(status, provided_status);
  IREE_EXPECT_STATUS_IS(IREE_STATUS_UNAVAILABLE, status);
}

}  // namespace
}  // namespace loom::cxx_import
