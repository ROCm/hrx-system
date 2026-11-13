// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cxx/diagnostics_client.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>
#include <cxx/triple.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace {

constexpr size_t kMaximumSourceSize = 64 * 1024;
constexpr std::string_view kVirtualIncludeRoot = "/fuzz/include";

struct VirtualHeader {
  std::string_view filename;
  std::string_view source;
};

constexpr VirtualHeader kVirtualHeaders[] = {
    {
        .filename = "/fuzz/include/fuzz/core.h",
        .source = R"(#pragma once
namespace fuzz {
template <class T>
struct box {
  T value;
  constexpr T get() const { return value; }
};
template <class T>
constexpr T select(bool condition, T true_value, T false_value) {
  return condition ? true_value : false_value;
}
inline constexpr int answer = 42;
}  // namespace fuzz
)",
    },
    {
        .filename = "/fuzz/include/fuzz/macros.h",
        .source = R"(#pragma once
#define FUZZ_CONCAT_IMPL(a, b) a##b
#define FUZZ_CONCAT(a, b) FUZZ_CONCAT_IMPL(a, b)
#define FUZZ_IDENTITY(value) value
)",
    },
    {
        .filename = "/fuzz/include/fuzz/nested.h",
        .source = R"(#pragma once
#include <fuzz/core.h>
namespace fuzz {
template <class T>
constexpr box<T> make_box(T value) { return {value}; }
}  // namespace fuzz
)",
    },
};

std::optional<std::string_view> lookup_virtual_header(
    std::string_view filename) {
  for (const auto& header : kVirtualHeaders) {
    if (header.filename == filename) {
      return header.source;
    }
  }
  return std::nullopt;
}

class DiscardingDiagnostics final : public cxx::DiagnosticsClient {
 public:
  void report(const cxx::Diagnostic&) override {}
};

// Matches the deterministic LP64 frontend environment used by source
// importers without consulting a host compiler, SDK, or filesystem.
class FuzzToolchain final : public cxx::Toolchain {
 public:
  explicit FuzzToolchain(cxx::Preprocessor* preprocessor)
      : cxx::Toolchain(preprocessor, cxx::Triple{"x86_64-unknown-linux-gnu"}) {
    const auto* standard = cxx::findLanguageStandard("c++26");
    setLanguage(standard->language);
    setLanguageStandard(standard);
    auto layout = std::make_unique<cxx::MemoryLayout>(64);
    layout->setSizeOfLongLong(8);
    layout->setTriple("x86_64-unknown-linux-gnu");
    setMemoryLayout(std::move(layout));
    initMemoryLayout();
    addPredefinedMacros();
    addSystemIncludePath(std::string(kVirtualIncludeRoot));
  }

  void addSystemIncludePaths() override {}
  void addSystemCppIncludePaths() override {}

  void addPredefinedMacros() override {
    defineMacro("__SIZE_TYPE__", "long unsigned int");
    defineMacro("__PTRDIFF_TYPE__", "long int");
    defineMacro("__SIZEOF_POINTER__", "8");
    defineMacro("__SIZEOF_LONG__", "8");
    defineMacro("__SIZEOF_INT__", "4");
    defineMacro("__SIZEOF_SHORT__", "2");
    defineMacro("__SIZEOF_LONG_LONG__", "8");
    addCommonMacros();
    addCommonCxx26Macros();
  }
};

void preprocess(cxx::TranslationUnit& unit, std::string source) {
  unit.beginPreprocessing(std::move(source), "input.cxx");
  for (;;) {
    auto state = unit.continuePreprocessing();
    if (std::holds_alternative<cxx::ProcessingComplete>(state)) {
      break;
    }
    if (auto* include = std::get_if<cxx::PendingInclude>(&state)) {
      std::optional<std::string> resolved_filename;
      bool is_system_header = false;
      for (const auto& candidate : include->candidates()) {
        if (lookup_virtual_header(candidate.fileName)) {
          resolved_filename = candidate.fileName;
          is_system_header = candidate.isSystemHeader;
          break;
        }
      }
      include->resolveWith(std::move(resolved_filename), is_system_header);
    } else if (auto* content = std::get_if<cxx::PendingFileContent>(&state)) {
      auto source = lookup_virtual_header(content->fileName);
      content->setContent(source ? std::optional<std::string>(*source)
                                 : std::nullopt);
    } else if (auto* queries = std::get_if<cxx::PendingHasIncludes>(&state)) {
      for (const auto& query : queries->requests) {
        bool exists = false;
        for (const auto& candidate : query.candidates()) {
          if (lookup_virtual_header(candidate.fileName)) {
            exists = true;
            break;
          }
        }
        query.setExists(exists);
      }
    } else if (auto* completion =
                   std::get_if<cxx::PendingIncludeCompletion>(&state)) {
      completion->setEntries({});
    } else if (std::holds_alternative<cxx::CanContinuePreprocessing>(state) ||
               std::holds_alternative<cxx::EnteringFile>(state) ||
               std::holds_alternative<cxx::LeavingFile>(state)) {
      // These states report progress without requesting a response.
      continue;
    } else {
      // A new producer continuation needs an explicit deterministic response.
      std::abort();
    }
  }
  unit.endPreprocessing();
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
  if (size > kMaximumSourceSize) {
    return 0;
  }

  DiscardingDiagnostics diagnostics;
  std::unique_ptr<FuzzToolchain> toolchain;
  {
    // The unit borrows the toolchain's memory layout and must be destroyed
    // before it.
    cxx::TranslationUnit unit(&diagnostics);
    toolchain = std::make_unique<FuzzToolchain>(unit.preprocessor());
    const char* source_data = size ? reinterpret_cast<const char*>(data) : "";
    preprocess(unit, std::string(source_data, size));
    unit.parse({
        .analysisMode = cxx::ParserAnalysisMode::kFull,
    });
  }
  return 0;
}
