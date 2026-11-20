// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/frontend.h"

#include <cxx/ast.h>
#include <cxx/gcc_linux_toolchain.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/triple.h>

#include <atomic>
#include <cstdlib>

#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

namespace {

std::atomic<uint64_t> preprocess_count{0};
std::atomic<uint64_t> parse_count{0};

// Supplies language macros and an explicit data model without discovering a
// host SDK, compiler driver, or machine-specific include directory.
class SourceToolchain final : public cxx::Toolchain {
 public:
  SourceToolchain(cxx::Preprocessor* preprocessor,
                  const loom_cxx_import_options_t& options)
      : cxx::Toolchain(preprocessor, cxx::Triple{string(options.triple)}) {
    auto spelling = string(loom_cxx_import_options_standard(&options));
    auto* standard = cxx::findLanguageStandard(spelling);
    if (!standard) {
      throw StatusError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                         "unsupported C/C++ standard '%s'",
                                         spelling.c_str()));
    }
    setLanguage(standard->language);
    setLanguageStandard(standard);
    auto layout = std::make_unique<cxx::MemoryLayout>(
        options.data_model == LOOM_CXX_DATA_MODEL_ILP32 ? 32 : 64);
    layout->setSizeOfLongLong(8);
    layout->setSizeOfLongDouble(8, 53);
    if (options.data_model == LOOM_CXX_DATA_MODEL_LLP64) {
      layout->setSizeOfLong(4);
      layout->setWideCharUnderlyingType(2, false);
    }
    layout->setTriple(string(options.triple));
    setMemoryLayout(std::move(layout));
    initMemoryLayout();
    addPredefinedMacros();
  }

  void addSystemIncludePaths() override {}
  void addSystemCppIncludePaths() override {}
  void addPredefinedMacros() override {
    // Loom's source model uses binary64 long double. Its definitions take
    // precedence over the common toolchain table on every host.
    static constexpr cxx::PredefinedMacro kPredefinedDefinitions[] = {
        "__LDBL_DECIMAL_DIG__ 17",
        "__LDBL_DENORM_MIN__ 4.9406564584124654e-324L",
        "__LDBL_DIG__ 15",
        "__LDBL_EPSILON__ 2.2204460492503131e-16L",
        "__LDBL_MANT_DIG__ 53",
        "__LDBL_MAX_10_EXP__ 308",
        "__LDBL_MAX_EXP__ 1024",
        "__LDBL_MAX__ 1.7976931348623157e+308L",
        "__LDBL_MIN_10_EXP__ (-307)",
        "__LDBL_MIN_EXP__ (-1021)",
        "__LDBL_MIN__ 2.2250738585072014e-308L",
        "__LDBL_NORM_MAX__ 1.7976931348623157e+308L",
        "__SIZEOF_LONG_DOUBLE__ 8",
    };
    static constexpr cxx::PredefinedMacros kModelMacros[] = {
        cxx::PredefinedMacros{kPredefinedDefinitions}};
    const auto* layout = memoryLayout();
    bool pointer_is_long = layout->sizeOfLong() == layout->sizeOfPointer();
    defineMacro("__SIZE_TYPE__", pointer_is_long ? "long unsigned int"
                                                 : "long long unsigned int");
    defineMacro("__PTRDIFF_TYPE__",
                pointer_is_long ? "long int" : "long long int");
    defineMacro("__SIZEOF_POINTER__", std::to_string(layout->sizeOfPointer()));
    defineMacro("__SIZEOF_LONG__", std::to_string(layout->sizeOfLong()));
    defineMacro("__loom__", "1");
    addCommonMacros(kModelMacros);
    if (language() == cxx::LanguageKind::kC) {
      addCommonC23Macros();
    } else {
      addCommonCxx26Macros();
      defineMacro("__STDCPP_FLOAT16_T__", "1");
      defineMacro("__STDCPP_BFLOAT16_T__", "1");
    }
  }
};

// Hosted single-source profile (LOOM_CXX_IMPORT_FLAG_HOSTED): the device pass
// parses the same headers as a host compile for options.triple, so it uses
// cxx's own Linux toolchain for that triple. Predefined macros, the memory
// layout (including long double) and the target architecture seen by
// __has_builtin then match the host compile. Include paths come from the
// options rather than toolchain discovery. Loom does not import exceptions,
// so they are disabled unless the driver's host compile enables them
// (LOOM_CXX_IMPORT_FLAG_EXCEPTIONS). Nothing marks the device pass: hosted
// source sees the same macros as the host compile.
std::unique_ptr<cxx::Toolchain> make_hosted_toolchain(
    cxx::Preprocessor* preprocessor, const loom_cxx_import_options_t& options) {
  cxx::Triple triple{string(options.triple)};
  if (triple.os() != cxx::TripleOS::kLinux ||
      (triple.arch() != cxx::TripleArch::kX86_64 &&
       triple.arch() != cxx::TripleArch::kAArch64)) {
    throw StatusError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hosted import requires an x86_64 or aarch64 Linux triple, got '%s'",
        string(options.triple).c_str()));
  }
  if (options.data_model != LOOM_CXX_DATA_MODEL_LP64) {
    throw StatusError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "hosted import takes its data model from the triple; use lp64"));
  }
  auto spelling = string(loom_cxx_import_options_standard(&options));
  auto* standard = cxx::findLanguageStandard(spelling);
  if (!standard) {
    throw StatusError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                       "unsupported C/C++ standard '%s'",
                                       spelling.c_str()));
  }
  auto toolchain =
      std::make_unique<cxx::GCCLinuxToolchain>(preprocessor, triple);
  toolchain->setLanguage(standard->language);
  toolchain->setExceptionsEnabled(
      (options.flags & LOOM_CXX_IMPORT_FLAG_EXCEPTIONS) != 0);
  toolchain->initMemoryLayout();
  toolchain->setLanguageStandard(standard);
  toolchain->addPredefinedMacros();
  return toolchain;
}

}  // namespace

Frontend::Frontend(iree_string_view_t filename,
                   const loom_cxx_import_options_t& options)
    : contents_(options),
      diagnostics_(options.diagnostic_sink),
      unit_(&diagnostics_) {
  auto* preprocessor = unit_.preprocessor();
  contents_.bind(preprocessor, string(filename));
  diagnostics_.bind(&unit_, &contents_);
  if (options.flags & LOOM_CXX_IMPORT_FLAG_HOSTED) {
    toolchain_ = make_hosted_toolchain(preprocessor, options);
  } else {
    toolchain_ = std::make_unique<SourceToolchain>(preprocessor, options);
  }
  for (size_t i = 0; i < options.include_path_count; ++i) {
    preprocessor->addUserIncludePath(string(options.include_paths[i]));
  }
  for (size_t i = 0; i < options.system_include_path_count; ++i) {
    preprocessor->addSystemIncludePath(string(options.system_include_paths[i]));
  }
  auto root = contents_.builtin_root();
  if (!root.empty()) {
    preprocessor->addSystemIncludePath(std::string(root));
  }
  for (size_t i = 0; i < options.define_count; ++i) {
    preprocessor->defineMacro(string(options.defines[i].name),
                              string(options.defines[i].value));
  }
  for (size_t i = 0; i < options.undefine_count; ++i) {
    preprocessor->undefMacro(string(options.undefines[i]));
  }
}

void Frontend::preprocess(iree_string_view_t source, bool preamble_only) {
  preprocess_count.fetch_add(1, std::memory_order_relaxed);
  auto* preprocessor = unit_.preprocessor();
  preprocessor->setPreambleOnly(preamble_only);
  unit_.beginPreprocessing(string(source),
                           std::string(contents_.primary_filename()));
  for (;;) {
    auto state = unit_.continuePreprocessing();
    diagnostics_.finish();
    if (std::holds_alternative<cxx::ProcessingComplete>(state)) {
      break;
    }
    if (auto* include = std::get_if<cxx::PendingInclude>(&state)) {
      std::optional<std::string> resolved;
      bool system = false;
      for (const auto& candidate : include->candidates()) {
        if (contents_.resolve(candidate.fileName)) {
          resolved = candidate.fileName;
          system = candidate.isSystemHeader;
          break;
        }
      }
      include->resolveWith(resolved, system);
    } else if (auto* content = std::get_if<cxx::PendingFileContent>(&state)) {
      const auto& resolved = contents_.resolve(content->fileName);
      content->setContent(resolved);
    } else if (auto* query = std::get_if<cxx::PendingHasIncludes>(&state)) {
      for (const auto& request : query->requests) {
        bool found = false;
        for (const auto& candidate : request.candidates()) {
          if (contents_.resolve(candidate.fileName)) {
            found = true;
            break;
          }
        }
        request.setExists(found);
      }
    }
  }
  unit_.endPreprocessing();
  diagnostics_.finish();
}

cxx::ParserConfiguration Frontend::parser_configuration() const {
  // Exception support follows the profile, as in a compiler driver: the
  // hosted profile parses and generates host code without exceptions.
  return {.analysisMode = cxx::ParserAnalysisMode::kFull,
          .exceptionsEnabled = toolchain_->exceptionsEnabled()};
}

void Frontend::parse() {
  parse_count.fetch_add(1, std::memory_order_relaxed);
  unit_.parse(parser_configuration());
  diagnostics_.finish();
}

void Frontend::resume() {
  parse_count.fetch_add(1, std::memory_order_relaxed);
  unit_.resume(parser_configuration());
  diagnostics_.finish();
}

FrontendStatistics frontend_statistics() {
  return {.preprocess_count = preprocess_count.load(std::memory_order_relaxed),
          .parse_count = parse_count.load(std::memory_order_relaxed)};
}

}  // namespace loom::cxx_import
