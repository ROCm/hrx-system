// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/source.h"

#include <cxx/ast.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/triple.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <unordered_map>

#include "loom/error/error_catalog.h"
#include "loom/import/cxx/source/catalog.h"
#include "loom/import/cxx/source/error.h"

namespace loom::cxx_import {

void Diagnostics::emit(const cxx::Token& token,
                       loom_diagnostic_severity_t severity,
                       const loom_error_def_t* error,
                       std::string_view message) {
  if (severity == LOOM_DIAGNOSTIC_ERROR) {
    has_error_ = true;
  }
  if (!sink_.fn || !iree_status_is_ok(status_)) {
    return;
  }
  loom_source_range_t range = {};
  range.provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE;
  if (token.fileId()) {
    auto first = preprocessor_->tokenStartPosition(token);
    auto last = preprocessor_->tokenEndPosition(token);
    range.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
    range.filename = view(first.fileName);
    range.source = view(preprocessor_->source(token.fileId()));
    range.start = token.offset();
    range.end = token.offset() + token.length();
    range.start_line = first.line;
    range.start_column = first.column;
    range.end_line = last.line;
    range.end_column = last.column;
  }
  loom_diagnostic_param_t parameter = loom_param_string(view(message));
  loom_diagnostic_t diagnostic = {.severity = severity,
                                  .error = error,
                                  .params = &parameter,
                                  .param_count = 1,
                                  .emitter = LOOM_EMITTER_PARSER,
                                  .origin = range,
                                  .source_location = range};
  status_ = sink_.fn(sink_.user_data, &diagnostic);
}

void Diagnostics::report(const cxx::Diagnostic& diagnostic) {
  auto severity = LOOM_DIAGNOSTIC_REMARK;
  switch (diagnostic.severity()) {
    case cxx::Severity::Fatal:
    case cxx::Severity::Error:
      severity = LOOM_DIAGNOSTIC_ERROR;
      break;
    case cxx::Severity::Warning:
      severity = LOOM_DIAGNOSTIC_WARNING;
      break;
    default:
      break;
  }
  emit(diagnostic.token(), severity, LOOM_ERR_PARSE_036, diagnostic.message());
}

void Diagnostics::finish() {
  check(std::exchange(status_, iree_ok_status()));
  if (has_error_) {
    throw SourceRejected();
  }
}

void Diagnostics::reject(cxx::TranslationUnit& unit, cxx::AST* ast,
                         std::string_view message) {
  cxx::Token token{};
  if (cxx::ast_cast<cxx::TranslationUnitAST>(ast)) {
    token.setFileId(preprocessor_->mainSourceFileId());
  } else {
    token = unit.tokenAt(ast->firstSourceLocation());
  }
  reject(token, message);
}

void Diagnostics::reject(const cxx::Token& token, std::string_view message) {
  emit(token, LOOM_DIAGNOSTIC_ERROR, LOOM_ERR_LOWERING_059, message);
  finish();
  throw SourceRejected();
}

namespace {

// Supplies language macros and an explicit data model without discovering a
// host SDK, compiler driver, or machine-specific include directory.
class SourceToolchain final : public cxx::Toolchain {
 public:
  SourceToolchain(cxx::Preprocessor* preprocessor,
                  const loom_cxx_import_options_t& options)
      : cxx::Toolchain(preprocessor, cxx::Triple{string(options.triple)}) {
    auto spelling = options.standard.size ? string(options.standard) : "c++26";
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
    if (options.data_model == LOOM_CXX_DATA_MODEL_LLP64) {
      layout->setSizeOfLong(4);
      layout->setSizeOfLongDouble(8, 53);
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
    const auto* layout = memoryLayout();
    bool pointer_is_long = layout->sizeOfLong() == layout->sizeOfPointer();
    defineMacro("__SIZE_TYPE__", pointer_is_long ? "long unsigned int"
                                                 : "long long unsigned int");
    defineMacro("__PTRDIFF_TYPE__",
                pointer_is_long ? "long int" : "long long int");
    defineMacro("__SIZEOF_POINTER__", std::to_string(layout->sizeOfPointer()));
    defineMacro("__SIZEOF_LONG__", std::to_string(layout->sizeOfLong()));
    defineMacro("__SIZEOF_INT__", "4");
    defineMacro("__SIZEOF_SHORT__", "2");
    defineMacro("__SIZEOF_LONG_LONG__", "8");
    defineMacro("__loom__", "1");
    addCommonMacros();
    if (language() == cxx::LanguageKind::kC) {
      addCommonC23Macros();
    } else {
      addCommonCxx26Macros();
      defineMacro("__STDCPP_FLOAT16_T__", "1");
      defineMacro("__STDCPP_BFLOAT16_T__", "1");
    }
  }
};

// Per-invocation lookups retain both hits and misses. Immutable providers can
// share source storage, while preprocessing and AST ownership stay local.
class Sources {
 public:
  Sources(loom_cxx_source_provider_t provider, std::string_view builtin_root)
      : provider_(provider), builtin_root_(builtin_root) {}

  const std::optional<std::string>& lookup(const std::string& path) {
    auto [entry, inserted] = contents_.try_emplace(path);
    if (!inserted) {
      return entry->second;
    }
    auto root = builtin_root_;
    if (!root.empty() && path.starts_with(root) && path.size() > root.size() &&
        path[root.size()] == '/') {
      auto contents =
          builtin_include(std::string_view(path).substr(root.size() + 1));
      if (contents) {
        entry->second = *contents;
      }
      return entry->second;
    }
    if (provider_.fn) {
      bool found = false;
      iree_string_view_t contents = iree_string_view_empty();
      check(provider_.fn(provider_.user_data, view(path), &found, &contents));
      if (found) {
        entry->second = string(contents);
      }
      return entry->second;
    }
    std::error_code error;
    auto status = std::filesystem::status(path, error);
    if (error == std::errc::no_such_file_or_directory ||
        error == std::errc::not_a_directory) {
      return entry->second;
    }
    if (error) {
      throw StatusError(iree_make_status(
          IREE_STATUS_UNAVAILABLE, "cannot inspect include '%s': %s",
          path.c_str(), error.message().c_str()));
    }
    if (!std::filesystem::is_regular_file(status)) {
      return entry->second;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input) {
      throw StatusError(iree_make_status(
          IREE_STATUS_UNAVAILABLE, "cannot open include '%s'", path.c_str()));
    }
    entry->second.emplace(std::istreambuf_iterator<char>(input),
                          std::istreambuf_iterator<char>());
    if (input.bad()) {
      throw StatusError(iree_make_status(
          IREE_STATUS_DATA_LOSS, "cannot read include '%s'", path.c_str()));
    }
    return entry->second;
  }

 private:
  // Borrowed provider state for this invocation.
  loom_cxx_source_provider_t provider_;
  // Enabled virtual root for this invocation, empty for external-only lookup.
  std::string_view builtin_root_;
  // Candidate path to immutable source bytes, or a retained miss.
  std::unordered_map<std::string, std::optional<std::string>> contents_;
};

}  // namespace

static void parse_source(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
                         iree_string_view_t source, iree_string_view_t filename,
                         const loom_cxx_import_options_t& options) {
  auto* preprocessor = unit.preprocessor();
  for (size_t i = 0; i < options.include_path_count; ++i) {
    preprocessor->addUserIncludePath(string(options.include_paths[i]));
  }
  for (size_t i = 0; i < options.system_include_path_count; ++i) {
    preprocessor->addSystemIncludePath(string(options.system_include_paths[i]));
  }
  auto root =
      iree_any_bit_set(options.flags, LOOM_CXX_IMPORT_FLAG_NO_BUILTIN_INCLUDES)
          ? std::string_view()
          : builtin_include_root();
  if (!root.empty()) {
    preprocessor->addSystemIncludePath(std::string(root));
  }
  for (size_t i = 0; i < options.define_count; ++i) {
    preprocessor->defineMacro(string(options.defines[i].name),
                              string(options.defines[i].value));
  }
  Sources sources(options.source_provider, root);
  unit.beginPreprocessing(string(source), string(filename));
  for (;;) {
    auto state = unit.continuePreprocessing();
    diagnostics.finish();
    if (std::holds_alternative<cxx::ProcessingComplete>(state)) {
      break;
    }
    if (auto* include = std::get_if<cxx::PendingInclude>(&state)) {
      std::optional<std::string> resolved;
      bool system = false;
      for (const auto& candidate : include->candidates()) {
        if (sources.lookup(candidate.fileName)) {
          resolved = candidate.fileName;
          system = candidate.isSystemHeader;
          break;
        }
      }
      include->resolveWith(resolved, system);
    } else if (auto* content = std::get_if<cxx::PendingFileContent>(&state)) {
      const auto& contents = sources.lookup(content->fileName);
      content->setContent(contents);
    } else if (auto* query = std::get_if<cxx::PendingHasIncludes>(&state)) {
      for (const auto& request : query->requests) {
        bool found = false;
        for (const auto& candidate : request.candidates()) {
          if (sources.lookup(candidate.fileName)) {
            found = true;
            break;
          }
        }
        request.setExists(found);
      }
    }
  }
  unit.endPreprocessing();
  diagnostics.finish();
  unit.parse(
      {.analysisMode = cxx::ParserAnalysisMode::kFull, .validateAst = true});
  diagnostics.finish();
}

Source::Source(iree_string_view_t contents, iree_string_view_t filename,
               const loom_cxx_import_options_t& options)
    : diagnostics_(options.diagnostic_sink), unit_(&diagnostics_) {
  diagnostics_.bind(unit_.preprocessor());
  toolchain_ = std::make_unique<SourceToolchain>(unit_.preprocessor(), options);
  parse_source(unit_, diagnostics_, contents, filename, options);
}

}  // namespace loom::cxx_import
