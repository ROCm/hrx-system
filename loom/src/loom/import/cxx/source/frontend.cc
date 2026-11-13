// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/frontend.h"

#include <cxx/ast.h>
#include <cxx/memory_layout.h>
#include <cxx/preprocessor.h>
#include <cxx/triple.h>

#include "loom/error/error_catalog.h"
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
  loom_source_range_t range = {
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
  };
  if (token.fileId()) {
    auto first = unit_->preprocessor()->tokenStartPosition(token);
    auto last = unit_->preprocessor()->tokenEndPosition(token);
    range.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
    range.filename = view(first.fileName);
    range.source = view(contents_->source(first.fileName));
    range.start = token.offset();
    range.end = token.offset() + token.length();
    range.start_line = first.line;
    range.start_column = first.column;
    range.end_line = last.line;
    range.end_column = last.column;
  }
  loom_diagnostic_param_t parameter = loom_param_string(view(message));
  loom_diagnostic_t diagnostic = {
      .severity = severity,
      .error = error,
      .params = &parameter,
      .param_count = 1,
      .emitter = LOOM_EMITTER_PARSER,
      .origin = range,
      .source_location = range,
  };
  status_ = sink_.fn(sink_.user_data, &diagnostic);
}

void Diagnostics::emit(cxx::SourceLocation location,
                       loom_diagnostic_severity_t severity,
                       const loom_error_def_t* error,
                       std::string_view message) {
  if (severity == LOOM_DIAGNOSTIC_ERROR) {
    has_error_ = true;
  }
  if (!sink_.fn || !iree_status_is_ok(status_)) {
    return;
  }
  loom_source_range_t range = {
      .provenance = LOOM_SOURCE_PROVENANCE_UNAVAILABLE_SOURCE,
  };
  if (location) {
    auto first = unit_->tokenStartPosition(location);
    auto last = unit_->tokenEndPosition(location);
    if (!first.fileName.empty() && first.line && first.column) {
      auto physical = contents_->range(first, last);
      range.provenance = LOOM_SOURCE_PROVENANCE_EXACT_SOURCE;
      range.filename = view(first.fileName);
      range.source = view(physical.source);
      range.start = physical.start;
      range.end = physical.end;
      range.start_line = first.line;
      range.start_column = first.column;
      range.end_line = last.line;
      range.end_column = last.column;
    }
  }
  loom_diagnostic_param_t parameter = loom_param_string(view(message));
  loom_diagnostic_t diagnostic = {
      .severity = severity,
      .error = error,
      .params = &parameter,
      .param_count = 1,
      .emitter = LOOM_EMITTER_PARSER,
      .origin = range,
      .source_location = range,
  };
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
  if (cxx::ast_cast<cxx::TranslationUnitAST>(ast)) {
    cxx::Token token{};
    token.setFileId(unit.preprocessor()->mainSourceFileId());
    emit(token, LOOM_DIAGNOSTIC_ERROR, LOOM_ERR_LOWERING_059, message);
  } else {
    emit(ast->firstSourceLocation(), LOOM_DIAGNOSTIC_ERROR,
         LOOM_ERR_LOWERING_059, message);
  }
  finish();
  throw SourceRejected();
}

void Diagnostics::reject(cxx::SourceLocation location,
                         std::string_view message) {
  emit(location, LOOM_DIAGNOSTIC_ERROR, LOOM_ERR_LOWERING_059, message);
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
    // Binary64 long-double limits and fixed source-model definitions stay as
    // immutable text until used; adding a definition adds no per-import setup.
    static constexpr cxx::PredefinedMacro kPredefinedDefinitions[] = {
        {"__LDBL_DECIMAL_DIG__", "17"},
        {"__LDBL_DENORM_MIN__", "4.9406564584124654e-324L"},
        {"__LDBL_DIG__", "15"},
        {"__LDBL_EPSILON__", "2.2204460492503131e-16L"},
        {"__LDBL_MANT_DIG__", "53"},
        {"__LDBL_MAX_10_EXP__", "308"},
        {"__LDBL_MAX_EXP__", "1024"},
        {"__LDBL_MAX__", "1.7976931348623157e+308L"},
        {"__LDBL_MIN_10_EXP__", "(-307)"},
        {"__LDBL_MIN_EXP__", "(-1021)"},
        {"__LDBL_MIN__", "2.2250738585072014e-308L"},
        {"__LDBL_NORM_MAX__", "1.7976931348623157e+308L"},
        {"__SIZEOF_LONG_DOUBLE__", "8"},
    };
    static constexpr cxx::PredefinedMacros kPredefinedMacros{
        kPredefinedDefinitions};
    const auto* layout = memoryLayout();
    bool pointer_is_long = layout->sizeOfLong() == layout->sizeOfPointer();
    defineMacro("__SIZE_TYPE__", pointer_is_long ? "long unsigned int"
                                                 : "long long unsigned int");
    defineMacro("__PTRDIFF_TYPE__",
                pointer_is_long ? "long int" : "long long int");
    defineMacro("__SIZEOF_POINTER__", std::to_string(layout->sizeOfPointer()));
    defineMacro("__SIZEOF_LONG__", std::to_string(layout->sizeOfLong()));
    defineMacro("__loom__", "1");
    addCommonMacros();
    if (language() == cxx::LanguageKind::kC) {
      addCommonC23Macros();
    } else {
      addCommonCxx26Macros();
      defineMacro("__STDCPP_FLOAT16_T__", "1");
      defineMacro("__STDCPP_BFLOAT16_T__", "1");
    }
    preprocessor()->setPredefinedMacros(kPredefinedMacros);
  }
};

}  // namespace

Frontend::Frontend(iree_string_view_t filename,
                   const loom_cxx_import_options_t& options)
    : contents_(options),
      diagnostics_(options.diagnostic_sink),
      unit_(&diagnostics_) {
  auto* preprocessor = unit_.preprocessor();
  contents_.bind(preprocessor, string(filename));
  diagnostics_.bind(&unit_, &contents_);
  toolchain_ = std::make_unique<SourceToolchain>(preprocessor, options);
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
}

void Frontend::preprocess(iree_string_view_t source, bool preamble_only) {
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

void Frontend::parse() {
  unit_.parse(
      {.analysisMode = cxx::ParserAnalysisMode::kFull, .validateAst = true});
  diagnostics_.finish();
}

void Frontend::resume() {
  unit_.resume(
      {.analysisMode = cxx::ParserAnalysisMode::kFull, .validateAst = true});
  diagnostics_.finish();
}

}  // namespace loom::cxx_import
