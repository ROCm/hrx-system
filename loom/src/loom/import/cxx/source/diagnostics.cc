// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/diagnostics.h"

#include <cxx/ast.h>
#include <cxx/preprocessor.h>
#include <cxx/translation_unit.h>

#include "loom/error/error_catalog.h"
#include "loom/import/cxx/source/contents.h"
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

}  // namespace loom::cxx_import
