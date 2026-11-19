// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_DIAGNOSTICS_H_
#define LOOM_IMPORT_CXX_SOURCE_DIAGNOSTICS_H_

#include <cxx/ast_fwd.h>
#include <cxx/diagnostics_client.h>

#include <string_view>

#include "loom/error/diagnostic.h"

namespace loom::cxx_import {

class SourceContents;

// Forwards frontend diagnostics without invoking its process-exiting renderer.
// Source bytes in each callback belong to the invocation's retained contents.
class Diagnostics final : public cxx::DiagnosticsClient {
 public:
  explicit Diagnostics(loom_diagnostic_sink_t sink) : sink_(sink) {}
  ~Diagnostics() override { iree_status_free(status_); }
  void bind(cxx::TranslationUnit* unit, SourceContents* contents) {
    unit_ = unit;
    contents_ = contents;
  }
  void report(const cxx::Diagnostic& diagnostic) override;
  // Translation-unit errors refer to the start of the primary source file,
  // independent of declarations injected by builtins or included headers.
  [[noreturn]] void reject(cxx::TranslationUnit& unit, cxx::AST* ast,
                           std::string_view message);
  // Diagnoses a semantic source location without requiring current-unit token
  // ownership, including declarations restored from an immutable prefix.
  [[noreturn]] void reject(cxx::SourceLocation location,
                           std::string_view message);
  // Uses a catalog diagnostic with one string parameter and its own fix hint.
  [[noreturn]] void reject(cxx::SourceLocation location,
                           const loom_error_def_t* error,
                           std::string_view argument);
  // Attaches a labeled second C++ source site to the primary error.
  [[noreturn]] void reject(cxx::SourceLocation location,
                           std::string_view message,
                           cxx::SourceLocation related, std::string_view label);
  SourceContents& contents() const { return *contents_; }
  bool has_error() const { return has_error_; }
  // Propagates sink failure and source rejection at a parser-safe boundary.
  // cxx forwards diagnostics from noexcept destructors, so report never throws.
  void finish();

 private:
  void emit(const cxx::Token& token, loom_diagnostic_severity_t severity,
            const loom_error_def_t* error, std::string_view message);
  loom_source_range_t source_range(cxx::SourceLocation location) const;
  void emit(cxx::SourceLocation location, loom_diagnostic_severity_t severity,
            const loom_error_def_t* error, std::string_view argument,
            const loom_diagnostic_related_location_t* related = nullptr);

  // Borrowed callback and state supplied by the current import request.
  loom_diagnostic_sink_t sink_;
  // Borrowed physical source positions and current frontend source storage.
  cxx::TranslationUnit* unit_ = nullptr;
  // Borrowed exact source snapshot used to materialize byte ranges.
  SourceContents* contents_ = nullptr;
  // Forwarded diagnostics can bypass cxx's counter; this owns admission.
  bool has_error_ = false;
  // First sink failure, retained until a safe point can unwind the frontend.
  iree_status_t status_ = iree_ok_status();
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_DIAGNOSTICS_H_
