// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_FRONTEND_H_
#define LOOM_IMPORT_CXX_SOURCE_FRONTEND_H_

#include <cxx/toolchain.h>
#include <cxx/translation_unit.h>

#include <cstdint>
#include <string>
#include <string_view>

#include "loom/import/cxx/source/contents.h"
#include "loom/import/cxx/source/diagnostics.h"
#include "loom/import/cxx/source/options.h"

namespace loom::cxx_import {

// Owns one mutable CXX frontend and its exact physical source snapshot. The
// caller selects cold parsing or restored-prefix resumption explicitly.
class Frontend {
 public:
  Frontend(iree_string_view_t filename,
           const loom_cxx_import_options_t& options);
  Frontend(const Frontend&) = delete;
  Frontend& operator=(const Frontend&) = delete;

  // Preprocesses one primary source. Preamble-only mode stops before the first
  // main-file input token and retains the resumable preprocessor state.
  void preprocess(iree_string_view_t source, bool preamble_only = false);
  // Parses a complete cold source or a producer's committed preamble.
  void parse();
  // Resumes parsing after an immutable semantic prefix has been adopted.
  void resume();

  cxx::TranslationUnit& unit() { return unit_; }
  // Source profile that configured the unit: target triple, memory layout,
  // language and exception support.
  const cxx::Toolchain& toolchain() const { return *toolchain_; }
  Diagnostics& diagnostics() { return diagnostics_; }
  SourceContents& contents() { return contents_; }

 private:
  // Outlives diagnostics and the frontend source store that borrow it.
  SourceContents contents_;
  // Outlives the frontend that can deliver diagnostics during teardown.
  Diagnostics diagnostics_;
  // Owns the memory layout borrowed by the translation unit.
  std::unique_ptr<cxx::Toolchain> toolchain_;
  // Destroyed before its borrowed layout, diagnostics and source contents.
  cxx::TranslationUnit unit_;
};

// Process-wide counts of frontend phases, for tools that show how often a
// translation unit was preprocessed and parsed.
struct FrontendStatistics {
  // Primary sources preprocessed, including preamble-only runs.
  uint64_t preprocess_count = 0;
  // Cold parses plus parses resumed after a restored preamble.
  uint64_t parse_count = 0;
};
FrontendStatistics frontend_statistics();

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_FRONTEND_H_
