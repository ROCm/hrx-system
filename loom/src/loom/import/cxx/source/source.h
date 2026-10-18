// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_SOURCE_SOURCE_H_
#define LOOM_IMPORT_CXX_SOURCE_SOURCE_H_

#include "loom/import/cxx/source/frontend.h"

namespace loom::cxx_import {

// Owns one fully preprocessed and type-checked source configuration. The
// translation unit and its mutable semantic state belong exclusively to this
// object. Borrowed AST, symbol, token and layout references expire with it.
// Construction throws SourceRejected after delivering source diagnostics or
// StatusError for provider, option or diagnostic-sink failures.
class Source {
 public:
  Source(iree_string_view_t contents, iree_string_view_t filename,
         const loom_cxx_import_options_t& options,
         cxx::ConstexprCallProvider* constexpr_call_provider = nullptr);
  Source(const Source&) = delete;
  Source& operator=(const Source&) = delete;

  cxx::TranslationUnit& unit() { return frontend_.unit(); }
  Diagnostics& diagnostics() { return frontend_.diagnostics(); }
  SourceContents& contents() { return frontend_.contents(); }

 private:
  // Mutable frontend state owned exclusively by this source invocation.
  Frontend frontend_;
};

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_SOURCE_SOURCE_H_
