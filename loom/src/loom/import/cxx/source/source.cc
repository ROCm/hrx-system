// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/import/cxx/source/source.h"

#include "loom/import/cxx/source/preamble.h"

namespace loom::cxx_import {

Source::Source(iree_string_view_t contents, iree_string_view_t filename,
               const loom_cxx_import_options_t& options,
               cxx::ConstexprCallProvider* constexpr_call_provider)
    : frontend_(filename, options) {
  frontend_.unit().setConstexprCallProvider(constexpr_call_provider);
  if (restore_preamble(options.preamble, contents, options, frontend_)) {
    return;
  }
  frontend_.preprocess(contents);
  frontend_.parse();
}

}  // namespace loom::cxx_import
