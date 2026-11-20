// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_IMPORT_CXX_TRANSLATION_MODULE_H_
#define LOOM_IMPORT_CXX_TRANSLATION_MODULE_H_

#include <vector>

#include "loom/import/cxx/symbol/kernel_launch.h"

struct loom_cxx_import_options_t;
struct loom_module_t;

namespace cxx {
class TranslationUnit;
}  // namespace cxx

namespace loom::cxx_import {

class Diagnostics;

// Projects checked source symbols and structured control flow into native IR.
// Throws SourceRejected after a source diagnostic, or StatusError on allocation
// or sink failure. The import entry point owns both exception boundaries.
// |out_kernel_launches|, if set, receives the host launch facts of every
// kernel_ref specialization reached from host code.
void translate(cxx::TranslationUnit& unit, Diagnostics& diagnostics,
               loom_module_t* module, const loom_cxx_import_options_t& options,
               std::vector<KernelLaunch>* out_kernel_launches = nullptr);

}  // namespace loom::cxx_import

#endif  // LOOM_IMPORT_CXX_TRANSLATION_MODULE_H_
