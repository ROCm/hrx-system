// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// C++ RAII helpers for Loom test modules.

#ifndef LOOM_TESTING_MODULE_PTR_H_
#define LOOM_TESTING_MODULE_PTR_H_

#include <memory>

#include "loom/ir/module.h"

namespace loom::testing {

// Deletes a Loom module owned by a C++ test smart pointer.
struct ModuleDeleter {
  void operator()(loom_module_t* module) const { loom_module_free(module); }
};

// Owns a parsed or generated Loom module for the lifetime of a C++ test scope.
using ModulePtr = std::unique_ptr<loom_module_t, ModuleDeleter>;

}  // namespace loom::testing

#endif  // LOOM_TESTING_MODULE_PTR_H_
