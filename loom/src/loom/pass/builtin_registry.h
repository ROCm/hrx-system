// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Builtin pass descriptor registry.
//
// This target owns the table that links concrete production pass
// implementations into standard Loom tools. Lower-level pipeline compiler and
// interpreter code should depend on pass_registry.h instead so tests and
// alternate tools can provide smaller registries.

#ifndef LOOM_PASS_BUILTIN_REGISTRY_H_
#define LOOM_PASS_BUILTIN_REGISTRY_H_

#include "loom/pass/registry.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns the builtin registry containing the pass implementations linked into
// the standard Loom compiler/tool build.
const loom_pass_registry_t* loom_pass_builtin_registry(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_PASS_BUILTIN_REGISTRY_H_
