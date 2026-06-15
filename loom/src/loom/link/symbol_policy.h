// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared linker symbol identity policy.
//
// The materialized linker and provider index must classify symbols the same
// way or the planner can select a symbol that the final clone step later
// renames as private. Keep in-memory IR policy here and bytecode metadata
// adaptation in the provider index.

#ifndef LOOM_LINK_SYMBOL_POLICY_H_
#define LOOM_LINK_SYMBOL_POLICY_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns true when |symbol| is a declaration-like anchor that may be satisfied
// by a later concrete definition with the same global link name.
bool loom_link_symbol_is_declaration(const loom_symbol_t* symbol);

// Returns true when |symbol| has a concrete defining operation.
bool loom_link_symbol_is_concrete_definition(const loom_symbol_t* symbol);

// Returns true when |symbol| participates in global link identity.
//
// Global-identity symbols are resolved by name across providers. Concrete
// private definitions are resolved only within their provider-local module and
// are renamed if they collide in the linked output.
bool loom_link_symbol_has_global_identity(const loom_module_t* module,
                                          const loom_symbol_t* symbol);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_SYMBOL_POLICY_H_
