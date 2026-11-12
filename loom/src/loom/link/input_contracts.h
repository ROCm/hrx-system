// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_LINK_INPUT_CONTRACTS_H_
#define LOOM_LINK_INPUT_CONTRACTS_H_

#include "loom/link/module_index.h"

#ifdef __cplusplus
extern "C" {
#endif

// Retains native template input requirements once during source admission.
// Symbol slots receive one-based ordinals in the returned unique contract
// table; zero denotes no additional requirements. The source's verified
// signatures own all coordinates. Temporary interning storage retires before
// return.
iree_status_t loom_link_project_input_contracts(
    const loom_module_t* module, loom_link_module_index_symbol_t* symbols,
    iree_arena_allocator_t* arena, const loom_input_contract_t*** out_contracts,
    uint32_t* out_count);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_LINK_INPUT_CONTRACTS_H_
