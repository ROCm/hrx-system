// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Source-to-Low target-contract query adaptation.

#ifndef LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_
#define LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_

#include "loom/codegen/low/lower/context.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns a target-contract query callback backed by |context|.
loom_target_contract_query_callback_t
loom_low_lower_context_contract_query_callback(
    loom_low_lower_context_t* context);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_CODEGEN_LOW_LOWER_SOURCE_QUERY_H_
