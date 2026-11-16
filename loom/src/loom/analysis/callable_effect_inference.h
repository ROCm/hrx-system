// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Whole-module callable effect inference.

#ifndef LOOM_ANALYSIS_CALLABLE_EFFECT_INFERENCE_H_
#define LOOM_ANALYSIS_CALLABLE_EFFECT_INFERENCE_H_

#include "iree/base/api.h"
#include "loom/ir/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

// Infers the strongest pure or deterministic read-only contract satisfied by
// each bodyful callable SCC and writes it to unannotated direct call sites.
// Explicit function and call contracts remain authoritative upper bounds and
// are never rewritten. Function definitions remain unannotated; the inferred
// contracts describe the concrete call graph finalized by the frontend.
//
// Every function root-region operation is classified during call-graph
// discovery. SCC propagation is linear in functions and distinct call
// edges. Transient graph and work storage is returned before this function
// exits; only call attributes and their maintained region summaries change.
// The module must have complete local symbol definitions and call references.
// Modules without semantic or native calls return without walking or
// allocating.
iree_status_t loom_callable_effect_inference_run(loom_module_t* module);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_ANALYSIS_CALLABLE_EFFECT_INFERENCE_H_
