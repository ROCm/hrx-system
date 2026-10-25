// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_PARAMETER_LAYOUT_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_PARAMETER_LAYOUT_H_

#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Checks whether independently compiled commands can share already-prepared
// parameter roots. Keys, offsets, extents, bindings and alignment must match;
// workspace and execution order are independent. This external model-source
// boundary does not infer numerical/state compatibility from equal placement.
iree_status_t loom_serve_parameter_layout_check(
    const loom_cmd_program_t* baseline, const loom_cmd_program_t* candidate);

// Owned canonical placement for one parameter domain. Keeps only reflected
// keys and layout facts, not an executable or its compiled/native products.
typedef struct loom_serve_parameter_layout_t loom_serve_parameter_layout_t;

// Copies one domain from a checked compiler product. The caller owns the
// selected binding ordinal and publishes this layout only after its complete
// preparation succeeds. No program storage is borrowed after this call.
iree_status_t loom_serve_parameter_layout_allocate_domain(
    const loom_cmd_program_t* program, uint32_t domain,
    iree_allocator_t host_allocator,
    loom_serve_parameter_layout_t** out_layout);

// Checks a new source product against the retained domain. A different binding
// ordinal is allowed; keys, order, placement, footprint, and alignment match.
iree_status_t loom_serve_parameter_layout_check_domain(
    const loom_serve_parameter_layout_t* layout,
    const loom_cmd_program_t* program, uint32_t domain);

void loom_serve_parameter_layout_free(loom_serve_parameter_layout_t* layout);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_PARAMETER_LAYOUT_H_
