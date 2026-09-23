// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Scalar transport for dynamic address-layout descriptors.

#ifndef LOOM_TRANSFORMS_ENCODING_LAYOUT_TRANSPORT_H_
#define LOOM_TRANSFORMS_ENCODING_LAYOUT_TRANSPORT_H_

#include "loom/pass/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns immutable metadata for CFG address-layout transport decomposition.
const loom_pass_info_t* loom_decompose_cfg_layout_transports_pass_info(void);

// Returns immutable metadata for structured address-layout transport
// decomposition.
const loom_pass_info_t* loom_decompose_scf_layout_transports_pass_info(void);

// Replaces strided-layout CFG block arguments with the index-valued strides
// that differ across incoming edges. A local encoding.layout.strided rebuilds
// the semantic layout at the destination. Exact axes remain static attributes.
iree_status_t loom_decompose_cfg_layout_transports_run(loom_pass_t* pass,
                                                       loom_module_t* module);

// Replaces varying strided-layout results of structured selections and loops
// with their index-valued strides. A local encoding.layout.strided rebuilds
// each semantic result and loop-region argument.
iree_status_t loom_decompose_scf_layout_transports_run(
    loom_pass_t* pass, loom_module_t* module, loom_func_like_t function);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_ENCODING_LAYOUT_TRANSPORT_H_
