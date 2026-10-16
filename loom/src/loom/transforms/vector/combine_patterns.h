// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_VECTOR_COMBINE_PATTERNS_H_
#define LOOM_TRANSFORMS_VECTOR_COMBINE_PATTERNS_H_

#include "loom/rewrite/pattern_registry.h"
#include "loom/rewrite/rewriter.h"

#ifdef __cplusplus
extern "C" {
#endif

// Source-combine patterns contributed by the vector dialect.
extern const loom_rewrite_pattern_provider_t
    loom_vector_source_combine_pattern_provider;

// Composes one verified vector width conversion with its immediate producer,
// including conversion through a splat. Creates source operations that still
// require target legalization. Legalizers use this before decomposing casts;
// universal post-legalization cleanup preserves selected conversion paths.
// |allow_bitfield| permits truncation-extension composition only after native
// conversion selection has failed; source combining preserves those choices.
iree_status_t loom_vector_combine_conversion(loom_op_t* op,
                                             loom_rewriter_t* rewriter,
                                             bool allow_bitfield,
                                             bool* out_changed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TRANSFORMS_VECTOR_COMBINE_PATTERNS_H_
