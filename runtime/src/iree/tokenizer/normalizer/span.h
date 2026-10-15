// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_TOKENIZER_NORMALIZER_SPAN_H_
#define IREE_TOKENIZER_NORMALIZER_SPAN_H_

#include "iree/tokenizer/normalizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Scopes |child| to independently finalized input spans. Finalization drains
// the child completely; the next nonempty process call starts a fresh child
// state in the same caller-provided storage. Empty process calls and repeated
// finalization do not start another span. As with any normalizer, callers must
// finish draining finalization before providing input for the next span.
//
// This preserves a component's lifetime when it is composed with transforms
// whose state spans the entire input, such as pre-tokenizer prefix policy.
// Takes ownership of |child| on success; the caller retains it on failure.
iree_status_t iree_tokenizer_normalizer_span_allocate(
    iree_tokenizer_normalizer_t* child, iree_allocator_t allocator,
    iree_tokenizer_normalizer_t** out_normalizer);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_TOKENIZER_NORMALIZER_SPAN_H_
