// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TRANSFORMS_BOUNDARY_PREDICATE_WORD_H_
#define LOOM_TRANSFORMS_BOUNDARY_PREDICATE_WORD_H_

#include "loom/ops/op_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

// Tests the scalar and statically shaped vector domain of a word carrier.
bool loom_predicate_word_supports_type(loom_type_t type,
                                       uint32_t maximum_element_count);

// Preserves the predicate shape while replacing each i1 element with i32.
loom_type_t loom_predicate_word_type(loom_type_t predicate_type);

// Captures a predicate as zero-or-one words at the current insertion point.
// Selection preserves lane participation; an i1 extension may be folded as
// an identity by source legalization and is not a durable transport boundary.
iree_status_t loom_predicate_word_materialize(loom_builder_t* builder,
                                              loom_value_id_t predicate,
                                              loom_type_t word_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_word);

// Reconstructs the predicate under the consumer's current participation.
iree_status_t loom_predicate_word_reconstruct(loom_builder_t* builder,
                                              loom_value_id_t word,
                                              loom_type_t predicate_type,
                                              loom_location_id_t location,
                                              loom_value_id_t* out_predicate);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_TRANSFORMS_BOUNDARY_PREDICATE_WORD_H_
