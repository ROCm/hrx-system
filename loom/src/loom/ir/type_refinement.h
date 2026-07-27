// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Monotonic type refinement helpers.
//
// Type refinement is a meet against more precise structural information. It
// may keep the current type unchanged or narrow dynamic properties to static
// ones, but it never widens a type and never invents semantic conversions. Ops
// that truly change kind, element type, rank, or storage meaning must do so via
// an explicit rewrite, not through these helpers.

#ifndef LOOM_IR_TYPE_REFINEMENT_H_
#define LOOM_IR_TYPE_REFINEMENT_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

// Result of meeting the current type with a candidate refinement.
typedef enum loom_type_refinement_result_e {
  // The candidate did not prove any strictly stronger type property.
  LOOM_TYPE_REFINEMENT_UNCHANGED = 0,

  // The output type is strictly more precise than the current type.
  LOOM_TYPE_REFINEMENT_NARROWED = 1,

  // The candidate contradicts an already-known type property.
  LOOM_TYPE_REFINEMENT_CONFLICT = 2,
} loom_type_refinement_result_t;

// Meets all comparable properties of |current_type| with |candidate_type|.
//
// Kind and rank must match. Static dimensions, concrete element types, static
// encodings, and encoding roles are treated as commitments. Dynamic dimensions
// may narrow to static dimensions, SSA encodings may narrow to static
// encodings, and encoding<> may narrow to encoding<role>. Function and dialect
// types are exact structural commitments and conflict if unequal.
iree_status_t loom_type_refine_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

// Refines only the element/role byte of compatible scalar, shaped, group, and
// encoding types. Unknown encoding roles may narrow to concrete roles; concrete
// scalar element types and group scopes must already match.
iree_status_t loom_type_refine_element_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

// Refines only rank/dimension properties using another type as the source of
// candidate dimensions. Type kind and rank must match.
iree_status_t loom_type_refine_shape_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

// Refines only rank/dimension properties using explicit candidate dimensions.
// |candidate_dims| must contain |candidate_rank| packed Loom dimensions.
iree_status_t loom_type_refine_shape_with_dims(
    loom_type_t current_type, const uint64_t* candidate_dims,
    uint8_t candidate_rank, iree_arena_allocator_t* arena,
    loom_type_t* out_type, loom_type_refinement_result_t* out_result);

// Refines only the encoding/layout attachment slot using another type as the
// candidate. Both types must have the same kind. Types that cannot carry
// encodings conflict if the candidate has an encoding attachment.
iree_status_t loom_type_refine_encoding_with_candidate(
    loom_type_t current_type, loom_type_t candidate_type,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

// Refines only the encoding/layout attachment slot using explicit attachment
// fields. This is useful when an analysis proves an SSA encoding is exactly a
// static encoding-table entry.
iree_status_t loom_type_refine_encoding_with_attachment(
    loom_type_t current_type, uint16_t candidate_encoding_id,
    loom_encoding_flags_t candidate_encoding_flags,
    iree_arena_allocator_t* arena, loom_type_t* out_type,
    loom_type_refinement_result_t* out_result);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_IR_TYPE_REFINEMENT_H_
