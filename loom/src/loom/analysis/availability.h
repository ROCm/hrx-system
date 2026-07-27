// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// SSA availability analysis for complete IR payloads.
//
// Dominance answers whether one value is visible at an ordinary use site. This
// layer answers the motion/refactoring question: can an op inserted immediately
// before another op legally reference every SSA value captured by an operand,
// type, attribute, predicate, static encoding parameter, or block argument
// type?
//
// The optional |moving_root_op| parameter models moving a subtree as a unit.
// Values defined inside that subtree are considered available because they move
// with the candidate. When |moving_root_op| is NULL, only values already
// available before |before_op| are accepted.

#ifndef LOOM_ANALYSIS_AVAILABILITY_H_
#define LOOM_ANALYSIS_AVAILABILITY_H_

#include "iree/base/api.h"
#include "iree/base/internal/arena.h"
#include "loom/ir/attribute.h"
#include "loom/ir/ir.h"
#include "loom/util/dominance.h"

#ifdef __cplusplus
extern "C" {
#endif

// Analysis state for availability queries. Structured dominance is lightweight
// today, but keeping it here gives CFG-backed dominance a stable cache home
// when Loom grows multi-block regions.
typedef struct loom_availability_analysis_t {
  // Module containing the queried IR.
  const loom_module_t* module;
  // Scratch arena reserved for dominance and future availability caches.
  iree_arena_allocator_t* arena;
  // Dominance state used for before-op value checks.
  loom_dominance_info_t dominance;
} loom_availability_analysis_t;

// Initializes availability analysis for |module|. The caller owns |arena| and
// must keep it live for the analysis object's lifetime.
iree_status_t loom_availability_analysis_initialize(
    const loom_module_t* module, iree_arena_allocator_t* arena,
    loom_availability_analysis_t* out_analysis);

// Returns true if |value_id| can be referenced by an op inserted immediately
// before |before_op|. If |moving_root_op| is non-NULL, values defined inside
// that subtree are also accepted because they move with the candidate.
bool loom_availability_value_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_value_id_t value_id);

// Returns whether every SSA value embedded in |type| can be referenced before
// |before_op|, subject to the same moving-subtree rule as value queries.
iree_status_t loom_availability_type_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_type_t type, bool* out_available);

// Returns whether the type of |value_id| can be materialized before
// |before_op|. Invalid values are conservatively unavailable.
iree_status_t loom_availability_value_type_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    loom_value_id_t value_id, bool* out_available);

// Returns whether every SSA value captured by |attr| can be referenced before
// |before_op|. This includes predicate-list value operands, TYPE attributes,
// nested DICT attributes, and recursively referenced static ENCODING parameter
// attributes.
iree_status_t loom_availability_attr_is_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_attribute_t* attr, bool* out_available);

// Returns whether every attribute on |op| is available before |before_op|.
iree_status_t loom_availability_op_attrs_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_op_t* op, bool* out_available);

// Returns whether every direct operand, result type capture, and attribute
// capture on |op| is available before |before_op|.
iree_status_t loom_availability_op_captures_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_op_t* op, bool* out_available);

// Returns whether all block argument type captures are available before
// |before_op|.
iree_status_t loom_availability_block_arg_types_are_available_before_op(
    const loom_availability_analysis_t* analysis,
    const loom_op_t* moving_root_op, const loom_op_t* before_op,
    const loom_block_t* block, bool* out_available);

#ifdef __cplusplus
}
#endif

#endif  // LOOM_ANALYSIS_AVAILABILITY_H_
