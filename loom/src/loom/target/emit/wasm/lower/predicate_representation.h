// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm physical representations for source predicate vectors.

#ifndef LOOM_TARGET_EMIT_WASM_LOWER_PREDICATE_REPRESENTATION_H_
#define LOOM_TARGET_EMIT_WASM_LOWER_PREDICATE_REPRESENTATION_H_

#include "loom/codegen/low/lower/representation_observer.h"
#include "loom/target/emit/wasm/lower/vector_carrier.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum loom_wasm_predicate_representation_e {
  // One all-zero/all-one byte per logical predicate lane.
  LOOM_WASM_PREDICATE_REPRESENTATION_I8X16 = 8,
  // One all-zero/all-one word per logical predicate lane.
  LOOM_WASM_PREDICATE_REPRESENTATION_I16X8 = 16,
  // One all-zero/all-one dword per logical predicate lane.
  LOOM_WASM_PREDICATE_REPRESENTATION_I32X4 = 32,
  // One all-zero/all-one qword per logical predicate lane.
  LOOM_WASM_PREDICATE_REPRESENTATION_I64X2 = 64,
} loom_wasm_predicate_representation_t;

enum {
  // Number of distinct physical predicate lane widths.
  LOOM_WASM_PREDICATE_REPRESENTATION_COUNT = 4,
};

// Returns true for a predicate vector that fits one v128 register in at least
// one representation and optionally returns its logical element count.
bool loom_wasm_predicate_type(loom_type_t source_type,
                              uint32_t* out_lane_count);

// Returns whether |representation| can carry |lane_count| predicate lanes in
// one v128 register.
bool loom_wasm_predicate_representation_available(
    uint32_t lane_count, loom_low_representation_id_t representation);

// Returns the widest available physical lane representation for |source_type|,
// or NONE when the type is not a supported predicate vector.
loom_low_representation_id_t loom_wasm_predicate_default_representation(
    loom_type_t source_type);

// Returns the one-packet carrier for |source_type| and |representation|. NONE
// selects the stable default representation. Invalid pairs return zero.
loom_wasm_vector_carrier_t loom_wasm_predicate_carrier(
    loom_type_t source_type, loom_low_representation_id_t representation);

// Enumerates every physical representation available to |source_type|. When
// |native_representation| is not NONE, alternatives include the cost of one
// byte shuffle from that native lane width.
iree_host_size_t loom_wasm_predicate_representation_candidates(
    loom_type_t source_type, loom_low_representation_id_t native_representation,
    loom_low_representation_candidate_t
        out_candidates[LOOM_WASM_PREDICATE_REPRESENTATION_COUNT]);

// Function-local Wasm predicate representation selection.
extern const loom_low_lower_source_plan_observer_t
    loom_wasm_predicate_representation_observer;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_LOWER_PREDICATE_REPRESENTATION_H_
