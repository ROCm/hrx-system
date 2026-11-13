// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared scalar semantics for address-domain casts.

#ifndef LOOM_OPS_INDEX_CAST_H_
#define LOOM_OPS_INDEX_CAST_H_

#include "loom/ir/integer_cast.h"
#include "loom/ir/scalar_type.h"

#ifdef __cplusplus
extern "C" {
#endif

// Describes the fixed-width payload interpretation for verified index.cast
// types. Address-to-address conversions and i64-to-offset instead require a
// source-domain proof and return false.
bool loom_index_cast_payload_interpretation(loom_scalar_type_t input_type,
                                            loom_scalar_type_t result_type,
                                            loom_integer_cast_kind_t* out_kind,
                                            uint8_t* out_bit_count);

// Applies index.cast semantics to one scalar fact summary. |input_type| and
// |result_type| must be fixed-width integer, index, or offset scalar types,
// with at least one side in an address domain.
loom_value_facts_t loom_index_cast_transfer_facts(
    loom_scalar_type_t input_type, loom_scalar_type_t result_type,
    loom_value_facts_t input_facts);

// Returns true when applying index.cast to every value in |input_facts|
// preserves its represented numeric value exactly. Representation changes that
// truncate, reinterpret a negative value as unsigned, or may wrap return false.
bool loom_index_cast_transfer_preserves_value(loom_scalar_type_t input_type,
                                              loom_scalar_type_t result_type,
                                              loom_value_facts_t input_facts);

// Returns true when an address -> fixed-width payload -> same address
// roundtrip preserves every value described by |source_facts|. Roundtrips from
// index require signed fit, roundtrips from offset require unsigned fit, and i1
// uses the 0/1 Boolean domain for either address type.
bool loom_index_cast_roundtrip_preserves_value(loom_scalar_type_t address_type,
                                               loom_scalar_type_t payload_type,
                                               loom_value_facts_t source_facts);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_OPS_INDEX_CAST_H_
