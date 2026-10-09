// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Wasm SIMD128 carriers for internal source vectors.

#ifndef LOOM_TARGET_EMIT_WASM_LOWER_VECTOR_CARRIER_H_
#define LOOM_TARGET_EMIT_WASM_LOWER_VECTOR_CARRIER_H_

#include <stdint.h>

#include "loom/ir/types.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
  // Physical bytes in one Wasm v128 register.
  LOOM_WASM_VECTOR_CARRIER_PACKET_BYTE_COUNT = 16,
  // Physical Wasm representation of source index and offset lanes.
  LOOM_WASM_ADDRESS_CARRIER_BIT_COUNT = 32,
};

typedef struct loom_wasm_vector_carrier_t {
  // Meaningful physical bytes in the source vector.
  uint16_t payload_byte_count;
  // Number of v128 register units occupied by the carrier.
  uint16_t packet_count;
  // Physical width of each source element in bits.
  uint8_t element_bit_count;
} loom_wasm_vector_carrier_t;

// Returns the fixed physical Wasm lane width for |element_type|. Address
// domains use their target representation instead of their abstract source
// width. Predicates return zero because their width is selected per value.
uint16_t loom_wasm_scalar_type_physical_bit_count(
    loom_scalar_type_t element_type);

// Returns the Wasm carrier for |type| when each logical element occupies
// |physical_element_bit_count| bits. The physical width must be byte-aligned;
// unsupported, dynamic, empty, and over-bound vectors return a zero carrier.
// Target representation policies use this after selecting a physical width
// that differs from the source element type.
loom_wasm_vector_carrier_t loom_wasm_vector_carrier_for_physical_element(
    loom_type_t type, uint16_t physical_element_bit_count);

// Returns the default internal Wasm carrier mapping for |type|. Predicate
// vectors use the widest 8/16/32/64-bit lane representation that fits one
// v128 packet. Unsupported, dynamic, empty, and over-bound vectors return a
// zero carrier.
loom_wasm_vector_carrier_t loom_wasm_vector_carrier_for_type(loom_type_t type);

// Returns true when |type| is one exact v128 vector admitted at a Wasm
// function boundary. Internal tuple carriers do not widen the callable ABI.
bool loom_wasm_vector_type_is_callable(loom_type_t type);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_EMIT_WASM_LOWER_VECTOR_CARRIER_H_
