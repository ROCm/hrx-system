// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven NPU2 register-field selection and encoding.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense process-local register-field identifier. Zero is invalid. Identifiers
// index the generated NPU2 corpus and are not part of the serialized ABI.
typedef uint16_t loom_xdna_register_field_id_t;

// Compile-time field selection from the same corpus as the encoding tables.
enum {
  LOOM_XDNA_REGISTER_FIELD_INVALID = 0,
#define LOOM_XDNA_REGISTER_FIELD(symbol, value) symbol = value,
#include "loom/target/arch/amd/xdna/array/register_field_ids.inl"
#undef LOOM_XDNA_REGISTER_FIELD
};

// Software-visible access contract of one register field. Fields in the same
// register may differ, such as a write-only reset strobe beside an event
// selector.
typedef enum loom_xdna_register_access_e {
  LOOM_XDNA_REGISTER_ACCESS_READ_WRITE = 1,
  LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY = 2,
  LOOM_XDNA_REGISTER_ACCESS_READ_ONLY = 3,
} loom_xdna_register_access_t;

// Public semantic facts for one register field.
typedef struct loom_xdna_register_field_info_t {
  // Stable target-relative field key.
  iree_string_view_t key;
  // Register module containing the field.
  loom_xdna_register_module_t module;
  // Software-visible register access contract.
  loom_xdna_register_access_t access;
  // Least-significant field bit in the 32-bit register.
  uint8_t least_significant_bit;
  // Encoded field width.
  uint8_t bit_width;
  // Whether raw input values use signed two's-complement interpretation.
  bool is_signed;
  // Number of indices required to form a concrete register address.
  uint8_t dimension_count;
  // Independent sources supporting this field and its address pattern.
  loom_xdna_provenance_bits_t provenance_bits;
} loom_xdna_register_field_info_t;

// One indexed dimension in a regular register-address pattern.
typedef struct loom_xdna_register_dimension_info_t {
  // Stable dimension name.
  iree_string_view_t name;
  // Exclusive upper bound of the dimension index.
  uint16_t count;
  // Byte stride between consecutive dimension values.
  uint32_t stride;
} loom_xdna_register_dimension_info_t;

// Returns the number of semantic fields in the selected NPU2 corpus.
iree_host_size_t loom_xdna_register_field_count(void);

// Returns public facts for one resolved field identifier.
iree_status_t loom_xdna_register_field_info(
    loom_xdna_register_field_id_t field_id,
    loom_xdna_register_field_info_t* out_info);

// Returns one address-pattern dimension by ordinal. The field identifier and
// dimension ordinal are compiler-owned selections from the generated corpus.
loom_xdna_register_dimension_info_t loom_xdna_register_field_dimension(
    loom_xdna_register_field_id_t field_id, iree_host_size_t ordinal);

// Encodes one raw semantic value into positioned 32-bit register bits.
//
// Signed fields accept exactly their two's-complement domain. Unsigned fields
// reject negative values and values wider than the declared field.
// This packs bits without performing IO or enforcing read/write access. Encoded
// values can also describe comparisons against read-only status fields.
iree_status_t loom_xdna_register_field_encode(
    loom_xdna_register_field_id_t field_id, int64_t value,
    uint32_t* out_register_bits);

// Encodes a field value admitted by its producing compiler stage.
//
// The field identifier must be generated and |value| must be in its signed or
// unsigned domain. Callers handling external values use the checked encoder.
uint32_t loom_xdna_register_field_encode_admitted(
    loom_xdna_register_field_id_t field_id, int64_t value);

// Forms one absolute register address for a field and concrete indices.
//
// |indices| must contain one value per field dimension in declaration order.
// The coordinate and field's register module are validated against |family|.
iree_status_t loom_xdna_register_field_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, iree_host_size_t index_count,
    const uint16_t* indices, uint64_t* out_address);

// Forms an address from coordinates and indices admitted by physical planning.
//
// The field identifier must be generated, |coordinate| must expose its module,
// and |indices| must contain the field's complete in-range dimension tuple.
uint64_t loom_xdna_register_field_address_admitted(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, const uint16_t* indices);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_
