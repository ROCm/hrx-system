// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven NPU2 configuration-register field selection and encoding.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_

#include "iree/base/api.h"
#include "loom/target/arch/amd/xdna/array/facts.h"

#ifdef __cplusplus
extern "C" {
#endif

// Dense process-local register-field identifier. Zero is invalid.
typedef uint16_t loom_xdna_register_field_id_t;

// Software-visible access contract of a configuration register.
typedef enum loom_xdna_register_access_e {
  LOOM_XDNA_REGISTER_ACCESS_READ_WRITE = 1,
  LOOM_XDNA_REGISTER_ACCESS_WRITE_ONLY = 2,
} loom_xdna_register_access_t;

// Public semantic facts for one register field.
typedef struct loom_xdna_register_field_info_t {
  // Stable target-relative field key.
  iree_string_view_t key;
  // Configuration-register module containing the field.
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

// Resolves one stable field key to a dense process-local identifier.
iree_status_t loom_xdna_register_field_lookup(
    iree_string_view_t key, loom_xdna_register_field_id_t* out_field_id);

// Returns public facts for one resolved field identifier.
iree_status_t loom_xdna_register_field_info(
    loom_xdna_register_field_id_t field_id,
    loom_xdna_register_field_info_t* out_info);

// Returns one address-pattern dimension by ordinal.
iree_status_t loom_xdna_register_field_dimension(
    loom_xdna_register_field_id_t field_id, iree_host_size_t ordinal,
    loom_xdna_register_dimension_info_t* out_info);

// Encodes one raw semantic value into positioned 32-bit register bits.
//
// Signed fields accept exactly their two's-complement domain. Unsigned fields
// reject negative values and values wider than the declared field.
iree_status_t loom_xdna_register_field_encode(
    loom_xdna_register_field_id_t field_id, int64_t value,
    uint32_t* out_register_bits);

// Forms one absolute register address for a field and concrete indices.
//
// |indices| must contain one value per field dimension in declaration order.
// The coordinate and field's register module are validated against |family|.
iree_status_t loom_xdna_register_field_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, iree_host_size_t index_count,
    const uint16_t* indices, uint64_t* out_address);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_
