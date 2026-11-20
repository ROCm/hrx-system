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
#include "loom/target/arch/amd/xdna/array_register_field_ids.inl"
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

// Returns public facts for a compiler-selected generated field identifier.
loom_xdna_register_field_info_t loom_xdna_register_field_info(
    loom_xdna_register_field_id_t field_id);

// Returns one address-pattern dimension by ordinal. The field identifier and
// dimension ordinal are compiler-owned selections from the generated corpus.
loom_xdna_register_dimension_info_t loom_xdna_register_field_dimension(
    loom_xdna_register_field_id_t field_id, iree_host_size_t ordinal);

// Selects the shim DMA mux (MM2S) or demux (S2MM) field for a south port.
// Returns INVALID for south lanes without a DMA connection in this direction.
loom_xdna_register_field_id_t loom_xdna_shim_dma_mux_field(
    loom_xdna_dma_direction_t direction, uint8_t south_channel);

// Encodes an admitted raw semantic value into positioned 32-bit register bits.
// The generated field identifier and value's signed or unsigned domain are
// established by their producing compiler stage. This packs bits without
// performing IO or enforcing read/write access; encoded values can also
// describe comparisons against read-only status fields.
uint32_t loom_xdna_register_field_encode(loom_xdna_register_field_id_t field_id,
                                         int64_t value);

// Forms an address from coordinates and indices admitted by physical planning.
// |indices| contains the generated field's complete in-range dimension tuple.
uint64_t loom_xdna_register_field_address(
    const loom_xdna_array_family_t* family,
    loom_xdna_register_field_id_t field_id,
    loom_xdna_tile_coordinate_t coordinate, const uint16_t* indices);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_ARRAY_REGISTERS_H_
