// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Table-driven AIE2P instruction and variable-width bundle encoding.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ENCODING_ENCODING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ENCODING_ENCODING_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

#define LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE 16u
#define LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT 8u

// Physical AIE2P VLIW slots. Slot widths are architectural and independent of
// the set of instructions currently selected into the compiler.
typedef enum loom_aie2p_slot_e {
  LOOM_AIE2P_SLOT_INVALID = 0,
  LOOM_AIE2P_SLOT_ALU = 1,
  LOOM_AIE2P_SLOT_LDA = 2,
  LOOM_AIE2P_SLOT_LDB = 3,
  LOOM_AIE2P_SLOT_LNG = 4,
  LOOM_AIE2P_SLOT_MV = 5,
  LOOM_AIE2P_SLOT_NOP = 6,
  LOOM_AIE2P_SLOT_ST = 7,
  LOOM_AIE2P_SLOT_VEC = 8,
  LOOM_AIE2P_SLOT_COUNT = 9,
} loom_aie2p_slot_t;

// Dense build-generated identifiers for target-owned instruction fields.
// Identifiers are private to one compiler build; stable names are the durable
// table identity used by source descriptors and reports. A dense identifier
// must not be used as a standalone serialized IR or cache identity.
typedef uint16_t loom_aie2p_encoding_field_id_t;
#define LOOM_AIE2P_ENCODING_FIELD_ID_INVALID ((loom_aie2p_encoding_field_id_t)0)

// Dense build-generated identifiers for concrete instruction forms. They have
// the same build-local lifetime as field identifiers above.
typedef uint16_t loom_aie2p_instruction_id_t;
#define LOOM_AIE2P_INSTRUCTION_ID_INVALID ((loom_aie2p_instruction_id_t)0)

// Dense build-generated identifiers for variable-width bundle formats. They
// have the same build-local lifetime as field identifiers above.
typedef uint16_t loom_aie2p_bundle_format_id_t;
#define LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID ((loom_aie2p_bundle_format_id_t)0)

// Immutable public metadata for one selected instruction form.
typedef struct loom_aie2p_instruction_info_t {
  // Stable upstream instruction name retained for reports and diagnostics.
  iree_string_view_t name;
  // Physical slot occupied by the instruction.
  loom_aie2p_slot_t slot;
  // Number of encoded bits consumed within the physical slot.
  uint8_t bit_count;
  // Number of subsequent bundles in the architectural delay window.
  uint8_t delay_slot_count;
} loom_aie2p_instruction_info_t;

// Immutable public metadata for one selected bundle format.
typedef struct loom_aie2p_bundle_format_info_t {
  // Stable upstream bundle format name retained for reports and diagnostics.
  iree_string_view_t name;
  // Total number of encoded bits in the variable-width bundle.
  uint8_t bit_count;
  // Number of physical slots carried by the bundle.
  uint8_t slot_count;
} loom_aie2p_bundle_format_info_t;

// One already-adapted instruction operand value keyed by its target field.
typedef struct loom_aie2p_encoding_field_value_t {
  // Target-owned field receiving the value.
  loom_aie2p_encoding_field_id_t field_id;
  // Encoded value before target bit scattering.
  uint64_t value;
} loom_aie2p_encoding_field_value_t;

// One encoded instruction value keyed by its physical bundle slot.
typedef struct loom_aie2p_encoded_slot_t {
  // Physical slot receiving the instruction value.
  loom_aie2p_slot_t slot;
  // Encoded instruction bits in the low bits of the value.
  uint64_t value;
} loom_aie2p_encoded_slot_t;

// Final little-endian bytes for one variable-width AIE2P bundle.
typedef struct loom_aie2p_encoding_packet_t {
  // Bundle bytes in architectural little-endian order.
  uint8_t data[LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE];
  // Number of populated bytes in |data|.
  uint8_t data_length;
} loom_aie2p_encoding_packet_t;

// Decoded physical slot values for one uniquely identified bundle format.
typedef struct loom_aie2p_decoded_bundle_t {
  // Identified bundle format.
  loom_aie2p_bundle_format_id_t format;
  // Number of populated entries in |slots|.
  uint8_t slot_count;
  // Encoded instruction values gathered from each physical slot.
  loom_aie2p_encoded_slot_t slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
} loom_aie2p_decoded_bundle_t;

// Returns the LLVM-AIE extraction-oracle revision for reports and diagnostics.
// This is provenance for the owned facts, not target identity or compatibility.
iree_string_view_t loom_aie2p_encoding_llvm_aie_source_commit(void);

// Returns the number of concrete instruction forms in this target.
iree_host_size_t loom_aie2p_encoding_instruction_count(void);

// Returns the number of target-owned instruction fields in this target.
iree_host_size_t loom_aie2p_encoding_field_count(void);

// Returns the number of variable-width bundle formats in this target.
iree_host_size_t loom_aie2p_encoding_bundle_format_count(void);

// Finds an instruction by its stable target name, or returns INVALID.
loom_aie2p_instruction_id_t loom_aie2p_encoding_find_instruction(
    iree_string_view_t name);

// Finds an instruction field by its stable target name, or returns INVALID.
loom_aie2p_encoding_field_id_t loom_aie2p_encoding_find_field(
    iree_string_view_t name);

// Finds a bundle format by its stable target name, or returns INVALID.
loom_aie2p_bundle_format_id_t loom_aie2p_encoding_find_bundle_format(
    iree_string_view_t name);

// Finds the unique bundle format carrying exactly |slots| in any order, or
// returns INVALID when the slot set is malformed or has no physical format.
// Generated table validation proves that no two formats share a slot set.
loom_aie2p_bundle_format_id_t loom_aie2p_encoding_find_bundle_format_for_slots(
    const loom_aie2p_slot_t* slots, iree_host_size_t slot_count);

// Returns the stable target field name, or an empty view for an invalid ID.
iree_string_view_t loom_aie2p_encoding_field_name(
    loom_aie2p_encoding_field_id_t field);

// Queries instruction metadata. Returns false for an invalid identifier or
// output pointer.
bool loom_aie2p_encoding_query_instruction_info(
    loom_aie2p_instruction_id_t instruction,
    loom_aie2p_instruction_info_t* out_info);

// Queries bundle metadata. Returns false for an invalid identifier or output
// pointer.
bool loom_aie2p_encoding_query_bundle_format_info(
    loom_aie2p_bundle_format_id_t format,
    loom_aie2p_bundle_format_info_t* out_info);

// Packs one instruction from target-adapted field values.
iree_status_t loom_aie2p_encoding_pack_instruction(
    loom_aie2p_instruction_id_t instruction,
    const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count, uint64_t* out_value);

// Packs one compiler-verified instruction without revalidating generated
// field joins. Register operand encoders may carry class-selector bits above a
// field's declared width; the instruction mapping consumes only its mapped low
// bits. Selection and descriptor generation prove that this truncation is
// injective for every legal register domain.
loom_aie2p_encoded_slot_t loom_aie2p_encoding_pack_verified_instruction(
    loom_aie2p_instruction_id_t instruction,
    const loom_aie2p_encoding_field_value_t* field_values,
    iree_host_size_t field_value_count);

// Decodes the fields of a known instruction form. |out_field_value_count| is
// always populated with the required count when non-NULL.
iree_status_t loom_aie2p_encoding_unpack_instruction(
    loom_aie2p_instruction_id_t instruction, uint64_t encoded_value,
    iree_host_size_t field_value_capacity,
    loom_aie2p_encoding_field_value_t* out_field_values,
    iree_host_size_t* out_field_value_count);

// Returns all instruction forms whose fixed pattern accepts |encoded_value|.
// At most |candidate_capacity| identifiers are written; the return value is the
// total candidate count. Operand domains may disambiguate overlapping forms
// later in decoding.
iree_host_size_t loom_aie2p_encoding_query_instruction_candidates(
    loom_aie2p_slot_t slot, uint64_t encoded_value,
    iree_host_size_t candidate_capacity,
    loom_aie2p_instruction_id_t* out_candidates);

// Packs all required physical slots into one variable-width bundle.
iree_status_t loom_aie2p_encoding_pack_bundle(
    loom_aie2p_bundle_format_id_t format,
    const loom_aie2p_encoded_slot_t* encoded_slots,
    iree_host_size_t encoded_slot_count,
    loom_aie2p_encoding_packet_t* out_packet);

// Identifies a bundle format and gathers its physical slot instruction values.
iree_status_t loom_aie2p_encoding_decode_bundle(
    iree_const_byte_span_t packet, loom_aie2p_decoded_bundle_t* out_bundle);

// Decodes the first complete bundle in |packet| and returns its byte length.
// Generated table validation proves that no shorter bundle prefix can alias a
// longer format.
iree_status_t loom_aie2p_encoding_decode_bundle_prefix(
    iree_const_byte_span_t packet, loom_aie2p_decoded_bundle_t* out_bundle,
    iree_host_size_t* out_packet_length);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_ENCODING_ENCODING_H_
