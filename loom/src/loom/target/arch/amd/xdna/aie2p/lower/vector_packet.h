// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// AIE2P 512-bit vector-packet emission.

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_PACKET_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_PACKET_H_

#include "loom/codegen/low/lower/lower.h"
#include "loom/target/arch/amd/xdna/aie2p/vector_carrier.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_aie2p_vector_packet_emitter_t {
  // Active target-low lowering context.
  loom_low_lower_context_t* context;
  // Source operation whose location is attached to emitted operations.
  const loom_op_t* source_op;
  // One scalar ER register used for instruction controls.
  loom_type_t scalar_type;
  // One 512-bit X packet represented by two vec256 units.
  loom_type_t vector_type;
  // One 64-bit predicate packet.
  loom_type_t predicate_type;
  // One 512-bit accumulator packet.
  loom_type_t accumulator_type;
  // Interned scalar-immediate attribute name.
  loom_string_id_t scalar_immediate_name;
  // Lazily materialized all-zero boolean bytes.
  loom_value_id_t zero_bytes;
  // Lazily materialized all-one boolean bytes.
  loom_value_id_t one_bytes;
} loom_aie2p_vector_packet_emitter_t;

enum {
  // Maximum packet count in any admitted AIE2P vector carrier.
  LOOM_AIE2P_VECTOR_PACKET_MAX_COUNT = 4,
};

// Initializes an emitter for 512-bit X-packet operations and carrier
// conversions. Initialization creates types and strings but emits no IR.
iree_status_t loom_aie2p_vector_packet_emitter_initialize(
    loom_low_lower_context_t* context, const loom_op_t* source_op,
    loom_aie2p_vector_packet_emitter_t* out_emitter);

// Returns the number of allocation units occupied by one 512-bit packet in
// |carrier_kind|.
uint8_t loom_aie2p_vector_packet_carrier_unit_count(
    loom_aie2p_vector_carrier_kind_t carrier_kind);

// Returns the number of logical scalar lanes in one 512-bit X packet. One-bit
// predicate elements use byte construction before conversion to predicate
// storage. Returns zero when |element_bit_count| has no scalar push family.
uint8_t loom_aie2p_vector_packet_scalar_lane_count(uint16_t element_bit_count);

// Seeds one X packet by broadcasting |scalar| in its physical scalar width.
iree_status_t loom_aie2p_vector_packet_emit_scalar_splat(
    loom_aie2p_vector_packet_emitter_t* emitter, uint16_t element_bit_count,
    loom_value_id_t scalar, loom_value_id_t* out_packet);

// Pushes |scalar| onto the high or low end of |packet| in its physical scalar
// width. The result is a fresh X packet.
iree_status_t loom_aie2p_vector_packet_emit_scalar_push(
    loom_aie2p_vector_packet_emitter_t* emitter, uint16_t element_bit_count,
    bool push_high, loom_value_id_t packet, loom_value_id_t scalar,
    loom_value_id_t* out_packet);

// Returns the low register type for |unit_count| units of |carrier_kind|.
iree_status_t loom_aie2p_vector_packet_make_carrier_type(
    loom_low_lower_context_t* context,
    loom_aie2p_vector_carrier_kind_t carrier_kind, uint32_t unit_count,
    loom_type_t* out_type);

// Emits one descriptor operation at the source operation's location.
iree_status_t loom_aie2p_vector_packet_emit_descriptor_op(
    loom_aie2p_vector_packet_emitter_t* emitter, uint32_t descriptor_ordinal,
    const loom_value_id_t* operands, iree_host_size_t operand_count,
    loom_named_attr_slice_t attrs, loom_type_t result_type,
    const loom_tied_result_t* tied_results, iree_host_size_t tied_result_count,
    loom_value_id_t* out_result);

// Emits one descriptor constant with scalar immediate |value|.
iree_status_t loom_aie2p_vector_packet_emit_constant(
    loom_aie2p_vector_packet_emitter_t* emitter, uint32_t descriptor_ordinal,
    int64_t value, loom_type_t result_type, loom_value_id_t* out_result);

// Materializes a 64-lane byte selector from |mask|. A set bit selects the
// corresponding lane from the second VSEL source.
iree_status_t loom_aie2p_vector_packet_emit_byte_selector(
    loom_aie2p_vector_packet_emitter_t* emitter, uint64_t mask,
    loom_value_id_t* out_selector);

// Extracts one native packet from a complete low carrier value.
iree_status_t loom_aie2p_vector_packet_read_native(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t low_value,
    loom_aie2p_vector_carrier_t carrier, uint8_t packet_index,
    loom_value_id_t* out_packet);

// Presents one native packet's payload as a 512-bit X vector. Predicate
// packets become zero/one bytes and accumulator packets are moved to X
// storage.
iree_status_t loom_aie2p_vector_packet_native_to_vector(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_kind_t carrier_kind,
    loom_value_id_t native_packet, loom_value_id_t* out_packet);

// Reads one native packet and presents its payload as a 512-bit X vector.
iree_status_t loom_aie2p_vector_packet_read_vector(
    loom_aie2p_vector_packet_emitter_t* emitter, loom_value_id_t low_value,
    loom_aie2p_vector_carrier_t carrier, uint8_t packet_index,
    loom_value_id_t* out_packet);

// Converts one 512-bit X vector to one native packet of |carrier_kind|.
// Boolean bytes become a predicate comparison and accumulator packets are
// moved to accumulator storage.
iree_status_t loom_aie2p_vector_packet_write_native(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_kind_t carrier_kind,
    loom_value_id_t vector_packet, loom_value_id_t* out_packet);

// Binds ordered native packets as one complete result carrier. Packets beyond
// |logical_packet_count| duplicate the final logical packet; those carrier
// lanes are outside the source value's observable domain.
iree_status_t loom_aie2p_vector_packet_bind_native_packets(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_t carrier, uint8_t logical_packet_count,
    loom_value_id_t* native_packets, loom_value_id_t result_value);

// Converts ordered X packets to |carrier| and binds the complete result value.
iree_status_t loom_aie2p_vector_packet_bind_vector_packets(
    loom_aie2p_vector_packet_emitter_t* emitter,
    loom_aie2p_vector_carrier_t carrier, uint8_t logical_packet_count,
    const loom_value_id_t* vector_packets, loom_value_id_t result_value);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_LOWER_VECTOR_PACKET_H_
