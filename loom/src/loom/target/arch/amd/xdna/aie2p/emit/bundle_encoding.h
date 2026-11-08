// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ENCODING_H_
#define LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ENCODING_H_

#include "loom/codegen/low/frame.h"
#include "loom/codegen/low/packet.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/encoding.h"

#ifdef __cplusplus
extern "C" {
#endif

// Returns generated machine facts for one verified descriptor encoding.
loom_aie2p_instruction_info_t loom_aie2p_bundle_encoding_instruction_info(
    loom_aie2p_instruction_id_t id);

// Returns the exact standalone packet width for one structural descriptor.
uint8_t loom_aie2p_bundle_encoding_single_slot_byte_length(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal);

// Returns the exact byte width of a verified bundle format and slot count.
uint8_t loom_aie2p_bundle_encoding_format_byte_length(
    loom_aie2p_bundle_format_id_t format, iree_host_size_t slot_count);

// Encodes the architectural NOP belonging to one physical bundle slot.
loom_aie2p_encoded_slot_t loom_aie2p_bundle_encoding_encode_slot_nop(
    loom_aie2p_slot_t slot);

// Returns allocation-planned moves attached to one structural register move.
const loom_low_allocation_packet_move_group_t*
loom_aie2p_bundle_encoding_packet_move_group(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet);

// Encodes one descriptor-backed packet with its allocated operands. A symbolic
// constant-data address is encoded as zero and reported for later relocation.
void loom_aie2p_bundle_encoding_encode_packet(
    const loom_low_emission_frame_t* frame,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot,
    uint32_t* out_read_only_data_ordinal);

// Encodes one admitted compiler-owned local-storage address. Bundle analysis
// retains its detached storage-space relocation target.
void loom_aie2p_bundle_encoding_encode_storage_address(
    const loom_low_emission_frame_t* frame,
    const loom_low_descriptor_set_t* descriptor_set,
    const loom_low_packet_view_t* packet,
    loom_aie2p_encoded_slot_t* out_encoded_slot);

// Encodes a structural descriptor with no represented operands or immediates.
loom_aie2p_encoded_slot_t
loom_aie2p_bundle_encoding_encode_structural_descriptor(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal);

// Encodes a single-immediate instruction acting only on fixed native state.
loom_aie2p_encoded_slot_t loom_aie2p_bundle_encoding_encode_state_immediate(
    const loom_low_descriptor_set_t* descriptor_set,
    uint32_t descriptor_ordinal, int64_t immediate);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_TARGET_ARCH_AMD_XDNA_AIE2P_EMIT_BUNDLE_ENCODING_H_
