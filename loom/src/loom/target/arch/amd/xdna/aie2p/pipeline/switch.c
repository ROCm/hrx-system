// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

typedef struct loom_aie2p_native_switch_admission_t {
  // Complete bank values retained for initialization after admission.
  loom_aie2p_native_switch_t* desired;
  // Encoded arbiter/selection, UINT8_MAX if free, UINT8_MAX - 1 if circuit.
  uint8_t* masters;
  // Packet-filter rule count per source slave, or UINT8_MAX if circuit.
  uint8_t* slaves;
  // Number of hardware packet-filter slots per source slave.
  uint16_t filter_slot_count;
  // Next physical arbiter in the current selection round.
  uint8_t next_arbiter;
  // Next selection round; exhausted at the tile's physical selection count.
  uint8_t next_master_select;
} loom_aie2p_native_switch_admission_t;

static iree_status_t loom_aie2p_native_switch_initialize(
    loom_aie2p_native_context_t* context, const loom_aie2p_native_tile_t* tile,
    loom_aie2p_native_switch_admission_t* state) {
  // All three tile kinds share word encodings. Generated register patterns
  // supply each bank's address and extent without including reserved holes.
  static const loom_xdna_register_field_id_t
      fields[][LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT] = {
          {LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_MASTER_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_SLAVE_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_SHIM_NOC_STREAM_SLAVE_SLOT_ENABLE},
          {LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_MASTER_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_MEMORY_TILE_STREAM_SLAVE_SLOT_ENABLE},
          {LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_MASTER_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_CONFIG_ENABLE,
           LOOM_XDNA_REGISTER_FIELD_CORE_STREAM_SLAVE_SLOT_ENABLE},
      };
  const loom_xdna_register_field_id_t* selected = fields[tile->facts->kind - 1];
  const loom_xdna_register_dimension_info_t filter_slots =
      loom_xdna_register_field_dimension(
          selected[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS], 1);
  state->filter_slot_count = filter_slots.count;
  uint32_t word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT];
  uint32_t total_word_count = 0;
  for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
    const loom_xdna_register_dimension_info_t ports =
        loom_xdna_register_field_dimension(selected[i], 0);
    word_counts[i] = ports.count;
    if (i == LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS) {
      word_counts[i] *= filter_slots.count;
    }
    total_word_count += word_counts[i];
  }
  const iree_host_size_t byte_length =
      sizeof(*state->desired) + total_word_count * sizeof(uint32_t) +
      word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS] +
      word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES];
  IREE_RETURN_IF_ERROR(iree_arena_allocate(context->pass->arena, byte_length,
                                           (void**)&state->desired));
  memset(state->desired, 0, byte_length);
  uint32_t* words = (uint32_t*)(state->desired + 1);
  const uint16_t first_indices[] = {0, 0};
  for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
    state->desired->banks[i] = (loom_aie2p_native_register_bank_t){
        .address = (uint32_t)loom_xdna_register_field_address_admitted(
            context->family, selected[i], tile->coordinate, first_indices),
        .word_count = word_counts[i],
        .words = words};
    words += word_counts[i];
  }
  state->masters = (uint8_t*)words;
  state->slaves =
      state->masters + word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS];
  memset(state->masters, UINT8_MAX,
         word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]);
  return iree_ok_status();
}

iree_status_t loom_aie2p_native_switches_build(
    loom_aie2p_native_context_t* context, const loom_op_t* entry,
    bool* out_valid) {
  const iree_host_size_t tile_count =
      context->family->column_count * context->family->row_count;
  loom_aie2p_native_switch_admission_t* switches;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, tile_count, sizeof(*switches), (void**)&switches));
  memset(switches, 0, tile_count * sizeof(*switches));
  loom_aie2p_native_switch_t** tail = &context->switches;
  for (iree_host_size_t i = 0; i < context->routing.route_count; ++i) {
    const loom_aie2p_native_route_t* selected = &context->routes[i];
    const loom_aie2p_array_route_plan_t* edge = selected->edge;
    if (edge->switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX) {
      continue;
    }
    const iree_host_size_t tile_index =
        edge->coordinate.column * context->family->row_count +
        edge->coordinate.row;
    const loom_aie2p_native_tile_t* tile = &context->tiles[tile_index];
    const loom_xdna_tile_kind_t kind = tile->facts->kind;
    loom_aie2p_native_switch_admission_t* state = &switches[tile_index];
    if (!state->desired) {
      IREE_RETURN_IF_ERROR(
          loom_aie2p_native_switch_initialize(context, tile, state));
      *tail = state->desired;
      tail = &state->desired->next;
    }
    const uint16_t master =
        loom_xdna_array_stream_port_range(context->family, kind,
                                          LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                          edge->destination_port)
            ->ordinal +
        edge->destination_channel;
    const uint16_t slave =
        loom_xdna_array_stream_port_range(context->family, kind,
                                          LOOM_XDNA_STREAM_DIRECTION_SLAVE,
                                          edge->source_port)
            ->ordinal +
        edge->source_channel;
    if (selected->packet == UINT8_MAX) {
      // A circuit source may fan out, but each destination master has exactly
      // one source. Packet filtering and circuit mode cannot share a port.
      if (state->masters[master] != UINT8_MAX ||
          (state->slaves[slave] != 0 && state->slaves[slave] != UINT8_MAX)) {
        return loom_aie2p_native_reject(
            context, entry,
            IREE_SV("exclusive circuit destinations and compatible stream-port "
                    "modes"));
      }
      state->masters[master] = UINT8_MAX - 1;
      state->slaves[slave] = UINT8_MAX;
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]
          .words[master] = UINT32_C(1) << 31 | slave;
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave] =
          UINT32_C(1) << 31;
      continue;
    }
    if (state->masters[master] == UINT8_MAX - 1 ||
        state->slaves[slave] == UINT8_MAX) {
      return loom_aie2p_native_reject(
          context, entry,
          IREE_SV("separate circuit and packet stream-switch ports"));
    }
    if (state->slaves[slave] == state->filter_slot_count ||
        (state->masters[master] == UINT8_MAX &&
         state->next_master_select ==
             tile->facts->packet_master_select_count)) {
      return loom_aie2p_native_reject(
          context, entry,
          IREE_SV("available packet filters and unicast master selections"));
    }
    if (state->masters[master] == UINT8_MAX) {
      state->masters[master] =
          state->next_arbiter | (state->next_master_select << 3);
      if (++state->next_arbiter == tile->facts->packet_arbiter_count) {
        state->next_arbiter = 0;
        ++state->next_master_select;
      }
    }
    const uint32_t arbiter = state->masters[master] & 7;
    const uint32_t master_select = state->masters[master] >> 3;
    const uint32_t slot = state->slaves[slave]++;
    // Control requests and one-word completion packets retain their header
    // at every hop, including TileControl and the receiving core.
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS].words[master] =
        UINT32_C(1) << 31 | UINT32_C(1) << 30 | arbiter |
        (UINT32_C(1) << (master_select + 3));
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave] =
        UINT32_C(1) << 31 | UINT32_C(1) << 30;
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS]
        .words[slave * state->filter_slot_count + slot] =
        (uint32_t)selected->packet << 24 | UINT32_C(31) << 16 |
        UINT32_C(1) << 8 | master_select << 4 | arbiter;
  }
  *out_valid = true;
  return iree_ok_status();
}
