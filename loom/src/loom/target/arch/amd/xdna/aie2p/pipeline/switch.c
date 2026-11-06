// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/native.h"

typedef struct loom_aie2p_native_packet_group_t {
  // Complete multicast output set for one input packet filter.
  uint32_t masters;
  // Physical filter word assigned when this input/packet pair first appears.
  uint16_t filter;
  // Packet identity matched by the filter.
  uint8_t packet;
  // Connected output component, assigned after all branches are admitted.
  uint8_t component;
  // Destination-set ordinal within the component's arbiter selections.
  uint8_t selection;
} loom_aie2p_native_packet_group_t;

typedef struct loom_aie2p_native_switch_admission_t {
  // Complete bank values retained for initialization after admission.
  loom_aie2p_native_switch_t* desired;
  // Packet-output union parents; UINT8_MAX if free, UINT8_MAX - 1 if circuit.
  uint8_t* masters;
  // Packet-filter rule count per source slave, or UINT8_MAX if circuit.
  uint8_t* slaves;
  // Number of hardware packet-filter slots per source slave.
  uint16_t filter_slot_count;
  // Input/packet groups in first-use order, bounded by physical filter slots.
  loom_aie2p_native_packet_group_t* groups;
  // Number of admitted packet groups.
  uint16_t group_count;
  // Group index per physical filter slot; only assigned slots are queried.
  uint16_t* filter_groups;
  // DMA output masters remove packet headers before delivering payload.
  uint32_t drop_headers;
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
        .address = (uint32_t)loom_xdna_register_field_address(
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
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS],
      sizeof(*state->groups), (void**)&state->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS],
      sizeof(*state->filter_groups), (void**)&state->filter_groups));
  return iree_ok_status();
}

static uint8_t loom_aie2p_native_packet_component(uint8_t* parents,
                                                  uint8_t master) {
  while (parents[master] != master) {
    parents[master] = parents[parents[master]];
    master = parents[master];
  }
  return master;
}

// Overlapping destination sets share one arbiter because each physical output
// selects only one arbiter. Disjoint components may use different selections
// of that same arbiter. Identical sets reuse a selection across packet IDs and
// inputs, while each input/packet pair owns exactly one filter.
static bool loom_aie2p_native_switch_select_packets(
    const loom_aie2p_native_tile_t* tile,
    loom_aie2p_native_switch_admission_t* state) {
  uint8_t selection_counts[32] = {0};
  uint8_t component_bases[32];
  memset(component_bases, UINT8_MAX, sizeof(component_bases));
  for (uint16_t i = 0; i < state->group_count; ++i) {
    loom_aie2p_native_packet_group_t* group = &state->groups[i];
    group->component = loom_aie2p_native_packet_component(
        state->masters,
        (uint8_t)iree_math_count_trailing_zeros_u32(group->masters));
    bool shared = false;
    for (uint16_t j = 0; j < i; ++j) {
      if (state->groups[j].masters == group->masters) {
        group->selection = state->groups[j].selection;
        shared = true;
        break;
      }
    }
    if (!shared) {
      group->selection = selection_counts[group->component]++;
      if (selection_counts[group->component] >
          tile->facts->packet_master_select_count) {
        return false;
      }
    }
  }
  uint8_t arbiter_uses[8] = {0};
  uint8_t next_arbiter = 0;
  // Largest components first prevent singleton sets from fragmenting the
  // four-selection arbiters needed by overlapping multicast sets.
  for (uint8_t count = tile->facts->packet_master_select_count; count;
       --count) {
    for (uint16_t i = 0; i < state->group_count; ++i) {
      const uint8_t component = state->groups[i].component;
      uint8_t* base = &component_bases[component];
      if (*base != UINT8_MAX || selection_counts[component] != count) {
        continue;
      }
      for (uint8_t j = 0; j < tile->facts->packet_arbiter_count; ++j) {
        const uint8_t arbiter =
            (next_arbiter + j) % tile->facts->packet_arbiter_count;
        if (arbiter_uses[arbiter] + count <=
            tile->facts->packet_master_select_count) {
          *base = arbiter | (arbiter_uses[arbiter] << 3);
          arbiter_uses[arbiter] += count;
          next_arbiter = (arbiter + 1) % tile->facts->packet_arbiter_count;
          break;
        }
      }
      if (*base == UINT8_MAX) {
        return false;
      }
    }
  }
  for (uint16_t i = 0; i < state->group_count; ++i) {
    const loom_aie2p_native_packet_group_t* group = &state->groups[i];
    const uint8_t base = component_bases[group->component];
    const uint32_t arbiter = base & 7;
    const uint32_t selection = (base >> 3) + group->selection;
    uint32_t masters = group->masters;
    while (masters) {
      const uint32_t master = iree_math_count_trailing_zeros_u32(masters);
      masters &= masters - 1;
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]
          .words[master] |= UINT32_C(1) << 31 | UINT32_C(1) << 30 | arbiter |
                            (UINT32_C(1) << (selection + 3)) |
                            ((state->drop_headers >> master) & 1) << 7;
    }
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_FILTERS]
        .words[group->filter] = (uint32_t)group->packet << 24 |
                                UINT32_C(31) << 16 | UINT32_C(1) << 8 |
                                selection << 4 | arbiter;
  }
  return true;
}

iree_status_t loom_aie2p_native_switches_build(
    loom_aie2p_native_context_t* context, const loom_op_t* entry,
    bool* out_valid) {
  const iree_host_size_t tile_count =
      (iree_host_size_t)context->family->column_count *
      context->family->row_count;
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
    loom_aie2p_native_packet_group_t* group = NULL;
    for (uint8_t j = 0; j < state->slaves[slave]; ++j) {
      loom_aie2p_native_packet_group_t* candidate =
          &state->groups[state->filter_groups[slave * state->filter_slot_count +
                                              j]];
      if (candidate->packet == selected->packet) {
        group = candidate;
        break;
      }
    }
    if (!group) {
      if (state->slaves[slave] == state->filter_slot_count) {
        return loom_aie2p_native_reject(
            context, entry,
            IREE_SV(
                "available packet filters and multicast master selections"));
      }
      const uint16_t filter =
          slave * state->filter_slot_count + state->slaves[slave]++;
      state->filter_groups[filter] = state->group_count;
      group = &state->groups[state->group_count++];
      *group = (loom_aie2p_native_packet_group_t){.filter = filter,
                                                  .packet = selected->packet};
    }
    if (state->masters[master] == UINT8_MAX) {
      state->masters[master] = (uint8_t)master;
    }
    if (group->masters) {
      const uint8_t first =
          (uint8_t)iree_math_count_trailing_zeros_u32(group->masters);
      state->masters[loom_aie2p_native_packet_component(state->masters,
                                                        (uint8_t)master)] =
          loom_aie2p_native_packet_component(state->masters, first);
    }
    group->masters |= UINT32_C(1) << master;
    if (edge->destination_port == LOOM_XDNA_STREAM_PORT_DMA) {
      state->drop_headers |= UINT32_C(1) << master;
    }
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave] =
        UINT32_C(1) << 31 | UINT32_C(1) << 30;
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    if (switches[i].desired && !loom_aie2p_native_switch_select_packets(
                                   &context->tiles[i], &switches[i])) {
      return loom_aie2p_native_reject(
          context, entry,
          IREE_SV("available packet filters and multicast master selections"));
    }
  }
  *out_valid = true;
  return iree_ok_status();
}
