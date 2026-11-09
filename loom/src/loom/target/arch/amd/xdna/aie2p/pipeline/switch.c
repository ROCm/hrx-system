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
  // Next packet group on the same input, or UINT16_MAX at the end.
  uint16_t next;
  // Physical filter word assigned after complete multicast aggregation.
  uint16_t filter;
  // Packet identity matched by the filter.
  uint8_t packet;
  // Packet ID bits compared by the hardware filter.
  uint8_t mask;
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
  // Input/packet groups in first-use order, bounded by this tile's routes and
  // the five-bit packet namespace of each input.
  loom_aie2p_native_packet_group_t* groups;
  // Number of admitted packet groups.
  uint16_t group_count;
  // First packet group per input, or UINT16_MAX for an unused/circuit input.
  uint16_t* first_groups;
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
  const uint32_t slave_count =
      word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES];
  const iree_host_size_t tile_index =
      tile->coordinate.column * context->family->row_count +
      tile->coordinate.row;
  const iree_host_size_t group_capacity = iree_min(
      context->routing.tile_route_counts[tile_index], slave_count * 32);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, group_capacity, sizeof(*state->groups),
      (void**)&state->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      context->pass->arena, slave_count, sizeof(*state->first_groups),
      (void**)&state->first_groups));
  memset(state->first_groups, 0xFF, slave_count * sizeof(*state->first_groups));
  return iree_ok_status();
}

typedef struct loom_aie2p_native_packet_filter_t {
  // Compared value of the packet ID bits selected by mask.
  uint8_t packet;
  // Packet ID bits compared by this filter.
  uint8_t mask;
} loom_aie2p_native_packet_filter_t;

// Finds a disjoint masked cover of one destination set's packet IDs within the
// given slot budget. The caller increases that budget to find a minimum cover.
// Every candidate contains the first uncovered ID, so every disjoint cover is
// considered without permuting its filters. Five ID bits bound each node to
// 32 candidates; the caller bounds depth to the four hardware filter slots.
// Unassigned IDs are never admitted, even if no current source sends them.
static uint8_t loom_aie2p_native_packet_cover(
    uint32_t packets, uint8_t slots,
    loom_aie2p_native_packet_filter_t* filters) {
  // Bit sets for every combination of varying ID bits, anchored at ID zero.
  static const uint32_t patterns[32] = {
      0x00000001, 0x00000003, 0x00000005, 0x0000000f, 0x00000011, 0x00000033,
      0x00000055, 0x000000ff, 0x00000101, 0x00000303, 0x00000505, 0x00000f0f,
      0x00001111, 0x00003333, 0x00005555, 0x0000ffff, 0x00010001, 0x00030003,
      0x00050005, 0x000f000f, 0x00110011, 0x00330033, 0x00550055, 0x00ff00ff,
      0x01010101, 0x03030303, 0x05050505, 0x0f0f0f0f, 0x11111111, 0x33333333,
      0x55555555, 0xffffffff,
  };
  const uint8_t first = (uint8_t)iree_math_count_trailing_zeros_u32(packets);
  uint8_t allowed = 0;
  for (uint8_t bit = 0; bit < 5; ++bit) {
    if (packets & (UINT32_C(1) << (first ^ (1u << bit)))) {
      allowed |= 1u << bit;
    }
  }
  if (packets == (patterns[allowed] << (first & ~allowed))) {
    filters[0] = (loom_aie2p_native_packet_filter_t){.packet = first & ~allowed,
                                                     .mask = 31 & ~allowed};
    return 1;
  }
  if (slots == 1) {
    return 0;
  }
  uint8_t varying = allowed;
  do {
    const uint8_t packet = first & ~varying;
    const uint32_t covered = patterns[varying] << packet;
    if ((packets & covered) == covered) {
      const uint8_t tail = loom_aie2p_native_packet_cover(
          packets ^ covered, slots - 1, filters + 1);
      if (tail) {
        filters[0] = (loom_aie2p_native_packet_filter_t){.packet = packet,
                                                         .mask = 31 & ~varying};
        return tail + 1;
      }
    }
    if (!varying) {
      break;
    }
    varying = (varying - 1) & allowed;
  } while (true);
  return 0;
}

static bool loom_aie2p_native_switch_cover_input(
    loom_aie2p_native_switch_admission_t* state, uint16_t slave) {
  uint32_t destinations[4] = {0};
  uint32_t packet_sets[4] = {0};
  uint16_t groups[32];
  uint8_t group_count = 0;
  uint8_t set_count = 0;
  for (uint16_t i = state->first_groups[slave]; i != UINT16_MAX;
       i = state->groups[i].next) {
    const loom_aie2p_native_packet_group_t* group = &state->groups[i];
    uint8_t set = 0;
    while (set < set_count && destinations[set] != group->masters) {
      ++set;
    }
    if (set == set_count) {
      if (set_count == state->filter_slot_count) {
        return false;
      }
      destinations[set_count++] = group->masters;
    }
    packet_sets[set] |= UINT32_C(1) << group->packet;
    groups[group_count++] = i;
  }
  uint8_t filter_count = 0;
  for (uint8_t set = 0; set < set_count; ++set) {
    loom_aie2p_native_packet_filter_t filters[4];
    uint8_t count = 0;
    // Distinct destination sets cannot share a filter. Solve each exactly,
    // retaining at least one slot for every remaining destination set.
    const uint8_t limit =
        state->filter_slot_count - filter_count - (set_count - set - 1);
    for (uint8_t slots = 1; slots <= limit && !count; ++slots) {
      count = loom_aie2p_native_packet_cover(packet_sets[set], slots, filters);
    }
    if (!count) {
      return false;
    }
    for (uint8_t j = 0; j < count; ++j) {
      loom_aie2p_native_packet_group_t* group =
          &state->groups[groups[filter_count]];
      *group = (loom_aie2p_native_packet_group_t){
          .masters = destinations[set],
          .filter = slave * state->filter_slot_count + filter_count,
          .packet = filters[j].packet,
          .mask = filters[j].mask};
      ++filter_count;
    }
  }
  for (uint8_t i = filter_count; i < group_count; ++i) {
    state->groups[groups[i]].masters = 0;
  }
  return true;
}

static bool loom_aie2p_native_switch_cover_packets(
    loom_aie2p_native_switch_admission_t* state) {
  const uint32_t slave_count =
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].word_count;
  bool compact = false;
  for (uint16_t slave = 0; slave < slave_count; ++slave) {
    if (state->slaves[slave] == UINT8_MAX) {
      continue;
    }
    if (state->slaves[slave] > state->filter_slot_count) {
      if (!loom_aie2p_native_switch_cover_input(state, slave)) {
        return false;
      }
      compact = true;
    } else {
      uint16_t filter = slave * state->filter_slot_count;
      for (uint16_t i = state->first_groups[slave]; i != UINT16_MAX;
           i = state->groups[i].next) {
        state->groups[i].filter = filter++;
      }
    }
  }
  if (compact) {
    uint16_t count = 0;
    for (uint16_t i = 0; i < state->group_count; ++i) {
      if (state->groups[i].masters) {
        state->groups[count++] = state->groups[i];
      }
    }
    state->group_count = count;
  }
  return true;
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
// selects only one arbiter. An arbiter holds its grant for the whole packet,
// including stalls: sharing it can prevent an otherwise independent input
// from making the progress needed to finish that packet. Disjoint components
// may share only when both have the same single input, which already serializes
// their packets. Identical sets reuse a selection across packet IDs and inputs.
static bool loom_aie2p_native_switch_select_packets(
    const loom_aie2p_native_tile_t* tile,
    loom_aie2p_native_switch_admission_t* state) {
  uint8_t selection_counts[32] = {0};
  uint32_t component_inputs[32] = {0};
  uint8_t component_bases[32];
  memset(component_bases, UINT8_MAX, sizeof(component_bases));
  for (uint16_t i = 0; i < state->group_count; ++i) {
    loom_aie2p_native_packet_group_t* group = &state->groups[i];
    group->component = loom_aie2p_native_packet_component(
        state->masters,
        (uint8_t)iree_math_count_trailing_zeros_u32(group->masters));
    component_inputs[group->component] |=
        UINT32_C(1) << (group->filter / state->filter_slot_count);
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
  uint32_t arbiter_inputs[8] = {0};
  for (uint16_t i = 0; i < state->group_count; ++i) {
    const uint8_t component = state->groups[i].component;
    uint8_t* base = &component_bases[component];
    if (*base != UINT8_MAX) {
      continue;
    }
    const uint8_t count = selection_counts[component];
    const uint32_t inputs = component_inputs[component];
    const bool single_input = (inputs & (inputs - 1)) == 0;
    // Allocated arbiters precede unused ones, so a compatible arbiter is
    // reused before opening another. One input's filter budget bounds the
    // total number of distinct destination sets that can share its arbiter.
    for (uint8_t arbiter = 0; arbiter < tile->facts->packet_arbiter_count;
         ++arbiter) {
      if (arbiter_uses[arbiter] &&
          (!single_input || arbiter_inputs[arbiter] != inputs)) {
        continue;
      }
      if (arbiter_uses[arbiter] + count <=
          tile->facts->packet_master_select_count) {
        *base = arbiter | (arbiter_uses[arbiter] << 3);
        arbiter_uses[arbiter] += count;
        arbiter_inputs[arbiter] = inputs;
        break;
      }
    }
    if (*base == UINT8_MAX) {
      return false;
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
                                (uint32_t)group->mask << 16 | UINT32_C(1) << 8 |
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
    bool private_control = false;
    if (selected->control_source != UINT32_MAX) {
      const loom_aie2p_native_control_source_t* source =
          &context->control_sources[selected->control_source];
      private_control =
          source->destination != UINT32_MAX &&
          context->control_destination_sources[source->destination] ==
              selected->control_source;
    }
    if (selected->packet == UINT8_MAX || private_control) {
      // A circuit source may fan out, but each destination master has exactly
      // one source. Private control streams retain their complete header and
      // body; neither endpoint needs switch filtering or header removal.
      // Repeated control identities can retain the same physical prefix.
      const uint32_t word = UINT32_C(1) << 31 | slave;
      if (state->masters[master] == UINT8_MAX - 1 &&
          state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]
                  .words[master] == word) {
        continue;
      }
      // Packet filtering and circuit mode cannot share a port.
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
          .words[master] = word;
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
    uint16_t* next = &state->first_groups[slave];
    while (*next != UINT16_MAX) {
      loom_aie2p_native_packet_group_t* candidate = &state->groups[*next];
      if (candidate->packet == selected->packet) {
        group = candidate;
        break;
      }
      next = &candidate->next;
    }
    if (!group) {
      ++state->slaves[slave];
      *next = state->group_count;
      group = &state->groups[state->group_count++];
      *group = (loom_aie2p_native_packet_group_t){
          .next = UINT16_MAX, .packet = selected->packet, .mask = 31};
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
    if (switches[i].desired &&
        (!loom_aie2p_native_switch_cover_packets(&switches[i]) ||
         !loom_aie2p_native_switch_select_packets(&context->tiles[i],
                                                  &switches[i]))) {
      return loom_aie2p_native_reject(
          context, entry,
          IREE_SV(
              "available packet filters and independent multicast arbiters"));
    }
  }
  *out_valid = true;
  return iree_ok_status();
}
