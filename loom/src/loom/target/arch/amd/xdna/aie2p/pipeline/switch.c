// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/pipeline/switch.h"

#include "iree/base/internal/math.h"
#include "loom/analysis/pipeline/lifetimes.h"
#include "loom/target/arch/amd/xdna/aie2p/pipeline/connectivity.h"
#include "loom/target/arch/amd/xdna/array/registers.h"

iree_status_t loom_aie2p_native_ingress_circuits(
    const loom_xdna_array_family_t* family,
    const loom_aie2p_native_connections_t* connections,
    iree_arena_allocator_t* arena, loom_aie2p_native_switch_t** out_switches) {
  *out_switches = NULL;
  iree_host_size_t capacity = 0;
  for (iree_host_size_t i = 0; i < connections->group_count; ++i) {
    const loom_aie2p_native_connection_t* group = &connections->groups[i];
    const uint16_t column = group->source.coordinate.column;
    for (iree_host_size_t j = 0; j < group->destination_count; ++j) {
      const loom_xdna_tile_coordinate_t destination =
          connections->destinations[group->destination_offset + j].coordinate;
      // Shim mux, one edge per link, and the receiver's local switch output.
      capacity += 2u + destination.row +
                  (iree_max(column, destination.column) -
                   iree_min(column, destination.column));
    }
  }
  loom_aie2p_array_route_builder_t routing;
  IREE_RETURN_IF_ERROR(loom_aie2p_array_route_builder_initialize(
      family, capacity, arena, &routing));
  for (iree_host_size_t i = 0; i < connections->group_count; ++i) {
    const loom_aie2p_native_connection_t* group = &connections->groups[i];
    loom_aie2p_array_route_tree_t tree = {0};
    for (iree_host_size_t j = 0; j < group->destination_count; ++j) {
      const loom_aie2p_native_connection_endpoint_t destination =
          connections->destinations[group->destination_offset + j];
      if (!loom_aie2p_array_route_stream(
              &routing, (uint32_t)i, &tree, group->source.coordinate,
              LOOM_XDNA_STREAM_PORT_DMA, group->source.channel,
              destination.coordinate, destination.port, destination.channel,
              /*packet=*/false)) {
        return iree_ok_status();
      }
    }
  }
  loom_aie2p_native_route_t* routes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, routing.route_count, sizeof(*routes), (void**)&routes));
  for (iree_host_size_t i = 0; i < routing.route_count; ++i) {
    routes[i] = (loom_aie2p_native_route_t){.edge = &routing.routes[i],
                                            .packet = UINT8_MAX,
                                            .control_source = UINT32_MAX};
  }
  const loom_aie2p_native_switch_input_t input = {.routing = &routing,
                                                  .routes = routes};
  iree_string_view_t requirement;
  IREE_RETURN_IF_ERROR(loom_aie2p_native_switches_build(
      &input, arena, out_switches, &requirement));
  if (!iree_string_view_is_empty(requirement)) {
    return iree_ok_status();
  }
  return loom_aie2p_native_connections_verify(&routing, *out_switches,
                                              connections, arena);
}

typedef struct loom_aie2p_native_packet_group_t {
  // Complete multicast output set for one input packet filter.
  uint32_t masters;
  // Next group on the same input, then in the same output component after
  // packet covering. UINT16_MAX marks the end of either list.
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
  // First contributing route plus one, or zero for scope-wide traffic.
  iree_host_size_t first_use;
  // Last contributing route plus one, used when covering several packet IDs.
  iree_host_size_t last_use;
} loom_aie2p_native_packet_group_t;

typedef struct loom_aie2p_native_switch_admission_t {
  // Complete bank values retained for initialization after admission.
  loom_aie2p_native_switch_t* desired;
  // Packet-output union parents; UINT8_MAX if free, UINT8_MAX - 1 if circuit.
  uint8_t* masters;
  // Sole input per packet output; UINT8_MAX if free, UINT8_MAX - 1 if shared.
  uint8_t* master_inputs;
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
  // Firmware-facing completion inputs retain their packet framing contract.
  uint32_t completion_inputs;
  // Outputs carrying any traffic that can wait for payload or core progress.
  uint32_t blocking_masters;
  // Retained physical routes and common lifetime facts for this build.
  const loom_aie2p_native_switch_input_t* input;
  // Next contributing route plus one, shared across this build's switches.
  iree_host_size_t* next_uses;
} loom_aie2p_native_switch_admission_t;

static iree_status_t loom_aie2p_native_switch_initialize(
    const loom_aie2p_array_route_builder_t* routing,
    loom_xdna_tile_coordinate_t coordinate, iree_arena_allocator_t* arena,
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
  const loom_xdna_array_family_t* family = routing->family;
  const loom_xdna_tile_facts_t* facts =
      loom_xdna_array_tile_facts(family, coordinate);
  const loom_xdna_register_field_id_t* selected = fields[facts->kind - 1];
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
      2 * word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS] +
      word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES];
  IREE_RETURN_IF_ERROR(
      iree_arena_allocate(arena, byte_length, (void**)&state->desired));
  memset(state->desired, 0, byte_length);
  state->desired->coordinate = coordinate;
  uint32_t* words = (uint32_t*)(state->desired + 1);
  const uint16_t first_indices[] = {0, 0};
  for (unsigned i = 0; i < LOOM_AIE2P_NATIVE_SWITCH_BANK_COUNT; ++i) {
    state->desired->banks[i] = (loom_aie2p_native_register_bank_t){
        .address = (uint32_t)loom_xdna_register_field_address(
            family, selected[i], coordinate, first_indices),
        .word_count = word_counts[i],
        .words = words};
    words += word_counts[i];
  }
  state->masters = (uint8_t*)words;
  state->master_inputs =
      state->masters + word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS];
  state->slaves =
      state->master_inputs + word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS];
  memset(state->masters, UINT8_MAX,
         2 * word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]);
  const uint32_t slave_count =
      word_counts[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES];
  const iree_host_size_t tile_index =
      loom_xdna_array_physical_tile_index(family, coordinate);
  const iree_host_size_t group_capacity =
      iree_min(routing->tile_route_counts[tile_index], slave_count * 32);
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, group_capacity, sizeof(*state->groups), (void**)&state->groups));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, slave_count,
                                                 sizeof(*state->first_groups),
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
  loom_aie2p_native_packet_group_t original[32];
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
    original[group_count] = *group;
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
      for (uint8_t k = 0; k < group_count; ++k) {
        const loom_aie2p_native_packet_group_t* use = &original[k];
        if (use->masters != group->masters ||
            ((use->packet ^ group->packet) & group->mask) || !use->first_use) {
          continue;
        }
        if (group->last_use) {
          state->next_uses[group->last_use - 1] = use->first_use;
        } else {
          group->first_use = use->first_use;
        }
        group->last_use = use->last_use;
      }
      ++filter_count;
    }
  }
  for (uint8_t i = filter_count; i < group_count; ++i) {
    state->groups[groups[i]].masters = 0;
  }
  return true;
}

// A private segment forwards all framed words unchanged when every input
// class has the same complete destination set. Decide from those classes and
// output ownership before assigning scarce filters and arbiters. Joins,
// differing destination sets, header removal, and firmware completion contracts
// retain packet mode.
static bool loom_aie2p_native_switch_select_circuits(
    loom_aie2p_native_switch_admission_t* state) {
  const uint32_t slave_count =
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].word_count;
  bool changed = false;
  for (uint16_t slave = 0; slave < slave_count; ++slave) {
    const uint16_t first = state->first_groups[slave];
    if (first == UINT16_MAX ||
        (state->completion_inputs & (UINT32_C(1) << slave))) {
      continue;
    }
    const uint32_t destinations = state->groups[first].masters;
    bool exclusive = true;
    for (uint16_t i = state->groups[first].next; i != UINT16_MAX;
         i = state->groups[i].next) {
      exclusive &= state->groups[i].masters == destinations;
    }
    uint32_t masters = destinations;
    while (masters) {
      const uint32_t master = iree_math_count_trailing_zeros_u32(masters);
      masters &= masters - 1;
      exclusive &= state->master_inputs[master] == slave;
    }
    if (!exclusive || (destinations & state->drop_headers)) {
      continue;
    }
    masters = destinations;
    while (masters) {
      const uint32_t master = iree_math_count_trailing_zeros_u32(masters);
      masters &= masters - 1;
      state->masters[master] = UINT8_MAX - 1;
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_MASTERS]
          .words[master] = UINT32_C(1) << 31 | slave;
    }
    state->slaves[slave] = UINT8_MAX;
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave] =
        UINT32_C(1) << 31;
    for (uint16_t i = first; i != UINT16_MAX; i = state->groups[i].next) {
      state->groups[i].masters = 0;
    }
    state->first_groups[slave] = UINT16_MAX;
    changed = true;
  }
  return changed;
}

static bool loom_aie2p_native_switch_cover_packets(
    loom_aie2p_native_switch_admission_t* state) {
  const uint32_t slave_count =
      state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].word_count;
  bool compact = loom_aie2p_native_switch_select_circuits(state);
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

static bool loom_aie2p_native_packet_uses_exclusive(
    const loom_aie2p_native_switch_admission_t* state,
    const loom_aie2p_native_packet_group_t* first,
    const loom_aie2p_native_packet_group_t* second) {
  if (first->filter / state->filter_slot_count ==
      second->filter / state->filter_slot_count) {
    return true;
  }
  if (!first->first_use || !second->first_use) {
    return false;
  }
  const loom_aie2p_native_switch_input_t* input = state->input;
  for (iree_host_size_t a = first->first_use; a; a = state->next_uses[a - 1]) {
    const iree_host_size_t head_a = input->routes[a - 1].lifetime_head;
    if (!head_a) {
      return false;
    }
    for (iree_host_size_t b = second->first_use; b;
         b = state->next_uses[b - 1]) {
      const iree_host_size_t head_b = input->routes[b - 1].lifetime_head;
      if (!head_b || head_a == head_b) {
        return false;
      }
      for (iree_host_size_t x = head_a; x; x = input->route_lifetimes[x].next) {
        for (iree_host_size_t y = head_b; y;
             y = input->route_lifetimes[y].next) {
          const iree_host_size_t ancestor =
              loom_pipeline_lifetimes_common_ancestor(
                  input->lifetimes, input->route_lifetimes[x].lifetime,
                  input->route_lifetimes[y].lifetime);
          if (input->lifetimes->nodes[ancestor].kind !=
              LOOM_PIPELINE_LIFETIME_SERIAL) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

static bool loom_aie2p_native_components_exclusive(
    const loom_aie2p_native_switch_admission_t* state,
    const uint16_t* group_heads, uint8_t component, uint32_t others) {
  while (others) {
    const uint8_t other = (uint8_t)iree_math_count_trailing_zeros_u32(others);
    others &= others - 1;
    for (uint16_t i = group_heads[component]; i != UINT16_MAX;
         i = state->groups[i].next) {
      for (uint16_t j = group_heads[other]; j != UINT16_MAX;
           j = state->groups[j].next) {
        if (!loom_aie2p_native_packet_uses_exclusive(state, &state->groups[i],
                                                     &state->groups[j])) {
          return false;
        }
      }
    }
  }
  return true;
}

// Overlapping destination sets share one arbiter because each physical output
// selects only one arbiter. An arbiter holds its grant for the whole packet,
// including stalls: sharing it can prevent an otherwise independent input
// from making the progress needed to finish that packet. Disjoint components
// may share when both have the same single input, which already serializes
// their packets. Complete register-command components can also share when their
// inputs overlap: their finite packets retire independently of payload credits.
// Different inputs may also share when every payload use is in a disjoint
// serial lifetime. Scope-wide traffic cannot establish that exclusion.
// Identical sets reuse a selection across packet IDs and inputs.
static bool loom_aie2p_native_switch_select_packets(
    const loom_xdna_tile_facts_t* facts,
    loom_aie2p_native_switch_admission_t* state) {
  uint8_t selection_counts[32] = {0};
  uint32_t component_inputs[32] = {0};
  uint32_t blocking_components = 0;
  uint8_t component_bases[32];
  uint16_t group_heads[32];
  memset(component_bases, UINT8_MAX, sizeof(component_bases));
  memset(group_heads, 0xFF, sizeof(group_heads));
  for (uint16_t i = 0; i < state->group_count; ++i) {
    loom_aie2p_native_packet_group_t* group = &state->groups[i];
    group->component = loom_aie2p_native_packet_component(
        state->masters,
        (uint8_t)iree_math_count_trailing_zeros_u32(group->masters));
    component_inputs[group->component] |=
        UINT32_C(1) << (group->filter / state->filter_slot_count);
    group->next = group_heads[group->component];
    group_heads[group->component] = i;
    if (group->masters & state->blocking_masters) {
      blocking_components |= UINT32_C(1) << group->component;
    }
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
          facts->packet_master_select_count) {
        return false;
      }
    }
  }
  uint8_t arbiter_uses[8] = {0};
  uint32_t arbiter_inputs[8] = {0};
  uint32_t arbiter_components[8] = {0};
  uint8_t command_arbiters = 0;
  for (uint16_t i = 0; i < state->group_count; ++i) {
    const uint8_t component = state->groups[i].component;
    uint8_t* base = &component_bases[component];
    if (*base != UINT8_MAX) {
      continue;
    }
    const uint8_t count = selection_counts[component];
    const uint32_t inputs = component_inputs[component];
    const bool single_input = (inputs & (inputs - 1)) == 0;
    const bool register_commands =
        !(blocking_components & (UINT32_C(1) << component));
    // Allocated arbiters precede unused ones, so a compatible arbiter is
    // reused before opening another. One input's filter budget bounds the
    // total number of distinct destination sets that can share its arbiter.
    for (uint8_t arbiter = 0; arbiter < facts->packet_arbiter_count;
         ++arbiter) {
      if (arbiter_uses[arbiter] &&
          !(single_input && arbiter_inputs[arbiter] == inputs) &&
          !(register_commands && (command_arbiters & (1u << arbiter)) &&
            (arbiter_inputs[arbiter] & inputs)) &&
          !loom_aie2p_native_components_exclusive(
              state, group_heads, component, arbiter_components[arbiter])) {
        continue;
      }
      if (arbiter_uses[arbiter] + count <= facts->packet_master_select_count) {
        *base = arbiter | (arbiter_uses[arbiter] << 3);
        if (!arbiter_uses[arbiter] && register_commands) {
          command_arbiters |= 1u << arbiter;
        } else if (!register_commands) {
          command_arbiters &= ~(1u << arbiter);
        }
        arbiter_uses[arbiter] += count;
        arbiter_inputs[arbiter] |= inputs;
        arbiter_components[arbiter] |= UINT32_C(1) << component;
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
    const loom_aie2p_native_switch_input_t* input,
    iree_arena_allocator_t* arena, loom_aie2p_native_switch_t** out_switches,
    iree_string_view_t* out_requirement) {
  *out_switches = NULL;
  *out_requirement = iree_string_view_empty();
  const loom_xdna_array_family_t* family = input->routing->family;
  const iree_host_size_t tile_count =
      loom_xdna_array_physical_tile_count(family);
  loom_aie2p_native_switch_admission_t* switches;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, tile_count, sizeof(*switches), (void**)&switches));
  memset(switches, 0, tile_count * sizeof(*switches));
  iree_host_size_t* next_uses = NULL;
  if (input->route_lifetimes) {
    IREE_RETURN_IF_ERROR(
        iree_arena_allocate_array(arena, input->routing->route_count,
                                  sizeof(*next_uses), (void**)&next_uses));
  }
  loom_aie2p_native_switch_t* head = NULL;
  loom_aie2p_native_switch_t** tail = &head;
  for (iree_host_size_t i = 0; i < input->routing->route_count; ++i) {
    const loom_aie2p_native_route_t* selected = &input->routes[i];
    const loom_aie2p_array_route_plan_t* edge = selected->edge;
    if (edge->switch_kind == LOOM_AIE2P_ARRAY_SWITCH_KIND_SHIM_MUX) {
      continue;
    }
    const iree_host_size_t tile_index =
        loom_xdna_array_physical_tile_index(family, edge->coordinate);
    const loom_xdna_tile_kind_t kind =
        loom_xdna_array_tile_facts(family, edge->coordinate)->kind;
    loom_aie2p_native_switch_admission_t* state = &switches[tile_index];
    if (!state->desired) {
      IREE_RETURN_IF_ERROR(loom_aie2p_native_switch_initialize(
          input->routing, edge->coordinate, arena, state));
      state->input = input;
      state->next_uses = next_uses;
      *tail = state->desired;
      tail = &state->desired->next;
    }
    const uint16_t master = loom_xdna_array_stream_port_range(
                                family, kind, LOOM_XDNA_STREAM_DIRECTION_MASTER,
                                edge->destination_port)
                                ->ordinal +
                            edge->destination_channel;
    const uint16_t slave =
        loom_xdna_array_stream_port_range(
            family, kind, LOOM_XDNA_STREAM_DIRECTION_SLAVE, edge->source_port)
            ->ordinal +
        edge->source_channel;
    if (!selected->register_command) {
      state->blocking_masters |= UINT32_C(1) << master;
    }
    bool private_control = false;
    if (selected->control_source != UINT32_MAX) {
      const loom_aie2p_native_control_source_t* source =
          &input->control_sources[selected->control_source];
      private_control =
          source->destination != UINT32_MAX &&
          input->control_destination_sources[source->destination] ==
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
        *out_requirement = IREE_SV(
            "exclusive circuit destinations and compatible stream-port "
            "modes");
        return iree_ok_status();
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
      *out_requirement =
          IREE_SV("separate circuit and packet stream-switch ports");
      return iree_ok_status();
    }
    if (kind == LOOM_XDNA_TILE_KIND_SHIM_NOC &&
        edge->destination_port == LOOM_XDNA_STREAM_PORT_SOUTH) {
      state->completion_inputs |= UINT32_C(1) << slave;
    }
    if (state->master_inputs[master] == UINT8_MAX) {
      state->master_inputs[master] = (uint8_t)slave;
    } else if (state->master_inputs[master] != slave) {
      state->master_inputs[master] = UINT8_MAX - 1;
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
    if (next_uses &&
        (!group->last_use || input->routes[group->last_use - 1].lifetime_head !=
                                 selected->lifetime_head)) {
      next_uses[i] = 0;
      if (group->last_use) {
        next_uses[group->last_use - 1] = i + 1;
      } else {
        group->first_use = i + 1;
      }
      group->last_use = i + 1;
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
    if (edge->dma_destination) {
      state->drop_headers |= UINT32_C(1) << master;
    }
    state->desired->banks[LOOM_AIE2P_NATIVE_SWITCH_BANK_SLAVES].words[slave] =
        UINT32_C(1) << 31 | UINT32_C(1) << 30;
  }
  for (iree_host_size_t i = 0; i < tile_count; ++i) {
    if (switches[i].desired &&
        (!loom_aie2p_native_switch_cover_packets(&switches[i]) ||
         !loom_aie2p_native_switch_select_packets(
             loom_xdna_array_tile_facts(family,
                                        switches[i].desired->coordinate),
             &switches[i]))) {
      *out_requirement = IREE_SV(
          "available packet filters and independent multicast arbiters");
      return iree_ok_status();
    }
  }
  *out_switches = head;
  return iree_ok_status();
}
