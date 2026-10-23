// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_alignment.h"

#include <string.h>

#include "iree/base/internal/math.h"
#include "loom/target/arch/amd/xdna/aie2p/descriptors/core_descriptors.h"
#include "loom/target/arch/amd/xdna/aie2p/emit/bundle_encoding.h"

// Format widening changes packet bytes without adding issue positions. Thirty
// bytes cover one full nonzero alignment residue after the largest minimal
// packet; the slot bound caps transactional storage growth.
enum {
  LOOM_AIE2P_CALL_RETURN_ALIGNMENT = LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES,
  LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT =
      LOOM_AIE2P_BUNDLE_ALIGNMENT_CALL_PACKET_COUNT,
  LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT =
      (LOOM_AIE2P_ENCODING_MAX_PACKET_SIZE - 2) *
      LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT,
  LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT =
      LOOM_AIE2P_BUNDLE_ALIGNMENT_BOUNDARY_BYTES,
  LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES = 30,
  LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_SLOTS =
      LOOM_AIE2P_BUNDLE_ALIGNMENT_LOOP_MAX_ADDED_SLOTS,
  LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_OPTION_CACHE_SIZE = 32,
};

// One packet in a direct call and its five architectural delay positions.
typedef struct loom_aie2p_call_alignment_packet_t {
  // Original bundle metadata, or synthesized metadata for an implicit NOP.
  loom_aie2p_planned_bundle_t bundle;
  // First original slot in the bounded region snapshot.
  uint8_t original_slot_start;
  // Number of original slot records in the packet.
  uint8_t original_slot_count;
  // Number of original slots with behavior other than alignment padding.
  uint8_t useful_slot_count;
  // Original encoded packet byte length.
  uint8_t byte_length;
  // Wider format selected to align the continuation.
  loom_aie2p_bundle_format_id_t selected_format;
} loom_aie2p_call_alignment_packet_t;

// Best predecessor reaching one exact count of bytes added to a call region.
typedef struct loom_aie2p_call_alignment_state_t {
  // Format selected for the current packet.
  loom_aie2p_bundle_format_id_t format;
  // Added-byte state before selecting |format|.
  uint8_t previous_added_byte_count;
  // Total slot records required by the selected formats so far.
  uint8_t slot_record_count;
  // Whether this exact added-byte count is reachable.
  bool reachable;
} loom_aie2p_call_alignment_state_t;

// Best predecessor reaching one exact byte increase in a hardware-loop region.
typedef struct loom_aie2p_loop_alignment_state_t {
  // Superset format selected for the current packet.
  loom_aie2p_bundle_format_id_t format;
  // Added-byte state before selecting |format|.
  uint8_t previous_added_byte_count;
  // Total output slot records required through this packet.
  uint32_t slot_record_count;
  // Whether this exact added-byte count is reachable.
  bool reachable;
} loom_aie2p_loop_alignment_state_t;

// Best legal superset format for one exact increase in packet byte length.
typedef struct loom_aie2p_loop_alignment_option_t {
  // Superset format providing this byte-length increase.
  loom_aie2p_bundle_format_id_t format;
  // Increase over the original packet's byte length.
  uint8_t added_byte_count;
  // Number of output slot records required by the format.
  uint8_t slot_record_count;
} loom_aie2p_loop_alignment_option_t;

// Packet-independent facts for one legal AIE2P bundle format.
typedef struct loom_aie2p_loop_alignment_format_t {
  // Dense encoding format identifier.
  loom_aie2p_bundle_format_id_t format;
  // Bit set for each physical slot carried by the format.
  uint16_t slot_mask;
  // Encoded packet byte length.
  uint8_t byte_length;
  // Number of output slot records required by the format.
  uint8_t slot_record_count;
} loom_aie2p_loop_alignment_format_t;

// Cached legal format choices for one recurring packet shape.
typedef struct loom_aie2p_loop_alignment_option_cache_entry_t {
  // Physical slots containing behavior in the original packet.
  uint16_t required_slot_mask;
  // Original encoded packet byte length, or zero for an empty entry.
  uint8_t original_byte_length;
  // Number of populated entries in |options|.
  uint8_t option_count;
  // Cheapest format for each reachable byte-length increase.
  loom_aie2p_loop_alignment_option_t
      options[LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES + 1u];
} loom_aie2p_loop_alignment_option_cache_entry_t;

static iree_host_size_t loom_aie2p_bundle_alignment_append_slot(
    loom_aie2p_bundle_alignment_storage_t* storage,
    loom_aie2p_planned_slot_t slot, loom_aie2p_slot_realization_t realization) {
  IREE_ASSERT_LT(storage->slot_count, storage->slot_capacity);
  const iree_host_size_t slot_index = storage->slot_count++;
  storage->slots[slot_index] = slot;
  storage->slot_realizations[slot_index] = realization;
  return slot_index;
}

// Enumerates format compatibility once for one packet. The alignment search
// only observes each format's byte-length increase and slot cost, so formats
// with the same increase are interchangeable after retaining the cheapest one.
static uint8_t loom_aie2p_bundle_alignment_query_loop_options(
    const loom_aie2p_loop_alignment_format_t* formats,
    iree_host_size_t format_count, uint16_t required_slot_mask,
    uint8_t original_byte_length,
    loom_aie2p_loop_alignment_option_t* out_options) {
  memset(out_options, 0,
         (LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES + 1u) *
             sizeof(*out_options));
  for (iree_host_size_t format_index = 0; format_index < format_count;
       ++format_index) {
    const loom_aie2p_loop_alignment_format_t* format = &formats[format_index];
    if (format->byte_length < original_byte_length ||
        (format->slot_mask & required_slot_mask) != required_slot_mask) {
      continue;
    }
    const uint8_t added_byte_count = format->byte_length - original_byte_length;
    if (added_byte_count > LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES) {
      continue;
    }
    loom_aie2p_loop_alignment_option_t* option = &out_options[added_byte_count];
    if (option->format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID &&
        option->slot_record_count <= format->slot_record_count) {
      continue;
    }
    *option = (loom_aie2p_loop_alignment_option_t){
        .format = format->format,
        .slot_record_count = format->slot_record_count,
    };
  }
  uint8_t option_count = 0;
  for (uint8_t added_byte_count = 0;
       added_byte_count <= LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES;
       ++added_byte_count) {
    loom_aie2p_loop_alignment_option_t option = out_options[added_byte_count];
    if (option.format == LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID) {
      continue;
    }
    option.added_byte_count = added_byte_count;
    out_options[option_count++] = option;
  }
  return option_count;
}

static uint8_t loom_aie2p_bundle_alignment_query_loop_packet_options(
    const loom_aie2p_bundle_alignment_storage_t* builder,
    const loom_aie2p_planned_bundle_t* bundle,
    const loom_aie2p_loop_alignment_format_t* formats,
    iree_host_size_t format_count,
    loom_aie2p_loop_alignment_option_cache_entry_t* option_cache,
    loom_aie2p_loop_alignment_option_t* out_options) {
  const loom_aie2p_planned_slot_t* original_slots =
      &builder->slots[bundle->slot_start];
  uint16_t required_slot_mask = 0;
  for (uint8_t i = 0; i < bundle->slot_count; ++i) {
    if (!iree_any_bit_set(original_slots[i].flags,
                          LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP)) {
      required_slot_mask |=
          (uint16_t)(1u << original_slots[i].encoded_slot.slot);
    }
  }
  const uint8_t original_byte_length =
      loom_aie2p_bundle_encoding_format_byte_length(bundle->format,
                                                    bundle->slot_count);
  uint32_t signature = required_slot_mask + 21u * original_byte_length;
  signature ^= signature >> 8;
  loom_aie2p_loop_alignment_option_cache_entry_t* cache_entry =
      &option_cache[signature &
                    (LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_OPTION_CACHE_SIZE -
                     1u)];
  if (cache_entry->required_slot_mask == required_slot_mask &&
      cache_entry->original_byte_length == original_byte_length) {
    memcpy(out_options, cache_entry->options,
           cache_entry->option_count * sizeof(*out_options));
    return cache_entry->option_count;
  }
  const uint8_t option_count = loom_aie2p_bundle_alignment_query_loop_options(
      formats, format_count, required_slot_mask, original_byte_length,
      out_options);
  cache_entry->required_slot_mask = required_slot_mask;
  cache_entry->original_byte_length = original_byte_length;
  cache_entry->option_count = option_count;
  memcpy(cache_entry->options, out_options,
         option_count * sizeof(*out_options));
  return option_count;
}

static bool loom_aie2p_bundle_alignment_bundle_is_nop(
    const loom_aie2p_bundle_alignment_storage_t* builder,
    const loom_aie2p_planned_bundle_t* bundle) {
  if (bundle->slot_count == 1 &&
      builder->slots[bundle->slot_start].encoded_slot.slot ==
          LOOM_AIE2P_SLOT_NOP) {
    return true;
  }
  for (uint8_t i = 0; i < bundle->slot_count; ++i) {
    if (!iree_any_bit_set(builder->slots[bundle->slot_start + i].flags,
                          LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP)) {
      return false;
    }
  }
  return true;
}

static iree_host_size_t loom_aie2p_bundle_alignment_trailing_nop_byte_length(
    const loom_aie2p_bundle_alignment_storage_t* builder) {
  iree_host_size_t cursor = builder->encoded_byte_length;
  iree_host_size_t trailing_byte_length = 0;
  for (iree_host_size_t i = builder->bundle_count; i != 0; --i) {
    const loom_aie2p_planned_bundle_t* bundle = &builder->bundles[i - 1u];
    const iree_host_size_t bundle_byte_length =
        loom_aie2p_bundle_encoding_format_byte_length(bundle->format,
                                                      bundle->slot_count);
    const iree_host_size_t bundle_end =
        bundle->byte_offset + bundle_byte_length;
    IREE_ASSERT_LE(bundle_end, cursor);
    trailing_byte_length += cursor - bundle_end;
    if (!loom_aie2p_bundle_alignment_bundle_is_nop(builder, bundle)) {
      break;
    }
    trailing_byte_length += bundle_byte_length;
    cursor = bundle->byte_offset;
  }
  return trailing_byte_length;
}

// Widens existing issue positions before |boundary_bundle_index| until the
// boundary starts at a 16-byte address. Selection is atomic: no plan storage is
// changed unless a legal set of superset formats has been found.
iree_status_t loom_aie2p_bundle_alignment_try_loop_boundary(
    loom_aie2p_bundle_alignment_storage_t* builder,
    iree_host_size_t region_bundle_start,
    iree_host_size_t boundary_bundle_index, bool* out_aligned) {
  *out_aligned = false;
  IREE_ASSERT_LE(region_bundle_start, boundary_bundle_index);
  IREE_ASSERT_LE(boundary_bundle_index, builder->bundle_count);
  const iree_host_size_t boundary_byte_offset =
      boundary_bundle_index < builder->bundle_count
          ? builder->bundles[boundary_bundle_index].byte_offset
          : builder->encoded_byte_length;
  if (boundary_byte_offset % LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT == 0) {
    *out_aligned = true;
    return iree_ok_status();
  }
  const iree_host_size_t packet_count =
      boundary_bundle_index - region_bundle_start;
  if (packet_count == 0) {
    return iree_ok_status();
  }

  const iree_host_size_t format_count =
      loom_aie2p_encoding_bundle_format_count();
  loom_aie2p_loop_alignment_format_t* formats = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(builder->scratch_arena,
                                                 format_count, sizeof(*formats),
                                                 (void**)&formats));
  for (iree_host_size_t format_index = 0; format_index < format_count;
       ++format_index) {
    const loom_aie2p_bundle_format_id_t format =
        (loom_aie2p_bundle_format_id_t)(format_index + 1u);
    loom_aie2p_bundle_format_info_t format_info;
    const bool found =
        loom_aie2p_encoding_query_bundle_format_info(format, &format_info);
    IREE_ASSERT(found && format_info.bit_count % 8 == 0);
    loom_aie2p_slot_t format_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    const iree_host_size_t format_slot_count =
        loom_aie2p_encoding_query_bundle_format_slots(
            format, IREE_ARRAYSIZE(format_slots), format_slots);
    IREE_ASSERT_EQ(format_slot_count, format_info.slot_count);
    uint16_t slot_mask = 0;
    for (iree_host_size_t i = 0; i < format_slot_count; ++i) {
      slot_mask |= (uint16_t)(1u << format_slots[i]);
    }
    formats[format_index] = (loom_aie2p_loop_alignment_format_t){
        .format = format,
        .slot_mask = slot_mask,
        .byte_length = format_info.bit_count / 8,
        .slot_record_count = format_info.slot_count,
    };
  }

  const uint8_t required_added_byte_count =
      (uint8_t)((LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT -
                 boundary_byte_offset % LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT) %
                LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT);
  loom_aie2p_bundle_format_id_t* selected_formats = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->scratch_arena, packet_count, sizeof(*selected_formats),
      (void**)&selected_formats));

  // Most boundaries need only one packet widened by the exact alignment
  // residue. Select that packet directly when its added slot cost reaches the
  // positive lower bound across every usable widening. Any multi-packet path
  // must then cost more, so the dynamic program cannot choose a better result.
  // Scanning in source order and replacing equal candidates retains its
  // existing preference for the latest equally cheap packet.
  uint32_t zero_added_slot_record_count = 0;
  int32_t minimum_nonzero_slot_delta = INT32_MAX;
  int32_t selected_direct_slot_delta = INT32_MAX;
  iree_host_size_t selected_direct_packet_index = 0;
  loom_aie2p_bundle_format_id_t selected_direct_format =
      LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID;
  loom_aie2p_loop_alignment_option_cache_entry_t
      option_cache[LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_OPTION_CACHE_SIZE];
  memset(option_cache, 0, sizeof(option_cache));
  for (iree_host_size_t packet_index = 0; packet_index < packet_count;
       ++packet_index) {
    const loom_aie2p_planned_bundle_t* bundle =
        &builder->bundles[region_bundle_start + packet_index];
    loom_aie2p_loop_alignment_option_t
        options[LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES + 1u];
    const uint8_t option_count =
        loom_aie2p_bundle_alignment_query_loop_packet_options(
            builder, bundle, formats, format_count, option_cache, options);
    IREE_ASSERT_GT(option_count, 0u);
    IREE_ASSERT_EQ(options[0].added_byte_count, 0u);
    const uint8_t zero_added_slot_count = options[0].slot_record_count;
    selected_formats[packet_index] = options[0].format;
    zero_added_slot_record_count += zero_added_slot_count;
    for (uint8_t option_index = 1; option_index < option_count;
         ++option_index) {
      const loom_aie2p_loop_alignment_option_t* option = &options[option_index];
      if (option->added_byte_count > required_added_byte_count) {
        break;
      }
      const int32_t slot_delta =
          (int32_t)option->slot_record_count - zero_added_slot_count;
      minimum_nonzero_slot_delta =
          iree_min(minimum_nonzero_slot_delta, slot_delta);
      if (option->added_byte_count == required_added_byte_count &&
          slot_delta <= selected_direct_slot_delta) {
        selected_direct_slot_delta = slot_delta;
        selected_direct_packet_index = packet_index;
        selected_direct_format = option->format;
      }
    }
  }

  uint8_t selected_added_byte_count = required_added_byte_count;
  uint32_t selected_slot_record_count = 0;
  const bool direct_selection_is_optimal =
      selected_direct_format != LOOM_AIE2P_BUNDLE_FORMAT_ID_INVALID &&
      (packet_count == 1u ||
       (minimum_nonzero_slot_delta > 0 &&
        selected_direct_slot_delta == minimum_nonzero_slot_delta));
  if (direct_selection_is_optimal) {
    selected_formats[selected_direct_packet_index] = selected_direct_format;
    selected_slot_record_count =
        selected_direct_slot_delta >= 0
            ? zero_added_slot_record_count +
                  (uint32_t)selected_direct_slot_delta
            : zero_added_slot_record_count -
                  (uint32_t)-selected_direct_slot_delta;
  } else {
    const iree_host_size_t state_width =
        LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES + 1u;
    iree_host_size_t state_count = 0;
    if (!iree_host_size_checked_mul(packet_count + 1u, state_width,
                                    &state_count)) {
      return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                              "AIE2P loop-alignment state exceeds host size");
    }
    loom_aie2p_loop_alignment_state_t* states = NULL;
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        builder->scratch_arena, state_count, sizeof(*states), (void**)&states));
    memset(states, 0, state_count * sizeof(*states));
    states[0].reachable = true;

    for (iree_host_size_t packet_index = 0; packet_index < packet_count;
         ++packet_index) {
      const loom_aie2p_planned_bundle_t* bundle =
          &builder->bundles[region_bundle_start + packet_index];
      loom_aie2p_loop_alignment_option_t
          options[LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES + 1u];
      const uint8_t option_count =
          loom_aie2p_bundle_alignment_query_loop_packet_options(
              builder, bundle, formats, format_count, option_cache, options);
      for (uint8_t added_byte_count = 0;
           added_byte_count <=
           LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES;
           ++added_byte_count) {
        const loom_aie2p_loop_alignment_state_t* state =
            &states[packet_index * state_width + added_byte_count];
        if (!state->reachable) {
          continue;
        }
        for (uint8_t option_index = 0; option_index < option_count;
             ++option_index) {
          const loom_aie2p_loop_alignment_option_t* option =
              &options[option_index];
          const unsigned next_added_byte_count =
              (unsigned)added_byte_count + option->added_byte_count;
          if (next_added_byte_count >
              LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES) {
            continue;
          }
          const uint32_t next_slot_record_count =
              state->slot_record_count + option->slot_record_count;
          loom_aie2p_loop_alignment_state_t* next_state =
              &states[(packet_index + 1u) * state_width +
                      next_added_byte_count];
          if (next_state->reachable &&
              next_state->slot_record_count <= next_slot_record_count) {
            continue;
          }
          *next_state = (loom_aie2p_loop_alignment_state_t){
              .format = option->format,
              .previous_added_byte_count = added_byte_count,
              .slot_record_count = next_slot_record_count,
              .reachable = true,
          };
        }
      }
    }

    while (selected_added_byte_count <=
               LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES &&
           !states[packet_count * state_width + selected_added_byte_count]
                .reachable) {
      selected_added_byte_count += LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT;
    }
    if (selected_added_byte_count >
        LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_BYTES) {
      return iree_ok_status();
    }
    const loom_aie2p_loop_alignment_state_t* selected_state =
        &states[packet_count * state_width + selected_added_byte_count];
    selected_slot_record_count = selected_state->slot_record_count;
    uint8_t backtrack_added_byte_count = selected_added_byte_count;
    for (iree_host_size_t packet_index = packet_count; packet_index != 0;
         --packet_index) {
      const loom_aie2p_loop_alignment_state_t* state =
          &states[packet_index * state_width + backtrack_added_byte_count];
      selected_formats[packet_index - 1u] = state->format;
      backtrack_added_byte_count = state->previous_added_byte_count;
    }
  }

  iree_host_size_t original_region_slot_count = 0;
  for (iree_host_size_t i = region_bundle_start; i < boundary_bundle_index;
       ++i) {
    original_region_slot_count += builder->bundles[i].slot_count;
  }
  if (selected_slot_record_count >
      original_region_slot_count +
          LOOM_AIE2P_HARDWARE_LOOP_ALIGNMENT_MAX_ADDED_SLOTS) {
    return iree_ok_status();
  }
  const iree_host_size_t new_slot_count = builder->slot_count -
                                          original_region_slot_count +
                                          selected_slot_record_count;
  if (new_slot_count > builder->slot_capacity ||
      builder->encoded_byte_length > LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE ||
      selected_added_byte_count >
          LOOM_AIE2P_CORE_PROGRAM_MEMORY_SIZE - builder->encoded_byte_length) {
    return iree_ok_status();
  }

  const iree_host_size_t tail_bundle_count =
      builder->bundle_count - region_bundle_start;
  const iree_host_size_t tail_slot_start =
      builder->bundles[region_bundle_start].slot_start;
  const iree_host_size_t tail_slot_count =
      builder->slot_count - tail_slot_start;
  loom_aie2p_planned_bundle_t* original_bundles = NULL;
  loom_aie2p_planned_slot_t* original_slots = NULL;
  loom_aie2p_slot_realization_t* original_realizations = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->scratch_arena, tail_bundle_count, sizeof(*original_bundles),
      (void**)&original_bundles));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->scratch_arena, tail_slot_count, sizeof(*original_slots),
      (void**)&original_slots));
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      builder->scratch_arena, tail_slot_count, sizeof(*original_realizations),
      (void**)&original_realizations));
  memcpy(original_bundles, &builder->bundles[region_bundle_start],
         tail_bundle_count * sizeof(*original_bundles));
  memcpy(original_slots, &builder->slots[tail_slot_start],
         tail_slot_count * sizeof(*original_slots));
  memcpy(original_realizations, &builder->slot_realizations[tail_slot_start],
         tail_slot_count * sizeof(*original_realizations));

  builder->slot_count = tail_slot_start;
  uint32_t cumulative_added_byte_count = 0;
  for (iree_host_size_t tail_index = 0; tail_index < tail_bundle_count;
       ++tail_index) {
    loom_aie2p_planned_bundle_t bundle = original_bundles[tail_index];
    const iree_host_size_t original_slot_offset =
        bundle.slot_start - tail_slot_start;
    bundle.byte_offset += cumulative_added_byte_count;
    bundle.slot_start = (uint32_t)builder->slot_count;
    if (tail_index < packet_count) {
      const uint8_t old_byte_length =
          loom_aie2p_bundle_encoding_format_byte_length(bundle.format,
                                                        bundle.slot_count);
      const loom_aie2p_bundle_format_id_t selected_format =
          selected_formats[tail_index];
      loom_aie2p_slot_t format_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
      const iree_host_size_t format_slot_count =
          loom_aie2p_encoding_query_bundle_format_slots(
              selected_format, IREE_ARRAYSIZE(format_slots), format_slots);
      for (iree_host_size_t i = 0; i < format_slot_count; ++i) {
        bool retained = false;
        for (uint8_t j = 0; j < bundle.slot_count; ++j) {
          const loom_aie2p_planned_slot_t* original_slot =
              &original_slots[original_slot_offset + j];
          if (iree_any_bit_set(original_slot->flags,
                               LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP) ||
              original_slot->encoded_slot.slot != format_slots[i]) {
            continue;
          }
          loom_aie2p_bundle_alignment_append_slot(
              builder, *original_slot,
              original_realizations[original_slot_offset + j]);
          retained = true;
          break;
        }
        if (!retained) {
          loom_aie2p_bundle_alignment_append_slot(
              builder,
              (loom_aie2p_planned_slot_t){
                  .encoded_slot = loom_aie2p_bundle_encoding_encode_slot_nop(
                      format_slots[i]),
                  .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
                  .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
              },
              (loom_aie2p_slot_realization_t){
                  .descriptor_ordinal = AIE2P_CORE_DESCRIPTOR_REF_NOP});
        }
      }
      loom_aie2p_bundle_format_info_t format_info;
      const bool found = loom_aie2p_encoding_query_bundle_format_info(
          selected_format, &format_info);
      IREE_ASSERT(found && format_info.slot_count == format_slot_count &&
                  format_info.bit_count % 8 == 0);
      bundle.format = selected_format;
      bundle.slot_count = (uint8_t)format_slot_count;
      cumulative_added_byte_count +=
          format_info.bit_count / 8 - old_byte_length;
    } else {
      for (uint8_t i = 0; i < bundle.slot_count; ++i) {
        loom_aie2p_bundle_alignment_append_slot(
            builder, original_slots[original_slot_offset + i],
            original_realizations[original_slot_offset + i]);
      }
    }
    builder->bundles[region_bundle_start + tail_index] = bundle;
  }
  IREE_ASSERT_EQ(cumulative_added_byte_count, selected_added_byte_count);
  IREE_ASSERT_EQ(builder->slot_count, new_slot_count);
  builder->encoded_byte_length += cumulative_added_byte_count;
  builder->trailing_nop_byte_length =
      loom_aie2p_bundle_alignment_trailing_nop_byte_length(builder);
  *out_aligned = true;
  return iree_ok_status();
}

static bool loom_aie2p_bundle_alignment_call_format_accepts_packet(
    loom_aie2p_bundle_format_id_t format,
    const loom_aie2p_call_alignment_packet_t* packet,
    const loom_aie2p_planned_slot_t* original_slots,
    loom_aie2p_bundle_format_info_t* out_format_info) {
  if (!loom_aie2p_encoding_query_bundle_format_info(format, out_format_info) ||
      out_format_info->bit_count % 8 != 0) {
    return false;
  }
  loom_aie2p_slot_t format_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  const iree_host_size_t format_slot_count =
      loom_aie2p_encoding_query_bundle_format_slots(
          format, IREE_ARRAYSIZE(format_slots), format_slots);
  if (format_slot_count != out_format_info->slot_count) {
    return false;
  }
  for (uint8_t i = 0; i < packet->original_slot_count; ++i) {
    const loom_aie2p_planned_slot_t* original_slot =
        &original_slots[packet->original_slot_start + i];
    if (iree_any_bit_set(original_slot->flags,
                         LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP)) {
      continue;
    }
    bool found = false;
    for (iree_host_size_t j = 0; j < format_slot_count; ++j) {
      found |= format_slots[j] == original_slot->encoded_slot.slot;
    }
    if (!found) {
      return false;
    }
  }
  return true;
}

static iree_status_t loom_aie2p_bundle_alignment_select_call_formats(
    loom_aie2p_call_alignment_packet_t* packets,
    const loom_aie2p_planned_slot_t* original_slots,
    iree_host_size_t encoded_byte_length) {
  loom_aie2p_call_alignment_state_t
      states[LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT + 1]
            [LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT + 1];
  memset(states, 0, sizeof(states));
  states[0][0].reachable = true;

  const iree_host_size_t format_count =
      loom_aie2p_encoding_bundle_format_count();
  for (iree_host_size_t packet_index = 0;
       packet_index < LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT; ++packet_index) {
    const loom_aie2p_call_alignment_packet_t* packet = &packets[packet_index];
    for (iree_host_size_t added_byte_count = 0;
         added_byte_count <= LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT;
         ++added_byte_count) {
      const loom_aie2p_call_alignment_state_t* state =
          &states[packet_index][added_byte_count];
      if (!state->reachable) {
        continue;
      }
      for (iree_host_size_t format_ordinal = 1; format_ordinal <= format_count;
           ++format_ordinal) {
        const loom_aie2p_bundle_format_id_t format =
            (loom_aie2p_bundle_format_id_t)format_ordinal;
        loom_aie2p_bundle_format_info_t format_info;
        if (!loom_aie2p_bundle_alignment_call_format_accepts_packet(
                format, packet, original_slots, &format_info)) {
          continue;
        }
        const uint8_t format_byte_length = format_info.bit_count / 8;
        if (format_byte_length < packet->byte_length) {
          continue;
        }
        const iree_host_size_t next_added_byte_count =
            added_byte_count + format_byte_length - packet->byte_length;
        if (next_added_byte_count >
            LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT) {
          continue;
        }
        const uint8_t next_slot_record_count =
            state->slot_record_count + format_info.slot_count;
        loom_aie2p_call_alignment_state_t* next_state =
            &states[packet_index + 1][next_added_byte_count];
        if (next_state->reachable &&
            next_state->slot_record_count <= next_slot_record_count) {
          continue;
        }
        *next_state = (loom_aie2p_call_alignment_state_t){
            .format = format,
            .previous_added_byte_count = (uint8_t)added_byte_count,
            .slot_record_count = next_slot_record_count,
            .reachable = true,
        };
      }
    }
  }

  const iree_host_size_t required_added_byte_count =
      (LOOM_AIE2P_CALL_RETURN_ALIGNMENT -
       encoded_byte_length % LOOM_AIE2P_CALL_RETURN_ALIGNMENT) %
      LOOM_AIE2P_CALL_RETURN_ALIGNMENT;
  iree_host_size_t selected_added_byte_count = required_added_byte_count;
  while (
      selected_added_byte_count <=
          LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT &&
      !states[LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT][selected_added_byte_count]
           .reachable) {
    selected_added_byte_count += LOOM_AIE2P_CALL_RETURN_ALIGNMENT;
  }
  if (selected_added_byte_count >
      LOOM_AIE2P_CALL_ALIGNMENT_MAX_ADDED_BYTE_COUNT) {
    return iree_make_status(
        IREE_STATUS_INTERNAL,
        "AIE2P call packets cannot form an aligned return address");
  }
  for (iree_host_size_t packet_index = LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT;
       packet_index != 0; --packet_index) {
    const loom_aie2p_call_alignment_state_t* state =
        &states[packet_index][selected_added_byte_count];
    packets[packet_index - 1].selected_format = state->format;
    selected_added_byte_count = state->previous_added_byte_count;
  }
  return iree_ok_status();
}

iree_status_t loom_aie2p_bundle_alignment_align_call_continuation(
    loom_aie2p_bundle_alignment_storage_t* builder,
    iree_host_size_t call_bundle_index, uint32_t continuation_cycle) {
  if (builder->encoded_byte_length % LOOM_AIE2P_CALL_RETURN_ALIGNMENT == 0) {
    return iree_ok_status();
  }
  const loom_aie2p_planned_bundle_t call_bundle =
      builder->bundles[call_bundle_index];
  IREE_ASSERT_EQ(continuation_cycle - call_bundle.issue_cycle,
                 LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT);

  loom_aie2p_call_alignment_packet_t
      packets[LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT];
  loom_aie2p_planned_slot_t
      original_slots[LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT *
                     LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  loom_aie2p_slot_realization_t
      original_realizations[LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT *
                            LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
  const loom_aie2p_slot_t nop_slot = LOOM_AIE2P_SLOT_NOP;
  const loom_aie2p_bundle_format_id_t nop_format =
      loom_aie2p_encoding_find_bundle_format_for_slots(&nop_slot, 1);
  const uint8_t nop_byte_length =
      loom_aie2p_bundle_encoding_format_byte_length(nop_format, 1);

  const iree_host_size_t region_slot_start = call_bundle.slot_start;
  iree_host_size_t original_slot_count = 0;
  iree_host_size_t bundle_index = call_bundle_index;
  uint32_t byte_offset = call_bundle.byte_offset;
  for (uint8_t packet_index = 0;
       packet_index < LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT; ++packet_index) {
    const uint32_t issue_cycle = call_bundle.issue_cycle + packet_index;
    loom_aie2p_call_alignment_packet_t* packet = &packets[packet_index];
    *packet = (loom_aie2p_call_alignment_packet_t){
        .bundle =
            {
                .issue_cycle = issue_cycle,
                .block_index = call_bundle.block_index,
                .logical_issue_cycle = call_bundle.logical_issue_cycle,
                .byte_offset = byte_offset,
                .format = nop_format,
            },
        .original_slot_start = (uint8_t)original_slot_count,
        .byte_length = nop_byte_length,
    };
    if (bundle_index < builder->bundle_count &&
        builder->bundles[bundle_index].issue_cycle == issue_cycle) {
      packet->bundle = builder->bundles[bundle_index++];
      packet->original_slot_count = packet->bundle.slot_count;
      packet->byte_length = loom_aie2p_bundle_encoding_format_byte_length(
          packet->bundle.format, packet->bundle.slot_count);
      IREE_ASSERT_EQ(packet->bundle.byte_offset, byte_offset);
      IREE_ASSERT_EQ(packet->bundle.slot_start,
                     region_slot_start + original_slot_count);
      for (uint8_t i = 0; i < packet->original_slot_count; ++i) {
        const iree_host_size_t source_slot_index =
            packet->bundle.slot_start + i;
        original_slots[original_slot_count] = builder->slots[source_slot_index];
        original_realizations[original_slot_count] =
            builder->slot_realizations[source_slot_index];
        packet->useful_slot_count +=
            !iree_any_bit_set(builder->slots[source_slot_index].flags,
                              LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP);
        ++original_slot_count;
      }
    }
    byte_offset += packet->byte_length;
  }
  IREE_ASSERT_EQ(bundle_index, builder->bundle_count);
  IREE_ASSERT_EQ(region_slot_start + original_slot_count, builder->slot_count);
  IREE_ASSERT_EQ(byte_offset, builder->encoded_byte_length);

  IREE_RETURN_IF_ERROR(loom_aie2p_bundle_alignment_select_call_formats(
      packets, original_slots, builder->encoded_byte_length));
  iree_host_size_t aligned_byte_length = call_bundle.byte_offset;
  for (uint8_t i = 0; i < LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT; ++i) {
    loom_aie2p_bundle_format_info_t format_info;
    const bool found = loom_aie2p_encoding_query_bundle_format_info(
        packets[i].selected_format, &format_info);
    IREE_ASSERT(found && format_info.bit_count % 8 == 0);
    aligned_byte_length += format_info.bit_count / 8;
  }
  builder->bundle_count = call_bundle_index;
  builder->slot_count = region_slot_start;
  builder->encoded_byte_length = call_bundle.byte_offset;
  for (uint8_t packet_index = 0;
       packet_index < LOOM_AIE2P_CALL_ALIGNMENT_PACKET_COUNT; ++packet_index) {
    loom_aie2p_call_alignment_packet_t* packet = &packets[packet_index];
    loom_aie2p_slot_t format_slots[LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT];
    const iree_host_size_t format_slot_count =
        loom_aie2p_encoding_query_bundle_format_slots(
            packet->selected_format, IREE_ARRAYSIZE(format_slots),
            format_slots);
    IREE_ASSERT_LE(format_slot_count,
                   LOOM_AIE2P_ENCODING_MAX_BUNDLE_SLOT_COUNT);
    const iree_host_size_t slot_start = builder->slot_count;
    for (iree_host_size_t i = 0; i < format_slot_count; ++i) {
      const loom_aie2p_slot_t physical_slot = format_slots[i];
      bool retained = false;
      for (uint8_t j = 0; j < packet->original_slot_count; ++j) {
        const iree_host_size_t original_slot_index =
            packet->original_slot_start + j;
        const loom_aie2p_planned_slot_t* original_slot =
            &original_slots[original_slot_index];
        if (iree_any_bit_set(original_slot->flags,
                             LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP) ||
            original_slot->encoded_slot.slot != physical_slot) {
          continue;
        }
        loom_aie2p_bundle_alignment_append_slot(
            builder, *original_slot,
            original_realizations[original_slot_index]);
        retained = true;
        break;
      }
      if (!retained) {
        loom_aie2p_bundle_alignment_append_slot(
            builder,
            (loom_aie2p_planned_slot_t){
                .encoded_slot =
                    loom_aie2p_bundle_encoding_encode_slot_nop(physical_slot),
                .scheduled_packet_index = LOOM_AIE2P_BUNDLE_PLAN_PACKET_NONE,
                .flags = LOOM_AIE2P_PLANNED_SLOT_FLAG_SYNTHETIC_NOP,
            },
            (loom_aie2p_slot_realization_t){.descriptor_ordinal =
                                                AIE2P_CORE_DESCRIPTOR_REF_NOP});
      }
    }
    loom_aie2p_bundle_format_info_t format_info;
    const bool found = loom_aie2p_encoding_query_bundle_format_info(
        packet->selected_format, &format_info);
    IREE_ASSERT(found && format_info.slot_count == format_slot_count &&
                format_info.bit_count % 8 == 0);
    const uint8_t format_byte_length = format_info.bit_count / 8;
    packet->bundle.byte_offset = (uint32_t)builder->encoded_byte_length;
    packet->bundle.slot_start = (uint32_t)slot_start;
    packet->bundle.format = packet->selected_format;
    packet->bundle.slot_count = (uint8_t)format_slot_count;
    builder->bundles[builder->bundle_count++] = packet->bundle;
    builder->encoded_byte_length += format_byte_length;
    builder->trailing_nop_byte_length =
        packet->useful_slot_count == 0
            ? builder->trailing_nop_byte_length + format_byte_length
            : 0;
  }
  IREE_ASSERT_EQ(builder->encoded_byte_length, aligned_byte_length);
  IREE_ASSERT_EQ(
      builder->encoded_byte_length % LOOM_AIE2P_CALL_RETURN_ALIGNMENT, 0u);
  return iree_ok_status();
}
