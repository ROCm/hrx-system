// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/codegen/low/packet_progress.h"

#include <string.h>

typedef struct loom_low_packet_progress_build_state_t {
  // Schedule table being walked.
  const loom_low_schedule_table_t* schedule;
  // Allocation table paired with |schedule|.
  const loom_low_allocation_table_t* allocation;
  // Target progress provider.
  const loom_low_packet_progress_provider_t* provider;
  // Packet currently being queried.
  const loom_low_packet_view_t* current_packet;
  // Mutable output record storage during the populate pass.
  loom_low_packet_progress_record_t* records;
  // Maximum entries available in |records|.
  iree_host_size_t record_capacity;
  // Number of records populated so far.
  iree_host_size_t record_count;
} loom_low_packet_progress_build_state_t;

static void loom_low_packet_progress_append_event(
    void* user_data, const loom_low_packet_progress_event_t* event) {
  loom_low_packet_progress_build_state_t* state =
      (loom_low_packet_progress_build_state_t*)user_data;
  IREE_ASSERT_LT(state->record_count, state->record_capacity);
  const loom_low_packet_view_t* packet = state->current_packet;
  state->records[state->record_count++] = (loom_low_packet_progress_record_t){
      .packet_index = packet->packet_index,
      .node_index = packet->node_index,
      .block_index = packet->node->block_index,
      .scheduled_ordinal = packet->node->scheduled_ordinal,
      .progress_class_id = event->progress_class_id,
      .progress_class_name = event->progress_class_name,
      .action = event->action,
      .units = event->units,
  };
}

static void loom_low_packet_progress_query_packets(
    loom_low_packet_progress_build_state_t* state) {
  for (iree_host_size_t packet_index = 0;
       packet_index < loom_low_packet_count(state->schedule); ++packet_index) {
    const loom_low_packet_view_t packet =
        loom_low_packet_at(state->schedule, packet_index);
    state->current_packet = &packet;
    state->provider->query(state->provider->user_data, state->schedule,
                           state->allocation, &packet,
                           loom_low_packet_progress_append_event, state);
    state->current_packet = NULL;
  }
}

iree_status_t loom_low_packet_progress_build(
    const loom_low_schedule_table_t* schedule,
    const loom_low_allocation_table_t* allocation,
    const loom_low_packet_progress_provider_t* provider,
    iree_arena_allocator_t* arena,
    loom_low_packet_progress_table_t* out_table) {
  memset(out_table, 0, sizeof(*out_table));

  loom_low_packet_progress_build_state_t state = {
      .schedule = schedule,
      .allocation = allocation,
      .provider = provider,
      .record_capacity = provider->event_count,
  };

  loom_low_packet_progress_record_t* records = NULL;
  if (state.record_capacity != 0) {
    IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
        arena, state.record_capacity, sizeof(*records), (void**)&records));
  }

  state.records = records;
  loom_low_packet_progress_query_packets(&state);
  IREE_ASSERT_EQ(state.record_count, state.record_capacity);

  *out_table = (loom_low_packet_progress_table_t){
      .schedule = schedule,
      .allocation = allocation,
      .records = records,
      .record_count = state.record_capacity,
  };
  return iree_ok_status();
}

static loom_low_packet_progress_class_chain_entry_t*
loom_low_packet_progress_class_chain_index_find_mutable(
    loom_low_packet_progress_class_chain_entry_t* classes, uint32_t class_count,
    uint16_t progress_class_id) {
  for (uint32_t i = 0; i < class_count; ++i) {
    if (classes[i].progress_class_id == progress_class_id) {
      return &classes[i];
    }
  }
  return NULL;
}

const loom_low_packet_progress_class_chain_entry_t*
loom_low_packet_progress_class_chain_index_lookup(
    const loom_low_packet_progress_class_chain_index_t* index,
    uint16_t progress_class_id) {
  if (index == NULL) return NULL;
  for (uint32_t i = 0; i < index->class_count; ++i) {
    if (index->classes[i].progress_class_id == progress_class_id) {
      return &index->classes[i];
    }
  }
  return NULL;
}

iree_status_t loom_low_packet_progress_class_chain_index_build(
    const loom_low_packet_progress_table_t* progress,
    iree_arena_allocator_t* arena,
    loom_low_packet_progress_class_chain_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  if (progress == NULL || progress->record_count == 0) {
    return iree_ok_status();
  }
  if (progress->record_count > UINT32_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "packet-progress record count exceeds uint32_t");
  }

  loom_low_packet_progress_class_chain_entry_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, progress->record_count, sizeof(*classes), (void**)&classes));
  uint32_t* next_record_indices = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(arena, progress->record_count,
                                                 sizeof(*next_record_indices),
                                                 (void**)&next_record_indices));

  uint32_t class_count = 0;
  for (iree_host_size_t i = progress->record_count; i > 0; --i) {
    const uint32_t record_index = (uint32_t)(i - 1);
    next_record_indices[record_index] =
        LOOM_LOW_PACKET_PROGRESS_RECORD_INDEX_NONE;

    const loom_low_packet_progress_record_t* record =
        &progress->records[record_index];
    loom_low_packet_progress_class_chain_entry_t* class_entry =
        loom_low_packet_progress_class_chain_index_find_mutable(
            classes, class_count, record->progress_class_id);
    if (class_entry == NULL) {
      class_entry = &classes[class_count++];
      *class_entry = (loom_low_packet_progress_class_chain_entry_t){
          .progress_class_id = record->progress_class_id,
          .first_record_index = LOOM_LOW_PACKET_PROGRESS_RECORD_INDEX_NONE,
      };
    }
    next_record_indices[record_index] = class_entry->first_record_index;
    class_entry->first_record_index = record_index;
    ++class_entry->record_count;
  }

  *out_index = (loom_low_packet_progress_class_chain_index_t){
      .progress = progress,
      .classes = classes,
      .class_count = class_count,
      .next_record_indices = next_record_indices,
  };
  return iree_ok_status();
}

uint32_t loom_low_packet_progress_class_chain_index_observed_progress(
    const loom_low_packet_progress_class_chain_index_t* index,
    iree_host_size_t start_packet_index, iree_host_size_t end_packet_index,
    uint16_t progress_class_id) {
  if (start_packet_index >= end_packet_index) return 0;
  const loom_low_packet_progress_class_chain_entry_t* class_entry =
      loom_low_packet_progress_class_chain_index_lookup(index,
                                                        progress_class_id);
  if (class_entry == NULL) return 0;

  uint32_t observed_progress = 0;
  const loom_low_packet_progress_table_t* progress = index->progress;
  for (uint32_t record_index = class_entry->first_record_index;
       record_index != LOOM_LOW_PACKET_PROGRESS_RECORD_INDEX_NONE;
       record_index = index->next_record_indices[record_index]) {
    const loom_low_packet_progress_record_t* record =
        &progress->records[record_index];
    if (record->packet_index <= start_packet_index) continue;
    if (record->packet_index >= end_packet_index) break;
    if (record->action == LOOM_LOW_PACKET_PROGRESS_ACTION_RESET) {
      return UINT32_MAX;
    } else if (observed_progress <= UINT32_MAX - record->units) {
      observed_progress += record->units;
    } else {
      observed_progress = UINT32_MAX;
    }
  }
  return observed_progress;
}

const loom_low_packet_progress_class_range_entry_t*
loom_low_packet_progress_class_range_index_lookup(
    const loom_low_packet_progress_class_range_index_t* index,
    uint16_t progress_class_id) {
  if (index == NULL) return NULL;
  for (uint32_t i = 0; i < index->class_count; ++i) {
    if (index->classes[i].progress_class_id == progress_class_id) {
      return &index->classes[i];
    }
  }
  return NULL;
}

iree_status_t loom_low_packet_progress_class_range_index_build(
    const loom_low_packet_progress_class_chain_index_t* chain_index,
    iree_arena_allocator_t* arena,
    loom_low_packet_progress_class_range_index_t* out_index) {
  memset(out_index, 0, sizeof(*out_index));
  const loom_low_packet_progress_table_t* progress = chain_index->progress;
  if (progress == NULL || progress->record_count == 0) {
    return iree_ok_status();
  }

  loom_low_packet_progress_class_range_entry_t* classes = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, chain_index->class_count, sizeof(*classes), (void**)&classes));
  loom_low_packet_progress_class_range_record_t* records = NULL;
  IREE_RETURN_IF_ERROR(iree_arena_allocate_array(
      arena, progress->record_count, sizeof(*records), (void**)&records));

  uint32_t range_record_count = 0;
  for (uint32_t class_index = 0; class_index < chain_index->class_count;
       ++class_index) {
    const loom_low_packet_progress_class_chain_entry_t* chain_entry =
        &chain_index->classes[class_index];
    classes[class_index] = (loom_low_packet_progress_class_range_entry_t){
        .progress_class_id = chain_entry->progress_class_id,
        .record_start = range_record_count,
        .record_count = chain_entry->record_count,
    };

    uint32_t progress_record_index = chain_entry->first_record_index;
    for (uint32_t i = 0; i < chain_entry->record_count; ++i) {
      records[range_record_count++].progress_record_index =
          progress_record_index;
      progress_record_index =
          chain_index->next_record_indices[progress_record_index];
    }
  }

  for (uint32_t class_index = 0; class_index < chain_index->class_count;
       ++class_index) {
    const loom_low_packet_progress_class_range_entry_t* range_entry =
        &classes[class_index];
    uint32_t cumulative_reset_count = 0;
    uint64_t cumulative_advance_units = 0;
    const uint32_t range_end =
        range_entry->record_start + range_entry->record_count;
    for (uint32_t i = range_entry->record_start; i < range_end; ++i) {
      loom_low_packet_progress_class_range_record_t* range_record = &records[i];
      const loom_low_packet_progress_record_t* progress_record =
          &progress->records[range_record->progress_record_index];
      if (progress_record->action == LOOM_LOW_PACKET_PROGRESS_ACTION_RESET) {
        ++cumulative_reset_count;
      } else {
        cumulative_advance_units += progress_record->units;
      }
      range_record->cumulative_reset_count = cumulative_reset_count;
      range_record->cumulative_advance_units = cumulative_advance_units;
    }
  }

  *out_index = (loom_low_packet_progress_class_range_index_t){
      .progress = progress,
      .classes = classes,
      .class_count = chain_index->class_count,
      .records = records,
      .record_count = range_record_count,
  };
  return iree_ok_status();
}

static uint32_t loom_low_packet_progress_class_range_index_lower_bound(
    const loom_low_packet_progress_class_range_index_t* index, uint32_t begin,
    uint32_t end, iree_host_size_t packet_index) {
  const loom_low_packet_progress_table_t* progress = index->progress;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const uint32_t progress_record_index =
        index->records[middle].progress_record_index;
    if (progress->records[progress_record_index].packet_index < packet_index) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

static uint32_t loom_low_packet_progress_class_range_index_upper_bound(
    const loom_low_packet_progress_class_range_index_t* index, uint32_t begin,
    uint32_t end, iree_host_size_t packet_index) {
  const loom_low_packet_progress_table_t* progress = index->progress;
  while (begin < end) {
    const uint32_t middle = begin + (end - begin) / 2;
    const uint32_t progress_record_index =
        index->records[middle].progress_record_index;
    if (progress->records[progress_record_index].packet_index <= packet_index) {
      begin = middle + 1;
    } else {
      end = middle;
    }
  }
  return begin;
}

uint32_t loom_low_packet_progress_class_range_index_observed_progress(
    const loom_low_packet_progress_class_range_index_t* index,
    iree_host_size_t start_packet_index, iree_host_size_t end_packet_index,
    uint16_t progress_class_id) {
  if (start_packet_index >= end_packet_index) return 0;
  const loom_low_packet_progress_class_range_entry_t* class_entry =
      loom_low_packet_progress_class_range_index_lookup(index,
                                                        progress_class_id);
  if (class_entry == NULL) return 0;

  const uint32_t class_begin = class_entry->record_start;
  const uint32_t class_end = class_begin + class_entry->record_count;
  const uint32_t range_begin =
      loom_low_packet_progress_class_range_index_upper_bound(
          index, class_begin, class_end, start_packet_index);
  const uint32_t range_end =
      loom_low_packet_progress_class_range_index_lower_bound(
          index, range_begin, class_end, end_packet_index);
  if (range_begin == range_end) return 0;

  const loom_low_packet_progress_class_range_record_t* last_record =
      &index->records[range_end - 1];
  const loom_low_packet_progress_class_range_record_t* preceding_record =
      range_begin > class_begin ? &index->records[range_begin - 1] : NULL;
  const uint32_t preceding_reset_count =
      preceding_record != NULL ? preceding_record->cumulative_reset_count : 0;
  if (last_record->cumulative_reset_count != preceding_reset_count) {
    return UINT32_MAX;
  }

  const uint64_t preceding_advance_units =
      preceding_record != NULL ? preceding_record->cumulative_advance_units : 0;
  const uint64_t observed_progress =
      last_record->cumulative_advance_units - preceding_advance_units;
  return observed_progress > UINT32_MAX ? UINT32_MAX
                                        : (uint32_t)observed_progress;
}
