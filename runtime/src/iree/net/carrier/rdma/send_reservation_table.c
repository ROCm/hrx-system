// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/send_reservation_table.h"

#include <string.h>

#define IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX UINT32_MAX

typedef uint8_t iree_net_rdma_send_reservation_slot_flags_t;
enum iree_net_rdma_send_reservation_slot_bits_e {
  IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE = 1u << 0,
  IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED = 1u << 1,
};

struct iree_net_rdma_send_reservation_slot_t {
  // Optional registered staging storage retained until posting or abort.
  iree_async_buffer_lease_t buffer_lease;

  // RDMA spans borrowed until completion or backed by buffer_lease.
  iree_async_span_t spans[IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE];

  // Number of valid entries in spans.
  iree_host_size_t span_count;

  // Number of bytes to send from the reservation payload.
  iree_host_size_t byte_length;

  // Completion behavior for the eventual posted send.
  iree_net_rdma_send_reservation_completion_t completion;

  // User data forwarded to user-visible completions.
  uint64_t user_data;

  // Generation encoded in the carrier send handle to reject stale handles.
  uint32_t generation;

  // Next slot index in the free list when not in use.
  uint32_t next_free;

  // Next slot index in the committed pending FIFO.
  uint32_t next_pending;

  // Bitfield of iree_net_rdma_send_reservation_slot_bits_e values.
  iree_net_rdma_send_reservation_slot_flags_t flags;
};

static iree_net_carrier_send_handle_t
iree_net_rdma_send_reservation_table_make_handle(uint32_t index,
                                                 uint32_t generation) {
  return ((uint64_t)generation << 32) | index;
}

static uint32_t iree_net_rdma_send_reservation_table_handle_index(
    iree_net_carrier_send_handle_t handle) {
  return (uint32_t)(handle & 0xFFFFFFFFu);
}

static uint32_t iree_net_rdma_send_reservation_table_handle_generation(
    iree_net_carrier_send_handle_t handle) {
  return (uint32_t)(handle >> 32);
}

static uint32_t iree_net_rdma_send_reservation_table_next_generation(
    uint32_t generation) {
  ++generation;
  return generation == 0 ? 1 : generation;
}

static iree_status_t iree_net_rdma_send_reservation_table_lookup_slot(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle, bool require_uncommitted,
    uint32_t* out_index, iree_net_rdma_send_reservation_slot_t** out_slot) {
  if (out_index) {
    *out_index = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  }
  if (out_slot) *out_slot = NULL;
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }

  uint32_t index = iree_net_rdma_send_reservation_table_handle_index(handle);
  if (index >= table->capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "send handle index %u is outside table capacity %u",
                            index, table->capacity);
  }

  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  uint32_t generation =
      iree_net_rdma_send_reservation_table_handle_generation(handle);
  if (!iree_any_bit_set(slot->flags,
                        IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE) ||
      slot->generation != generation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "send handle does not reference an active reservation");
  }
  if (require_uncommitted &&
      iree_any_bit_set(slot->flags,
                       IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED)) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "send handle references a committed reservation");
  }

  if (out_index) *out_index = index;
  if (out_slot) *out_slot = slot;
  return iree_ok_status();
}

static void iree_net_rdma_send_reservation_table_release_slot(
    iree_net_rdma_send_reservation_table_t* table, uint32_t index,
    iree_net_rdma_send_reservation_t* out_reservation) {
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  out_reservation->buffer_lease = slot->buffer_lease;
  memcpy(out_reservation->spans, slot->spans, sizeof(out_reservation->spans));
  out_reservation->span_count = slot->span_count;
  out_reservation->byte_length = slot->byte_length;
  out_reservation->completion = slot->completion;
  out_reservation->user_data = slot->user_data;

  memset(&slot->buffer_lease, 0, sizeof(slot->buffer_lease));
  memset(slot->spans, 0, sizeof(slot->spans));
  slot->span_count = 0;
  slot->byte_length = 0;
  slot->completion = IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL;
  slot->user_data = 0;
  slot->generation =
      iree_net_rdma_send_reservation_table_next_generation(slot->generation);
  slot->next_free = table->free_head;
  slot->next_pending = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  slot->flags = 0;
  table->free_head = index;
  ++table->available_capacity;
}

static iree_status_t iree_net_rdma_send_reservation_table_unlink_pending_slot(
    iree_net_rdma_send_reservation_table_t* table, uint32_t index) {
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  if (!iree_any_bit_set(slot->flags,
                        IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED)) {
    return iree_ok_status();
  }
  if (table->pending_count == 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "committed reservation is missing from pending FIFO");
  }

  uint32_t previous_index = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  uint32_t cursor = table->pending_head;
  while (cursor != IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX &&
         cursor != index) {
    previous_index = cursor;
    cursor = table->slots[cursor].next_pending;
  }
  if (cursor == IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "committed reservation is missing from pending FIFO");
  }

  uint32_t next_index = slot->next_pending;
  if (previous_index == IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    table->pending_head = next_index;
  } else {
    table->slots[previous_index].next_pending = next_index;
  }
  if (table->pending_tail == index) {
    table->pending_tail = previous_index;
  }
  --table->pending_count;
  slot->next_pending = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  slot->flags &= ~IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED;
  return iree_ok_status();
}

static iree_status_t iree_net_rdma_send_reservation_table_validate_completion(
    iree_net_rdma_send_reservation_completion_t completion) {
  switch (completion) {
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_INTERNAL:
    case IREE_NET_RDMA_SEND_RESERVATION_COMPLETION_SEND:
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "unsupported reservation completion mode %u",
                              (uint32_t)completion);
  }
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_net_rdma_send_reservation_table_t* out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_table must not be NULL");
  }
  memset(out_table, 0, sizeof(*out_table));
  out_table->host_allocator = host_allocator;
  out_table->free_head = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  out_table->pending_head = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  out_table->pending_tail = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;

  iree_status_t status = iree_ok_status();
  if (capacity == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "capacity must be non-zero");
  }

  iree_net_rdma_send_reservation_slot_t* slots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*slots), (void**)&slots);
  }

  if (iree_status_is_ok(status)) {
    memset(slots, 0, capacity * sizeof(*slots));
    for (uint32_t index = 0; index < capacity; ++index) {
      slots[index].generation = 1;
      slots[index].next_free =
          index + 1 < capacity
              ? index + 1
              : IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
      slots[index].next_pending =
          IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
    }
    out_table->slots = slots;
    out_table->capacity = capacity;
    out_table->available_capacity = capacity;
    out_table->free_head = 0;
  }

  return status;
}

IREE_API_EXPORT void iree_net_rdma_send_reservation_table_deinitialize(
    iree_net_rdma_send_reservation_table_t* table) {
  if (!table) return;

  if (table->slots) {
    for (uint32_t index = 0; index < table->capacity; ++index) {
      iree_async_buffer_lease_release(&table->slots[index].buffer_lease);
    }
  }
  iree_allocator_free(table->host_allocator, table->slots);
  memset(table, 0, sizeof(*table));
  table->free_head = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  table->pending_head = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  table->pending_tail = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
}

IREE_API_EXPORT uint32_t
iree_net_rdma_send_reservation_table_available_capacity(
    const iree_net_rdma_send_reservation_table_t* table) {
  return table ? table->available_capacity : 0;
}

IREE_API_EXPORT uint32_t iree_net_rdma_send_reservation_table_pending_count(
    const iree_net_rdma_send_reservation_table_t* table) {
  return table ? table->pending_count : 0;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_acquire(
    iree_net_rdma_send_reservation_table_t* table, iree_async_span_list_t spans,
    iree_async_buffer_lease_t* buffer_lease,
    iree_net_rdma_send_reservation_completion_t completion, uint64_t user_data,
    iree_net_carrier_send_handle_t* out_handle) {
  if (!out_handle) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_handle must not be NULL");
  }
  *out_handle = 0;
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }
  if (spans.count == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "spans must not be empty");
  }
  if (!spans.values) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "span values must not be NULL");
  }
  if (spans.count > IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE) {
    return iree_make_status(
        IREE_STATUS_OUT_OF_RANGE,
        "span count %" PRIhsz " exceeds RDMA reservation limit %u", spans.count,
        IREE_NET_RDMA_CONNECTION_DATA_MAX_SEND_SGE);
  }
  IREE_RETURN_IF_ERROR(
      iree_net_rdma_send_reservation_table_validate_completion(completion));

  iree_host_size_t byte_length = 0;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0; i < spans.count && iree_status_is_ok(status);
       ++i) {
    iree_host_size_t new_byte_length = 0;
    if (iree_host_size_checked_add(byte_length, spans.values[i].length,
                                   &new_byte_length)) {
      byte_length = new_byte_length;
    } else {
      status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                "span length total overflows host size");
    }
  }
  if (iree_status_is_ok(status) && byte_length == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "span list must not be empty");
  }
  if (iree_status_is_ok(status) &&
      (table->available_capacity == 0 ||
       table->free_head ==
           IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX)) {
    status = iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "RDMA send reservation table is full");
  }

  if (iree_status_is_ok(status)) {
    uint32_t index = table->free_head;
    iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
    table->free_head = slot->next_free;
    --table->available_capacity;

    if (buffer_lease && buffer_lease->release.fn) {
      slot->buffer_lease = *buffer_lease;
      memset(buffer_lease, 0, sizeof(*buffer_lease));
    } else {
      memset(&slot->buffer_lease, 0, sizeof(slot->buffer_lease));
    }
    memset(slot->spans, 0, sizeof(slot->spans));
    memcpy(slot->spans, spans.values, spans.count * sizeof(spans.values[0]));
    slot->span_count = spans.count;
    slot->byte_length = byte_length;
    slot->completion = completion;
    slot->user_data = user_data;
    slot->next_free = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
    slot->next_pending = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
    slot->flags = IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE;
    *out_handle = iree_net_rdma_send_reservation_table_make_handle(
        index, slot->generation);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_peek(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle,
    iree_net_rdma_send_reservation_t* out_reservation) {
  if (!out_reservation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_reservation must not be NULL");
  }
  memset(out_reservation, 0, sizeof(*out_reservation));
  iree_net_rdma_send_reservation_slot_t* slot = NULL;
  IREE_RETURN_IF_ERROR(iree_net_rdma_send_reservation_table_lookup_slot(
      table, handle, /*require_uncommitted=*/true, /*out_index=*/NULL, &slot));

  out_reservation->buffer_lease = slot->buffer_lease;
  memcpy(out_reservation->spans, slot->spans, sizeof(out_reservation->spans));
  out_reservation->span_count = slot->span_count;
  out_reservation->byte_length = slot->byte_length;
  out_reservation->completion = slot->completion;
  out_reservation->user_data = slot->user_data;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_commit(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle) {
  uint32_t index = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  iree_net_rdma_send_reservation_slot_t* slot = NULL;
  IREE_RETURN_IF_ERROR(iree_net_rdma_send_reservation_table_lookup_slot(
      table, handle, /*require_uncommitted=*/true, &index, &slot));

  slot->flags |= IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED;
  slot->next_pending = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  if (table->pending_tail ==
      IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    table->pending_head = index;
  } else {
    table->slots[table->pending_tail].next_pending = index;
  }
  table->pending_tail = index;
  ++table->pending_count;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_send_reservation_table_peek_pending_front(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t* out_handle,
    iree_net_rdma_send_reservation_t* out_reservation) {
  if (out_handle) *out_handle = 0;
  if (!out_reservation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_reservation must not be NULL");
  }
  memset(out_reservation, 0, sizeof(*out_reservation));
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }
  if (table->pending_head ==
      IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pending reservation FIFO is empty");
  }

  uint32_t index = table->pending_head;
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  if (out_handle) {
    *out_handle = iree_net_rdma_send_reservation_table_make_handle(
        index, slot->generation);
  }
  out_reservation->buffer_lease = slot->buffer_lease;
  memcpy(out_reservation->spans, slot->spans, sizeof(out_reservation->spans));
  out_reservation->span_count = slot->span_count;
  out_reservation->byte_length = slot->byte_length;
  out_reservation->completion = slot->completion;
  out_reservation->user_data = slot->user_data;
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_net_rdma_send_reservation_table_resolve_pending_front(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t* out_handle,
    iree_net_rdma_send_reservation_t* out_reservation) {
  if (out_handle) *out_handle = 0;
  if (!out_reservation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_reservation must not be NULL");
  }
  memset(out_reservation, 0, sizeof(*out_reservation));
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }
  if (table->pending_head ==
      IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "pending reservation FIFO is empty");
  }

  uint32_t index = table->pending_head;
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  if (!iree_any_bit_set(slot->flags,
                        IREE_NET_RDMA_SEND_RESERVATION_SLOT_COMMITTED)) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "pending FIFO head does not reference a committed reservation");
  }
  if (out_handle) {
    *out_handle = iree_net_rdma_send_reservation_table_make_handle(
        index, slot->generation);
  }
  table->pending_head = slot->next_pending;
  if (table->pending_head ==
      IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    table->pending_tail = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  }
  --table->pending_count;
  iree_net_rdma_send_reservation_table_release_slot(table, index,
                                                    out_reservation);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_resolve(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle,
    iree_net_rdma_send_reservation_t* out_reservation) {
  if (!out_reservation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_reservation must not be NULL");
  }
  memset(out_reservation, 0, sizeof(*out_reservation));
  uint32_t index = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  IREE_RETURN_IF_ERROR(iree_net_rdma_send_reservation_table_lookup_slot(
      table, handle, /*require_uncommitted=*/true, &index,
      /*out_slot=*/NULL));

  iree_net_rdma_send_reservation_table_release_slot(table, index,
                                                    out_reservation);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_resolve_next(
    iree_net_rdma_send_reservation_table_t* table, uint32_t* inout_cursor,
    iree_net_rdma_send_reservation_t* out_reservation, bool* out_found) {
  if (out_found) *out_found = false;
  if (!out_reservation) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_reservation must not be NULL");
  }
  memset(out_reservation, 0, sizeof(*out_reservation));
  if (!out_found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_found must not be NULL");
  }
  if (!inout_cursor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inout_cursor must not be NULL");
  }
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }

  iree_status_t status = iree_ok_status();
  for (uint32_t index = *inout_cursor; index < table->capacity; ++index) {
    *inout_cursor = index + 1;
    iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
    if (!iree_any_bit_set(slot->flags,
                          IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE)) {
      continue;
    }

    status =
        iree_net_rdma_send_reservation_table_unlink_pending_slot(table, index);
    if (iree_status_is_ok(status)) {
      iree_net_rdma_send_reservation_table_release_slot(table, index,
                                                        out_reservation);
      *out_found = true;
    }
    break;
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_abort(
    iree_net_rdma_send_reservation_table_t* table,
    iree_net_carrier_send_handle_t handle) {
  iree_net_rdma_send_reservation_t reservation;
  iree_status_t status =
      iree_net_rdma_send_reservation_table_resolve(table, handle, &reservation);
  if (iree_status_is_ok(status)) {
    iree_async_buffer_lease_release(&reservation.buffer_lease);
  }
  return status;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_abort_all(
    iree_net_rdma_send_reservation_table_t* table,
    uint32_t* out_aborted_count) {
  if (out_aborted_count) *out_aborted_count = 0;
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }

  uint32_t aborted_count = 0;
  for (uint32_t index = 0; index < table->capacity; ++index) {
    iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
    if (!iree_any_bit_set(slot->flags,
                          IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE)) {
      continue;
    }
    iree_net_rdma_send_reservation_t reservation;
    memset(&reservation, 0, sizeof(reservation));
    iree_net_rdma_send_reservation_table_release_slot(table, index,
                                                      &reservation);
    iree_async_buffer_lease_release(&reservation.buffer_lease);
    ++aborted_count;
  }
  if (out_aborted_count) *out_aborted_count = aborted_count;
  table->pending_head = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  table->pending_tail = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  table->pending_count = 0;
  return iree_ok_status();
}
