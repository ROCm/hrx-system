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
};

struct iree_net_rdma_send_reservation_slot_t {
  // Buffer lease retained between begin_send and commit/abort.
  iree_async_buffer_lease_t buffer_lease;

  // Number of valid caller-written bytes in buffer_lease.
  iree_host_size_t byte_length;

  // Generation encoded in the carrier send handle to reject stale handles.
  uint32_t generation;

  // Next slot index in the free list when not in use.
  uint32_t next_free;

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

static void iree_net_rdma_send_reservation_table_release_slot(
    iree_net_rdma_send_reservation_table_t* table, uint32_t index,
    iree_net_rdma_send_reservation_t* out_reservation) {
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  out_reservation->buffer_lease = slot->buffer_lease;
  out_reservation->byte_length = slot->byte_length;

  memset(&slot->buffer_lease, 0, sizeof(slot->buffer_lease));
  slot->byte_length = 0;
  slot->generation =
      iree_net_rdma_send_reservation_table_next_generation(slot->generation);
  slot->next_free = table->free_head;
  slot->flags = 0;
  table->free_head = index;
  ++table->available_capacity;
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
}

IREE_API_EXPORT uint32_t
iree_net_rdma_send_reservation_table_available_capacity(
    const iree_net_rdma_send_reservation_table_t* table) {
  return table ? table->available_capacity : 0;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_send_reservation_table_acquire(
    iree_net_rdma_send_reservation_table_t* table,
    iree_async_buffer_lease_t* buffer_lease, iree_host_size_t byte_length,
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
  if (!buffer_lease || !buffer_lease->release.fn) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "buffer_lease must be live");
  }
  if (byte_length == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "byte_length must be non-zero");
  }
  if (byte_length > buffer_lease->span.length) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "byte_length %" PRIhsz
                            " exceeds lease length %" PRIhsz,
                            byte_length, buffer_lease->span.length);
  }
  if (table->available_capacity == 0 ||
      table->free_head == IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "RDMA send reservation table is full");
  }

  uint32_t index = table->free_head;
  iree_net_rdma_send_reservation_slot_t* slot = &table->slots[index];
  table->free_head = slot->next_free;
  --table->available_capacity;

  slot->buffer_lease = *buffer_lease;
  memset(buffer_lease, 0, sizeof(*buffer_lease));
  slot->byte_length = byte_length;
  slot->next_free = IREE_NET_RDMA_SEND_RESERVATION_TABLE_INVALID_INDEX;
  slot->flags = IREE_NET_RDMA_SEND_RESERVATION_SLOT_IN_USE;
  *out_handle =
      iree_net_rdma_send_reservation_table_make_handle(index, slot->generation);
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

  iree_net_rdma_send_reservation_table_release_slot(table, index,
                                                    out_reservation);
  return iree_ok_status();
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
  return iree_ok_status();
}
