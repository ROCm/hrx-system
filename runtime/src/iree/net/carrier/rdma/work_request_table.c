// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/rdma/work_request_table.h"

#include <string.h>

#define IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX UINT32_MAX

typedef uint8_t iree_net_rdma_work_request_slot_flags_t;
enum iree_net_rdma_work_request_slot_bits_e {
  IREE_NET_RDMA_WORK_REQUEST_SLOT_IN_USE = 1u << 0,
};

struct iree_net_rdma_work_request_slot_t {
  // Buffer lease retained for the native work request lifetime.
  iree_async_buffer_lease_t retained_buffer_lease;

  // User data returned when the work request completes.
  uint64_t user_data;

  // Byte length returned when the work request completes.
  iree_host_size_t byte_length;

  // Generation encoded in wr_id to reject stale completions.
  uint32_t generation;

  // Next slot index in the free list when not in use.
  uint32_t next_free;

  // Operation class returned when the work request completes.
  iree_net_rdma_work_request_operation_t operation;

  // Bitfield of iree_net_rdma_work_request_slot_bits_e values.
  iree_net_rdma_work_request_slot_flags_t flags;
};

static uint64_t iree_net_rdma_work_request_table_make_wr_id(
    uint32_t index, uint32_t generation) {
  return ((uint64_t)generation << 32) | index;
}

static uint32_t iree_net_rdma_work_request_table_wr_id_index(uint64_t wr_id) {
  return (uint32_t)(wr_id & 0xFFFFFFFFu);
}

static uint32_t iree_net_rdma_work_request_table_wr_id_generation(
    uint64_t wr_id) {
  return (uint32_t)(wr_id >> 32);
}

static uint32_t iree_net_rdma_work_request_table_next_generation(
    uint32_t generation) {
  ++generation;
  return generation == 0 ? 1 : generation;
}

static bool iree_net_rdma_work_request_operation_is_valid(
    iree_net_rdma_work_request_operation_t operation) {
  switch (operation) {
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_SEND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_RECV:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_WRITE:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_DIRECT_READ:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_COMMITTED_SEND:
    case IREE_NET_RDMA_WORK_REQUEST_OPERATION_CREDIT_GRANT:
      return true;
    default:
      return false;
  }
}

static void iree_net_rdma_work_request_table_release_slot(
    iree_net_rdma_work_request_table_t* table, uint32_t index,
    iree_net_rdma_work_request_completion_t* out_completion) {
  iree_net_rdma_work_request_slot_t* slot = &table->slots[index];
  out_completion->operation = slot->operation;
  out_completion->user_data = slot->user_data;
  out_completion->byte_length = slot->byte_length;
  out_completion->retained_buffer_lease = slot->retained_buffer_lease;

  memset(&slot->retained_buffer_lease, 0, sizeof(slot->retained_buffer_lease));
  slot->user_data = 0;
  slot->byte_length = 0;
  slot->operation = IREE_NET_RDMA_WORK_REQUEST_OPERATION_NONE;
  slot->generation =
      iree_net_rdma_work_request_table_next_generation(slot->generation);
  slot->next_free = table->free_head;
  slot->flags = 0;
  table->free_head = index;
  ++table->available_capacity;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_initialize(
    uint32_t capacity, iree_allocator_t host_allocator,
    iree_net_rdma_work_request_table_t* out_table) {
  if (!out_table) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_table must not be NULL");
  }
  memset(out_table, 0, sizeof(*out_table));
  out_table->host_allocator = host_allocator;
  out_table->free_head = IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX;

  iree_status_t status = iree_ok_status();
  if (capacity == 0) {
    status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "capacity must be non-zero");
  }

  iree_net_rdma_work_request_slot_t* slots = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc_array(host_allocator, capacity,
                                         sizeof(*slots), (void**)&slots);
  }

  if (iree_status_is_ok(status)) {
    memset(slots, 0, capacity * sizeof(*slots));
    for (uint32_t index = 0; index < capacity; ++index) {
      slots[index].generation = 1;
      slots[index].next_free =
          index + 1 < capacity ? index + 1
                               : IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX;
    }
    out_table->slots = slots;
    out_table->capacity = capacity;
    out_table->available_capacity = capacity;
    out_table->free_head = 0;
  }

  return status;
}

IREE_API_EXPORT void iree_net_rdma_work_request_table_deinitialize(
    iree_net_rdma_work_request_table_t* table) {
  if (!table) return;

  if (table->slots) {
    for (uint32_t index = 0; index < table->capacity; ++index) {
      iree_net_rdma_work_request_slot_t* slot = &table->slots[index];
      if (iree_any_bit_set(slot->flags,
                           IREE_NET_RDMA_WORK_REQUEST_SLOT_IN_USE)) {
        iree_async_buffer_lease_release(&slot->retained_buffer_lease);
      }
    }
  }
  iree_allocator_free(table->host_allocator, table->slots);
  memset(table, 0, sizeof(*table));
  table->free_head = IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX;
}

IREE_API_EXPORT uint32_t iree_net_rdma_work_request_table_available_capacity(
    const iree_net_rdma_work_request_table_t* table) {
  return table ? table->available_capacity : 0;
}

IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_acquire(
    iree_net_rdma_work_request_table_t* table,
    iree_net_rdma_work_request_operation_t operation, uint64_t user_data,
    iree_host_size_t byte_length,
    iree_async_buffer_lease_t* retained_buffer_lease, uint64_t* out_wr_id) {
  if (!out_wr_id) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_wr_id must not be NULL");
  }
  *out_wr_id = 0;
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }
  if (!iree_net_rdma_work_request_operation_is_valid(operation)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "operation %u is not valid", operation);
  }
  if (table->available_capacity == 0 ||
      table->free_head == IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "RDMA work request table is full");
  }

  iree_net_rdma_work_request_slot_t* slots = table->slots;
  uint32_t index = table->free_head;
  iree_net_rdma_work_request_slot_t* slot = &slots[index];
  table->free_head = slot->next_free;
  --table->available_capacity;

  slot->user_data = user_data;
  slot->byte_length = byte_length;
  if (retained_buffer_lease) {
    slot->retained_buffer_lease = *retained_buffer_lease;
    memset(retained_buffer_lease, 0, sizeof(*retained_buffer_lease));
  } else {
    memset(&slot->retained_buffer_lease, 0,
           sizeof(slot->retained_buffer_lease));
  }
  slot->operation = operation;
  slot->next_free = IREE_NET_RDMA_WORK_REQUEST_TABLE_INVALID_INDEX;
  slot->flags = IREE_NET_RDMA_WORK_REQUEST_SLOT_IN_USE;
  *out_wr_id =
      iree_net_rdma_work_request_table_make_wr_id(index, slot->generation);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_complete(
    iree_net_rdma_work_request_table_t* table, uint64_t wr_id,
    iree_net_rdma_work_request_completion_t* out_completion) {
  if (!out_completion) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_completion must not be NULL");
  }
  memset(out_completion, 0, sizeof(*out_completion));
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }

  uint32_t index = iree_net_rdma_work_request_table_wr_id_index(wr_id);
  if (index >= table->capacity) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "wr_id index %u is outside table capacity %u",
                            index, table->capacity);
  }

  iree_net_rdma_work_request_slot_t* slots = table->slots;
  iree_net_rdma_work_request_slot_t* slot = &slots[index];
  uint32_t generation =
      iree_net_rdma_work_request_table_wr_id_generation(wr_id);
  if (!iree_any_bit_set(slot->flags, IREE_NET_RDMA_WORK_REQUEST_SLOT_IN_USE) ||
      slot->generation != generation) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "wr_id does not reference an in-flight work request");
  }

  iree_net_rdma_work_request_table_release_slot(table, index, out_completion);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_net_rdma_work_request_table_drain_next(
    iree_net_rdma_work_request_table_t* table, uint32_t* inout_cursor,
    iree_net_rdma_work_request_completion_t* out_completion, bool* out_found) {
  if (out_completion) memset(out_completion, 0, sizeof(*out_completion));
  if (out_found) *out_found = false;
  if (!table || !table->slots) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "table must be initialized");
  }
  if (!inout_cursor) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "inout_cursor must not be NULL");
  }
  if (!out_completion) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_completion must not be NULL");
  }
  if (!out_found) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "out_found must not be NULL");
  }

  for (uint32_t index = *inout_cursor; index < table->capacity; ++index) {
    iree_net_rdma_work_request_slot_t* slot = &table->slots[index];
    if (!iree_any_bit_set(slot->flags,
                          IREE_NET_RDMA_WORK_REQUEST_SLOT_IN_USE)) {
      continue;
    }
    iree_net_rdma_work_request_table_release_slot(table, index, out_completion);
    *inout_cursor = index + 1;
    *out_found = true;
    return iree_ok_status();
  }

  *inout_cursor = table->capacity;
  return iree_ok_status();
}
