// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/channel/bulk/transfer_table.h"

#include <string.h>

#define IREE_NET_BULK_TRANSFER_INDEX_EMPTY UINT32_MAX

//===----------------------------------------------------------------------===//
// Storage
//===----------------------------------------------------------------------===//

struct iree_net_bulk_transfer_t {
  // Transfer ID used on the wire.
  uint64_t transfer_id;

  // Total transfer byte length announced by the transfer owner.
  uint64_t total_size;

  // Owner-managed scalar associated with this transfer.
  uint64_t user_value;

  // Descriptor array index for returning this descriptor to the free list.
  uint32_t descriptor_index;

  // Next descriptor index while this descriptor is free.
  uint32_t next_free_index;

  // Owner-managed storage attached to this transfer.
  iree_byte_span_t user_storage;
};

struct iree_net_bulk_transfer_table_t {
  // Host allocator used for the table allocation.
  iree_allocator_t host_allocator;

  // Maximum active transfers retained by the table.
  iree_host_size_t capacity;

  // Current active transfer count.
  iree_host_size_t count;

  // Power-of-two capacity of the open-addressed ID map.
  iree_host_size_t map_capacity;

  // Byte length of owner-managed storage per transfer.
  iree_host_size_t user_storage_size;

  // Byte stride between owner-managed storage slots.
  iree_host_size_t user_storage_stride;

  // Next generated transfer ID candidate.
  uint64_t next_transfer_id;

  // Transfer ID increment for generated IDs.
  uint64_t transfer_id_stride;

  // Descriptor index at the head of the free list, or empty.
  uint32_t free_head;

  // Stable descriptor array indexed by descriptor index.
  iree_net_bulk_transfer_t* transfers;

  // Open-addressed map from transfer ID to descriptor index.
  uint32_t* map_indices;

  // Owner-managed per-transfer storage block.
  uint8_t* user_storage;
};

static uint64_t iree_net_bulk_transfer_table_mix_u64(uint64_t value) {
  value ^= value >> 33;
  value *= 0xff51afd7ed558ccdull;
  value ^= value >> 33;
  value *= 0xc4ceb9fe1a85ec53ull;
  value ^= value >> 33;
  return value;
}

static iree_host_size_t iree_net_bulk_transfer_table_map_slot(
    uint64_t transfer_id, iree_host_size_t capacity) {
  return (iree_host_size_t)(iree_net_bulk_transfer_table_mix_u64(transfer_id) &
                            (capacity - 1));
}

static void iree_net_bulk_transfer_table_reset_transfer(
    iree_net_bulk_transfer_table_t* table, uint32_t descriptor_index,
    uint32_t next_free_index) {
  iree_net_bulk_transfer_t* transfer = &table->transfers[descriptor_index];
  transfer->transfer_id = 0;
  transfer->total_size = 0;
  transfer->user_value = 0;
  transfer->descriptor_index = descriptor_index;
  transfer->next_free_index = next_free_index;
  if (transfer->user_storage.data_length > 0) {
    memset(transfer->user_storage.data, 0, transfer->user_storage.data_length);
  }
}

static iree_status_t iree_net_bulk_transfer_table_resolve_options(
    const iree_net_bulk_transfer_table_options_t* options,
    iree_net_bulk_transfer_table_options_t* out_options,
    iree_host_size_t* out_map_capacity,
    iree_host_size_t* out_user_storage_stride) {
  iree_net_bulk_transfer_table_options_t resolved_options =
      iree_net_bulk_transfer_table_options_default();
  if (options) resolved_options = *options;
  if (resolved_options.capacity == 0) {
    resolved_options.capacity = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY;
  }
  if (resolved_options.capacity >= IREE_NET_BULK_TRANSFER_INDEX_EMPTY) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk transfer table capacity too large: %" PRIhsz,
                            resolved_options.capacity);
  }

  if (resolved_options.user_storage_alignment == 0) {
    resolved_options.user_storage_alignment = iree_max_align_t;
  }
  if (!iree_host_size_is_valid_alignment(
          resolved_options.user_storage_alignment)) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "bulk transfer user storage alignment must be a power of two: %" PRIhsz,
        resolved_options.user_storage_alignment);
  }

  if (resolved_options.initial_transfer_id == 0) {
    resolved_options.initial_transfer_id =
        IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_INITIAL_ID;
  }
  if (resolved_options.transfer_id_stride == 0) {
    resolved_options.transfer_id_stride =
        IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_ID_STRIDE;
  }

  iree_host_size_t map_capacity_lower_bound = 0;
  if (!iree_host_size_checked_mul(resolved_options.capacity, 2,
                                  &map_capacity_lower_bound)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk transfer table map capacity overflow");
  }
  iree_host_size_t map_capacity =
      iree_host_size_next_power_of_two(iree_max(map_capacity_lower_bound, 1));
  if (map_capacity == IREE_HOST_SIZE_MAX) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk transfer table map capacity overflow");
  }

  iree_host_size_t user_storage_stride = 0;
  if (resolved_options.user_storage_size > 0 &&
      !iree_host_size_checked_align(resolved_options.user_storage_size,
                                    resolved_options.user_storage_alignment,
                                    &user_storage_stride)) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "bulk transfer user storage stride overflow");
  }

  *out_options = resolved_options;
  *out_map_capacity = map_capacity;
  *out_user_storage_stride = user_storage_stride;
  return iree_ok_status();
}

static iree_status_t iree_net_bulk_transfer_table_calculate_layout(
    const iree_net_bulk_transfer_table_options_t* options,
    iree_host_size_t map_capacity, iree_host_size_t user_storage_stride,
    iree_host_size_t* out_total_size, iree_host_size_t* out_transfers_offset,
    iree_host_size_t* out_map_indices_offset,
    iree_host_size_t* out_user_storage_offset) {
  iree_host_size_t user_storage_alignment =
      user_storage_stride > 0 ? options->user_storage_alignment : 1;
  return IREE_STRUCT_LAYOUT(
      sizeof(iree_net_bulk_transfer_table_t), out_total_size,
      IREE_STRUCT_FIELD_ALIGNED(options->capacity, iree_net_bulk_transfer_t,
                                iree_alignof(iree_net_bulk_transfer_t),
                                out_transfers_offset),
      IREE_STRUCT_FIELD_ALIGNED(map_capacity, uint32_t, iree_alignof(uint32_t),
                                out_map_indices_offset),
      IREE_STRUCT_ARRAY_FIELD_ALIGNED(options->capacity, user_storage_stride,
                                      uint8_t, user_storage_alignment,
                                      out_user_storage_offset));
}

//===----------------------------------------------------------------------===//
// Map operations
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_net_bulk_transfer_table_find_map_slot(
    const iree_net_bulk_transfer_table_t* table, uint64_t transfer_id,
    bool* out_found) {
  iree_host_size_t slot =
      iree_net_bulk_transfer_table_map_slot(transfer_id, table->map_capacity);
  while (table->map_indices[slot] != IREE_NET_BULK_TRANSFER_INDEX_EMPTY) {
    uint32_t descriptor_index = table->map_indices[slot];
    if (table->transfers[descriptor_index].transfer_id == transfer_id) {
      *out_found = true;
      return slot;
    }
    slot = (slot + 1) & (table->map_capacity - 1);
  }
  *out_found = false;
  return slot;
}

static bool iree_net_bulk_transfer_table_should_shift_map_slot(
    const iree_net_bulk_transfer_table_t* table, iree_host_size_t hole_slot,
    iree_host_size_t occupied_slot) {
  uint32_t descriptor_index = table->map_indices[occupied_slot];
  uint64_t transfer_id = table->transfers[descriptor_index].transfer_id;
  iree_host_size_t home_slot =
      iree_net_bulk_transfer_table_map_slot(transfer_id, table->map_capacity);
  iree_host_size_t slot_distance =
      (occupied_slot - home_slot) & (table->map_capacity - 1);
  iree_host_size_t hole_distance =
      (hole_slot - home_slot) & (table->map_capacity - 1);
  return hole_distance < slot_distance;
}

static void iree_net_bulk_transfer_table_remove_map_slot(
    iree_net_bulk_transfer_table_t* table, iree_host_size_t slot) {
  iree_host_size_t hole_slot = slot;
  iree_host_size_t next_slot = (slot + 1) & (table->map_capacity - 1);
  while (table->map_indices[next_slot] != IREE_NET_BULK_TRANSFER_INDEX_EMPTY) {
    if (iree_net_bulk_transfer_table_should_shift_map_slot(table, hole_slot,
                                                           next_slot)) {
      table->map_indices[hole_slot] = table->map_indices[next_slot];
      hole_slot = next_slot;
    }
    next_slot = (next_slot + 1) & (table->map_capacity - 1);
  }
  table->map_indices[hole_slot] = IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
}

//===----------------------------------------------------------------------===//
// iree_net_bulk_transfer_table_t
//===----------------------------------------------------------------------===//

iree_status_t iree_net_bulk_transfer_table_allocate(
    const iree_net_bulk_transfer_table_options_t* options,
    iree_allocator_t host_allocator,
    iree_net_bulk_transfer_table_t** out_table) {
  IREE_ASSERT_ARGUMENT(out_table);
  *out_table = NULL;

  iree_net_bulk_transfer_table_options_t resolved_options;
  iree_host_size_t map_capacity = 0;
  iree_host_size_t user_storage_stride = 0;
  IREE_RETURN_IF_ERROR(iree_net_bulk_transfer_table_resolve_options(
      options, &resolved_options, &map_capacity, &user_storage_stride));

  iree_host_size_t total_size = 0;
  iree_host_size_t transfers_offset = 0;
  iree_host_size_t map_indices_offset = 0;
  iree_host_size_t user_storage_offset = 0;
  IREE_RETURN_IF_ERROR(iree_net_bulk_transfer_table_calculate_layout(
      &resolved_options, map_capacity, user_storage_stride, &total_size,
      &transfers_offset, &map_indices_offset, &user_storage_offset));

  iree_net_bulk_transfer_table_t* table = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&table));
  memset(table, 0, total_size);

  uint8_t* table_storage = (uint8_t*)table;
  table->host_allocator = host_allocator;
  table->capacity = resolved_options.capacity;
  table->map_capacity = map_capacity;
  table->user_storage_size = resolved_options.user_storage_size;
  table->user_storage_stride = user_storage_stride;
  table->next_transfer_id = resolved_options.initial_transfer_id;
  table->transfer_id_stride = resolved_options.transfer_id_stride;
  table->free_head =
      table->capacity > 0 ? 0 : IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
  table->transfers =
      (iree_net_bulk_transfer_t*)(table_storage + transfers_offset);
  table->map_indices = (uint32_t*)(table_storage + map_indices_offset);
  table->user_storage = (uint8_t*)(table_storage + user_storage_offset);

  for (iree_host_size_t i = 0; i < table->map_capacity; ++i) {
    table->map_indices[i] = IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
  }
  for (iree_host_size_t i = 0; i < table->capacity; ++i) {
    iree_net_bulk_transfer_t* transfer = &table->transfers[i];
    if (table->user_storage_size > 0) {
      transfer->user_storage = iree_make_byte_span(
          table->user_storage + i * table->user_storage_stride,
          table->user_storage_size);
    }
    uint32_t next_free_index = i + 1 < table->capacity
                                   ? (uint32_t)(i + 1)
                                   : IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
    iree_net_bulk_transfer_table_reset_transfer(table, (uint32_t)i,
                                                next_free_index);
  }

  *out_table = table;
  return iree_ok_status();
}

void iree_net_bulk_transfer_table_free(iree_net_bulk_transfer_table_t* table) {
  if (!table) return;
  iree_allocator_t host_allocator = table->host_allocator;
  iree_allocator_free(host_allocator, table);
}

iree_host_size_t iree_net_bulk_transfer_table_capacity(
    const iree_net_bulk_transfer_table_t* table) {
  IREE_ASSERT_ARGUMENT(table);
  return table->capacity;
}

iree_host_size_t iree_net_bulk_transfer_table_count(
    const iree_net_bulk_transfer_table_t* table) {
  IREE_ASSERT_ARGUMENT(table);
  return table->count;
}

void iree_net_bulk_transfer_table_visit(
    iree_net_bulk_transfer_table_t* table,
    iree_net_bulk_transfer_visit_fn_t visitor, void* user_data) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(visitor);
  for (iree_host_size_t i = 0; i < table->capacity; ++i) {
    iree_net_bulk_transfer_t* transfer = &table->transfers[i];
    if (transfer->transfer_id != 0) visitor(user_data, transfer);
  }
}

void iree_net_bulk_transfer_table_clear(iree_net_bulk_transfer_table_t* table) {
  IREE_ASSERT_ARGUMENT(table);
  for (iree_host_size_t i = 0; i < table->map_capacity; ++i) {
    table->map_indices[i] = IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
  }
  for (iree_host_size_t i = 0; i < table->capacity; ++i) {
    uint32_t next_free_index = i + 1 < table->capacity
                                   ? (uint32_t)(i + 1)
                                   : IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
    iree_net_bulk_transfer_table_reset_transfer(table, (uint32_t)i,
                                                next_free_index);
  }
  table->free_head =
      table->capacity > 0 ? 0 : IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
  table->count = 0;
}

iree_status_t iree_net_bulk_transfer_table_insert(
    iree_net_bulk_transfer_table_t* table, uint64_t transfer_id,
    uint64_t total_size, uint64_t user_value,
    iree_net_bulk_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_transfer);
  *out_transfer = NULL;

  if (transfer_id == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk transfer ID must be non-zero");
  }
  if (table->free_head == IREE_NET_BULK_TRANSFER_INDEX_EMPTY) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  bool found = false;
  iree_host_size_t map_slot =
      iree_net_bulk_transfer_table_find_map_slot(table, transfer_id, &found);
  if (found) {
    return iree_make_status(IREE_STATUS_ALREADY_EXISTS,
                            "bulk transfer ID already exists: %" PRIu64,
                            transfer_id);
  }

  uint32_t descriptor_index = table->free_head;
  iree_net_bulk_transfer_t* transfer = &table->transfers[descriptor_index];
  table->free_head = transfer->next_free_index;

  transfer->transfer_id = transfer_id;
  transfer->total_size = total_size;
  transfer->user_value = user_value;
  transfer->next_free_index = IREE_NET_BULK_TRANSFER_INDEX_EMPTY;
  if (transfer->user_storage.data_length > 0) {
    memset(transfer->user_storage.data, 0, transfer->user_storage.data_length);
  }

  table->map_indices[map_slot] = descriptor_index;
  ++table->count;
  *out_transfer = transfer;
  return iree_ok_status();
}

iree_status_t iree_net_bulk_transfer_table_allocate_transfer(
    iree_net_bulk_transfer_table_t* table, uint64_t total_size,
    uint64_t user_value, iree_net_bulk_transfer_t** out_transfer) {
  IREE_ASSERT_ARGUMENT(table);
  IREE_ASSERT_ARGUMENT(out_transfer);
  *out_transfer = NULL;

  if (table->free_head == IREE_NET_BULK_TRANSFER_INDEX_EMPTY) {
    return iree_status_from_code(IREE_STATUS_RESOURCE_EXHAUSTED);
  }

  uint64_t transfer_id = table->next_transfer_id;
  bool found = true;
  for (iree_host_size_t i = 0; found && i < table->capacity; ++i) {
    (void)iree_net_bulk_transfer_table_find_map_slot(table, transfer_id,
                                                     &found);
    if (found) transfer_id += table->transfer_id_stride;
    if (transfer_id == 0) transfer_id += table->transfer_id_stride;
  }
  if (found) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "bulk transfer ID space exhausted");
  }

  iree_status_t status = iree_net_bulk_transfer_table_insert(
      table, transfer_id, total_size, user_value, out_transfer);
  if (iree_status_is_ok(status)) {
    table->next_transfer_id = transfer_id + table->transfer_id_stride;
    if (table->next_transfer_id == 0) {
      table->next_transfer_id += table->transfer_id_stride;
    }
  }
  return status;
}

iree_net_bulk_transfer_t* iree_net_bulk_transfer_table_lookup(
    iree_net_bulk_transfer_table_t* table, uint64_t transfer_id) {
  IREE_ASSERT_ARGUMENT(table);
  if (transfer_id == 0) return NULL;
  bool found = false;
  iree_host_size_t map_slot =
      iree_net_bulk_transfer_table_find_map_slot(table, transfer_id, &found);
  return found ? &table->transfers[table->map_indices[map_slot]] : NULL;
}

bool iree_net_bulk_transfer_table_remove(iree_net_bulk_transfer_table_t* table,
                                         uint64_t transfer_id) {
  IREE_ASSERT_ARGUMENT(table);
  if (transfer_id == 0) return false;

  bool found = false;
  iree_host_size_t map_slot =
      iree_net_bulk_transfer_table_find_map_slot(table, transfer_id, &found);
  if (!found) return false;

  uint32_t descriptor_index = table->map_indices[map_slot];
  iree_net_bulk_transfer_table_remove_map_slot(table, map_slot);

  iree_net_bulk_transfer_table_reset_transfer(table, descriptor_index,
                                              table->free_head);
  table->free_head = descriptor_index;
  --table->count;
  return true;
}

uint64_t iree_net_bulk_transfer_id(const iree_net_bulk_transfer_t* transfer) {
  IREE_ASSERT_ARGUMENT(transfer);
  return transfer->transfer_id;
}

uint64_t iree_net_bulk_transfer_total_size(
    const iree_net_bulk_transfer_t* transfer) {
  IREE_ASSERT_ARGUMENT(transfer);
  return transfer->total_size;
}

uint64_t iree_net_bulk_transfer_user_value(
    const iree_net_bulk_transfer_t* transfer) {
  IREE_ASSERT_ARGUMENT(transfer);
  return transfer->user_value;
}

void iree_net_bulk_transfer_set_user_value(iree_net_bulk_transfer_t* transfer,
                                           uint64_t user_value) {
  IREE_ASSERT_ARGUMENT(transfer);
  transfer->user_value = user_value;
}

iree_byte_span_t iree_net_bulk_transfer_user_storage(
    iree_net_bulk_transfer_t* transfer) {
  IREE_ASSERT_ARGUMENT(transfer);
  return transfer->user_storage;
}
