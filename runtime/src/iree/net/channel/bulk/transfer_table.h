// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Fixed-capacity active bulk transfer table.
//
// The table maps arbitrary 64-bit transfer IDs to stable descriptors without
// allocating on insert/remove. It is intended for bulk transfer engines that
// need bounded active-transfer state while keeping queue/control channels free
// to make progress independently from bulk movement.

#ifndef IREE_NET_CHANNEL_BULK_TRANSFER_TABLE_H_
#define IREE_NET_CHANNEL_BULK_TRANSFER_TABLE_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default maximum number of active transfers tracked by a table.
#define IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY 64

// Default first generated transfer ID.
#define IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_INITIAL_ID 1

// Default increment between generated transfer IDs.
#define IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_ID_STRIDE 1

typedef struct iree_net_bulk_transfer_t iree_net_bulk_transfer_t;
typedef struct iree_net_bulk_transfer_table_t iree_net_bulk_transfer_table_t;

// Visits an active transfer.
typedef void (*iree_net_bulk_transfer_visit_fn_t)(
    void* user_data, iree_net_bulk_transfer_t* transfer);

// Bulk transfer table creation options.
typedef struct iree_net_bulk_transfer_table_options_t {
  // Maximum active transfers retained by the table.
  iree_host_size_t capacity;

  // Byte length of owner-managed storage attached to each transfer.
  iree_host_size_t user_storage_size;

  // Alignment of owner-managed storage. Zero selects max host alignment.
  iree_host_size_t user_storage_alignment;

  // First transfer ID produced by allocate_transfer. Zero selects the default.
  uint64_t initial_transfer_id;

  // Transfer ID increment used by allocate_transfer. Zero selects the default.
  uint64_t transfer_id_stride;
} iree_net_bulk_transfer_table_options_t;

// Returns conservative default transfer table options.
static inline iree_net_bulk_transfer_table_options_t
iree_net_bulk_transfer_table_options_default(void) {
  iree_net_bulk_transfer_table_options_t options = {0};
  options.capacity = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_CAPACITY;
  options.user_storage_alignment = iree_max_align_t;
  options.initial_transfer_id = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_INITIAL_ID;
  options.transfer_id_stride = IREE_NET_BULK_TRANSFER_TABLE_DEFAULT_ID_STRIDE;
  return options;
}

// Allocates a fixed-capacity transfer table.
//
// The returned table owns one allocation containing the table, stable transfer
// descriptors, the ID map, and optional per-transfer user storage. No steady
// state insert/remove operation allocates.
iree_status_t iree_net_bulk_transfer_table_allocate(
    const iree_net_bulk_transfer_table_options_t* options,
    iree_allocator_t host_allocator,
    iree_net_bulk_transfer_table_t** out_table);

// Frees a transfer table. The owner must have released resources referenced by
// active transfers before calling this.
void iree_net_bulk_transfer_table_free(iree_net_bulk_transfer_table_t* table);

// Returns the maximum number of active transfers.
iree_host_size_t iree_net_bulk_transfer_table_capacity(
    const iree_net_bulk_transfer_table_t* table);

// Returns the current number of active transfers.
iree_host_size_t iree_net_bulk_transfer_table_count(
    const iree_net_bulk_transfer_table_t* table);

// Visits all active transfers in unspecified order.
//
// The visitor must not mutate |table|. Use this for teardown/failure handling
// before releasing owner-managed resources and then clear/free the table.
void iree_net_bulk_transfer_table_visit(
    iree_net_bulk_transfer_table_t* table,
    iree_net_bulk_transfer_visit_fn_t visitor, void* user_data);

// Removes all active transfers from the table.
void iree_net_bulk_transfer_table_clear(iree_net_bulk_transfer_table_t* table);

// Inserts a caller-specified transfer ID into the table.
//
// |transfer_id| must be non-zero and unique within |table|. On success
// |out_transfer| is a stable descriptor pointer that remains valid until that
// specific transfer is removed or the table is freed.
iree_status_t iree_net_bulk_transfer_table_insert(
    iree_net_bulk_transfer_table_t* table, uint64_t transfer_id,
    uint64_t total_size, uint64_t user_value,
    iree_net_bulk_transfer_t** out_transfer);

// Allocates a generated transfer ID and inserts it into the table.
//
// Generated IDs use the configured initial ID and stride, skipping IDs already
// present in |table|. This is intended for local transfer initiation; protocol
// layers that need direction-specific ID partitioning can use an odd/even or
// otherwise disjoint initial/stride pair.
iree_status_t iree_net_bulk_transfer_table_allocate_transfer(
    iree_net_bulk_transfer_table_t* table, uint64_t total_size,
    uint64_t user_value, iree_net_bulk_transfer_t** out_transfer);

// Looks up an active transfer by ID. Returns NULL when absent.
iree_net_bulk_transfer_t* iree_net_bulk_transfer_table_lookup(
    iree_net_bulk_transfer_table_t* table, uint64_t transfer_id);

// Removes a transfer by ID. Returns true when a transfer was removed.
bool iree_net_bulk_transfer_table_remove(iree_net_bulk_transfer_table_t* table,
                                         uint64_t transfer_id);

// Returns a transfer's ID.
uint64_t iree_net_bulk_transfer_id(const iree_net_bulk_transfer_t* transfer);

// Returns a transfer's total byte length.
uint64_t iree_net_bulk_transfer_total_size(
    const iree_net_bulk_transfer_t* transfer);

// Returns a transfer's owner-managed scalar value.
uint64_t iree_net_bulk_transfer_user_value(
    const iree_net_bulk_transfer_t* transfer);

// Updates a transfer's owner-managed scalar value.
void iree_net_bulk_transfer_set_user_value(iree_net_bulk_transfer_t* transfer,
                                           uint64_t user_value);

// Returns the owner-managed storage attached to the transfer.
iree_byte_span_t iree_net_bulk_transfer_user_storage(
    iree_net_bulk_transfer_t* transfer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_TRANSFER_TABLE_H_
