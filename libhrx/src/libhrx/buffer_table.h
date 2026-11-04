// Copyright 2026 The HRX Authors
// SPDX-License-Identifier: Apache-2.0
//
// Per-device buffer table: maps device/host pointers to hrx_buffer_t.
// Thread-safe via internal mutex.  Also carries an opaque user_data pointer
// per entry so that higher layers (e.g. the HIP binding) can attach their
// own wrapper object alongside each hrx_buffer_t.

#ifndef HRX_BUFFER_TABLE_H_
#define HRX_BUFFER_TABLE_H_

#include "hrx_runtime.h"
#include "iree/base/api.h"
#include "iree/base/threading/mutex.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct hrx_buffer_table_entry_t {
  // Base device address of the allocation.
  uint64_t device_ptr;
  // Base host address of the allocation, or NULL when not host-addressable.
  void* host_ptr;
  // Allocation length in bytes.
  size_t size;
  // Borrowed allocation handle owned by the inserting caller.
  hrx_buffer_t buffer;
  // Opaque payload owned by the inserting caller.
  void* user_data;
  // Generation of the last bulk lookup that matched this entry.
  uint64_t bulk_lookup_generation;
  // Retained-reference index assigned by |bulk_lookup_generation|.
  size_t bulk_lookup_ref_index;
} hrx_buffer_table_entry_t;

typedef struct hrx_buffer_table_range_index_t {
  // Base address of one device or host allocation alias.
  uint64_t base;
  // Index of the owning entry in |hrx_buffer_table_t.entries|.
  size_t entry_index;
} hrx_buffer_table_range_index_t;

typedef struct hrx_buffer_table_t {
  // Guards entries, the range index, and insertion reservations.
  iree_slim_mutex_t mutex;
  // Dense allocation entries in unspecified order.
  hrx_buffer_table_entry_t* entries;
  // Address-sorted device and distinct host aliases for |entries|.
  hrx_buffer_table_range_index_t* range_index;
  // Number of initialized entries in |entries|.
  size_t count;
  // Number of initialized aliases in |range_index|.
  size_t range_count;
  // Entry capacity; |range_index| has twice this capacity.
  size_t capacity;
  // Slots promised to callers that require allocation-free rollback.
  size_t reserved_insert_count;
  // Monotonic generation used to deduplicate bulk lookup results.
  uint64_t bulk_lookup_generation;
} hrx_buffer_table_t;

// Stable allocation metadata copied while the table lock protects the entry.
// |buffer| is retained and must be released by the caller.
typedef struct hrx_buffer_table_retained_ref_t {
  // Retained allocation wrapper.
  hrx_buffer_t buffer;
  // Base device address recorded by the table entry.
  uint64_t device_ptr;
  // Base host address recorded by the table entry, or NULL.
  void* host_ptr;
  // Allocation length in bytes.
  size_t size;
  // Byte offset of the requested pointer from its matching device or host base.
  size_t offset;
  // Opaque entry payload captured while the table lock was held.
  void* user_data;
} hrx_buffer_table_retained_ref_t;

typedef struct hrx_buffer_table_range_request_t {
  // First address in the requested range.
  uint64_t address;
  // Requested range length in bytes.
  size_t length;
} hrx_buffer_table_range_request_t;

typedef struct hrx_buffer_table_range_match_t {
  // Index into the returned retained references, or SIZE_MAX when unmatched.
  size_t ref_index;
  // Byte offset of the request from the matching allocation alias.
  size_t offset;
} hrx_buffer_table_range_match_t;

// Validates or acquires entry-local state while the table lock is held.
// Implementations must not wait for work that can require another table
// operation to complete. A callback returning an error must release any state
// it acquired before returning.
typedef hrx_status_t (*hrx_buffer_table_entry_callback_t)(
    const hrx_buffer_table_entry_t* entry, size_t offset, void* user_data);

void hrx_buffer_table_initialize(hrx_buffer_table_t* table);
void hrx_buffer_table_deinitialize(hrx_buffer_table_t* table);

hrx_status_t hrx_buffer_table_insert(hrx_buffer_table_t* table,
                                     uint64_t device_ptr, void* host_ptr,
                                     size_t size, hrx_buffer_t buffer,
                                     void* user_data);

// Reserves capacity for one future insertion. Regular insertions cannot consume
// the reserved slot. A successful call must be paired with either
// hrx_buffer_table_insert_reserved or hrx_buffer_table_cancel_reserved_insert.
hrx_status_t hrx_buffer_table_reserve_insert(hrx_buffer_table_t* table);

// Inserts an entry using capacity reserved by hrx_buffer_table_reserve_insert.
// This consumes one reservation even if pointer validation rejects the entry.
hrx_status_t hrx_buffer_table_insert_reserved(hrx_buffer_table_t* table,
                                              uint64_t device_ptr,
                                              void* host_ptr, size_t size,
                                              hrx_buffer_t buffer,
                                              void* user_data);

// Cancels one insertion reservation without inserting an entry.
void hrx_buffer_table_cancel_reserved_insert(hrx_buffer_table_t* table);

hrx_status_t hrx_buffer_table_remove(hrx_buffer_table_t* table,
                                     uint64_t any_ptr);

// Looks up a buffer containing |any_ptr| (device or host) without creating a
// status for an ordinary miss. Outputs borrow the inserting caller's handles;
// a miss clears them. Any out-parameter may be NULL if not needed.
bool hrx_buffer_table_try_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                               hrx_buffer_t* out_buffer, size_t* out_offset,
                               void** out_user_data);

// Looks up a required buffer containing |any_ptr| (device or host).
// Returns the buffer, byte offset within it, and optional user_data.
// Any out-parameter may be NULL if not needed.
hrx_status_t hrx_buffer_table_find(hrx_buffer_table_t* table, uint64_t any_ptr,
                                   hrx_buffer_t* out_buffer, size_t* out_offset,
                                   void** out_user_data);

// Looks up a buffer containing the entire range [any_ptr, any_ptr + size).
// Returns NOT_FOUND if no single buffer covers the full range.
hrx_status_t hrx_buffer_table_find_range(hrx_buffer_table_t* table,
                                         uint64_t any_ptr, size_t size,
                                         hrx_buffer_t* out_buffer,
                                         size_t* out_offset,
                                         void** out_user_data);

// Looks up a buffer containing the entire range [any_ptr, any_ptr + size),
// retains it, and snapshots immutable entry metadata before releasing the
// table lock. On success, |out_ref->buffer| must be released by the caller.
hrx_status_t hrx_buffer_table_find_range_retain(
    hrx_buffer_table_t* table, uint64_t any_ptr, size_t size,
    hrx_buffer_table_retained_ref_t* out_ref);

// Looks up and retains a range after |callback| accepts the matching entry.
// |callback| runs while the table lock protects both the entry and its opaque
// payload. On success, |out_ref->buffer| and any ownership acquired by the
// callback belong to the caller.
hrx_status_t hrx_buffer_table_find_range_retain_if(
    hrx_buffer_table_t* table, uint64_t any_ptr, size_t size,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    hrx_buffer_table_retained_ref_t* out_ref);

// Looks up arbitrary address ranges as one table transaction. Each unique
// allocation is accepted by |callback| and retained at most once. Every match
// names its retained reference; unmatched requests use SIZE_MAX. On failure,
// |out_ref_count| reports the successfully retained prefix for caller cleanup.
hrx_status_t hrx_buffer_table_find_ranges_retain_if(
    hrx_buffer_table_t* table, size_t request_count,
    const hrx_buffer_table_range_request_t* requests,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    size_t ref_capacity, hrx_buffer_table_retained_ref_t* out_refs,
    size_t* out_ref_count, hrx_buffer_table_range_match_t* out_matches);

// Validates and removes the entry containing |any_ptr| as one table
// transaction. A successful removal reserves its vacated slot so the caller
// can restore the exact entry without allocation. The caller must pair success
// with either hrx_buffer_table_insert_reserved or
// hrx_buffer_table_cancel_reserved_insert. |out_entry| contains borrowed
// handles whose ownership remains with the caller that originally inserted
// them.
hrx_status_t hrx_buffer_table_remove_reserved_if(
    hrx_buffer_table_t* table, uint64_t any_ptr,
    hrx_buffer_table_entry_callback_t callback, void* callback_user_data,
    hrx_buffer_table_entry_t* out_entry, size_t* out_offset);

#ifdef __cplusplus
}
#endif

#endif  // HRX_BUFFER_TABLE_H_
