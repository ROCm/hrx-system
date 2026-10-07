// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_TEXT_STATE_H_
#define EXPERIMENTAL_LOOM_SERVE_TEXT_STATE_H_

#include "experimental/loom_serve/runtime/retirement.h"
#include "experimental/loom_serve/storage/block_pool.h"
#include "experimental/loom_serve/text/model.h"
#include "experimental/loom_serve/text/storage.h"

#ifdef __cplusplus
extern "C" {
#endif

// Binding roles of the private text adapter. Source defines their extents and
// placement; retained state and command workspace have distinct lifetimes.
enum {
  TEXT_RESIDUAL = 0,
  TEXT_CONTROL = 1,
  TEXT_RECURRENT = 2,
  TEXT_ATTENTION = 3,
  TEXT_TOKENS = 4,
  TEXT_PROGRESS = 5,
  TEXT_WORKSPACE = 6,
  TEXT_BINDING_COUNT = 7,
};

enum loom_serve_text_state_flag_bits_e {
  LOOM_SERVE_TEXT_STATE_FLAG_PACKED = 1u << 0,
  LOOM_SERVE_TEXT_STATE_FLAG_MTP = 1u << 1,
};
typedef uint32_t loom_serve_text_state_flags_t;

typedef struct loom_serve_text_state_t loom_serve_text_state_t;

// Retained device state at a consumed-token frontier. Host predictions and
// metrics belong to the model row and do not own device storage.
typedef struct loom_serve_text_state_row_t {
  // Borrowed single-owner state arena, outliving this row.
  loom_serve_text_state_t* owner;
  // Private views in slots 1, 3-5; recurrence borrows its owned pool slot.
  // Residual/workspace borrow model-wide storage.
  iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  // Current destination and the immutable source of an in-flight fork.
  struct {
    // Owned slot, or UINT32_MAX while empty or suspended.
    uint32_t slot;
    // Owned reader until the cohort retires, or UINT32_MAX when in-place.
    uint32_t anchor;
  } recurrent;
  // Old partial KV tail owned through COW retirement, or UINT32_MAX.
  uint32_t retiring_tail;
  // Number of inputs actually consumed into KV/recurrent state.
  iree_host_size_t position;
  // Owned physical pages, including any in-flight speculative suffix.
  uint32_t block_count;
  // Owned DRAM image while suspended; no physical KV IDs remain owned.
  loom_serve_snapshot_t* snapshot;
} loom_serve_text_state_row_t;

// Explicit endpoint holding immutable KV and recurrence by reference. MTP carry
// lives in the source-declared carry slab after the ordinary row entries.
typedef struct loom_serve_text_state_checkpoint_t {
  // Borrowed storage owner, outliving this endpoint.
  loom_serve_text_state_t* owner;
  // Exact consumed input frontier.
  iree_host_size_t position;
  // Retained recurrent slot.
  uint32_t recurrent_slot;
  // Number of retained logical pages.
  uint32_t block_count;
  // Logical-order IDs borrowing the owner's fixed checkpoint map bank.
  uint32_t* blocks;
} loom_serve_text_state_checkpoint_t;

// Retained text storage, independent of source compilation and VM invocation.
// All mutations use the device owner's serialized execution domain. External
// model input is validated before growth; source geometry is validated once by
// storage_initialize. Borrowed views remain valid until deinitialize retires
// every exported root, including failed queued uses.
struct loom_serve_text_state_t {
  // Allocation policy for host metadata and tracked views.
  iree_allocator_t allocator;
  // Borrowed device, outliving this state and all virtual reservations.
  iree_hal_device_t* device;
  // Borrowed shared execution timelines for state and model work.
  loom_serve_execution_t* execution;
  // Final queue ownership enclosing host payloads and virtual backing.
  loom_serve_retirement_t retirement;
  // Enabled packed and speculative storage roles.
  loom_serve_text_state_flags_t flags;
  // Number of fixed row records and source-defined private views.
  iree_host_size_t row_count;
  // Owned rows, stable through final queue retirement.
  loom_serve_text_state_row_t* rows;
  // Validated source geometry and retained initialization payloads.
  loom_serve_text_storage_t storage;
  // Explicit endpoint records, separate from active row capacity.
  struct {
    // Configured cold record count, including before allocation.
    uint32_t capacity;
    // Available checkpoint IDs; each live endpoint owns one.
    loom_serve_block_pool_t pool;
    // Owned records indexed by checkpoint ID.
    loom_serve_text_state_checkpoint_t* values;
    // Owned logical page maps for all checkpoint records.
    uint32_t* maps;
  } checkpoints;
  // Recurrent ownership is independent of row-private control and KV maps.
  struct {
    // Free slot IDs; release follows retirement or completed capture.
    loom_serve_block_pool_t pool;
    // Owned arena subviews, stable through final queue retirement.
    iree_hal_buffer_t** buffers;
    // Rows whose new byte origins need source encoding before packed work.
    uint32_t dirty_rows;
  } recurrent;
  // Referenced page ownership shared by target and draft cache planes.
  struct {
    // Addressable pooled token capacity; zero selects dense comparison.
    iree_host_size_t capacity;
    // Free-ID metadata; ownership returns only after model work retires.
    loom_serve_block_pool_t pool;
    // Row-major host maps retained through queued uploads.
    uint32_t* maps;
    // Cold compaction plan indexed by original physical ID.
    uint32_t* destinations;
  } cache;
  // Physical commitment, separate from logical row/block ownership.
  struct {
    // Borrowed shared allocation domain for state and model parameters.
    loom_serve_memory_pool_t* pool;
    // Mutable virtual state only, excluding weights and fixed workspace.
    loom_serve_memory_statistics_t statistics;
    // Optional reservations in source allocation order.
    loom_serve_virtual_buffer_t*
        buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    // Always-live initialized prefixes outside row-private storage.
    iree_device_size_t
        private_lengths[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  } memory;
  // Private row views followed by pooled KV, or dense row KV.
  iree_hal_buffer_t* row_arena;
  // Model-wide residual storage serialized by the execution timeline.
  iree_hal_buffer_t* residual;
  // Model-wide transient storage with undefined contents on stage entry.
  iree_hal_buffer_t* workspace;
  // Packed bindings; slots 1, 2, 4 and 5 are owned, others borrow roots.
  struct {
    // Residual, metadata, origins, arena, inputs, outputs, workspace.
    iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  } epoch;
  // Optional retained draft state and speculative proposal/feedback storage.
  struct {
    // Committed target hidden state indexed by retained row.
    iree_hal_buffer_t* carry;
    // Accepted span metadata consumed by draft catch-up.
    iree_hal_buffer_t* committed;
    // Source-defined feedback records.
    iree_hal_buffer_t* results;
    // Owned view of the second result bank and original-span tags.
    iree_hal_buffer_t* next_results;
    // Draft attention cache with source-defined pooled planes.
    iree_hal_buffer_t* cache;
    // Draft row origins and logical-to-physical block maps.
    iree_hal_buffer_t* row_table;
  } mtp;
};

// Initializes ownership before bootstrap can fail. Options have passed the
// model boundary's capacity checks. Source storage is initialized separately
// before allocate consumes its declared roots and private views.
void loom_serve_text_state_initialize(const loom_serve_text_options_t* options,
                                      loom_serve_text_state_t* out_state,
                                      iree_allocator_t allocator);
iree_status_t loom_serve_text_state_allocate(
    loom_serve_text_state_t* state, loom_serve_device_t* device,
    iree_device_size_t workspace_length,
    iree_device_size_t workspace_alignment);

// The model first drains execution and releases its VM and command references.
// This releases views, joins actual queue ownership, then frees maps, rows and
// physical backing. Borrowed model host payloads must outlive this call even
// when the initial drain failed. Platform unmap/free failure is terminal.
iree_status_t loom_serve_text_state_deinitialize(
    loom_serve_text_state_t* state);

// A validated cohort already fits the logical free-ID capacity. Growth
// publishes new map entries and orders them before subsequent model work.
// Physical/platform failure is terminal; accepted work still owns its views.
iree_status_t loom_serve_text_state_grow(loom_serve_text_state_t* state,
                                         iree_host_size_t span_count,
                                         const loom_serve_text_span_t* spans,
                                         const uint32_t* extents);

// Releases fork readers, COW sources and the rejected speculative suffix only
// after the cohort has retired and the consumed frontier is published.
void loom_serve_text_state_row_trim(loom_serve_text_state_row_t* row);
// Unique pages required for append, including detaching a shared partial tail.
uint32_t loom_serve_text_state_row_growth(
    const loom_serve_text_state_row_t* row, uint32_t extent);
// Activates an empty row's recurrent slot and initializes private state on the
// existing work timeline. An already resident row performs no work.
iree_status_t loom_serve_text_state_row_activate(
    loom_serve_text_state_row_t* row);
iree_status_t loom_serve_text_state_row_reset(loom_serve_text_state_row_t* row);
iree_status_t loom_serve_text_state_row_suspend(
    loom_serve_text_state_row_t* row);
iree_status_t loom_serve_text_state_row_try_resume(
    loom_serve_text_state_row_t* row, bool* out_resumed);
iree_status_t loom_serve_text_state_row_try_pin(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t** out_checkpoint);
iree_status_t loom_serve_text_state_row_try_restore(
    loom_serve_text_state_row_t* row,
    const loom_serve_text_state_checkpoint_t* checkpoint, bool* out_restored);
void loom_serve_text_state_checkpoint_release(
    loom_serve_text_state_checkpoint_t* checkpoint);
iree_status_t loom_serve_text_state_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_TEXT_STATE_H_
