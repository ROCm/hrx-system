// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_STATE_H_
#define LOOM_SERVE_SERVICE_TEXT_STATE_H_

#include "iree/base/internal/arena.h"
#include "loom/serve/runtime/execution/retirement.h"
#include "loom/serve/runtime/storage/block_pool.h"
#include "loom/serve/service/text/model.h"
#include "loom/serve/service/text/storage.h"

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
  TEXT_BINDING_COUNT = 6,
};

// Resource arities of text stages, excluding the reflected workspace. Ordinary
// runner stages use at most TEXT_STAGE_ARITY resources; resource ABI 2 prefill
// group leaves bind exactly TEXT_GROUP_DRAFT_ARITY packed and draft roots.
enum {
  TEXT_STAGE_ARITY = 7,
  TEXT_GROUP_DRAFT_ARITY = 12,
};

enum loom_serve_text_state_flag_bits_e {
  LOOM_SERVE_TEXT_STATE_FLAG_PACKED = 1u << 0,
  LOOM_SERVE_TEXT_STATE_FLAG_DRAFT = 1u << 1,
};
typedef uint32_t loom_serve_text_state_flags_t;

// Next owner action after the current maintenance frontier succeeds.
typedef enum loom_serve_text_trim_phase_e {
  LOOM_SERVICE_TEXT_TRIM_COMPLETE,
  LOOM_SERVICE_TEXT_TRIM_RELOCATE,
  LOOM_SERVICE_TEXT_TRIM_RELEASE,
} loom_serve_text_trim_phase_t;

typedef struct loom_serve_text_state_t loom_serve_text_state_t;

// Retained device state at a consumed-token frontier. Host predictions and
// metrics belong to the model row and do not own device storage.
typedef struct loom_serve_text_state_row_t {
  // Borrowed single-owner state arena, outliving this row.
  loom_serve_text_state_t* owner;
  // Stable source-visible identity, independent of this record's address.
  iree_host_size_t index;
  // Private views in slots 1, 3-5; recurrence borrows its owned pool slot.
  // Residual borrows model-wide storage.
  iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  // Current destination and the immutable source of an in-flight fork.
  struct {
    // Owned slot, or UINT32_MAX while empty or suspended.
    uint32_t slot;
    // Owned reader until the cohort retires, or UINT32_MAX when in-place.
    uint32_t anchor;
    // Changed origins not yet captured by the source state encoder.
    bool dirty;
  } recurrent;
  // Old partial KV tail owned through COW retirement, or UINT32_MAX.
  uint32_t retiring_tail;
  // Source-owned logical position at the consumed frontier.
  uint32_t source_position;
  // Number of inputs actually consumed into KV/recurrent state.
  uint32_t position;
  // Owned physical pages, including any in-flight speculative suffix.
  uint32_t block_count;
  // Request high-water in token positions; zero outside a reservation.
  uint32_t reserved_extent;
  // Owned capture payload, or NULL. Successful suspension releases physical
  // IDs; terminal transfer failure preserves both until teardown.
  loom_serve_snapshot_t* snapshot;
} loom_serve_text_state_row_t;

// Explicit endpoint holding immutable KV and recurrence by reference. Draft
// carry lives in the source-declared carry slab after the ordinary row entries.
typedef struct loom_serve_text_state_checkpoint_t {
  // Borrowed storage owner, outliving this endpoint.
  loom_serve_text_state_t* owner;
  // Exact consumed input frontier.
  uint32_t position;
  // Exact source-owned logical position at that frontier.
  uint32_t source_position;
  // Retained recurrent slot, or UINT32_MAX while suspended.
  uint32_t recurrent_slot;
  // Number of retained physical pages; zero while suspended.
  uint32_t block_count;
  // Logical-order IDs borrowing the owner's fixed checkpoint map bank.
  uint32_t* blocks;
  // Owned capture payload independent of active rows. Successful suspension
  // releases physical IDs; terminal failure retains them until teardown.
  loom_serve_snapshot_t* snapshot;
} loom_serve_text_state_checkpoint_t;

// Retained text storage, independent of source compilation and VM invocation.
// All mutations use the model's serialized execution domain. External
// model input is validated before growth; source geometry is validated once by
// storage_initialize. Borrowed views remain valid until deinitialize retires
// every exported root, including failed queued uses.
struct loom_serve_text_state_t {
  // Allocation policy for host metadata and tracked views.
  iree_allocator_t host_allocator;
  // Borrowed device, outliving this state and all virtual reservations.
  iree_hal_device_t* device;
  // Borrowed storage owner, preserved across state growth and compaction.
  loom_serve_pool_t* pool;
  // Borrowed model timelines shared by its state maintenance and model work.
  loom_serve_execution_t* execution;
  // Final queue ownership enclosing host payloads and virtual backing.
  loom_serve_retirement_t retirement;
  // Enabled packed and speculative storage roles.
  loom_serve_text_state_flags_t flags;
  // Number of published resident records and source-defined private views.
  iree_host_size_t row_count;
  // Stable records allocated from the shared application's host block pool.
  iree_arena_allocator_t row_records;
  // Owned directory; records outlive every asynchronous borrower.
  loom_serve_text_state_row_t** rows;
  // Prepared logical plans whose selected IDs exclude compaction until freed.
  struct loom_serve_text_state_reservation_t* prepared_reservations;
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
    // Number of rows with unpublished origins; zero skips the row scan.
    iree_host_size_t dirty_count;
  } recurrent;
  // Referenced page ownership shared by target and draft cache planes.
  struct {
    // Addressable pooled token capacity; zero selects dense comparison.
    iree_host_size_t capacity;
    // Demand-grown free-ID metadata; ownership returns only after model work
    // retires. Capacity is a live high-water mark, not addressability.
    loom_serve_block_pool_t pool;
    // Row-major host maps retained through queued uploads.
    uint32_t* maps;
    // Cold compaction plan indexed by original physical ID.
    uint32_t* destinations;
    // Allocated entries in destinations; grown only by cold maintenance.
    uint32_t destination_capacity;
    // Reusable target/draft map uploads, at most two per participating row.
    iree_hal_transfer_operation_t* uploads;
  } cache;
  // Physical commitment, separate from logical row/block ownership.
  struct {
    // Borrowed shared allocation domain for state and model parameters.
    loom_serve_memory_pool_t* pool;
    // Mutable virtual state only, excluding weights and execution workspace.
    loom_runtime_storage_statistics_t statistics;
    // Native backing changes exclude state mutation through their readiness.
    struct {
      // Owned native borrower, joined separately from failed queue readiness.
      loom_serve_memory_operation_t* operation;
      // Owned timeline imported before dependent transfers or maintenance.
      iree_hal_semaphore_t* semaphore;
      // Last admitted native readiness value.
      uint64_t value;
    } backing;
    // Independent native mapping borrowers during cold reservation growth.
    // Each owns its own readiness value; completion of one cannot publish
    // another reservation's translations.
    struct {
      // Owned native operations, joined before replacing their directories.
      loom_serve_memory_operation_t*
          operations[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
      // Owned readiness semaphores imported into the model timeline.
      iree_hal_semaphore_t*
          semaphores[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    } growth;
    // Compaction spans native releases and device copies on one model owner.
    struct {
      // Action admitted after successful readiness of the preceding phase.
      loom_serve_text_trim_phase_t phase;
      // State-local release counter before this maintenance cut.
      uint64_t released_before;
    } trim;
    // Optional reservations in source allocation order.
    loom_serve_virtual_buffer_t*
        buffers[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
    // Always-live initialized prefixes outside row-private storage.
    iree_device_size_t
        private_lengths[LOOM_SERVE_TEXT_STORAGE_ALLOCATION_COUNT];
  } memory;
  // Source-positioned private views, recurrent slots, and target KV planes.
  iree_hal_buffer_t* row_arena;
  // Model-wide residual storage serialized by the execution timeline.
  iree_hal_buffer_t* residual;
  // Optional source-defined committed state indexed by retained row or
  // endpoint. Presence follows the source carry stride, independently of draft
  // scheduling.
  iree_hal_buffer_t* carry;
  // Packed bindings; slots 1, 2, 4 and 5 are owned, others borrow roots.
  struct {
    // Residual, metadata, origins, arena, inputs and outputs.
    iree_hal_buffer_t* buffers[TEXT_BINDING_COUNT];
  } epoch;
  // Optional speculative proposal, cache, and feedback storage.
  struct {
    // Accepted span metadata consumed by draft catch-up.
    iree_hal_buffer_t* committed;
    // Source-defined feedback records.
    iree_hal_buffer_t* results;
    // Owned view of the second result bank and original-span tags.
    iree_hal_buffer_t* next_results;
    // Optional draft attention cache with source-defined pooled planes.
    iree_hal_buffer_t* cache;
    // Optional draft row origins and logical-to-physical block maps. Presence
    // is independent of a separate cache: source may reuse the target arena.
    iree_hal_buffer_t* row_table;
  } draft;
};

// Initializes ownership before bootstrap can fail. Options have passed the
// model boundary's capacity checks. Source storage is initialized separately
// before allocate consumes its declared roots and private views.
void loom_serve_text_state_initialize(const loom_serve_text_options_t* options,
                                      iree_allocator_t host_allocator,
                                      loom_serve_text_state_t* out_state);

// Allocates source-described storage and submits initialization on execution's
// work timeline without waiting. The owner retains source payloads through
// successful readiness of its captured execution frontier, or actual consumer
// retirement on failure. Later model work depends on this initialization.
iree_status_t loom_serve_text_state_allocate(loom_serve_text_state_t* state,
                                             loom_serve_device_t* device,
                                             loom_serve_pool_t* pool,
                                             loom_serve_execution_t* execution);

// Source slot to its owned root, shared by allocation and cold maintenance.
iree_hal_buffer_t** loom_serve_text_state_allocation(
    loom_serve_text_state_t* state, iree_host_size_t index);

// Moves a validated enlarged source layout into retired state. Existing
// private views, recurrent slots, and cache planes retain their byte offsets.
// The caller has joined CPU/device consumers and prepared reservations.
// Accepted copies and mappings remain owned through normal terminal cleanup.
iree_status_t loom_serve_text_state_resize(loom_serve_text_state_t* state,
                                           iree_host_size_t row_count,
                                           loom_serve_text_storage_t* storage);

// Releases cold layout payloads and native mapping borrowers after successful
// completion of the captured resize frontier.
void loom_serve_text_state_complete_resize(loom_serve_text_state_t* state);

// Begins irreversible retirement after external row/checkpoint users and
// cached commands relinquish their references. Releases owned device views
// without waiting. Host packets and virtual backing remain owned by state.
// The wake only schedules the serialized owner; it must not reenter state.
void loom_serve_text_state_retire(loom_serve_text_state_t* state,
                                  void (*wake)(void* user_data),
                                  void* user_data);

// Joins native backing borrowers as well as all exported device views.
// Failed queue readiness alone does not make state safe to destroy.
bool loom_serve_text_state_is_retired(loom_serve_text_state_t* state);

// Outer-thread ownership join after retire; never used by scheduler tasks.
void loom_serve_text_state_await_retirement(loom_serve_text_state_t* state);

// Final destruction after state_is_retired succeeds. Never waits
// for device work. On platform release failure, remaining handles and host
// metadata stay intact for owner-side retry; otherwise state is consumed.
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
// Releases the row's logical ownership at its successfully retired boundary.
void loom_serve_text_state_row_reset(loom_serve_text_state_row_t* row);
iree_status_t loom_serve_text_state_row_suspend(
    loom_serve_text_state_row_t* row);
iree_status_t loom_serve_text_state_row_try_resume(
    loom_serve_text_state_row_t* row, bool* out_resumed);
// Retains the retired row's exact endpoint and queues its optional carry copy.
// An active reservation admits its private continuation before sharing state;
// unused completion pages remain private to the row. The owner excludes
// mutation through successful execution readiness. Failure retains acquired IDs
// for state teardown; capacity denial returns NULL without changing the row.
iree_status_t loom_serve_text_state_row_submit_pin(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t** out_checkpoint);
iree_status_t loom_serve_text_state_row_try_restore(
    loom_serve_text_state_row_t* row,
    loom_serve_text_state_checkpoint_t* checkpoint, bool* out_restored);
// Joins a caller's complete working set to the state planner's exact range
// union. Called synchronously on the serialized owner after logical IDs are
// tentatively acquired, before publishing any row changes. Denial submits no
// work and leaves no pin; success returns accepted native backing ownership.
// Fixed backing supplies zero ranges and no readiness. The callback must not
// yield or change the selected row/checkpoint. Denial leaves the prepared plan
// intact; the immediate wrapper returns its tentative IDs in this owner turn.
typedef iree_status_t (*loom_serve_text_state_admit_fn_t)(
    void* user_data, iree_host_size_t range_count,
    const loom_serve_memory_range_t* ranges, iree_hal_semaphore_t* readiness,
    uint64_t value, loom_serve_memory_admission_t* out_admission,
    loom_serve_memory_operation_t** out_operation);

// A prepared continuation owns its selected logical IDs without changing the
// row, mapping backing, or submitting work. The row and optional checkpoint
// remain borrowed and immutable until submission or cancellation; other rows
// may execute, but state compaction waits for prepared continuations to retire.
typedef struct loom_serve_text_state_reservation_t
    loom_serve_text_state_reservation_t;

// Uses the same KV/COW planner as immediate reservation. NULL means logical-ID
// capacity is unavailable; no row/checkpoint changes or device work occur.
// The caller supplies the validated extent and a successfully retired row.
iree_status_t loom_serve_text_state_row_prepare_reserve(
    loom_serve_text_state_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_state_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_state_reservation_t** out_reservation);

// Whole continuation's unique physical slab footprint, including shared
// epoch/proposal storage and COW sources through retirement. Captured with the
// plan, independent of current backing. Fixed backing has no finite budget and
// returns zero rather than claiming a physical estimate.
uint64_t loom_serve_text_state_reservation_footprint(
    const loom_serve_text_state_reservation_t* reservation);

// Commits the prepared union and installs its selected IDs on success. Denial
// keeps the plan intact for retry. The owner excludes concurrent state edits
// and joins prior backing work before submission. Native failure is terminal;
// state keeps any installed ownership through ordinary retirement.
iree_status_t loom_serve_text_state_reservation_submit(
    loom_serve_text_state_reservation_t* reservation,
    loom_serve_text_state_admit_fn_t admit, void* user_data,
    loom_serve_memory_admission_t* out_admission);

// Before submission, returns the prepared logical ownership without changing
// the row/checkpoint. After successful submission, only host planning storage
// remains here; the row/checkpoint retain accepted device work. NULL is valid.
void loom_serve_text_state_reservation_free(
    loom_serve_text_state_reservation_t* reservation);

// Starts at a successfully retired row boundary.
// Extent and starting frontier have passed model validation. Denial changes no
// selected row or checkpoint contents/ownership. Success installs ownership
// before submitting transfers; new backing becomes ready on the model timeline
// before any initialization or restore. All maps and backing survive until
// completion, including the native borrower after failed queue readiness.
// CONTINUE may replace an active reservation while retaining its exact
// frontier. AVAILABLE distinguishes admission from an unchanged earlier
// reservation. DEFERRED preserves observed physical progress independently of
// later collection; logical page exhaustion reports CAPACITY.
// admit joins new-request parameter admission with state backing; NULL uses
// ordinary state admission for an already-admitted continuation/COW operation.
iree_status_t loom_serve_text_state_row_submit_reserve(
    loom_serve_text_state_row_t* row, loom_serve_text_reserve_mode_t mode,
    loom_serve_text_state_checkpoint_t* checkpoint, iree_host_size_t extent,
    loom_serve_text_state_admit_fn_t admit, void* user_data,
    loom_serve_memory_admission_t* out_admission);
// After successful readiness, releases the copied tail source. The recurrent
// reader survives until the first epoch or cancellation.
void loom_serve_text_state_row_complete_reserve(
    loom_serve_text_state_row_t* row);
// Returns unused page credit at a retired boundary, preserving the consumed
// frontier even when a reserved fork was never advanced.
void loom_serve_text_state_row_release_reservation(
    loom_serve_text_state_row_t* row);
// Captures a retired endpoint without releasing its source IDs. The caller
// excludes state mutation until successful readiness and completion
// consumption.
iree_status_t loom_serve_text_state_checkpoint_submit_suspend(
    loom_serve_text_state_checkpoint_t* checkpoint);
// Releases captured source IDs after successful readiness. Already-cold
// endpoints retain their image without releasing any IDs twice.
void loom_serve_text_state_checkpoint_complete_suspend(
    loom_serve_text_state_checkpoint_t* checkpoint);
void loom_serve_text_state_checkpoint_release(
    loom_serve_text_state_checkpoint_t* checkpoint);
// Starts at a successfully completed model frontier and submits dead-backing
// release. The owner excludes mutation until complete_trim finishes all phases;
// old block ownership and backing remain live if accepted work fails.
iree_status_t loom_serve_text_state_submit_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* out_result);
// After successful ALL readiness, advances relocation or old-backing release.
// False requires another execution frontier wait before advancing again.
// result is the unchanged submission record, finalized when complete is true.
iree_status_t loom_serve_text_state_complete_trim(
    loom_serve_text_state_t* state, loom_serve_text_trim_result_t* result,
    bool* out_complete);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_SERVICE_TEXT_STATE_H_
