// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_H_
#define IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_H_

#include "iree/async/frontier.h"
#include "iree/base/api.h"
#include "iree/base/internal/atomics.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Fixed-Block Offset Allocator
//===----------------------------------------------------------------------===//
//
// A lock-free fixed-size allocator that manages a contiguous range of offsets
// [0, block_count * block_size). All blocks are the same size. The allocator
// uses an atomic bitmap for O(1) allocation and deallocation, with no mutexes
// and full concurrency support for multiple threads.
//
// Like the TLSF allocator, this is pure bookkeeping: it tracks which blocks
// are allocated and which are free, but has no knowledge of physical memory.
// All metadata lives in host memory.
//
// ## Why a fixed-block allocator
//
// For allocation patterns where all items are the same size, a fixed-block
// allocator is simpler and faster than a general-purpose allocator like TLSF:
//
//   - O(1) allocation via bitmap scan (count-trailing-zeros on inverted word)
//   - O(1) deallocation via bit clear (atomic AND)
//   - Zero external fragmentation (all blocks are uniform)
//   - Lock-free: allocation uses atomic_fetch_or, deallocation uses
//     atomic_fetch_and. No mutexes, no CAS retry loops on the common path.
//   - Device-side compatible: the bitmap could be placed in device-visible
//     shared memory for future GPU-side allocation via atomicOR/AND.
//
// ## How it works
//
// The allocator maintains a flat array of 64-bit bitmap words. Bit i in word j
// represents block (j * 64 + i). A set bit means the block is allocated; a
// clear bit means it is free.
//
//   ┌─────────────────────────────────────────────────────────────────────┐
//   │  bitmap[0]: bits 0-63     │  bitmap[1]: bits 64-127    │  ...       │
//   │  1=allocated, 0=free      │  1=allocated, 0=free       │            │
//   └─────────────────────────────────────────────────────────────────────┘
//
// ### Allocation
//
// 1. Starting from a roving hint (the last word that had free blocks), scan
//    bitmap words for one with a clear bit.
// 2. Use ctz(~word) to find the first free bit in that word.
// 3. Claim the bit with atomic_fetch_or. If the bit was already set (another
//    thread raced us), retry within the same word or move to the next.
// 4. Return the block index = word_index * 64 + bit.
//
// ### Deallocation
//
// 1. Clear the block's bit with atomic_fetch_and. Ownership transfers
//    immediately; the block is available for reallocation.
//
// ### Death frontiers
//
// Each block carries an inline death frontier (`iree_async_frontier_t`) with
// a configurable maximum entry capacity. Unlike TLSF, blocks never coalesce,
// so frontiers are never merged. The frontier records the causal position at
// which the block was freed; when the block is reallocated, the caller can
// check whether its own frontier dominates the death frontier for zero-sync
// reuse.
//
// Memory ordering ensures frontier visibility: the freeing thread writes the
// frontier before the atomic bitmap clear (release semantics), and the
// allocating thread reads the frontier after the atomic bitmap set (acquire
// semantics).
//
// If a death frontier exceeds the inline capacity, the block is marked
// tainted. Taint is self-healing: when a tainted block is freed with a
// frontier that fits, the taint is cleared.
//
// ### Capacity
//
// The bitmap is stored inline in the allocator structure: up to
// IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BITMAP_WORDS words * 64 bits =
// IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS blocks. This is sufficient
// for most fixed-block allocation use cases (KV-cache slabs, MoE expert slots,
// signal ring entries). If larger allocators are needed, shard across multiple
// allocator instances or switch to a dynamic bitmap plus a summary layer.
//
// ### Thread safety
//
// Acquire and release are lock-free and safe to call from any thread.
// Allocator creation and destruction are NOT thread-safe and must be called
// from a single thread with no concurrent acquire/release operations.

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// Maximum number of blocks in a single allocator instance. Limited by the
// inline bitmap array size (IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BITMAP_WORDS
// words * 64 bits).
#define IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS 4096

// Default death frontier capacity per block. Covers single-queue and moderate
// multi-queue workloads. Since blocks never coalesce, frontiers don't grow
// through merging; the capacity only needs to cover the number of distinct
// axes a single block's user touches.
#define IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_DEFAULT_FRONTIER_CAPACITY 4

// Number of 64-bit words in the inline bitmap array.
#define IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BITMAP_WORDS 64

// Bitfield flags for a block's current state.
typedef uint32_t iree_hal_memory_fixed_block_allocator_block_flags_t;
enum iree_hal_memory_fixed_block_allocator_block_flag_bits_e {
  IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_NONE = 0u,

  // Block's death frontier is tainted: the frontier passed to release()
  // exceeded the inline capacity. The frontier data is zeroed and cannot be
  // used for dominance checking. The async allocator must treat this block as
  // conservatively "not yet safe for zero-sync reuse."
  //
  // Taint is self-healing: when a tainted block is freed with a frontier that
  // fits within capacity, the taint is cleared.
  IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED = 1u << 0,
};

// Fixed-size metadata for a single block. Flags and backing generation are
// followed by inline frontier storage (accessed via stride arithmetic).
// This struct is exposed for sizeof() in IREE_STRUCT_LAYOUT; callers must
// not access block fields directly.
typedef struct iree_hal_memory_fixed_block_allocator_block_t {
  // Reuse validity of the inline frontier following this header.
  iree_hal_memory_fixed_block_allocator_block_flags_t flags;
  // Backing generation in which this block's own history was last written.
  uint32_t generation;
} iree_hal_memory_fixed_block_allocator_block_t;

// Options for creating a fixed-block allocator instance.
typedef struct iree_hal_memory_fixed_block_allocator_options_t {
  // Size of each block in bytes. Must be > 0. All allocations return offsets
  // aligned to this size (offset = block_index * block_size). The allocator
  // does not impose alignment constraints; the caller is responsible for
  // choosing a block_size that meets their alignment requirements.
  iree_device_size_t block_size;

  // Number of blocks in the allocator. Must be > 0 and
  // <= IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS. The managed offset
  // range is [0, block_count * block_size).
  uint32_t block_count;

  // Maximum number of frontier entries per block. Set to 0 to use
  // IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_DEFAULT_FRONTIER_CAPACITY.
  uint16_t frontier_capacity;
  // Exact prerequisite for the entire initial range, copied during creation.
  // NULL means no prerequisite. A frontier wider than capacity is rejected.
  const iree_async_frontier_t* initial_frontier;
} iree_hal_memory_fixed_block_allocator_options_t;

// Describes an acquired block or a read-only candidate for acquisition.
typedef struct iree_hal_memory_fixed_block_allocator_allocation_t {
  // Start offset of the allocated block within [0, block_count * block_size).
  iree_device_size_t offset;

  // Block index (0-based). Pass to
  // iree_hal_memory_fixed_block_allocator_release() to return the block to the
  // allocator.
  uint32_t block_index;

  // The death frontier from the block's previous deallocation or its backing's
  // initial history, valid and stable while allocated. Candidate queries borrow
  // it only while acquisitions remain externally serialized. NULL if empty.
  const iree_async_frontier_t* death_frontier;

  // Flags from the block at allocation time. Check
  // IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED to determine
  // whether the death frontier is trustworthy for dominance checking.
  iree_hal_memory_fixed_block_allocator_block_flags_t block_flags;
} iree_hal_memory_fixed_block_allocator_allocation_t;

// Result of a fixed-block try-acquire that can run out of blocks during normal
// allocator search.
typedef uint32_t iree_hal_memory_fixed_block_allocator_acquire_result_t;
enum iree_hal_memory_fixed_block_allocator_acquire_result_e {
  IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_OK = 0u,
  IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED = 1u,
};

// Running statistics for a fixed-block allocator. All values are atomic
// snapshots; they may be momentarily inconsistent under concurrent
// modifications.
typedef struct iree_hal_memory_fixed_block_allocator_stats_t {
  // Total blocks in the allocator (immutable after init).
  uint32_t block_count;

  // Number of currently allocated blocks.
  uint32_t allocation_count;
} iree_hal_memory_fixed_block_allocator_stats_t;

// The fixed-block allocator instance. Allocated with
// iree_hal_memory_fixed_block_allocator_allocate() and freed with
// iree_hal_memory_fixed_block_allocator_free(). All fields are implementation
// details and must not be accessed directly by callers.
//
// Layout is organized by access pattern for cache efficiency:
//   - Read-only config fields share one cache line (no coherence traffic).
//   - Each contended atomic gets its own cache line to avoid false sharing.
//   - The bitmap starts on a cache line boundary; adjacent words intentionally
//     share lines for scan locality.
//   - Per-block metadata is a trailing flexible array member (FAM). Untouched
//     blocks share one initial frontier instead of duplicating that history.
typedef struct iree_hal_memory_fixed_block_allocator_t {
  // --- Read-only while backing is attached (one cache line) --------------

  // Size of each block in bytes.
  iree_device_size_t block_size;

  // Total number of blocks.
  uint32_t block_count;

  // Maximum frontier entries per block.
  uint16_t frontier_capacity;

  // Number of 64-bit bitmap words in use: ceil(block_count / 64).
  uint16_t word_count;

  // Byte stride between adjacent blocks' metadata, computed during creation.
  iree_host_size_t block_stride;
  // Byte offset of the inline frontier within each block's metadata.
  iree_host_size_t frontier_offset;

  // Current backing generation; zero is reserved for untouched metadata.
  uint32_t generation;
  // Owned initial history for blocks not yet returned in this generation.
  iree_async_frontier_t* initial_frontier;
  // One bit per block whose history has been written in this generation.
  iree_atomic_uint64_t* returned_bitmap;

  // --- Contended atomics (each on its own cache line) ---------------------

  // Roving allocation hint: the word index where the last successful
  // allocation was found. Relaxed atomic; concurrent writes may produce
  // stale values, which is harmless (the allocator scans all words if the
  // hint misses). Reduces scan overhead under steady-state patterns.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_int32_t alloc_hint_word;

  // Number of currently allocated blocks. Updated on every alloc (add) and
  // free (sub) with relaxed ordering; pure bookkeeping for stats queries.
  iree_alignas(iree_hardware_destructive_interference_size)
      iree_atomic_uint32_t allocation_count;

  // Cold allocator ownership, occupying padding beside the count cache line.
  iree_allocator_t host_allocator;

  // --- Bitmap (cache-line-aligned, compact for scan locality) -------------

  // Atomic bitmap: bit set = allocated, bit clear = free.
  // Trailing invalid bits in the last word (when block_count is not a multiple
  // of 64) are set to 1 during initialization so they are never allocated.
  // Adjacent words share cache lines by design; the bitmap is meant to be
  // scanned sequentially.
  iree_alignas(iree_hardware_destructive_interference_size) iree_atomic_uint64_t
      bitmap[IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BITMAP_WORDS];

  // --- Per-block metadata (FAM, cache-line-aligned) -----------------------

  // Flat array of per-block metadata followed by the owned initial frontier
  // and returned-block bitmap. Each block_stride-byte entry contains a
  // flags/generation header followed by its frontier and entries.
  // Accessed on every alloc (to read death frontier) and free (to write it).
  iree_alignas(iree_hardware_destructive_interference_size) uint8_t
      block_storage[];
} iree_hal_memory_fixed_block_allocator_t;

//===----------------------------------------------------------------------===//
// API
//===----------------------------------------------------------------------===//

// Allocates a fixed-block allocator managing block_count blocks of block_size
// bytes. The managed offset range is [0, block_count * block_size). All blocks
// start free with empty death frontiers.
//
// The allocator is a single cache-line-aligned allocation (struct + trailing
// per-block metadata). |host_allocator| is used for this allocation and stored
// for iree_hal_memory_fixed_block_allocator_free().
//
// Returns IREE_STATUS_INVALID_ARGUMENT if options are invalid (zero block_size,
// zero block_count, block_count exceeds
// IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_MAX_BLOCKS).
//
// Thread-safety: NOT thread-safe. Must be called before any concurrent access.
iree_status_t iree_hal_memory_fixed_block_allocator_allocate(
    iree_hal_memory_fixed_block_allocator_options_t options,
    iree_allocator_t host_allocator,
    iree_hal_memory_fixed_block_allocator_t** out_pool);

// Frees a fixed-block allocator, releasing all memory.
// In debug builds, asserts if any blocks are still acquired.
//
// Thread-safety: NOT thread-safe. Must be called after all concurrent access
// has ceased.
void iree_hal_memory_fixed_block_allocator_free(
    iree_hal_memory_fixed_block_allocator_t* pool);

// Rebinds a completely free allocator to backing with the same geometry.
// The caller has transferred the old range's complete history to its owner.
// Copies |initial_frontier|, whose width must fit the configured capacity.
// Requires exclusion of all acquisitions, releases, and candidate readers.
void iree_hal_memory_fixed_block_allocator_reset(
    iree_hal_memory_fixed_block_allocator_t* pool,
    const iree_async_frontier_t* initial_frontier);

// Merges the exact histories of a completely free allocator. Reads only blocks
// returned in the current generation and includes untouched history once.
// False leaves a partial output; every source history remains unchanged.
// An optional overflow handler replaces the accumulated history with a compact
// proof covering both inputs. It must use presecured storage and cannot fail.
// Requires the same quiescent ownership as reset().
bool iree_hal_memory_fixed_block_allocator_merge_free_frontiers(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint8_t capacity,
    void (*overflow)(void* user_data, iree_async_frontier_t* target,
                     const iree_async_frontier_t* source),
    void* user_data, iree_async_frontier_t* out_frontier);

// Finds the first available block at or after |start_block_index| without
// claiming it. Returns false when no candidate remains. The returned frontier
// is borrowed; copy it before allowing another acquisition to proceed.
//
// The caller must exclude all acquisitions while querying and reading the
// candidate. Releases may proceed concurrently: their bitmap publication makes
// the complete frontier visible before the candidate becomes available.
bool iree_hal_memory_fixed_block_allocator_query_candidate(
    const iree_hal_memory_fixed_block_allocator_t* pool,
    uint32_t start_block_index,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_candidate);

// Returns whether the candidate's block remains available with the same exact
// frontier and flags. |candidate| points to a caller-owned frontier snapshot.
// Requires the same acquisition exclusion as query_candidate().
bool iree_hal_memory_fixed_block_allocator_candidate_is_current(
    const iree_hal_memory_fixed_block_allocator_t* pool,
    const iree_hal_memory_fixed_block_allocator_allocation_t* candidate);

// Claims a current candidate and returns its acquired block, with the frontier
// borrowed until release. The caller must exclude other acquisitions from its
// successful candidate_is_current() check through this call. This permits
// validating a complete transaction before making any claims.
void iree_hal_memory_fixed_block_allocator_acquire_candidate(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_allocation);

// Attempts to acquire a block from the allocator.
//
// Returns an error status only for invalid arguments or infrastructure
// failures. If all blocks are acquired, returns OK with
// out_result=IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_ACQUIRE_EXHAUSTED so pool
// callers can route expected transient exhaustion without constructing status
// backtraces.
//
// Thread-safety: Lock-free. Safe to call concurrently from any thread.
iree_status_t iree_hal_memory_fixed_block_allocator_try_acquire(
    iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_allocation,
    iree_hal_memory_fixed_block_allocator_acquire_result_t* out_result);

// Acquires a block from the allocator.
//
// The returned offset is block_index * block_size. The acquisition result
// includes the death frontier and flags from the block's previous release
// (if any), for dominance checking by the async allocator.
//
// Returns IREE_STATUS_RESOURCE_EXHAUSTED if all blocks are acquired.
//
// Thread-safety: Lock-free. Safe to call concurrently from any thread.
iree_status_t iree_hal_memory_fixed_block_allocator_acquire(
    iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_allocation_t* out_allocation);

// Releases a previously acquired block, returning it to the allocator.
//
// |block_index| is the index returned by
// iree_hal_memory_fixed_block_allocator_acquire(). |death_frontier| is the
// causal snapshot to attach to the released block. May be NULL for an empty
// frontier. The frontier entries are copied into the block's inline storage;
// the caller retains ownership of |death_frontier|.
//
// If |death_frontier| exceeds the per-block capacity, the block is marked
// tainted (IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_BLOCK_FLAG_TAINTED).
//
// Thread-safety: Lock-free. Safe to call concurrently from any thread.
void iree_hal_memory_fixed_block_allocator_release(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index,
    const iree_async_frontier_t* death_frontier);

// Releases a previously acquired block without modifying its existing death
// frontier or taint metadata.
//
// Use this when an allocation was acquired speculatively to inspect its death
// frontier and then rejected because the requester does not dominate that
// frontier. Unlike iree_hal_memory_fixed_block_allocator_release(), this keeps
// the prior dependency metadata intact so the block remains conservatively
// unavailable for zero-sync reuse until the original frontier is satisfied.
//
// Thread-safety: Lock-free. Safe to call concurrently from any thread.
void iree_hal_memory_fixed_block_allocator_restore(
    iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index);

// Copies the allocator's running statistics into |out_stats|. Values are atomic
// snapshots and may be momentarily inconsistent under concurrent modifications.
//
// Thread-safety: Safe to call concurrently from any thread.
void iree_hal_memory_fixed_block_allocator_query_stats(
    const iree_hal_memory_fixed_block_allocator_t* pool,
    iree_hal_memory_fixed_block_allocator_stats_t* out_stats);

// Returns the death frontier of a block. The returned pointer is into the
// block's inline frontier storage and remains valid while the block is
// allocated. Returns NULL if the frontier has zero entries.
//
// Thread-safety: Safe to call on allocated blocks. The caller must ensure the
// block is not concurrently freed.
const iree_async_frontier_t*
iree_hal_memory_fixed_block_allocator_block_death_frontier(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index);

// Returns the flags of a block. Useful for inspecting TAINTED status.
//
// Thread-safety: Safe to call on allocated blocks. The caller must ensure the
// block is not concurrently freed.
iree_hal_memory_fixed_block_allocator_block_flags_t
iree_hal_memory_fixed_block_allocator_block_flags(
    const iree_hal_memory_fixed_block_allocator_t* pool, uint32_t block_index);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_MEMORY_FIXED_BLOCK_ALLOCATOR_H_
