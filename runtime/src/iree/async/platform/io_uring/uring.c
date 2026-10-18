// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/async/platform/io_uring/uring.h"

#include <errno.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "iree/base/internal/atomics.h"
#include "iree/base/internal/math.h"

//===----------------------------------------------------------------------===//
// Syscall wrappers
//===----------------------------------------------------------------------===//

static inline int iree_io_uring_setup(uint32_t entries,
                                      iree_io_uring_params_t* params) {
  return (int)syscall(IREE_IO_URING_SYSCALL_SETUP, entries, params);
}

static inline int iree_io_uring_enter(int ring_fd, uint32_t to_submit,
                                      uint32_t min_complete, uint32_t flags,
                                      void* arg, size_t argsz) {
  return (int)syscall(IREE_IO_URING_SYSCALL_ENTER, ring_fd, to_submit,
                      min_complete, flags, arg, argsz);
}

//===----------------------------------------------------------------------===//
// Ring initialization
//===----------------------------------------------------------------------===//

// Translates a threading mode into the appropriate io_uring setup flags.
static uint32_t iree_io_uring_ring_threading_mode_flags(
    iree_io_uring_ring_threading_mode_t mode) {
  switch (mode) {
    default:
    case IREE_IO_URING_RING_THREADING_SAME_THREAD:
      return IREE_IORING_SETUP_SINGLE_ISSUER | IREE_IORING_SETUP_DEFER_TASKRUN;
    case IREE_IO_URING_RING_THREADING_CROSS_THREAD:
      return IREE_IORING_SETUP_SINGLE_ISSUER | IREE_IORING_SETUP_DEFER_TASKRUN |
             IREE_IORING_SETUP_R_DISABLED;
  }
}

// Attempts io_uring_setup with progressively fewer flags on failure.
// On success, stores ring fd, features, and fills out_params with kernel data.
static iree_status_t iree_io_uring_ring_try_setup(
    uint32_t entries, uint32_t requested_flags, iree_io_uring_ring_t* ring,
    iree_io_uring_params_t* out_params) {
  memset(out_params, 0, sizeof(*out_params));

  // Attempt 1: Try with all requested flags.
  out_params->flags = requested_flags;
  int fd = iree_io_uring_setup(entries, out_params);
  if (fd >= 0) {
    ring->ring_fd = fd;
    ring->features = out_params->features;
    return iree_ok_status();
  }

  // All io_uring_setup failures other than EINVAL (which triggers flag
  // fallbacks below) mean io_uring is not usable in this environment.
  // We return UNAVAILABLE with the errno in the message for diagnostics.
  // This matches the API contract in api.h and the industry-standard approach
  // (MariaDB, sled, etc. all treat ENOSYS/EPERM/ENOMEM uniformly as
  // "io_uring not available, fall back").
  if (errno == ENOSYS) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring not supported by kernel (ENOSYS)");
  }
  if (errno == EPERM) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring blocked by security policy (EPERM)");
  }

  // EINVAL may mean unsupported flags - try fallbacks.
  if (errno != EINVAL) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "io_uring_setup failed (errno %d)", errno);
  }

  // Attempt 2: Remove DEFER_TASKRUN (requires 6.1+).
  if (requested_flags & IREE_IORING_SETUP_DEFER_TASKRUN) {
    memset(out_params, 0, sizeof(*out_params));
    out_params->flags = requested_flags & ~IREE_IORING_SETUP_DEFER_TASKRUN;
    fd = iree_io_uring_setup(entries, out_params);
    if (fd >= 0) {
      ring->ring_fd = fd;
      ring->features = out_params->features;
      return iree_ok_status();
    }
    if (errno != EINVAL) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "io_uring_setup failed (errno %d)", errno);
    }
  }

  // Attempt 3: Remove SINGLE_ISSUER (requires 6.0+). Also remove R_DISABLED
  // since it is only meaningful with SINGLE_ISSUER.
  if (iree_any_bit_set(requested_flags, IREE_IORING_SETUP_SINGLE_ISSUER)) {
    memset(out_params, 0, sizeof(*out_params));
    out_params->flags = requested_flags & ~(IREE_IORING_SETUP_SINGLE_ISSUER |
                                            IREE_IORING_SETUP_DEFER_TASKRUN |
                                            IREE_IORING_SETUP_R_DISABLED);
    fd = iree_io_uring_setup(entries, out_params);
    if (fd >= 0) {
      ring->ring_fd = fd;
      ring->features = out_params->features;
      return iree_ok_status();
    }
    if (errno != EINVAL) {
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "io_uring_setup failed (errno %d)", errno);
    }
  }

  // Attempt 4: No optional flags (minimal 5.1 support).
  memset(out_params, 0, sizeof(*out_params));
  out_params->flags = 0;
  fd = iree_io_uring_setup(entries, out_params);
  if (fd >= 0) {
    ring->ring_fd = fd;
    ring->features = out_params->features;
    return iree_ok_status();
  }

  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "io_uring_setup failed with no flags (errno %d)",
                          errno);
}

// Validates that an offset + size fits within a mapped region.
// Returns true if valid, false if out of bounds.
static inline bool iree_io_uring_validate_offset(uint32_t offset, size_t size,
                                                 size_t region_size) {
  // Check for overflow in addition and bounds.
  return offset <= region_size && size <= region_size - offset;
}

// Maps the ring buffers using offsets from io_uring_setup params.
static iree_status_t iree_io_uring_ring_map_buffers(
    iree_io_uring_ring_t* ring, const iree_io_uring_params_t* params) {
  // The kernel tells us the sizes in params.sq_entries and params.cq_entries.
  uint32_t sq_entries = params->sq_entries;
  uint32_t cq_entries = params->cq_entries;

  // Compute SQ ring mmap size: we need enough to cover the array at the end.
  // The array is at sq_off.array and has sq_entries uint32_t indices.
  size_t sq_ring_size = params->sq_off.array + sq_entries * sizeof(uint32_t);
  void* sq_ring_ptr =
      mmap(NULL, sq_ring_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_POPULATE, ring->ring_fd, IREE_IORING_OFF_SQ_RING);
  if (sq_ring_ptr == MAP_FAILED) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SQ ring mmap failed (%d)", errno);
  }
  ring->sq_ring_ptr = sq_ring_ptr;
  ring->sq_ring_size = sq_ring_size;

  // Validate SQ ring offsets before dereferencing.
  // A buggy or mismatched kernel could provide offsets outside the mapped
  // region, causing segfaults.
  if (!iree_io_uring_validate_offset(params->sq_off.head, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.tail, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.ring_mask, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.ring_entries,
                                     sizeof(uint32_t), sq_ring_size) ||
      !iree_io_uring_validate_offset(params->sq_off.flags, sizeof(uint32_t),
                                     sq_ring_size) ||
      !iree_io_uring_validate_offset(
          params->sq_off.array, sq_entries * sizeof(uint32_t), sq_ring_size)) {
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "SQ ring offset out of bounds");
  }

  // Parse SQ ring pointers using kernel-provided offsets.
  ring->sq_head = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.head);
  ring->sq_tail = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.tail);
  ring->sq_ring_mask =
      (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.ring_mask);
  ring->sq_ring_entries =
      (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.ring_entries);
  ring->sq_flags = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.flags);
  ring->sq_array = (uint32_t*)((uint8_t*)sq_ring_ptr + params->sq_off.array);

  ring->sq_mask = *ring->sq_ring_mask;
  ring->sq_entries = *ring->sq_ring_entries;

  // Map the SQE array (separate mmap region).
  size_t sqes_size = sq_entries * sizeof(iree_io_uring_sqe_t);
  void* sqes_ptr =
      mmap(NULL, sqes_size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE,
           ring->ring_fd, IREE_IORING_OFF_SQES);
  if (sqes_ptr == MAP_FAILED) {
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    return iree_make_status(iree_status_code_from_errno(errno),
                            "SQE array mmap failed (%d)", errno);
  }
  ring->sqes = (iree_io_uring_sqe_t*)sqes_ptr;
  ring->sqes_size = sqes_size;

  // Compute CQ ring mmap size: we need enough to cover the CQE array.
  // The cqes offset tells us where in the mmap the CQE array starts.
  size_t cq_ring_size =
      params->cq_off.cqes + cq_entries * sizeof(iree_io_uring_cqe_t);
  void* cq_ring_ptr =
      mmap(NULL, cq_ring_size, PROT_READ | PROT_WRITE,
           MAP_SHARED | MAP_POPULATE, ring->ring_fd, IREE_IORING_OFF_CQ_RING);
  if (cq_ring_ptr == MAP_FAILED) {
    munmap(sqes_ptr, sqes_size);
    munmap(sq_ring_ptr, sq_ring_size);
    ring->sq_ring_ptr = NULL;
    ring->sqes = NULL;
    return iree_make_status(iree_status_code_from_errno(errno),
                            "CQ ring mmap failed (%d)", errno);
  }
  ring->cq_ring_ptr = cq_ring_ptr;
  ring->cq_ring_size = cq_ring_size;

  // Validate CQ ring offsets before dereferencing.
  if (!iree_io_uring_validate_offset(params->cq_off.head, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.tail, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.ring_mask, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.ring_entries,
                                     sizeof(uint32_t), cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.overflow, sizeof(uint32_t),
                                     cq_ring_size) ||
      !iree_io_uring_validate_offset(params->cq_off.cqes,
                                     cq_entries * sizeof(iree_io_uring_cqe_t),
                                     cq_ring_size)) {
    munmap(cq_ring_ptr, cq_ring_size);
    munmap(sqes_ptr, sqes_size);
    munmap(sq_ring_ptr, sq_ring_size);
    ring->cq_ring_ptr = NULL;
    ring->sq_ring_ptr = NULL;
    ring->sqes = NULL;
    return iree_make_status(IREE_STATUS_INTERNAL,
                            "CQ ring offset out of bounds");
  }

  // Parse CQ ring pointers using kernel-provided offsets.
  ring->cq_head = (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.head);
  ring->cq_tail = (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.tail);
  ring->cq_ring_mask =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.ring_mask);
  ring->cq_ring_entries =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.ring_entries);
  ring->cq_overflow =
      (uint32_t*)((uint8_t*)cq_ring_ptr + params->cq_off.overflow);
  ring->cqes =
      (iree_io_uring_cqe_t*)((uint8_t*)cq_ring_ptr + params->cq_off.cqes);

  ring->cq_mask = *ring->cq_ring_mask;
  ring->cq_entries = *ring->cq_ring_entries;

  // Initialize local tail to match kernel's view.
  ring->sq_local_tail = *ring->sq_tail;

  return iree_ok_status();
}

iree_status_t iree_io_uring_ring_initialize(
    iree_io_uring_ring_options_t options, iree_io_uring_ring_t* out_ring) {
  IREE_TRACE_ZONE_BEGIN(z0);
  IREE_ASSERT_ARGUMENT(out_ring);
  memset(out_ring, 0, sizeof(*out_ring));
  out_ring->ring_fd = -1;

  uint32_t entries = options.sq_entries;
  if (entries == 0) {
    entries = 256;
  }
  // Round up to power of 2 (kernel requirement).
  entries = iree_math_round_up_to_pow2_u32(entries);
  if (entries < 1) {
    entries = 1;
  }
  if (entries > IREE_IORING_MAX_ENTRIES) {
    entries = IREE_IORING_MAX_ENTRIES;
  }

  // Translate threading mode to kernel flags and try setup with fallbacks.
  uint32_t setup_flags =
      iree_io_uring_ring_threading_mode_flags(options.threading_mode);
  iree_io_uring_params_t params;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0,
      iree_io_uring_ring_try_setup(entries, setup_flags, out_ring, &params));

  // Record whether the ring needs enabling before io_uring_enter can be
  // called. The actual flags used may differ from requested (fallback path).
  out_ring->setup_flags = params.flags;
  iree_atomic_store(&out_ring->needs_enable,
                    (params.flags & IREE_IORING_SETUP_R_DISABLED) != 0,
                    iree_memory_order_relaxed);
  iree_io_uring_registration_initialize(out_ring->ring_fd, params.flags,
                                        &out_ring->registration);

  iree_status_t status = iree_io_uring_ring_map_buffers(out_ring, &params);
  if (iree_status_is_ok(status)) {
    status = iree_allocator_malloc(
        iree_allocator_system(),
        ((out_ring->sq_entries + 63) / 64) * sizeof(uint64_t),
        (void**)&out_ring->sq_progress_boundaries);
  }
  if (!iree_status_is_ok(status)) {
    iree_io_uring_ring_deinitialize(out_ring);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

iree_status_t iree_io_uring_ring_enable(iree_io_uring_ring_t* ring) {
  if (!iree_io_uring_ring_needs_enable(ring)) {
    return iree_ok_status();
  }

  int result = iree_io_uring_registration_enable(&ring->registration);
  if (result < 0) {
    int error_number = -result;
    return iree_make_status(iree_status_code_from_errno(error_number),
                            "IORING_REGISTER_ENABLE_RINGS failed (%d)",
                            error_number);
  }

  iree_atomic_store(&ring->needs_enable, 0, iree_memory_order_release);
  return iree_ok_status();
}

void iree_io_uring_ring_set_registration_wake_callback(
    iree_io_uring_ring_t* ring,
    iree_io_uring_registration_wake_callback_t callback) {
  iree_io_uring_registration_set_wake_callback(&ring->registration, callback);
}

int iree_io_uring_ring_register(iree_io_uring_ring_t* ring, uint32_t opcode,
                                void* arg, uint32_t argument_count) {
  return iree_io_uring_registration_execute(&ring->registration, opcode, arg,
                                            argument_count);
}

void iree_io_uring_ring_drain_registration_requests(
    iree_io_uring_ring_t* ring) {
  iree_io_uring_registration_drain(&ring->registration);
}

void iree_io_uring_ring_end_polling(iree_io_uring_ring_t* ring) {
  iree_io_uring_registration_end_polling(&ring->registration);
}

void iree_io_uring_ring_deinitialize(iree_io_uring_ring_t* ring) {
  IREE_TRACE_ZONE_BEGIN(z0);
  if (ring->ring_fd >= 0) {
    iree_io_uring_registration_deinitialize(&ring->registration);
  }
  iree_allocator_free(iree_allocator_system(), ring->sq_progress_boundaries);
  ring->sq_progress_boundaries = NULL;
  if (ring->cq_ring_ptr) {
    munmap(ring->cq_ring_ptr, ring->cq_ring_size);
    ring->cq_ring_ptr = NULL;
  }
  if (ring->sqes) {
    munmap(ring->sqes, ring->sqes_size);
    ring->sqes = NULL;
  }
  if (ring->sq_ring_ptr) {
    munmap(ring->sq_ring_ptr, ring->sq_ring_size);
    ring->sq_ring_ptr = NULL;
  }
  if (ring->ring_fd >= 0) {
    close(ring->ring_fd);
    ring->ring_fd = -1;
  }
  IREE_TRACE_ZONE_END(z0);
}

//===----------------------------------------------------------------------===//
// Submission queue operations
//===----------------------------------------------------------------------===//

iree_io_uring_sqe_t* iree_io_uring_ring_get_sqe(iree_io_uring_ring_t* ring) {
  uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                   iree_memory_order_acquire);
  uint32_t tail = ring->sq_local_tail;

  // Check if SQ is full (including uncommitted SQEs).
  if (tail - head >= ring->sq_entries) {
    return NULL;
  }

  uint32_t index = tail & ring->sq_mask;
  iree_io_uring_sqe_t* sqe = &ring->sqes[index];
  ring->sq_progress_boundaries[index / 64] &= ~(UINT64_C(1) << (index % 64));

  // Zero the SQE for the caller.
  memset(sqe, 0, sizeof(*sqe));

  // Update the SQ array to point to this SQE index.
  ring->sq_array[index] = index;

  // Increment local tail only. The kernel-visible *sq_tail is updated
  // during submit(), allowing rollback if encoding fails.
  ring->sq_local_tail = tail + 1;

  return sqe;
}

// Publishes prepared entries and returns the submission extent. The caller
// holds the SQ lock so another submitter cannot be mid-fill.
static uint32_t iree_io_uring_ring_flush(iree_io_uring_ring_t* ring) {
  uint32_t tail = ring->sq_local_tail;
  if (tail != *ring->sq_tail) {
    // Release barrier ensures all SQE field writes complete before kernel sees
    // the new tail.
    iree_atomic_store((iree_atomic_int32_t*)ring->sq_tail, tail,
                      iree_memory_order_release);
  }
  return tail;
}

// A submission allocation failure bypasses deferred task work in the kernel.
// Run that work without allocating new requests, then reattempt the frozen
// extent. Repeated failure is terminal to this attempt, never a busy retry.
static iree_status_t iree_io_uring_ring_recover_submission(
    iree_io_uring_ring_t* ring, uint32_t submission_tail, uint32_t min_complete,
    uint32_t flags) {
  int ret = 0;
  do {
    ret = iree_io_uring_enter(ring->ring_fd, 0, 0, IREE_IORING_ENTER_GETEVENTS,
                              NULL, 0);
  } while (ret < 0 && errno == EINTR);
  if (ret < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring recovery progress failed (%d)", errno);
  }
  do {
    uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                     iree_memory_order_acquire);
    ret = iree_io_uring_enter(ring->ring_fd, submission_tail - head,
                              min_complete, flags, NULL, 0);
  } while (ret < 0 && errno == EINTR);
  if (ret < 0) {
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring admission failed after progress (%d)",
                            errno);
  }
  return iree_ok_status();
}

static iree_status_t iree_io_uring_ring_submit_extent(
    iree_io_uring_ring_t* ring, uint32_t submission_tail, uint32_t min_complete,
    uint32_t flags) {
  // A published entry still needs submission until the kernel consumes it.
  // Recompute the remaining extent on EINTR as well: the kernel may have
  // advanced its head before returning, or may not have consumed anything.
  int ret = 0;
  uint32_t to_submit = 0;
  do {
    uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                     iree_memory_order_acquire);
    to_submit = submission_tail - head;
    // GETEVENTS must run deferred task work even when no SQEs remain.
    if (to_submit == 0 && min_complete == 0 &&
        !iree_any_bit_set(flags, IREE_IORING_ENTER_GETEVENTS)) {
      return iree_ok_status();
    }
    ret = iree_io_uring_enter(ring->ring_fd, to_submit, min_complete, flags,
                              NULL, 0);
  } while (ret < 0 && errno == EINTR);

  if (ret < 0) {
    if (errno == EAGAIN && to_submit > 0) {
      return iree_io_uring_ring_recover_submission(ring, submission_tail,
                                                   min_complete, flags);
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring_enter failed (%d)", errno);
  }

  return iree_ok_status();
}

// Selects the next complete chain within the frozen extent. The caller holds
// the SQ lock so producers cannot replace a marker while it is being read.
static uint32_t iree_io_uring_ring_progress_tail(
    const iree_io_uring_ring_t* ring, uint32_t submission_tail) {
  uint32_t position = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                       iree_memory_order_acquire);
  while (position != submission_tail) {
    uint32_t index = position & ring->sq_mask;
    uint32_t shift = index % 64;
    uint32_t count = iree_min(submission_tail - position,
                              iree_min(ring->sq_entries - index, 64 - shift));
    uint64_t boundaries = (ring->sq_progress_boundaries[index / 64] >> shift) &
                          (UINT64_MAX >> (64 - count));
    if (boundaries) {
      return position + iree_math_count_trailing_zeros_u64(boundaries) + 1;
    }
    position += count;
  }
  return submission_tail;
}

iree_status_t iree_io_uring_ring_progress(iree_io_uring_ring_t* ring,
                                          uint32_t flags) {
  iree_io_uring_ring_sq_lock(ring);
  uint32_t submission_tail = iree_io_uring_ring_flush(ring);
  uint32_t progress_tail =
      iree_io_uring_ring_progress_tail(ring, submission_tail);
  iree_io_uring_ring_sq_unlock(ring);
  for (;;) {
    IREE_RETURN_IF_ERROR(iree_io_uring_ring_submit_extent(
        ring, progress_tail, /*min_complete=*/0, flags));
    // A successful enter may consume only part of the requested extent.
    // Without ready completions, finish admitting it before callers can wait.
    if (iree_io_uring_ring_cq_ready(ring) ||
        iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                         iree_memory_order_acquire) == submission_tail) {
      return iree_ok_status();
    }
    iree_io_uring_ring_sq_lock(ring);
    progress_tail = iree_io_uring_ring_progress_tail(ring, submission_tail);
    iree_io_uring_ring_sq_unlock(ring);
  }
}

iree_status_t iree_io_uring_ring_submit(iree_io_uring_ring_t* ring,
                                        uint32_t min_complete, uint32_t flags) {
  iree_io_uring_ring_sq_lock(ring);
  uint32_t submission_tail = iree_io_uring_ring_flush(ring);
  iree_io_uring_ring_sq_unlock(ring);
  return iree_io_uring_ring_submit_extent(ring, submission_tail, min_complete,
                                          flags);
}

iree_status_t iree_io_uring_ring_submit_pending_locked(
    iree_io_uring_ring_t* ring) {
  return iree_io_uring_ring_submit_extent(ring, iree_io_uring_ring_flush(ring),
                                          /*min_complete=*/0, /*flags=*/0);
}

//===----------------------------------------------------------------------===//
// Completion queue operations
//===----------------------------------------------------------------------===//

iree_status_t iree_io_uring_ring_wait_cqe(iree_io_uring_ring_t* ring,
                                          uint32_t min_complete,
                                          bool flush_pending,
                                          iree_duration_t timeout_ns) {
  // Check if we already have enough CQEs.
  if (iree_io_uring_ring_cq_count(ring) >= min_complete) {
    return iree_ok_status();
  }

  // Freeze the published extent for this wait. Cross-thread preparation can
  // continue, but only the poll owner publishes and consumes entries.
  uint32_t submission_tail = 0;
  if (flush_pending) {
    iree_io_uring_ring_sq_lock(ring);
    submission_tail = iree_io_uring_ring_flush(ring);
    iree_io_uring_ring_sq_unlock(ring);
  }

  // Compute absolute deadline so EINTR retries use the remaining time rather
  // than restarting the full timeout duration. Without this, frequent signals
  // (profiler sampling, etc.) cause the effective wait to grow unboundedly.
  const bool is_immediate = timeout_ns == IREE_DURATION_ZERO;
  const bool has_timeout =
      timeout_ns > 0 && timeout_ns != IREE_DURATION_INFINITE;
  iree_time_t deadline = IREE_TIME_INFINITE_FUTURE;
  if (has_timeout) {
    deadline = iree_time_now() + timeout_ns;
  }

  for (;;) {
    // Build flags and args for io_uring_enter.
    uint32_t flags = IREE_IORING_ENTER_GETEVENTS;

    iree_kernel_timespec_t ts;
    iree_io_uring_getevents_arg_t arg;
    void* arg_ptr = NULL;
    size_t arg_sz = 0;

    if (has_timeout) {
      iree_duration_t remaining = deadline - iree_time_now();
      if (remaining <= 0) {
        if (iree_io_uring_ring_cq_count(ring) > 0) {
          return iree_ok_status();
        }
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      flags |= IREE_IORING_ENTER_EXT_ARG;
      ts.tv_sec = remaining / 1000000000LL;
      ts.tv_nsec = remaining % 1000000000LL;
      memset(&arg, 0, sizeof(arg));
      arg.ts = (uint64_t)&ts;
      arg_ptr = &arg;
      arg_sz = sizeof(arg);
    }

    uint32_t to_submit = 0;
    if (flush_pending) {
      uint32_t head = iree_atomic_load((iree_atomic_int32_t*)ring->sq_head,
                                       iree_memory_order_acquire);
      to_submit = submission_tail - head;
    }
    // An immediate wait still flushes pending submissions and deferred kernel
    // work, but cannot require the kernel to produce a completion before
    // return.
    int ret = iree_io_uring_enter(ring->ring_fd, to_submit,
                                  is_immediate ? 0 : min_complete, flags,
                                  arg_ptr, arg_sz);

    if (ret >= 0) {
      if (is_immediate && iree_io_uring_ring_cq_count(ring) == 0) {
        return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
      }
      return iree_ok_status();
    }

    if (errno == ETIME) {
      if (iree_io_uring_ring_cq_count(ring) > 0) {
        return iree_ok_status();
      }
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    if (errno == EINTR) {
      if (iree_io_uring_ring_cq_count(ring) > 0) {
        return iree_ok_status();
      }
      // Loop retries with remaining time computed from the deadline.
      continue;
    }
    if (errno == EAGAIN && to_submit > 0) {
      // Recover admission without extending the caller's wait deadline. The
      // next pass recomputes both the remaining extent and remaining time.
      IREE_RETURN_IF_ERROR(iree_io_uring_ring_recover_submission(
          ring, submission_tail, 0, IREE_IORING_ENTER_GETEVENTS));
      continue;
    }
    return iree_make_status(iree_status_code_from_errno(errno),
                            "io_uring_enter failed (%d)", errno);
  }
}
