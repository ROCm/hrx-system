// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/net/carrier/shm/carrier.h"

#include "iree/async/notification.h"
#include "iree/async/operations/scheduling.h"
#include "iree/base/internal/mpsc_queue.h"
#include "iree/net/carrier/shm/file_transfer.h"
#include "iree/net/carrier/shm/shared_wake.h"

typedef struct iree_net_shm_carrier_t {
  // Base carrier (must be first for safe upcasting).
  iree_net_carrier_t base;

  bool is_client;
  iree_atomic_int32_t shutdown_initiated;
  bool peer_shutdown_received;

  // First asynchronous carrier error without a completion callback.
  iree_atomic_intptr_t failure_status;

  // Opaque ownership context released during destroy. Typically a pair_context
  // for create_pair or a per-carrier resource bundle for factory-created
  // carriers.
  struct {
    void (*fn)(void* context);
    void* context;
  } release;

  iree_mpsc_queue_t tx_queue;
  iree_mpsc_queue_t rx_queue;

  // Our proactor's shared wake. Owns a local notification and the sleeping
  // carrier list. Retained.
  iree_net_shm_shared_wake_t* shared_wake;

  // Peer proactor's shared notification (signaled to wake the peer's scan
  // callback). Retained.
  iree_async_notification_t* peer_wake_notification;

  // Intrusive linkage for the shared_wake's sleeping carrier list.
  // NULL when not in the list (poll mode or inactive).
  struct iree_net_shm_carrier_t* sleeping_next;

  // Adaptive signaling pointers into the SHM region. Each is on its own cache
  // line (64-byte aligned) to prevent false sharing.
  iree_atomic_int32_t* our_armed;   // We set before sleeping.
  iree_atomic_int32_t* peer_armed;  // Peer sets before sleeping; we check.

  struct {
    iree_net_carrier_deactivate_callback_fn_t fn;
    void* user_data;
  } deactivate_callback;

  // NOP operation used by activate to defer sleeping list registration to the
  // poll thread. Embedded to avoid allocation on the activate path.
  iree_async_nop_operation_t activate_nop;

  // Adaptive polling state (progress callback mode).
  // When traffic is hot, the carrier registers a progress callback with the
  // proactor that polls the MPSC queue directly each poll() iteration,
  // bypassing the kernel notification path (~50ns acquire-load vs ~1-5us
  // notification).
  struct {
    // Progress callback entry registered with the proactor. The entry's fn and
    // user_data are set once during the first sleep-to-poll transition and
    // remain stable for the carrier's lifetime.
    iree_async_progress_entry_t entry;

    // True when in poll mode (progress callback registered, no
    // NOTIFICATION_WAIT posted). False when in sleep mode.
    bool active;

    // True when removal retires the poll-mode pending operation instead of
    // transferring it to the shared-wake sleeping list.
    bool retire_on_remove;

    // Consecutive poll iterations with no data. Reset to 0 on any progress.
    // When this exceeds IREE_NET_SHM_IDLE_SPIN_THRESHOLD, the carrier
    // transitions back to notification-based sleep mode.
    int32_t idle_spin_count;

    // AIMD adaptive window: number of RX entries to drain per progress
    // callback invocation.
    int32_t window;

    // True if the previous progress callback drained exactly |window| entries
    // (ring was at least as full as the window). Two consecutive full polls
    // trigger additive increase.
    bool previous_was_full;

    // Minimum number of RX entries drained in a single notification callback
    // before transitioning from sleep mode to poll mode.
    int32_t mode_threshold;
  } polling;

  // Known SHM regions for buffer registration and direct read/write.
  // Stored as a FAM at the end of the struct.
  iree_host_size_t region_count;
  // Optional sideband used to transfer external file handle rights.
  iree_net_shm_file_transfer_t* file_transfer;
  // Known SHM region table entries.
  iree_net_shm_region_info_t regions[];
} iree_net_shm_carrier_t;

static inline iree_net_shm_carrier_t* iree_net_shm_carrier_cast(
    iree_net_carrier_t* base_carrier) {
  return (iree_net_shm_carrier_t*)base_carrier;
}

//===----------------------------------------------------------------------===//
// Sleeping list accessors (used by shared_wake.c)
//===----------------------------------------------------------------------===//

iree_net_shm_carrier_t* iree_net_shm_carrier_sleeping_next(
    iree_net_shm_carrier_t* carrier) {
  return carrier->sleeping_next;
}

void iree_net_shm_carrier_set_sleeping_next(iree_net_shm_carrier_t* carrier,
                                            iree_net_shm_carrier_t* next) {
  carrier->sleeping_next = next;
}

//===----------------------------------------------------------------------===//
// Adaptive signaling helpers
//===----------------------------------------------------------------------===//

// Signals the peer notification if the peer's armed flag is set.
//
// The seq_cst fence prevents StoreLoad reordering between our data write (ring
// commit or ring consume) and the armed flag read. Without it, on ARM the peer
// could store armed=1 and load ring=empty, while we store data and load
// armed=0 — a Dekker-style deadlock where neither side sees the other's store.
// The fence ensures our ring store is globally visible before we read the armed
// flag, and the peer's armed store is globally visible before it reads the
// ring.
static void iree_net_shm_signal_peer(iree_net_shm_carrier_t* carrier) {
  iree_atomic_thread_fence(iree_memory_order_seq_cst);
  int32_t armed =
      iree_atomic_load(carrier->peer_armed, iree_memory_order_acquire);
  if (armed) {
    iree_async_notification_signal(carrier->peer_wake_notification, 1);
  }
}

static void iree_net_shm_carrier_maybe_complete_deactivation(
    iree_net_shm_carrier_t* carrier) {
  int32_t pending = iree_atomic_load(&carrier->base.pending_operations,
                                     iree_memory_order_acquire);
  if (pending > 0) return;
  if (!iree_net_carrier_try_transition_state(
          &carrier->base, IREE_NET_CARRIER_STATE_DRAINING,
          IREE_NET_CARRIER_STATE_DEACTIVATED)) {
    return;
  }

  iree_net_carrier_deactivate_callback_fn_t callback =
      carrier->deactivate_callback.fn;
  void* callback_user_data = carrier->deactivate_callback.user_data;
  carrier->deactivate_callback.fn = NULL;
  carrier->deactivate_callback.user_data = NULL;
  if (callback) callback(callback_user_data);
}

// Retires one operation and attempts terminal callback delivery only when this
// caller owned the final count. No carrier access is permitted after this call.
static void iree_net_shm_carrier_retire_pending_operation(
    iree_net_shm_carrier_t* carrier) {
  if (iree_net_carrier_retire_pending_operation(&carrier->base)) {
    iree_net_shm_carrier_maybe_complete_deactivation(carrier);
  }
}

//===----------------------------------------------------------------------===//
// Drain callback (unified RX data + TX completions)
//===----------------------------------------------------------------------===//

static iree_host_size_t iree_net_shm_carrier_progress(void* user_data);

// Default RX drain budget for the notification-based drain callback.
// The progress callback uses the AIMD window instead.
#define IREE_NET_SHM_DRAIN_RX_BUDGET 256

static void iree_net_shm_carrier_set_failure_status(
    iree_net_shm_carrier_t* carrier, iree_status_t status) {
  intptr_t expected = 0;
  intptr_t desired = (intptr_t)status;
  if (!iree_atomic_compare_exchange_strong(&carrier->failure_status, &expected,
                                           desired, iree_memory_order_release,
                                           iree_memory_order_relaxed)) {
    iree_status_free(status);
  }
}

static void iree_net_shm_carrier_drop_failure_status(
    iree_net_shm_carrier_t* carrier) {
  iree_status_t status = (iree_status_t)iree_atomic_exchange(
      &carrier->failure_status, 0, iree_memory_order_acq_rel);
  iree_status_free(status);
}

static void iree_net_shm_carrier_notify_error(iree_net_shm_carrier_t* carrier,
                                              iree_status_t status) {
  if (carrier->base.callback.fn) {
    carrier->base.callback.fn(carrier->base.callback.user_data,
                              IREE_NET_CARRIER_COMPLETION_ERROR,
                              /*operation_user_data=*/0, status,
                              /*bytes_transferred=*/0, /*recv_lease=*/NULL);
  } else if (!iree_status_is_ok(status)) {
    iree_net_shm_carrier_set_failure_status(carrier, status);
  }
}

typedef struct iree_net_shm_drain_rx_result_t {
  enum {
    // Ring empty — all available data was processed.
    IREE_NET_SHM_DRAIN_RX_EMPTY = 0,
    // Peer shutdown marker or unknown entry type — RX is done permanently.
    IREE_NET_SHM_DRAIN_RX_PEER_DONE,
    // Budget exhausted — data remains in the ring but we should yield to let
    // other proactor operations run.
    IREE_NET_SHM_DRAIN_RX_BUDGET_HIT,
  } status;
  // Number of RX entries actually drained.
  int32_t drained_count;
} iree_net_shm_drain_rx_result_t;

// Dispatches a single received ring entry to the data or signal handler.
// Returns true on success (entry was valid and delivered), false if the entry
// is malformed, out-of-bounds, or rejected by the handler.
static bool iree_net_shm_carrier_dispatch_rx_entry(
    iree_net_shm_carrier_t* carrier, const void* payload,
    iree_host_size_t entry_length) {
  const uint8_t* data = (const uint8_t*)payload;
  if (entry_length < sizeof(iree_net_shm_entry_header_t)) {
    iree_net_shm_carrier_notify_error(
        carrier, iree_make_status(
                     IREE_STATUS_INVALID_ARGUMENT,
                     "SHM RX entry length %" PRIhsz
                     " is smaller than the %" PRIhsz " byte header",
                     entry_length,
                     (iree_host_size_t)sizeof(iree_net_shm_entry_header_t)));
    return false;
  }
  const iree_net_shm_entry_header_t* header =
      (const iree_net_shm_entry_header_t*)data;
  uint32_t entry_type = header->type;
  const uint8_t* entry_data = data + sizeof(*header);
  iree_host_size_t data_length = entry_length - sizeof(*header);

  if (entry_type == IREE_NET_SHM_ENTRY_TYPE_INLINE) {
    if (carrier->base.recv_handler.fn) {
      iree_async_span_t span =
          iree_async_span_from_ptr((void*)entry_data, data_length);
      iree_status_t recv_status = carrier->base.recv_handler.fn(
          carrier->base.recv_handler.user_data, span, NULL);
      if (!iree_status_is_ok(recv_status)) {
        iree_net_shm_carrier_notify_error(carrier, recv_status);
        return false;
      }
    }
    iree_atomic_fetch_add(&carrier->base.bytes_received, (int64_t)data_length,
                          iree_memory_order_relaxed);
    return true;
  }

  if (entry_type == IREE_NET_SHM_ENTRY_TYPE_REFERENCE) {
    if (data_length != sizeof(iree_net_shm_reference_descriptor_t)) {
      iree_net_shm_carrier_notify_error(
          carrier,
          iree_make_status(
              IREE_STATUS_INVALID_ARGUMENT,
              "SHM reference descriptor has length %" PRIhsz
              ", expected %" PRIhsz,
              data_length,
              (iree_host_size_t)sizeof(iree_net_shm_reference_descriptor_t)));
      return false;
    }
    const iree_net_shm_reference_descriptor_t descriptor =
        *(const iree_net_shm_reference_descriptor_t*)entry_data;
    uint64_t range_end = 0;
    if (descriptor.region_id >= carrier->region_count ||
        !iree_checked_add_u64(descriptor.offset, descriptor.length,
                              &range_end) ||
        range_end > carrier->regions[descriptor.region_id].size) {
      iree_net_shm_carrier_notify_error(
          carrier, iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                    "SHM reference region %u offset %" PRIu64
                                    " length %" PRIu64 " is out of range",
                                    descriptor.region_id, descriptor.offset,
                                    descriptor.length));
      return false;
    }
    if (!carrier->base.signal_handler.fn) {
      iree_net_shm_carrier_notify_error(
          carrier, iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                                    "SHM direct_write signal handler not set"));
      return false;
    }
    iree_status_t signal_status = carrier->base.signal_handler.fn(
        carrier->base.signal_handler.user_data, descriptor.reserved);
    if (!iree_status_is_ok(signal_status)) {
      iree_net_shm_carrier_notify_error(carrier, signal_status);
      return false;
    }
    iree_atomic_fetch_add(&carrier->base.bytes_received,
                          (int64_t)descriptor.length,
                          iree_memory_order_relaxed);
    return true;
  }

  // Unknown entry type — cannot safely interpret subsequent entries.
  iree_net_shm_carrier_notify_error(
      carrier,
      iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                       "unknown SHM RX entry type 0x%08" PRIX32, entry_type));
  return false;
}

// Drains received entries from the RX ring up to |budget| entries.
// Returns both the drain status and the count of entries processed.
static iree_net_shm_drain_rx_result_t iree_net_shm_carrier_drain_rx(
    iree_net_shm_carrier_t* carrier, int32_t budget) {
  iree_net_shm_drain_rx_result_t result = {IREE_NET_SHM_DRAIN_RX_EMPTY, 0};
  iree_host_size_t entry_length = 0;
  const void* payload = NULL;
  while ((payload = iree_mpsc_queue_peek(&carrier->rx_queue, &entry_length)) !=
         NULL) {
    // Check for a header-only shutdown marker.
    if (entry_length == sizeof(iree_net_shm_entry_header_t) &&
        ((const iree_net_shm_entry_header_t*)payload)->type ==
            IREE_NET_SHM_ENTRY_TYPE_SHUTDOWN) {
      iree_mpsc_queue_consume(&carrier->rx_queue);
      iree_net_shm_signal_peer(carrier);
      carrier->peer_shutdown_received = true;
      result.status = IREE_NET_SHM_DRAIN_RX_PEER_DONE;
      return result;
    }

    if (!iree_net_shm_carrier_dispatch_rx_entry(carrier, payload,
                                                entry_length)) {
      // Malformed or unrecognized entry — consume it and stop.
      iree_mpsc_queue_consume(&carrier->rx_queue);
      iree_net_shm_signal_peer(carrier);
      carrier->peer_shutdown_received = true;
      result.status = IREE_NET_SHM_DRAIN_RX_PEER_DONE;
      return result;
    }

    iree_mpsc_queue_consume(&carrier->rx_queue);
    iree_net_shm_signal_peer(carrier);
    ++result.drained_count;

    if (--budget <= 0) {
      result.status = IREE_NET_SHM_DRAIN_RX_BUDGET_HIT;
      return result;
    }
  }
  return result;
}

// Releases the sleep-mode slot and attempts deactivation completion.
// Used when drain cannot continue (deactivation in progress or unexpected
// state). The caller is responsible for ensuring the carrier is removed from
// the shared_wake sleeping list (drain_from_wake returns false).
static void iree_net_shm_carrier_drain_stop(iree_net_shm_carrier_t* carrier) {
  iree_net_shm_carrier_retire_pending_operation(carrier);
}

//===----------------------------------------------------------------------===//
// Progress callback (poll mode)
//===----------------------------------------------------------------------===//
//
// When in poll mode, this callback runs every proactor poll() iteration
// instead of using NOTIFICATION_WAIT. It checks the MPSC ring positions with
// acquire-loads (~50ns) and drains data inline. The AIMD adaptive window
// controls how many entries to drain per invocation to prevent a single hot
// carrier from monopolizing the poll thread.
//
// Idle-to-sleep transition uses the Dekker-fence protocol:
//   1. Set armed → seq_cst fence → re-check ring
//   2. If data appeared: clear armed, stay in poll mode
//   3. If ring still empty: post NOTIFICATION_WAIT, request removal

// Called by run_progress() after the progress entry is safely unlinked from
// the proactor's list. Triggers deactivation completion if pending_operations
// has reached zero. This runs after the entry is unlinked, so it is safe for
// the deactivate callback to release/free the carrier.
static void iree_net_shm_carrier_progress_removed(void* user_data) {
  iree_net_shm_carrier_t* carrier = (iree_net_shm_carrier_t*)user_data;
  if (!carrier->polling.retire_on_remove) return;
  carrier->polling.retire_on_remove = false;
  iree_net_shm_carrier_retire_pending_operation(carrier);
}

// Stops the progress callback. Its pending operation remains live until
// on_remove runs after the proactor has unlinked the entry; that is the first
// point where terminal callback delivery may safely free the carrier.
static void iree_net_shm_carrier_progress_stop(
    iree_net_shm_carrier_t* carrier) {
  carrier->polling.active = false;
  carrier->polling.retire_on_remove = true;
  carrier->polling.entry.remove_requested = true;
}

static iree_host_size_t iree_net_shm_carrier_progress(void* user_data) {
  iree_net_shm_carrier_t* carrier = (iree_net_shm_carrier_t*)user_data;

  // Check carrier state — deactivation may have been requested.
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_shm_carrier_progress_stop(carrier);
    // on_remove may complete deactivation and invoke an external callback.
    // Count the removal as progress so proactor_poll returns instead of
    // blocking after the final progress entry disappears.
    return 1;
  }

  iree_host_size_t completions = 0;

  // Drain RX ring up to the AIMD window.
  bool peer_rx_stopped = carrier->peer_shutdown_received;
  if (!peer_rx_stopped) {
    iree_net_shm_drain_rx_result_t rx_result =
        iree_net_shm_carrier_drain_rx(carrier, carrier->polling.window);
    completions += (iree_host_size_t)rx_result.drained_count;

    if (rx_result.status == IREE_NET_SHM_DRAIN_RX_PEER_DONE) {
      peer_rx_stopped = true;
    }

    // AIMD window adaptation.
    if (rx_result.drained_count > 0) {
      if (rx_result.status == IREE_NET_SHM_DRAIN_RX_BUDGET_HIT) {
        // Drained exactly window entries (budget hit = ring was at least as
        // full as the window).
        if (carrier->polling.previous_was_full) {
          // Two consecutive full polls → additive increase.
          if (carrier->polling.window < IREE_NET_SHM_POLL_WINDOW_MAX) {
            carrier->polling.window++;
          }
        }
        carrier->polling.previous_was_full = true;
      } else {
        // Underconsumed (drained fewer than window) → multiplicative decrease.
        carrier->polling.window =
            iree_max(carrier->polling.window / 2, IREE_NET_SHM_POLL_WINDOW_MIN);
        carrier->polling.previous_was_full = false;
      }
    }
  }

  // Signal peer after draining (peer may be waiting for us to consume data
  // before it can write more).
  if (completions > 0) {
    iree_net_shm_signal_peer(carrier);
    carrier->polling.idle_spin_count = 0;
    return completions;
  }

  // No progress this iteration.
  carrier->polling.previous_was_full = false;
  carrier->polling.idle_spin_count++;
  if (carrier->polling.idle_spin_count < IREE_NET_SHM_IDLE_SPIN_THRESHOLD) {
    return 0;
  }

  // Idle threshold exceeded — transition back to notification-based sleep mode.
  // Dekker protocol: set armed → fence → re-check ring.
  iree_atomic_store(carrier->our_armed, 1, iree_memory_order_release);
  iree_atomic_thread_fence(iree_memory_order_seq_cst);

  // Re-check: committed data may have arrived between our last check and the
  // armed store. We use peek (not can_read) because can_read returns true for
  // reserved-but-uncommitted entries, which would prevent the sleep transition
  // while a peer holds an outstanding begin_send reservation.
  iree_host_size_t recheck_length = 0;
  if (!peer_rx_stopped &&
      iree_mpsc_queue_peek(&carrier->rx_queue, &recheck_length) != NULL) {
    // Data arrived — stay in poll mode.
    iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);
    carrier->polling.idle_spin_count = 0;
    return 0;
  }

  // Ring is truly empty — transition to sleep mode. Register with the
  // shared_wake sleeping list and request removal from the progress callback
  // list. The pending_operations slot transfers from the progress callback to
  // the sleeping list registration.
  iree_status_t register_status =
      iree_net_shm_shared_wake_register(carrier->shared_wake, carrier);
  if (IREE_UNLIKELY(!iree_status_is_ok(register_status))) {
    // Cannot register for wake — stay in poll mode. This keeps the carrier
    // functional (progress callback continues firing) at the cost of spinning.
    iree_status_free(register_status);
    iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);
    carrier->polling.idle_spin_count = 0;
    return 0;
  }
  carrier->polling.active = false;
  carrier->polling.retire_on_remove = false;
  carrier->polling.entry.remove_requested = true;
  carrier->polling.window = IREE_NET_SHM_POLL_WINDOW_DEFAULT;

  return 0;
}

// Per-carrier drain logic called from the shared_wake scan callback.
//
// Returns true if the carrier should remain in the sleeping list (sleep mode),
// false if it was removed (transitioned to poll mode or stopped).
//
// Armed flag protocol (Dekker-fence):
//   1. Clear armed (we're awake and processing)
//   2. Drain RX ring (up to budget)
//   3. Set armed (going back to sleep)
//   4. seq_cst fence (Dekker — pairs with producer's fence in signal_peer)
//   5. Re-check: if data available, clear armed and loop to step 2
//   6. Return true to stay in sleeping list (shared_wake re-posts its wait)
//
// The double-check after arming (step 5) prevents the race where a producer
// writes between our "no data found" and "set armed."
//
// Peer shutdown (PEER_DONE from drain_rx) stops RX draining but does NOT stop
// the callback loop — only an explicit deactivate() causes cancellation.
//
// Budget exhaustion (BUDGET_HIT from drain_rx) breaks the Dekker loop and
// self-signals the shared_wake notification to guarantee re-entry on the next
// scan.
bool iree_net_shm_carrier_drain_from_wake(iree_net_shm_carrier_t* carrier) {
  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_DRAINING) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return false;  // Remove from sleeping list.
  }

  bool peer_rx_stopped = carrier->peer_shutdown_received;
  bool budget_hit = false;
  int32_t total_drained = 0;
  for (;;) {
    // Step 1: clear armed — we're actively processing.
    iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);

    // Step 2: drain RX ring (up to budget).
    if (!peer_rx_stopped) {
      iree_net_shm_drain_rx_result_t rx_result =
          iree_net_shm_carrier_drain_rx(carrier, IREE_NET_SHM_DRAIN_RX_BUDGET);
      total_drained += rx_result.drained_count;
      if (rx_result.status == IREE_NET_SHM_DRAIN_RX_PEER_DONE) {
        peer_rx_stopped = true;
      } else if (rx_result.status == IREE_NET_SHM_DRAIN_RX_BUDGET_HIT) {
        budget_hit = true;
      }
    }

    // Budget exhaustion: break the Dekker loop to yield to other proactor
    // work. The shared_wake will be self-signaled below to guarantee
    // re-entry on the next scan.
    if (budget_hit) break;

    // Step 3: set armed — going back to sleep.
    iree_atomic_store(carrier->our_armed, 1, iree_memory_order_release);

    // Step 4: seq_cst fence — Dekker pairing with signal_peer's fence.
    iree_atomic_thread_fence(iree_memory_order_seq_cst);

    // Step 5: re-check RX ring for committed data. If data appeared between
    // our last drain and the armed store, loop back (clear armed and process
    // it). We use peek (not can_read) because can_read returns true for
    // reserved-but-uncommitted entries, which would spin this loop while a
    // peer holds an outstanding begin_send reservation.
    iree_host_size_t recheck_length = 0;
    if (!peer_rx_stopped &&
        iree_mpsc_queue_peek(&carrier->rx_queue, &recheck_length) != NULL) {
      continue;
    }

    break;
  }

  // Decide whether to transition to poll mode, stay sleeping, or stop.
  state = iree_net_carrier_state(&carrier->base);
  if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
    // Transition to poll mode if traffic exceeds the threshold. In poll mode
    // the progress callback checks the MPSC ring directly each poll()
    // iteration, eliminating kernel notification overhead on the hot path.
    // The pending_operations slot transfers from the sleeping list to the
    // progress callback (no decrement/increment — count stays at 1).
    if (total_drained >= carrier->polling.mode_threshold) {
      carrier->polling.active = true;
      carrier->polling.idle_spin_count = 0;
      carrier->polling.window = IREE_NET_SHM_POLL_WINDOW_DEFAULT;
      carrier->polling.previous_was_full = false;
      // Clear armed — in poll mode we're actively checking the ring every
      // iteration so the peer must not waste syscalls signaling us. If the
      // Dekker loop exited normally (not budget-hit), armed may be 1 from
      // step 3; clear it unconditionally.
      iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);
      iree_async_proactor_register_progress(carrier->shared_wake->proactor,
                                            &carrier->polling.entry);
      return false;  // Remove from sleeping list (now in poll mode).
    }

    // Traffic below threshold — stay in sleep mode.
    if (budget_hit) {
      // Self-signal the shared_wake notification to ensure the scan fires
      // again promptly. We are NOT armed (cleared in step 1), so the peer's
      // signal_peer won't wake us — we must wake ourselves.
      iree_async_notification_signal(carrier->shared_wake->notification, 1);
    }
    return true;  // Stay in sleeping list.
  } else {
    iree_net_shm_carrier_drain_stop(carrier);
    return false;  // Remove from sleeping list (stopped).
  }
}

static iree_status_t iree_net_shm_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data);

//===----------------------------------------------------------------------===//
// Carrier vtable methods
//===----------------------------------------------------------------------===//

static void iree_net_shm_carrier_free(iree_net_shm_carrier_t* carrier) {
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_shm_carrier_drop_failure_status(carrier);
  iree_async_notification_release(carrier->peer_wake_notification);
  if (carrier->release.fn) {
    carrier->release.fn(carrier->release.context);
  }
  iree_net_shm_shared_wake_release(carrier->shared_wake);

  iree_allocator_t allocator = carrier->base.host_allocator;
  iree_allocator_free(allocator, carrier);
  IREE_TRACE_ZONE_END(z0);
}

static void iree_net_shm_carrier_deferred_destroy(void* user_data) {
  iree_net_shm_carrier_t* carrier = (iree_net_shm_carrier_t*)user_data;
  iree_net_shm_carrier_free(carrier);
}

static void iree_net_shm_carrier_destroy(iree_net_carrier_t* base_carrier) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);

  if (state == IREE_NET_CARRIER_STATE_ACTIVE) {
    // Carrier was never properly deactivated. Start async deactivation and
    // defer the actual free until all pending operations complete.
    base_carrier->recv_handler.fn = NULL;
    base_carrier->recv_handler.user_data = NULL;
    base_carrier->signal_handler.fn = NULL;
    base_carrier->signal_handler.user_data = NULL;
    iree_status_t status = iree_net_shm_carrier_deactivate(
        base_carrier, iree_net_shm_carrier_deferred_destroy, carrier);
    if (!iree_status_is_ok(status)) iree_status_abort(status);
    return;
  }

  if (state == IREE_NET_CARRIER_STATE_DRAINING) {
    // Already deactivating — replace the callback so we get notified when
    // draining completes.
    carrier->deactivate_callback.fn = iree_net_shm_carrier_deferred_destroy;
    carrier->deactivate_callback.user_data = carrier;
    return;
  }

  IREE_ASSERT(state == IREE_NET_CARRIER_STATE_DEACTIVATED ||
              state == IREE_NET_CARRIER_STATE_CREATED);
  iree_net_shm_carrier_free(carrier);
}

static void iree_net_shm_carrier_set_recv_handler(
    iree_net_carrier_t* base_carrier, iree_net_carrier_recv_handler_t handler) {
  base_carrier->recv_handler = handler;
}

// NOP callback for deferred sleeping list registration. Fires on the poll
// thread, making it safe to manipulate the sleeping list. If deactivation was
// requested between the NOP submission and this callback, the carrier skips
// registration and completes deactivation instead.
static void iree_net_shm_carrier_activate_on_poll(
    void* user_data, iree_async_operation_t* operation, iree_status_t status,
    iree_async_completion_flags_t flags) {
  (void)operation;
  (void)flags;
  iree_net_shm_carrier_t* carrier = (iree_net_shm_carrier_t*)user_data;
  if (!iree_status_is_ok(status)) iree_status_abort(status);

  iree_net_carrier_state_t state = iree_net_carrier_state(&carrier->base);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    // Deactivation was requested before we reached the poll thread. Don't
    // register — cancel any pending TX completions (which are also tracked in
    // pending_operations and would otherwise be orphaned since the carrier
    // never entered the sleeping list where drain logic runs), release the
    // NOP's pending_operations slot, and attempt deactivation completion.
    iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);
    iree_net_shm_carrier_drain_stop(carrier);
    return;
  }

  iree_status_t register_status =
      iree_net_shm_shared_wake_register(carrier->shared_wake, carrier);
  if (IREE_UNLIKELY(!iree_status_is_ok(register_status))) {
    // Registration failed — the carrier cannot receive data. Begin
    // deactivation so subsequent operations fail visibly rather than hanging.
    // This is a catastrophic failure (proactor cannot submit NOTIFICATION_WAIT
    // despite having just completed a NOP on the same proactor).
    iree_status_abort(register_status);
  }

  // Self-signal to catch data that arrived between activate() setting armed=1
  // and this NOP callback registering the carrier. The peer may have already
  // signaled (saw armed=1) but the signal was consumed by a scan that ran
  // before we were in the sleeping list. The self-signal forces a re-scan
  // that will find us and dispatch any pending data.
  iree_async_notification_signal(carrier->shared_wake->notification, 1);
}

static iree_status_t iree_net_shm_carrier_activate(
    iree_net_carrier_t* base_carrier) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in CREATED state to activate");
  }
  if (!base_carrier->recv_handler.fn) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "recv handler must be set before activation");
  }

  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_ACTIVE);

  // Start armed and submit a NOP to the proactor. The NOP callback fires on
  // the poll thread and registers the carrier in the sleeping list. This
  // ensures all sleeping list mutations happen on the poll thread.
  iree_atomic_store(carrier->our_armed, 1, iree_memory_order_release);
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);
  memset(&carrier->activate_nop, 0, sizeof(carrier->activate_nop));
  iree_async_operation_initialize(
      &carrier->activate_nop.base, IREE_ASYNC_OPERATION_TYPE_NOP,
      IREE_ASYNC_OPERATION_FLAG_NONE, iree_net_shm_carrier_activate_on_poll,
      carrier);
  iree_status_t status = iree_async_proactor_submit_one(
      carrier->shared_wake->proactor, &carrier->activate_nop.base);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) {
    // NOP submission failed — undo armed flag and pending_ops.
    iree_atomic_store(carrier->our_armed, 0, iree_memory_order_release);
    iree_atomic_fetch_sub(&base_carrier->pending_operations, 1,
                          iree_memory_order_release);
    iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_CREATED);
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_status_t iree_net_shm_carrier_deactivate(
    iree_net_carrier_t* base_carrier,
    iree_net_carrier_deactivate_callback_fn_t callback, void* user_data) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE &&
      state != IREE_NET_CARRIER_STATE_CREATED) {
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "carrier must be in ACTIVE or CREATED state to deactivate");
  }

  carrier->deactivate_callback.fn = callback;
  carrier->deactivate_callback.user_data = user_data;

  if (state == IREE_NET_CARRIER_STATE_CREATED) {
    iree_net_carrier_set_state(base_carrier,
                               IREE_NET_CARRIER_STATE_DEACTIVATED);
    if (callback) callback(user_data);
    IREE_TRACE_ZONE_END(z0);
    return iree_ok_status();
  }

  // Keep one pending count owned by this call until state publication and wake
  // submission are complete. This prevents an operation callback from
  // delivering the terminal callback while deactivate still accesses carrier
  // state, and makes the final retirement the lifetime boundary.
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  // Publish DRAINING and wake the poll thread. In sleep mode the shared-wake
  // scan removes the carrier; in poll mode the progress callback observes the
  // state on its next iteration. Signaling unconditionally avoids reading the
  // proactor-owned polling mode from an arbitrary deactivation caller thread.
  iree_net_carrier_set_state(base_carrier, IREE_NET_CARRIER_STATE_DRAINING);
  iree_async_notification_signal(carrier->shared_wake->notification, 1);

  IREE_TRACE_ZONE_END(z0);
  iree_net_shm_carrier_retire_pending_operation(carrier);
  return iree_ok_status();
}

static iree_net_carrier_send_budget_t iree_net_shm_carrier_query_send_budget(
    iree_net_carrier_t* base_carrier) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Ring buffer available bytes, adjusted for per-entry overhead. Each MPSC
  // entry consumes align_up(4 + payload, entry_alignment) bytes where the 4 is
  // the uint32_t length header. Our payload includes a 4-byte type header
  // before the user data, so the total ring cost for a send of N data bytes is
  // align_up(4 + 4 + N, 8). We subtract the maximum overhead (4 + 4 + 7 = 15)
  // so the reported budget is always achievable in a single send.
  iree_host_size_t ring_bytes =
      iree_mpsc_queue_write_available(&carrier->tx_queue);
  iree_host_size_t per_entry_overhead =
      sizeof(uint32_t) + sizeof(iree_net_shm_entry_header_t) + 7;

  // Slots are unlimited — send completions fire synchronously after data is
  // committed to the ring, so there are no in-flight completion entries.
  iree_net_carrier_send_budget_t budget;
  budget.slots = UINT32_MAX;
  budget.bytes =
      ring_bytes > per_entry_overhead ? ring_bytes - per_entry_overhead : 0;
  return budget;
}

static iree_net_carrier_send_budget_t
iree_net_shm_carrier_query_direct_write_budget(
    iree_net_carrier_t* base_carrier, iree_net_direct_write_flags_t flags) {
  iree_net_direct_write_flags_t known_flags =
      IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER;
  iree_net_carrier_send_budget_t budget = {0};
  if (iree_any_bit_set(flags, ~known_flags)) return budget;

  if (iree_net_carrier_state(base_carrier) != IREE_NET_CARRIER_STATE_ACTIVE) {
    return budget;
  }
  budget.bytes = IREE_HOST_SIZE_MAX;
  if (!iree_any_bit_set(flags, IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER)) {
    budget.slots = UINT32_MAX;
    return budget;
  }

  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  iree_host_size_t ring_entry_size =
      sizeof(iree_net_shm_entry_header_t) +
      sizeof(iree_net_shm_reference_descriptor_t);
  // Account for the uint32_t length prefix plus worst-case 8-byte alignment.
  iree_host_size_t per_entry_overhead = sizeof(uint32_t) + 7;
  iree_host_size_t total_entry_size = ring_entry_size + per_entry_overhead;
  iree_host_size_t available_entries =
      iree_mpsc_queue_write_available(&carrier->tx_queue) / total_entry_size;
  budget.slots =
      available_entries > UINT32_MAX ? UINT32_MAX : (uint32_t)available_entries;
  return budget;
}

static iree_status_t iree_net_shm_carrier_send(
    iree_net_carrier_t* base_carrier, const iree_net_send_params_t* params) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Overflow-checked scatter-gather size accumulation. Validated before
  // touching any carrier state so early errors have no side effects.
  iree_host_size_t total_size = 0;
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    if (IREE_UNLIKELY(!iree_host_size_checked_add(
            total_size, params->data.values[i].length, &total_size))) {
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "scatter-gather total size overflow");
    }
  }
  if (total_size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }
  iree_host_size_t ring_entry_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_net_shm_entry_header_t), total_size, &ring_entry_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring entry size overflow");
  }

  // Increment pending_operations BEFORE state check (TOCTOU protection).
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  if (iree_atomic_load(&carrier->shutdown_initiated,
                       iree_memory_order_seq_cst)) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  iree_mpsc_queue_reservation_t reservation;
  uint8_t* payload = (uint8_t*)iree_mpsc_queue_begin_write(
      &carrier->tx_queue, ring_entry_size, &reservation);
  if (!payload) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TX ring buffer full");
  }

  // Post-CAS shutdown check. If shutdown set shutdown_initiated (seq_cst store)
  // before our CAS on reserve_position, this seq_cst load sees 1 and we cancel
  // to prevent data from appearing after the EOS marker. If shutdown set it
  // after our CAS, our data precedes the EOS marker in reservation order — the
  // peer will drain it before reaching EOS.
  if (IREE_UNLIKELY(iree_atomic_load(&carrier->shutdown_initiated,
                                     iree_memory_order_seq_cst))) {
    iree_mpsc_queue_cancel_write(&carrier->tx_queue, reservation);
    iree_net_shm_signal_peer(carrier);
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  iree_net_shm_entry_header_t* header = (iree_net_shm_entry_header_t*)payload;
  header->type = IREE_NET_SHM_ENTRY_TYPE_INLINE;
  uint8_t* write_ptr = payload + sizeof(*header);
  for (iree_host_size_t i = 0; i < params->data.count; ++i) {
    memcpy(write_ptr, iree_async_span_ptr(params->data.values[i]),
           params->data.values[i].length);
    write_ptr += params->data.values[i].length;
  }

  iree_mpsc_queue_commit_write(&carrier->tx_queue, reservation);

  // Signal the peer that new data is available. The peer's notification wait
  // will fire, and its drain callback will process the data.
  iree_net_shm_signal_peer(carrier);

  // Fire completion synchronously — the data is already copied into the ring,
  // so the caller's buffers are immediately free to reuse. This matches TCP's
  // "commit to transport" semantics (data accepted by the transport layer).
  iree_atomic_fetch_add(&base_carrier->bytes_sent, (int64_t)total_size,
                        iree_memory_order_relaxed);
  if (base_carrier->callback.fn) {
    base_carrier->callback.fn(
        base_carrier->callback.user_data, IREE_NET_CARRIER_COMPLETION_SEND,
        params->user_data, iree_ok_status(), total_size, NULL);
  }

  iree_net_shm_carrier_retire_pending_operation(carrier);

  return iree_ok_status();
}

static iree_status_t iree_net_shm_carrier_begin_send(
    iree_net_carrier_t* base_carrier, iree_host_size_t size, void** out_ptr,
    iree_net_carrier_send_handle_t* out_handle) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  if (size == 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "empty sends are not allowed");
  }
  iree_host_size_t ring_entry_size = 0;
  if (IREE_UNLIKELY(!iree_host_size_checked_add(
          sizeof(iree_net_shm_entry_header_t), size, &ring_entry_size))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "ring entry size overflow");
  }

  // Increment pending_operations BEFORE state check (TOCTOU protection).
  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to send");
  }

  if (iree_atomic_load(&carrier->shutdown_initiated,
                       iree_memory_order_seq_cst)) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  iree_mpsc_queue_reservation_t reservation;
  uint8_t* payload = (uint8_t*)iree_mpsc_queue_begin_write(
      &carrier->tx_queue, ring_entry_size, &reservation);
  if (!payload) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TX ring buffer full");
  }

  // Post-CAS shutdown check (see send() for rationale).
  if (IREE_UNLIKELY(iree_atomic_load(&carrier->shutdown_initiated,
                                     iree_memory_order_seq_cst))) {
    iree_mpsc_queue_cancel_write(&carrier->tx_queue, reservation);
    iree_net_shm_signal_peer(carrier);
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier has been shut down for sending");
  }

  iree_net_shm_entry_header_t* header = (iree_net_shm_entry_header_t*)payload;
  header->type = IREE_NET_SHM_ENTRY_TYPE_INLINE;
  *out_ptr = payload + sizeof(*header);

  // Pack the MPSC reservation into the handle. The reservation is 8 bytes
  // (two uint32_t fields), same size as the uint64_t handle.
  iree_net_carrier_send_handle_t handle;
  memcpy(&handle, &reservation, sizeof(handle));
  *out_handle = handle;
  return iree_ok_status();
}

static iree_status_t iree_net_shm_carrier_commit_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Unpack the MPSC reservation from the handle.
  iree_mpsc_queue_reservation_t reservation;
  memcpy(&reservation, &handle, sizeof(reservation));

  iree_mpsc_queue_commit_write(&carrier->tx_queue, reservation);

  iree_net_shm_signal_peer(carrier);

  // Update stats: reservation.payload_length is ring_entry_size which includes
  // the SHM entry header, so subtract it to recover the user data size.
  iree_host_size_t data_size =
      (iree_host_size_t)(reservation.payload_length -
                         sizeof(iree_net_shm_entry_header_t));
  iree_atomic_fetch_add(&base_carrier->bytes_sent, (int64_t)data_size,
                        iree_memory_order_relaxed);

  iree_net_shm_carrier_retire_pending_operation(carrier);

  return iree_ok_status();
}

static void iree_net_shm_carrier_abort_send(
    iree_net_carrier_t* base_carrier, iree_net_carrier_send_handle_t handle) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Unpack the MPSC reservation from the handle.
  iree_mpsc_queue_reservation_t reservation;
  memcpy(&reservation, &handle, sizeof(reservation));

  iree_mpsc_queue_cancel_write(&carrier->tx_queue, reservation);

  // Wake the consumer past the canceled entry to prevent head-of-line blocking.
  iree_net_shm_signal_peer(carrier);

  iree_net_shm_carrier_retire_pending_operation(carrier);
}

static iree_status_t iree_net_shm_carrier_shutdown(
    iree_net_carrier_t* base_carrier) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state to shutdown");
  }

  // Atomically set shutdown_initiated. seq_cst ordering pairs with the seq_cst
  // load in send()/begin_send() to prevent data from being committed after the
  // EOS marker in the ring. If the flag was already set (prior call that may
  // have failed to enqueue the marker), we try the marker again.
  iree_atomic_store(&carrier->shutdown_initiated, 1, iree_memory_order_seq_cst);

  // Write a header-only shutdown marker to the TX ring.
  iree_mpsc_queue_reservation_t reservation;
  iree_net_shm_entry_header_t* marker_header =
      (iree_net_shm_entry_header_t*)iree_mpsc_queue_begin_write(
          &carrier->tx_queue, sizeof(*marker_header), &reservation);
  if (!marker_header) {
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TX ring full, cannot enqueue shutdown marker");
  }
  marker_header->type = IREE_NET_SHM_ENTRY_TYPE_SHUTDOWN;
  iree_mpsc_queue_commit_write(&carrier->tx_queue, reservation);

  iree_net_shm_signal_peer(carrier);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Buffer registration and direct access
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_carrier_query_region(
    iree_net_carrier_t* base_carrier, uint32_t region_id,
    iree_net_shm_region_info_t* out_region_info) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  if (region_id >= carrier->region_count) {
    memset(out_region_info, 0, sizeof(*out_region_info));
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "region_id %" PRIu32 " >= region_count %" PRIhsz,
                            region_id, carrier->region_count);
  }
  *out_region_info = carrier->regions[region_id];
  return iree_ok_status();
}

static iree_status_t iree_net_shm_carrier_register_buffer(
    iree_net_carrier_t* base_carrier, iree_async_region_t* region,
    iree_net_remote_handle_t* out_handle) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  *out_handle = iree_net_remote_handle_null();

  uint8_t* region_base = (uint8_t*)region->base_ptr;
  iree_host_size_t region_length = region->length;

  for (iree_host_size_t i = 0; i < carrier->region_count; ++i) {
    uint8_t* shm_base = (uint8_t*)carrier->regions[i].base_ptr;
    iree_host_size_t shm_size = carrier->regions[i].size;
    if (region_base >= shm_base &&
        region_base + region_length <= shm_base + shm_size) {
      iree_host_size_t offset = (iree_host_size_t)(region_base - shm_base);
      iree_net_remote_handle_t handle = {{(uint64_t)i, (uint64_t)offset}};
      *out_handle = handle;
      return iree_ok_status();
    }
  }

  return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                          "buffer [%p, +%" PRIhsz
                          ") does not fall within any registered SHM "
                          "region",
                          region_base, region_length);
}

static void iree_net_shm_carrier_unregister_buffer(
    iree_net_carrier_t* base_carrier, iree_net_remote_handle_t handle) {
  // SHM regions have no kernel resources to release. Validate the handle to
  // catch misuse (e.g., double-unregister or handles from a different carrier).
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  IREE_ASSERT((uint64_t)handle.opaque[0] < carrier->region_count,
              "unregister_buffer: region_id %" PRIu64 " out of range",
              handle.opaque[0]);
  (void)carrier;
}

// Resolves a remote handle to a local SHM pointer and bounds-checks the
// requested length. Returns NULL and sets |out_status| on failure.
static uint8_t* iree_net_shm_resolve_handle(iree_net_shm_carrier_t* carrier,
                                            iree_net_remote_handle_t handle,
                                            iree_host_size_t length,
                                            iree_status_t* out_status) {
  uint64_t region_id = handle.opaque[0];
  uint64_t offset = handle.opaque[1];
  if (IREE_UNLIKELY(region_id >= carrier->region_count)) {
    *out_status =
        iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                         "region_id %" PRIu64 " >= region_count %" PRIhsz,
                         region_id, carrier->region_count);
    return NULL;
  }
  iree_net_shm_region_info_t* region = &carrier->regions[region_id];
  uint64_t range_end = 0;
  if (IREE_UNLIKELY(
          !iree_checked_add_u64(offset, (uint64_t)length, &range_end) ||
          range_end > region->size)) {
    *out_status = iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                                   "offset %" PRIu64 " + length %" PRIhsz
                                   " exceeds region size %" PRIhsz,
                                   offset, length, region->size);
    return NULL;
  }
  *out_status = iree_ok_status();
  return (uint8_t*)region->base_ptr + offset;
}

static iree_status_t iree_net_shm_carrier_direct_write(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_write_params_t* params) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Resolve and bounds-check the remote handle.
  iree_status_t status;
  uint8_t* destination = iree_net_shm_resolve_handle(
      carrier, params->remote, params->local.length, &status);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) return status;

  // Copy local data to SHM destination.
  memcpy(destination, iree_async_span_ptr(params->local), params->local.length);

  if (!(params->flags & IREE_NET_DIRECT_WRITE_FLAG_SIGNAL_RECEIVER)) {
    // Non-signaling write: no ring entry, no completion callback. The write
    // is synchronous — data is in SHM when we return.
    return iree_ok_status();
  }

  // Signaling write: write a REFERENCE entry to the TX ring so the peer's
  // drain callback delivers the immediate value. Completion fires
  // synchronously after the reference entry is committed to the ring.
  iree_host_size_t ring_entry_size =
      sizeof(iree_net_shm_entry_header_t) +
      sizeof(iree_net_shm_reference_descriptor_t);

  iree_atomic_fetch_add(&base_carrier->pending_operations, 1,
                        iree_memory_order_acq_rel);

  iree_net_carrier_state_t state = iree_net_carrier_state(base_carrier);
  if (state != IREE_NET_CARRIER_STATE_ACTIVE) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "carrier must be in ACTIVE state for "
                            "signaling direct_write");
  }

  iree_mpsc_queue_reservation_t reservation;
  uint8_t* payload = (uint8_t*)iree_mpsc_queue_begin_write(
      &carrier->tx_queue, ring_entry_size, &reservation);
  if (!payload) {
    iree_net_shm_carrier_retire_pending_operation(carrier);
    return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                            "TX ring buffer full");
  }

  iree_net_shm_entry_header_t* header = (iree_net_shm_entry_header_t*)payload;
  header->type = IREE_NET_SHM_ENTRY_TYPE_REFERENCE;
  iree_net_shm_reference_descriptor_t* descriptor =
      (iree_net_shm_reference_descriptor_t*)(payload + sizeof(*header));
  *descriptor = (iree_net_shm_reference_descriptor_t){
      .region_id = (uint32_t)params->remote.opaque[0],
      .reserved = params->immediate,
      .offset = params->remote.opaque[1],
      .length = params->local.length,
  };

  iree_mpsc_queue_commit_write(&carrier->tx_queue, reservation);

  iree_net_shm_signal_peer(carrier);

  // Fire completion synchronously — the reference entry is committed to the
  // ring and the data was already memcpy'd to SHM above.
  iree_atomic_fetch_add(&base_carrier->bytes_sent,
                        (int64_t)params->local.length,
                        iree_memory_order_relaxed);
  if (base_carrier->callback.fn) {
    base_carrier->callback.fn(base_carrier->callback.user_data,
                              IREE_NET_CARRIER_COMPLETION_DIRECT_WRITE,
                              params->user_data, iree_ok_status(),
                              params->local.length, NULL);
  }

  iree_net_shm_carrier_retire_pending_operation(carrier);

  return iree_ok_status();
}

static iree_status_t iree_net_shm_carrier_direct_read(
    iree_net_carrier_t* base_carrier,
    const iree_net_direct_read_params_t* params) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);

  // Resolve and bounds-check the remote handle.
  iree_status_t status;
  uint8_t* source = iree_net_shm_resolve_handle(carrier, params->remote,
                                                params->local.length, &status);
  if (IREE_UNLIKELY(!iree_status_is_ok(status))) return status;

  // Copy SHM source to local destination. Direct reads are synchronous —
  // no ring entry, no peer notification, no completion callback.
  memcpy(iree_async_span_ptr(params->local), source, params->local.length);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// File handle transfer
//===----------------------------------------------------------------------===//

static iree_status_t iree_net_shm_carrier_query_file_handle_transfer(
    iree_net_carrier_t* base_carrier, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t* out_transfer_type,
    iree_host_size_t* out_payload_length) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  return iree_net_shm_file_transfer_query(carrier->file_transfer, file_handle,
                                          out_transfer_type,
                                          out_payload_length);
}

static iree_status_t iree_net_shm_carrier_export_file_handle(
    iree_net_carrier_t* base_carrier, iree_io_file_handle_t* file_handle,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_byte_span_t transfer_payload) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  return iree_net_shm_file_transfer_export(carrier->file_transfer, file_handle,
                                           transfer_type, transfer_payload);
}

static iree_status_t iree_net_shm_carrier_import_file_handle(
    iree_net_carrier_t* base_carrier,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload, iree_allocator_t host_allocator,
    iree_io_file_handle_t** out_file_handle) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  return iree_net_shm_file_transfer_import(carrier->file_transfer,
                                           transfer_type, transfer_payload,
                                           host_allocator, out_file_handle);
}

static iree_status_t iree_net_shm_carrier_release_file_handle_transfer(
    iree_net_carrier_t* base_carrier,
    iree_net_file_handle_transfer_type_t transfer_type,
    iree_const_byte_span_t transfer_payload) {
  iree_net_shm_carrier_t* carrier = iree_net_shm_carrier_cast(base_carrier);
  return iree_net_shm_file_transfer_release_export(
      carrier->file_transfer, transfer_type, transfer_payload);
}

//===----------------------------------------------------------------------===//
// Vtable
//===----------------------------------------------------------------------===//

static const iree_net_carrier_vtable_t iree_net_shm_carrier_vtable = {
    .destroy = iree_net_shm_carrier_destroy,
    .set_recv_handler = iree_net_shm_carrier_set_recv_handler,
    .activate = iree_net_shm_carrier_activate,
    .deactivate = iree_net_shm_carrier_deactivate,
    .query_send_budget = iree_net_shm_carrier_query_send_budget,
    .send = iree_net_shm_carrier_send,
    .begin_send = iree_net_shm_carrier_begin_send,
    .commit_send = iree_net_shm_carrier_commit_send,
    .abort_send = iree_net_shm_carrier_abort_send,
    .shutdown = iree_net_shm_carrier_shutdown,
    .direct_write = iree_net_shm_carrier_direct_write,
    .direct_read = iree_net_shm_carrier_direct_read,
    .register_buffer = iree_net_shm_carrier_register_buffer,
    .unregister_buffer = iree_net_shm_carrier_unregister_buffer,
    .query_file_handle_transfer =
        iree_net_shm_carrier_query_file_handle_transfer,
    .export_file_handle = iree_net_shm_carrier_export_file_handle,
    .import_file_handle = iree_net_shm_carrier_import_file_handle,
    .release_file_handle_transfer =
        iree_net_shm_carrier_release_file_handle_transfer,
    .query_direct_write_budget = iree_net_shm_carrier_query_direct_write_budget,
};

//===----------------------------------------------------------------------===//
// iree_net_shm_carrier_create
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t iree_net_shm_carrier_create(
    const iree_net_shm_carrier_create_params_t* params,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier) {
  IREE_ASSERT_ARGUMENT(params);
  IREE_ASSERT_ARGUMENT(params->shared_wake);
  IREE_ASSERT_ARGUMENT(params->peer_wake_notification);
  IREE_ASSERT_ARGUMENT(params->our_armed);
  IREE_ASSERT_ARGUMENT(params->peer_armed);
  IREE_ASSERT_ARGUMENT(out_carrier);
  IREE_TRACE_ZONE_BEGIN(z0);
  *out_carrier = NULL;

  // Compute allocation size with overflow-checked FAM layout.
  iree_host_size_t total_size = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, IREE_STRUCT_LAYOUT(
              iree_sizeof_struct(iree_net_shm_carrier_t), &total_size,
              IREE_STRUCT_FIELD_FAM(params->region_count,
                                    iree_net_shm_region_info_t)));

  iree_net_shm_carrier_t* carrier = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, total_size, (void**)&carrier));

  memset(carrier, 0, total_size);

  iree_net_carrier_capabilities_t capabilities =
      IREE_NET_CARRIER_CAPABILITY_RELIABLE |
      IREE_NET_CARRIER_CAPABILITY_ORDERED |
      IREE_NET_CARRIER_CAPABILITY_ZERO_COPY_TX;
  if (params->region_count > 0) {
    capabilities |= IREE_NET_CARRIER_CAPABILITY_REGISTERED_REGIONS |
                    IREE_NET_CARRIER_CAPABILITY_DIRECT_WRITE |
                    IREE_NET_CARRIER_CAPABILITY_DIRECT_READ;
  }
  if (params->file_transfer) {
    switch (iree_net_shm_file_transfer_type(params->file_transfer)) {
      case IREE_NET_FILE_HANDLE_TRANSFER_TYPE_POSIX_FD:
        capabilities |= IREE_NET_CARRIER_CAPABILITY_POSIX_FD_TRANSFER;
        break;
      case IREE_NET_FILE_HANDLE_TRANSFER_TYPE_WIN32_HANDLE:
        capabilities |= IREE_NET_CARRIER_CAPABILITY_WIN32_HANDLE_TRANSFER;
        break;
      default:
        break;
    }
  }
  iree_net_carrier_initialize(&iree_net_shm_carrier_vtable, capabilities, 0,
                              SIZE_MAX, callback, host_allocator,
                              &carrier->base);

  carrier->is_client = params->is_client;
  carrier->release.fn = params->release_context_fn;
  carrier->release.context = params->release_context;
  carrier->tx_queue = params->tx_queue;
  carrier->rx_queue = params->rx_queue;
  carrier->shared_wake = params->shared_wake;
  iree_net_shm_shared_wake_retain(params->shared_wake);
  carrier->peer_wake_notification = params->peer_wake_notification;
  iree_async_notification_retain(params->peer_wake_notification);
  carrier->our_armed = params->our_armed;
  carrier->peer_armed = params->peer_armed;
  // Initialize adaptive polling state.
  carrier->polling.entry.fn = iree_net_shm_carrier_progress;
  carrier->polling.entry.user_data = carrier;
  carrier->polling.entry.next = NULL;
  carrier->polling.entry.remove_requested = false;
  carrier->polling.entry.on_remove = iree_net_shm_carrier_progress_removed;
  carrier->polling.active = false;
  carrier->polling.retire_on_remove = false;
  carrier->polling.idle_spin_count = 0;
  carrier->polling.window = IREE_NET_SHM_POLL_WINDOW_DEFAULT;
  carrier->polling.previous_was_full = false;
  carrier->polling.mode_threshold =
      params->poll_mode_threshold > 0 ? params->poll_mode_threshold : 1;

  // Copy region table from params.
  carrier->region_count = params->region_count;
  carrier->file_transfer = params->file_transfer;
  for (iree_host_size_t i = 0; i < params->region_count; ++i) {
    carrier->regions[i] = params->regions[i];
  }

  *out_carrier = &carrier->base;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}
