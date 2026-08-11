// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared memory carrier: high-performance IPC via MPSC rings in shared memory.
//
// The SHM carrier provides zero-kernel-copy data transfer between two endpoints
// (typically in different processes) using MPSC ring buffers in a shared memory
// region. A single shared notification per carrier drives the proactor poll
// loop to drain received data and fire send completion callbacks.
//
// Design:
//   - Two carriers share a single SHM region (connected via create_pair or a
//     cross-process factory). The region contains two MPSC queues (one per
//     direction) plus armed flags for adaptive signaling.
//   - send() reserves MPSC queue entries directly from concurrent producers
//     and writes data inline into the TX queue. Completion fires synchronously
//     after the data is committed to the queue, matching TCP's "commit to
//     transport" semantics. The caller's buffers are copied during send() and
//     immediately free.
//   - A single NOTIFICATION_WAIT per carrier wakes the proactor when the peer
//     signals. The drain callback checks the RX ring for new data — a single
//     acquire-load, so spurious wakes cost ~50ns.
//
// Signaling:
//   Each carrier has one shared notification (epoch in SHM, event pair for
//   proactor integration). The armed flag in SHM controls adaptive wake:
//   a Dekker-fence protocol ensures no missed wakes between the producer's
//   write and the consumer's arm-then-sleep sequence.
//
//   This is the same code path for both create_pair (in-process testing) and
//   cross-process operation via a factory. The only difference is how handles
//   and SHM are exchanged — the carrier-level code is identical.
//
// Ring entry framing (layered on MPSC queue's length-prefixed entries):
//   length >= 4: [uint32_t type][type-specific payload]
//     type 0x00 (INLINE):    data bytes (memcpy'd from scatter-gather spans)
//     type 0x01 (REFERENCE): region descriptor for zero-copy direct_write
//     type 0x02 (SHUTDOWN):  peer initiated graceful shutdown
//
// MPSC entries begin on 8-byte boundaries and have a 4-byte length prefix. The
// 4-byte SHM type header therefore places type-specific payloads on the next
// 8-byte boundary, satisfying IREE_NET_MESSAGE_ALIGNMENT for inline data and
// naturally aligning reference descriptors.
//
// Capabilities:
//   - RELIABLE: no drops (shared memory with explicit queue backpressure).
//   - ORDERED: FIFO by MPSC reservation order.
//   - ZERO_COPY_TX: advertised for large-data CTS tests; inline sends copy but
//     one-sided writes into registered buffers avoid the copy entirely.
//
// Thread safety:
//   - send() is thread-safe across concurrent MPSC producers.
//   - All RX delivery and TX completion callbacks fire from the proactor
//   thread.

#ifndef IREE_NET_CARRIER_SHM_CARRIER_H_
#define IREE_NET_CARRIER_SHM_CARRIER_H_

#include "iree/async/notification.h"
#include "iree/async/proactor.h"
#include "iree/base/api.h"
#include "iree/base/internal/mpsc_queue.h"
#include "iree/net/carrier.h"

typedef struct iree_net_shm_shared_wake_t iree_net_shm_shared_wake_t;
typedef struct iree_net_shm_file_transfer_t iree_net_shm_file_transfer_t;

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Default ring capacity in bytes (256KB). Must be a power of two.
#define IREE_NET_SHM_CARRIER_DEFAULT_RING_CAPACITY ((uint32_t)(256 * 1024))

// Number of consecutive empty progress callback iterations before the carrier
// transitions from poll mode back to notification-based sleep mode.
// At ~50ns per poll iteration, 256 iterations ≈ 13µs of spinning.
#define IREE_NET_SHM_IDLE_SPIN_THRESHOLD 256

// AIMD adaptive window bounds for progress callback batching.
// Controls how many RX entries are drained per progress callback invocation.
// Additive increase (+1) after two consecutive full windows; multiplicative
// decrease (/2) on underconsumed polls. Matches UCX mm transport behavior.
#define IREE_NET_SHM_POLL_WINDOW_MIN 1
#define IREE_NET_SHM_POLL_WINDOW_MAX 256
#define IREE_NET_SHM_POLL_WINDOW_DEFAULT IREE_NET_SHM_POLL_WINDOW_MAX

// Inline data: payload bytes follow the type tag directly.
#define IREE_NET_SHM_ENTRY_TYPE_INLINE ((uint32_t)0x00)

// Reference descriptor: a region ID + offset + length identifying data in a
// registered shared memory region. The consumer delivers |reserved| to the
// signal handler so protocols can locate or interpret the written bytes.
#define IREE_NET_SHM_ENTRY_TYPE_REFERENCE ((uint32_t)0x01)

// Shutdown marker: signals the peer that no more data will be sent. The
// consumer stops draining upon encountering this entry.
#define IREE_NET_SHM_ENTRY_TYPE_SHUTDOWN ((uint32_t)0x02)

// Fixed ring entry header preceding every type-specific payload.
typedef struct iree_net_shm_entry_header_t {
  // IREE_NET_SHM_ENTRY_TYPE_* value identifying the payload representation.
  uint32_t type;
} iree_net_shm_entry_header_t;
static_assert(sizeof(iree_net_shm_entry_header_t) == 4, "");

// Reference descriptor for zero-copy sends via registered shared memory
// regions. Written into the ring as the payload of a REFERENCE entry.
// Uses 64-bit offset and length to match iree_async_span_t's iree_host_size_t
// fields and avoid artificial 4GB caps on registered region ranges.
typedef struct iree_net_shm_reference_descriptor_t {
  uint32_t region_id;  // Assigned during register_buffer.
  uint32_t reserved;   // Direct-write signal immediate value.
  uint64_t offset;     // Byte offset within the registered region.
  uint64_t length;     // Byte length of the referenced data.
} iree_net_shm_reference_descriptor_t;

// Describes a memory region known to the carrier for buffer registration
// and direct read/write. Both sides of a carrier pair must populate matching
// region IDs — the region_id in a remote handle is the array index.
typedef struct iree_net_shm_region_info_t {
  void* base_ptr;
  iree_host_size_t size;
} iree_net_shm_region_info_t;

// Mode bits for SHM carrier behavior. Bitfield for future extension.
// New modes are added as new bit values — no API churn.
typedef uint32_t iree_net_shm_carrier_mode_t;
enum iree_net_shm_carrier_mode_bits_e {
  IREE_NET_SHM_CARRIER_MODE_DEFAULT = 0u,
};

typedef struct iree_net_shm_carrier_options_t {
  // Ring buffer data capacity in bytes. Must be a power of two and >=
  // IREE_MPSC_QUEUE_MIN_CAPACITY.
  uint32_t ring_capacity;
  // Mode bitfield controlling carrier behavior.
  iree_net_shm_carrier_mode_t mode;
  // Maximum endpoints per connection. Each open_endpoint() call consumes one
  // slot. Sessions need one control endpoint plus any application endpoints
  // declared in their session options. Used by the factory when creating
  // connections; individual carriers ignore this field.
  uint16_t max_endpoint_count;
} iree_net_shm_carrier_options_t;

// Returns default options.
static inline iree_net_shm_carrier_options_t
iree_net_shm_carrier_options_default(void) {
  iree_net_shm_carrier_options_t options;
  memset(&options, 0, sizeof(options));
  options.ring_capacity = IREE_NET_SHM_CARRIER_DEFAULT_RING_CAPACITY;
  options.max_endpoint_count = 4;
  return options;
}

//===----------------------------------------------------------------------===//
// iree_net_shm_carrier_create
//===----------------------------------------------------------------------===//

// Parameters for creating a single SHM carrier from pre-assembled
// dependencies. All pointer fields must remain valid for the lifetime of the
// carrier (the carrier retains the shared_wake, peer_wake_notification,
// and calls release_context_fn during destroy).
typedef struct iree_net_shm_carrier_create_params_t {
  bool is_client;
  iree_mpsc_queue_t tx_queue;
  iree_mpsc_queue_t rx_queue;
  // Our proactor's shared wake (owns notification + sleeping carrier list).
  // Retained by the carrier. Provides the proactor reference.
  iree_net_shm_shared_wake_t* shared_wake;
  // Peer proactor's shared notification (signaled to wake the peer's scan).
  // Retained by the carrier.
  iree_async_notification_t* peer_wake_notification;
  // Adaptive signaling flag pointers into the SHM region. Each must be on its
  // own cache line (64-byte aligned) in the shared region.
  iree_atomic_int32_t* our_armed;
  iree_atomic_int32_t* peer_armed;
  // Opaque ownership context released when the carrier is destroyed. Called
  // after the carrier allocation is freed. For create_pair: a ref-counted pair
  // context holding SHM mappings. For factory-created carriers: a per-carrier
  // resource bundle. May be NULL if no cleanup is needed.
  void* release_context;
  void (*release_context_fn)(void* context);

  // Minimum number of RX entries drained in a single notification callback
  // before the carrier transitions from sleep mode to inline poll mode.
  // 0 or 1 = transition on first data arrival (lowest latency).
  // Higher values avoid spinning for low-rate traffic — e.g., 64 means "only
  // spin-poll when receiving bursts of 64+ entries per wake."
  // Default: 1 (immediate transition to poll mode on first data).
  int32_t poll_mode_threshold;

  // Known SHM regions for buffer registration and direct read/write.
  // Region 0 is typically the carrier's main SHM region. Both sides of a
  // carrier pair must populate matching region IDs. When region_count > 0,
  // the carrier advertises REGISTERED_REGIONS | DIRECT_WRITE | DIRECT_READ
  // capabilities.
  const iree_net_shm_region_info_t* regions;
  // Number of entries in |regions|.
  iree_host_size_t region_count;

  // Optional sideband used to transfer external file handle rights.
  iree_net_shm_file_transfer_t* file_transfer;
} iree_net_shm_carrier_create_params_t;

// Creates a single SHM carrier from pre-assembled dependencies.
//
// The carrier retains |params->shared_wake| and
// |params->peer_wake_notification|. It takes ownership of one reference to
// |params->release_context| (will call release_context_fn during destroy).
//
// Capabilities: RELIABLE | ORDERED | ZERO_COPY_TX (always), plus
// REGISTERED_REGIONS | DIRECT_WRITE | DIRECT_READ when regions are provided.
//
// |callback| receives send completion notifications. For inline sends, the
// callback fires synchronously during send() after data is committed to the
// ring — the caller's buffers are immediately free to reuse. For signaling
// direct_write, the callback fires synchronously after the reference entry
// is committed. Pass a callback with fn=NULL to skip completion callbacks.
IREE_API_EXPORT iree_status_t iree_net_shm_carrier_create(
    const iree_net_shm_carrier_create_params_t* params,
    iree_net_carrier_callback_t callback, iree_allocator_t host_allocator,
    iree_net_carrier_t** out_carrier);

// Returns the base pointer and size of a registered region by index.
// This is an SHM-specific API (not on the base carrier vtable) used by tests
// and buffer pool creation to discover the SHM layout.
//
// Returns IREE_STATUS_OUT_OF_RANGE if |region_id| >= the carrier's region
// count.
IREE_API_EXPORT iree_status_t iree_net_shm_carrier_query_region(
    iree_net_carrier_t* base_carrier, uint32_t region_id,
    iree_net_shm_region_info_t* out_region_info);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CARRIER_SHM_CARRIER_H_
