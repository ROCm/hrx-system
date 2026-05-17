// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Bulk channel wire format.
//
// Bulk frames carry large data transfers with chunking and progress tracking.
// The format is optimized for transfers exceeding 4GB with minimal overhead.
// CREDIT frames provide channel-level receiver flow control for DATA chunks by
// carrying the sender's cumulative receive credit grant.
//
// ## Frame header layout
//
//   ┌────────────────────────────────────────────────────────────────────────┐
//   │ Bytes 0-3:   magic (0x42455249 "IREB" - IREE Bulk)                     │
//   │ Byte 4:      version (currently 1)                                     │
//   │ Byte 5:      type (see iree_net_bulk_frame_type_t)                     │
//   │ Byte 6:      flags                                                     │
//   │ Byte 7:      reserved (must be 0)                                      │
//   │ Bytes 8-15:  transfer_id (little-endian uint64)                        │
//   │ Bytes 16-23: total_size (little-endian uint64, START/cumulative CREDIT)│
//   │ Bytes 24-31: chunk_offset (little-endian uint64)                       │
//   │ Bytes 32-35: chunk_length (little-endian uint32)                       │
//   │ Bytes 36-39: sequence (little-endian uint32, for datagram ordering)    │
//   └────────────────────────────────────────────────────────────────────────┘
//
#ifndef IREE_NET_CHANNEL_BULK_FRAME_H_
#define IREE_NET_CHANNEL_BULK_FRAME_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

// Bulk frame magic bytes ("IREB" - IREE Bulk, little-endian).
#define IREE_NET_BULK_FRAME_MAGIC 0x42455249u

// Current bulk frame format version.
#define IREE_NET_BULK_FRAME_VERSION 1

// Size of the bulk frame header in bytes.
#define IREE_NET_BULK_FRAME_HEADER_SIZE 40

//===----------------------------------------------------------------------===//
// Frame types
//===----------------------------------------------------------------------===//

// Bulk frame type identifiers.
typedef enum iree_net_bulk_frame_type_e {
  // Transfer start: announces a new transfer with total_size.
  IREE_NET_BULK_FRAME_TYPE_START = 0x01,

  // Data chunk: carries chunk_length bytes at chunk_offset.
  IREE_NET_BULK_FRAME_TYPE_DATA = 0x02,

  // Transfer complete: signals successful completion.
  IREE_NET_BULK_FRAME_TYPE_COMPLETE = 0x03,

  // Transfer abort: signals transfer cancellation.
  IREE_NET_BULK_FRAME_TYPE_ABORT = 0x04,

  // Receiver credit: announces the cumulative DATA chunk receive credit grant.
  IREE_NET_BULK_FRAME_TYPE_CREDIT = 0x05,
} iree_net_bulk_frame_type_e;
typedef uint8_t iree_net_bulk_frame_type_t;

//===----------------------------------------------------------------------===//
// Frame flags
//===----------------------------------------------------------------------===//

// Bulk frame flag bits.
typedef enum iree_net_bulk_frame_flag_bits_e {
  IREE_NET_BULK_FRAME_FLAG_NONE = 0u,

  // Payload is compressed.
  IREE_NET_BULK_FRAME_FLAG_COMPRESSED = 1u << 0,

  // This is the final chunk (can deliver before all chunks received).
  IREE_NET_BULK_FRAME_FLAG_FINAL_CHUNK = 1u << 2,
} iree_net_bulk_frame_flag_bits_e;
typedef uint8_t iree_net_bulk_frame_flags_t;

//===----------------------------------------------------------------------===//
// Frame header
//===----------------------------------------------------------------------===//

// On-wire bulk frame header. All fields are little-endian.
typedef struct iree_net_bulk_frame_header_t {
  uint32_t magic;         // IREE_NET_BULK_FRAME_MAGIC
  uint8_t version;        // IREE_NET_BULK_FRAME_VERSION
  uint8_t type;           // iree_net_bulk_frame_type_t
  uint8_t flags;          // iree_net_bulk_frame_flags_t
  uint8_t reserved;       // Must be 0.
  uint64_t transfer_id;   // Unique transfer identifier.
  uint64_t total_size;    // Total size in START; cumulative CREDIT grant.
  uint64_t chunk_offset;  // Offset of this chunk within transfer.
  uint32_t chunk_length;  // Length of this chunk's payload.
  uint32_t sequence;      // Sequence number for datagram ordering.
} iree_net_bulk_frame_header_t;

static_assert(sizeof(iree_net_bulk_frame_header_t) ==
                  IREE_NET_BULK_FRAME_HEADER_SIZE,
              "Bulk frame header must be exactly 40 bytes");
static_assert(offsetof(iree_net_bulk_frame_header_t, magic) == 0, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, version) == 4, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, type) == 5, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, flags) == 6, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, reserved) == 7, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, transfer_id) == 8, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, total_size) == 16, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, chunk_offset) == 24, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, chunk_length) == 32, "");
static_assert(offsetof(iree_net_bulk_frame_header_t, sequence) == 36, "");

//===----------------------------------------------------------------------===//
// Frame header initialization
//===----------------------------------------------------------------------===//

// Initializes a bulk frame header with the given parameters.
static inline void iree_net_bulk_frame_header_initialize(
    iree_net_bulk_frame_type_t type, iree_net_bulk_frame_flags_t flags,
    uint64_t transfer_id, uint64_t total_size, uint64_t chunk_offset,
    uint32_t chunk_length, uint32_t sequence,
    iree_net_bulk_frame_header_t* out_header) {
  out_header->magic = IREE_NET_BULK_FRAME_MAGIC;
  out_header->version = IREE_NET_BULK_FRAME_VERSION;
  out_header->type = type;
  out_header->flags = flags;
  out_header->reserved = 0;
  out_header->transfer_id = transfer_id;
  out_header->total_size = total_size;
  out_header->chunk_offset = chunk_offset;
  out_header->chunk_length = chunk_length;
  out_header->sequence = sequence;
}

//===----------------------------------------------------------------------===//
// Frame header accessors
//===----------------------------------------------------------------------===//

// Validates a bulk frame header's magic, version, and reserved fields.
static inline iree_status_t iree_net_bulk_frame_header_validate(
    iree_net_bulk_frame_header_t header) {
  if (header.magic != IREE_NET_BULK_FRAME_MAGIC) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid bulk frame magic: 0x%08X", header.magic);
  }
  if (header.version != IREE_NET_BULK_FRAME_VERSION) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "unsupported bulk frame version: %u",
                            header.version);
  }
  if (header.reserved != 0) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "bulk frame reserved field must be 0");
  }
  return iree_ok_status();
}

// Returns the frame type.
static inline iree_net_bulk_frame_type_t iree_net_bulk_frame_header_type(
    iree_net_bulk_frame_header_t header) {
  return header.type;
}

// Returns the frame flags.
static inline iree_net_bulk_frame_flags_t iree_net_bulk_frame_header_flags(
    iree_net_bulk_frame_header_t header) {
  return header.flags;
}

// Returns the transfer ID.
static inline uint64_t iree_net_bulk_frame_header_transfer_id(
    iree_net_bulk_frame_header_t header) {
  return header.transfer_id;
}

// Returns the total transfer size.
static inline uint64_t iree_net_bulk_frame_header_total_size(
    iree_net_bulk_frame_header_t header) {
  return header.total_size;
}

// Returns the chunk byte offset.
static inline uint64_t iree_net_bulk_frame_header_chunk_offset(
    iree_net_bulk_frame_header_t header) {
  return header.chunk_offset;
}

// Returns the chunk payload length.
static inline uint32_t iree_net_bulk_frame_header_chunk_length(
    iree_net_bulk_frame_header_t header) {
  return header.chunk_length;
}

// Returns the datagram sequence number.
static inline uint32_t iree_net_bulk_frame_header_sequence(
    iree_net_bulk_frame_header_t header) {
  return header.sequence;
}

// Returns true if the specified flag is set.
static inline bool iree_net_bulk_frame_header_has_flag(
    iree_net_bulk_frame_header_t header, iree_net_bulk_frame_flag_bits_e flag) {
  return iree_any_bit_set(header.flags, flag);
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_NET_CHANNEL_BULK_FRAME_H_
