// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/loom_serve/telemetry/stream.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

static_assert(sizeof(loom_serve_telemetry_record_t) % sizeof(uint64_t) == 0,
              "record storage is a whole number of atomic words");
enum {
  RECORD_WORDS = sizeof(loom_serve_telemetry_record_t) / sizeof(uint64_t),
};

typedef struct telemetry_slot_t {
  // Odd while being overwritten; even = 2 * (absolute position + 1).
  iree_atomic_uint64_t sequence;
  // Atomic payload prevents a C data race even when a reader loses its slot.
  iree_atomic_uint64_t words[RECORD_WORDS];
} telemetry_slot_t;

struct loom_serve_telemetry_stream_t {
  // Allocator owning this object and its contiguous bounded slots.
  iree_allocator_t allocator;
  // Immutable lifetime identity, independent of physical wrap or model name.
  loom_serve_telemetry_identity_t identity;
  // Power-of-two physical record capacity.
  iree_host_size_t capacity;
  // Writer-owned next position, never modified by a reader.
  uint64_t next;
  // Published one-past-last position, independently sampled by consumers.
  iree_atomic_uint64_t head;
  // Fixed-size slots addressed by the low bits of absolute positions.
  telemetry_slot_t slots[];
};

static iree_atomic_uint64_t telemetry_ordinal = IREE_ATOMIC_VAR_INIT(0);

void loom_serve_telemetry_diagnostic_capture(
    iree_status_t status, loom_serve_telemetry_diagnostic_t* out_diagnostic) {
  memset(out_diagnostic, 0, sizeof(*out_diagnostic));
  out_diagnostic->code = iree_status_code(status);
  const iree_string_view_t message = iree_status_message(status);
  out_diagnostic->length =
      iree_min(message.size, sizeof(out_diagnostic->message));
  out_diagnostic->truncated = message.size > out_diagnostic->length;
  if (out_diagnostic->length) {
    memcpy(out_diagnostic->message, message.data, out_diagnostic->length);
  }
}

iree_status_t loom_serve_telemetry_stream_create(
    iree_host_size_t capacity, loom_serve_telemetry_stream_t** out_stream,
    iree_allocator_t allocator) {
  *out_stream = NULL;
  if (!capacity || (capacity & (capacity - 1)) ||
      capacity > (SIZE_MAX - sizeof(loom_serve_telemetry_stream_t)) /
                     sizeof(telemetry_slot_t)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "telemetry capacity must be a representable power "
                            "of two");
  }
  if (!iree_atomic_int64_is_lock_free()) {
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "telemetry requires lock-free 64-bit atomics");
  }
  loom_serve_telemetry_stream_t* stream = NULL;
  IREE_RETURN_IF_ERROR(iree_allocator_malloc(
      allocator, sizeof(*stream) + capacity * sizeof(*stream->slots),
      (void**)&stream));
  stream->allocator = allocator;
  stream->capacity = capacity;
  stream->identity.created_ns = iree_time_now();
  stream->identity.ordinal =
      iree_atomic_fetch_add(&telemetry_ordinal, 1, iree_memory_order_relaxed) +
      1;
  *out_stream = stream;
  return iree_ok_status();
}

void loom_serve_telemetry_stream_destroy(
    loom_serve_telemetry_stream_t* stream) {
  if (stream) {
    iree_allocator_free(stream->allocator, stream);
  }
}

loom_serve_telemetry_identity_t loom_serve_telemetry_stream_identity(
    const loom_serve_telemetry_stream_t* stream) {
  return stream->identity;
}

uint64_t loom_serve_telemetry_stream_head(
    const loom_serve_telemetry_stream_t* stream) {
  return iree_atomic_load(&stream->head, iree_memory_order_acquire);
}

iree_host_size_t loom_serve_telemetry_stream_capacity(
    const loom_serve_telemetry_stream_t* stream) {
  return stream->capacity;
}

void loom_serve_telemetry_stream_append(
    loom_serve_telemetry_stream_t* stream,
    const loom_serve_telemetry_record_t* record) {
  const uint64_t position = stream->next++;
  IREE_ASSERT(position < INT64_MAX - 1);
  telemetry_slot_t* slot = &stream->slots[position & (stream->capacity - 1)];
  iree_atomic_store(&slot->sequence, position * 2 + 1,
                    iree_memory_order_relaxed);
  for (iree_host_size_t i = 0; i < RECORD_WORDS; ++i) {
    uint64_t word;
    memcpy(&word, (const uint8_t*)record + i * sizeof(word), sizeof(word));
    // If a reader sees even one word from this overwrite, acquire orders the
    // in-progress sequence before its final sequence check. Plain/relaxed
    // payload accesses would not establish that cross-object ordering.
    iree_atomic_store(&slot->words[i], word, iree_memory_order_release);
  }
  iree_atomic_store(&slot->sequence, (position + 1) * 2,
                    iree_memory_order_release);
  iree_atomic_store(&stream->head, position + 1, iree_memory_order_release);
}

loom_serve_telemetry_read_t loom_serve_telemetry_stream_read(
    const loom_serve_telemetry_stream_t* stream, uint64_t* cursor,
    loom_serve_telemetry_record_t* out_record) {
  const uint64_t head = loom_serve_telemetry_stream_head(stream);
  if (*cursor >= head) {
    return LOOM_SERVE_TELEMETRY_READ_EMPTY;
  }
  const uint64_t oldest = head > stream->capacity ? head - stream->capacity : 0;
  if (*cursor < oldest) {
    *cursor = oldest;
    return LOOM_SERVE_TELEMETRY_READ_GAP;
  }
  const telemetry_slot_t* slot =
      &stream->slots[*cursor & (stream->capacity - 1)];
  const uint64_t sequence = (*cursor + 1) * 2;
  if (iree_atomic_load(&slot->sequence, iree_memory_order_acquire) !=
      sequence) {
    return LOOM_SERVE_TELEMETRY_READ_BUSY;
  }
  for (iree_host_size_t i = 0; i < RECORD_WORDS; ++i) {
    const uint64_t word =
        iree_atomic_load(&slot->words[i], iree_memory_order_acquire);
    memcpy((uint8_t*)out_record + i * sizeof(word), &word, sizeof(word));
  }
  if (iree_atomic_load(&slot->sequence, iree_memory_order_acquire) !=
      sequence) {
    return LOOM_SERVE_TELEMETRY_READ_BUSY;
  }
  ++*cursor;
  return LOOM_SERVE_TELEMETRY_READ_RECORD;
}
