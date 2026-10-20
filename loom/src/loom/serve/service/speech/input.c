// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/speech/input.h"

#include <string.h>

typedef struct loom_serve_speech_utf8_t {
  // Continuation bytes still required by the current scalar.
  uint8_t remaining;
  // Inclusive lower bound for the next continuation byte.
  uint8_t minimum;
  // Inclusive upper bound for the next continuation byte.
  uint8_t maximum;
} loom_serve_speech_utf8_t;

typedef struct loom_serve_speech_utf8_lead_t {
  // Inclusive first leading byte represented by this row.
  uint8_t first;
  // Inclusive last leading byte represented by this row.
  uint8_t last;
  // Continuation bounds excluding overlong, surrogate, and out-of-range forms.
  loom_serve_speech_utf8_t next;
} loom_serve_speech_utf8_lead_t;

static const loom_serve_speech_utf8_lead_t speech_utf8_leads[] = {
    {0xc2, 0xdf, {1, 0x80, 0xbf}}, {0xe0, 0xe0, {2, 0xa0, 0xbf}},
    {0xe1, 0xec, {2, 0x80, 0xbf}}, {0xed, 0xed, {2, 0x80, 0x9f}},
    {0xee, 0xef, {2, 0x80, 0xbf}}, {0xf0, 0xf0, {3, 0x90, 0xbf}},
    {0xf1, 0xf3, {3, 0x80, 0xbf}}, {0xf4, 0xf4, {3, 0x80, 0x8f}},
};

struct loom_serve_speech_input_t {
  // Serialized serving-owner references; source tasks only borrow snapshots.
  uint32_t reference_count;
  // Allocator owning this object and its optional trailing ring.
  iree_allocator_t host_allocator;
  // Original builder allocator owning finite text; unused for a trailing ring.
  iree_allocator_t text_allocator;
  // Transferred finite bytes or a fixed ring in the trailing allocation.
  iree_byte_span_t storage;
  // Maximum complete-scalar window offered to one CPU turn, in bytes.
  iree_host_size_t capacity;
  // Maximum total raw UTF-8 bytes accepted by this request.
  uint64_t limit;
  // Absolute byte position after all accepted writes, including partial UTF-8.
  uint64_t received;
  // Absolute byte position after the most recent complete UTF-8 scalar.
  uint64_t ready;
  // Absolute byte position acknowledged by retired source CPU calls.
  uint64_t consumed;
  // Validation cursor for the incomplete suffix after ready.
  loom_serve_speech_utf8_t utf8;
  // True for a fixed ring; false for transferred immutable finite storage.
  bool ring;
  // The producer has published EOF at a complete scalar boundary.
  bool complete;
};

iree_status_t loom_serve_speech_input_allocate(
    iree_host_size_t capacity, uint64_t limit, iree_string_builder_t* text,
    bool complete, iree_allocator_t host_allocator,
    loom_serve_speech_input_t** out_input) {
  *out_input = NULL;
  const iree_host_size_t length = iree_string_builder_size(text);
  if (length > limit) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech text exceeds the model byte limit");
  }
  iree_host_size_t total_size = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(loom_serve_speech_input_t), &total_size,
      IREE_STRUCT_FIELD(complete ? 0 : capacity, uint8_t, NULL)));
  loom_serve_speech_input_t* input = NULL;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&input));
  input->reference_count = 1;
  input->host_allocator = host_allocator;
  input->capacity = capacity;
  input->limit = limit;
  input->ring = !complete;
  input->complete = complete;
  if (complete) {
    input->text_allocator = text->allocator;
    input->storage =
        iree_make_byte_span(iree_string_builder_take_storage(text), length);
    input->received = input->ready = length;
  } else {
    input->storage = iree_make_byte_span(input + 1, capacity);
  }
  *out_input = input;
  return iree_ok_status();
}

void loom_serve_speech_input_retain(loom_serve_speech_input_t* input) {
  ++input->reference_count;
}

void loom_serve_speech_input_release(loom_serve_speech_input_t* input) {
  if (!input || --input->reference_count) {
    return;
  }
  if (!input->ring) {
    iree_allocator_free(input->text_allocator, input->storage.data);
  }
  iree_allocator_free(input->host_allocator, input);
}

// The local cursor is committed only if the complete accepted prefix validates.
static bool loom_serve_speech_input_validate(iree_const_byte_span_t text,
                                             loom_serve_speech_utf8_t* utf8,
                                             iree_host_size_t* out_ready) {
  *out_ready = 0;
  bool valid = true;
  for (iree_host_size_t i = 0; i < text.data_length && valid; ++i) {
    const uint8_t byte = text.data[i];
    if (utf8->remaining) {
      valid = byte >= utf8->minimum && byte <= utf8->maximum;
      --utf8->remaining;
      utf8->minimum = 0x80;
      utf8->maximum = 0xbf;
    } else if (byte >= 0x80) {
      iree_host_size_t row = 0;
      while (row < IREE_ARRAYSIZE(speech_utf8_leads) &&
             (byte < speech_utf8_leads[row].first ||
              byte > speech_utf8_leads[row].last)) {
        ++row;
      }
      valid = row < IREE_ARRAYSIZE(speech_utf8_leads);
      if (valid) {
        *utf8 = speech_utf8_leads[row].next;
      }
    }
    if (!utf8->remaining) {
      *out_ready = i + 1;
    }
  }
  return valid;
}

iree_host_size_t loom_serve_speech_input_credit(
    const loom_serve_speech_input_t* input) {
  if (input->complete) {
    return 0;
  }
  return (iree_host_size_t)iree_min(
      (uint64_t)input->capacity - (input->received - input->consumed),
      input->limit - input->received);
}

iree_status_t loom_serve_speech_input_write(loom_serve_speech_input_t* input,
                                            iree_const_byte_span_t text,
                                            iree_host_size_t* out_byte_count) {
  *out_byte_count = 0;
  if (input->complete) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "speech text input is already complete");
  }
  if (text.data_length > input->limit - input->received) {
    return iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                            "speech text exceeds the model byte limit");
  }
  const iree_host_size_t count =
      iree_min(text.data_length, loom_serve_speech_input_credit(input));
  if (!count) {
    return iree_ok_status();
  }
  loom_serve_speech_utf8_t utf8 = input->utf8;
  iree_host_size_t ready = 0;
  if (!loom_serve_speech_input_validate(
          iree_make_const_byte_span(text.data, count), &utf8, &ready)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech text must be UTF-8");
  }
  const iree_host_size_t tail = input->received % input->capacity;
  const iree_host_size_t first = iree_min(count, input->capacity - tail);
  memcpy(input->storage.data + tail, text.data, first);
  memcpy(input->storage.data, text.data + first, count - first);
  if (ready) {
    input->ready = input->received + ready;
  }
  input->received += count;
  input->utf8 = utf8;
  *out_byte_count = count;
  return iree_ok_status();
}

iree_status_t loom_serve_speech_input_finish(loom_serve_speech_input_t* input) {
  if (input->utf8.remaining) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "speech input ends inside a UTF-8 scalar");
  }
  input->complete = true;
  return iree_ok_status();
}

static iree_host_size_t loom_serve_speech_input_offset(
    const loom_serve_speech_input_t* input, uint64_t position) {
  return input->ring ? (iree_host_size_t)(position % input->capacity)
                     : (iree_host_size_t)position;
}

loom_serve_speech_input_block_t loom_serve_speech_input_peek(
    const loom_serve_speech_input_t* input) {
  const uint64_t available = input->ready - input->consumed;
  iree_host_size_t count =
      (iree_host_size_t)iree_min(available, input->capacity);
  if (count < available) {
    // The complete input is validated; a bounded cut backs over at most three
    // continuation bytes instead of rescanning the offered window.
    while ((input->storage.data[loom_serve_speech_input_offset(
                input, input->consumed + count)] &
            0xc0) == 0x80) {
      --count;
    }
  }
  loom_serve_speech_input_block_t block = {
      .position = input->consumed,
      .byte_count = count,
      .complete = input->complete && input->consumed + count == input->received,
  };
  if (count) {
    const iree_host_size_t head =
        loom_serve_speech_input_offset(input, input->consumed);
    const iree_host_size_t first =
        iree_min(count, input->storage.data_length - head);
    block.spans[0] =
        iree_make_const_byte_span(input->storage.data + head, first);
    block.spans[1] =
        iree_make_const_byte_span(input->storage.data, count - first);
  }
  return block;
}

void loom_serve_speech_input_consume(loom_serve_speech_input_t* input,
                                     iree_host_size_t byte_count) {
  input->consumed += byte_count;
}
