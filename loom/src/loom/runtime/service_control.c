// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/runtime/service_control.h"

#include <string.h>

#include "iree/base/internal/atomics.h"

// The wire layout consists exclusively of little-endian 8-byte words. Cold
// descriptors never change while borrowed. Only guarded payload words are
// atomic; decoded public values are ordinary copies, not views of these types.
struct loom_runtime_service_control_t {
  // Immutable header in the order documented by control_layout_t.
  uint64_t words[8];
};

typedef struct loom_service_control_summary_t {
  // Odd during publication, even when a complete observation is available.
  iree_atomic_uint64_t sequence;
  // Owned requests, waiting requests, offers, operations, and two device masks.
  iree_atomic_uint64_t words[6];
  // Reserved zero word, independent of the publication guard.
  uint64_t reserved;
} loom_service_control_summary_t;

typedef struct loom_service_control_offer_t {
  // Immutable control-relative offset of the choice array.
  uint64_t choices_offset;
  // Immutable number of allocated choices.
  uint64_t choice_capacity;
  // Guards this slot, its choices, and every choice's uses.
  iree_atomic_uint64_t sequence;
  // Revision, work flags, and actual choice count (zero means empty).
  iree_atomic_uint64_t words[3];
  // Reserved zero words.
  uint64_t reserved[2];
} loom_service_control_offer_t;

typedef struct loom_service_control_choice_t {
  // Immutable control-relative offset of the use array.
  uint64_t uses_offset;
  // Immutable number of allocated uses.
  uint64_t use_capacity;
  // Instance, requests, devices, use count, estimate validity, and duration.
  iree_atomic_uint64_t words[6];
} loom_service_control_choice_t;

typedef struct loom_service_control_operation_t {
  // Immutable control-relative offset of independently retained uses.
  uint64_t uses_offset;
  // Immutable number of allocated uses.
  uint64_t use_capacity;
  // Guards this slot and its uses independently of the original offer.
  iree_atomic_uint64_t sequence;
  // Serial, origin ordinal/revision, choice, work, waits, devices, use count,
  // estimate validity, and remaining whole-unit duration.
  iree_atomic_uint64_t words[10];
  // Reserved zero words.
  uint64_t reserved[3];
} loom_service_control_operation_t;

typedef struct loom_service_control_use_t {
  // Resource, estimate validity, duration, remaining, first use, and last use.
  iree_atomic_uint64_t words[6];
} loom_service_control_use_t;

static_assert(sizeof(iree_atomic_uint64_t) == 8,
              "publication requires 8-byte atomic words");
static_assert(sizeof(loom_runtime_service_control_t) == 64,
              "control header ABI");
static_assert(sizeof(loom_service_control_summary_t) == 64, "summary ABI");
static_assert(sizeof(loom_service_control_offer_t) == 64, "offer ABI");
static_assert(sizeof(loom_service_control_choice_t) == 64, "choice ABI");
static_assert(sizeof(loom_service_control_operation_t) == 128, "operation ABI");
static_assert(sizeof(loom_service_control_use_t) == 48, "resource use ABI");

static uint64_t loom_service_control_word(uint64_t value) {
  return iree_unaligned_convert_native_le_u64(value);
}

static uint64_t loom_service_control_load(const iree_atomic_uint64_t* word) {
  return loom_service_control_word(
      iree_atomic_load(word, iree_memory_order_acquire));
}

static void loom_service_control_store(iree_atomic_uint64_t* word,
                                       uint64_t value) {
  iree_atomic_store(word, loom_service_control_word(value),
                    iree_memory_order_release);
}

// Release/acquire on payloads is essential: seeing any overwrite orders the
// preceding odd sequence before the reader's final sequence load. A rejected
// read never exposes its partial copy. Plain memcpy or relaxed payload access
// would lose that ordering even on a platform with indivisible 64-bit loads.
static uint64_t loom_service_control_begin(iree_atomic_uint64_t* sequence) {
  const uint64_t previous = loom_service_control_word(
      iree_atomic_load(sequence, iree_memory_order_relaxed));
  IREE_ASSERT(previous < UINT64_MAX - 1);
  iree_atomic_store(sequence, loom_service_control_word(previous + 1),
                    iree_memory_order_relaxed);
  return previous + 2;
}

static void loom_service_control_load_words(const iree_atomic_uint64_t* source,
                                            iree_host_size_t count,
                                            uint64_t* target) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    target[i] = loom_service_control_load(&source[i]);
  }
}

static void loom_service_control_store_words(iree_atomic_uint64_t* target,
                                             iree_host_size_t count,
                                             const uint64_t* source) {
  for (iree_host_size_t i = 0; i < count; ++i) {
    loom_service_control_store(&target[i], source[i]);
  }
}

static bool loom_service_control_unchanged(const iree_atomic_uint64_t* sequence,
                                           uint64_t publication) {
  return !(publication & 1) &&
         loom_service_control_load(sequence) == publication;
}

// One cold layout walk serves both checked sizing and initialization. The
// initializing walk only runs after the complete extent has been checked.
static iree_status_t loom_service_control_layout(
    const loom_runtime_service_control_capacity_t* capacity,
    loom_runtime_service_control_t* control, iree_host_size_t* out_length) {
  if ((capacity->offer_capacity && !capacity->offers) ||
      (capacity->operation_capacity && !capacity->operation_use_capacities)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "service control needs each slot's capacities");
  }
  iree_host_size_t length = 0, summary_offset = 0, offers_offset = 0,
                   operations_offset = 0;
  IREE_RETURN_IF_ERROR(IREE_STRUCT_LAYOUT(
      sizeof(*control), &length,
      IREE_STRUCT_FIELD_ALIGNED(1, loom_service_control_summary_t, 8,
                                &summary_offset),
      IREE_STRUCT_FIELD_ALIGNED(capacity->offer_capacity,
                                loom_service_control_offer_t, 8,
                                &offers_offset),
      IREE_STRUCT_FIELD_ALIGNED(capacity->operation_capacity,
                                loom_service_control_operation_t, 8,
                                &operations_offset)));
  uint8_t* base = (uint8_t*)control;
  iree_status_t status = iree_ok_status();
  for (iree_host_size_t i = 0;
       i < capacity->offer_capacity && iree_status_is_ok(status); ++i) {
    const loom_runtime_service_offer_capacity_t* offer = &capacity->offers[i];
    if (offer->choice_capacity && !offer->use_capacities) {
      status = iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                                "service offer needs each choice's capacity");
      break;
    }
    iree_host_size_t choices_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        length, &length,
        IREE_STRUCT_FIELD_ALIGNED(offer->choice_capacity,
                                  loom_service_control_choice_t, 8,
                                  &choices_offset));
    if (control && iree_status_is_ok(status)) {
      loom_service_control_offer_t* slots = (void*)(base + offers_offset);
      slots[i].choices_offset = loom_service_control_word(choices_offset);
      slots[i].choice_capacity =
          loom_service_control_word(offer->choice_capacity);
    }
    for (iree_host_size_t j = 0;
         j < offer->choice_capacity && iree_status_is_ok(status); ++j) {
      iree_host_size_t uses_offset = 0;
      status =
          IREE_STRUCT_LAYOUT(length, &length,
                             IREE_STRUCT_FIELD_ALIGNED(
                                 offer->use_capacities[j],
                                 loom_service_control_use_t, 8, &uses_offset));
      if (control && iree_status_is_ok(status)) {
        loom_service_control_choice_t* slots = (void*)(base + choices_offset);
        slots[j].uses_offset = loom_service_control_word(uses_offset);
        slots[j].use_capacity =
            loom_service_control_word(offer->use_capacities[j]);
      }
    }
  }
  for (iree_host_size_t i = 0;
       i < capacity->operation_capacity && iree_status_is_ok(status); ++i) {
    iree_host_size_t uses_offset = 0;
    status = IREE_STRUCT_LAYOUT(
        length, &length,
        IREE_STRUCT_FIELD_ALIGNED(capacity->operation_use_capacities[i],
                                  loom_service_control_use_t, 8, &uses_offset));
    if (control && iree_status_is_ok(status)) {
      loom_service_control_operation_t* slots =
          (void*)(base + operations_offset);
      slots[i].uses_offset = loom_service_control_word(uses_offset);
      slots[i].use_capacity =
          loom_service_control_word(capacity->operation_use_capacities[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    if (control) {
      control->words[0] = loom_service_control_word(length);
      control->words[2] = loom_service_control_word(summary_offset);
      control->words[3] = loom_service_control_word(offers_offset);
      control->words[4] = loom_service_control_word(capacity->offer_capacity);
      control->words[5] = loom_service_control_word(operations_offset);
      control->words[6] =
          loom_service_control_word(capacity->operation_capacity);
    }
    *out_length = length;
  }
  return status;
}

iree_status_t loom_runtime_service_control_size(
    const loom_runtime_service_control_capacity_t* capacity,
    iree_host_size_t* out_byte_length) {
  *out_byte_length = 0;
  return loom_service_control_layout(capacity, NULL, out_byte_length);
}

iree_status_t loom_runtime_service_control_initialize(
    const loom_runtime_service_control_capacity_t* capacity,
    uint64_t incarnation, iree_byte_span_t storage,
    loom_runtime_service_control_t** out_control) {
  *out_control = NULL;
  if (!iree_atomic_int64_is_lock_free()) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "service observation requires lock-free u64 atomics");
  }
  iree_host_size_t length = 0;
  IREE_RETURN_IF_ERROR(loom_runtime_service_control_size(capacity, &length));
  if (!incarnation || !storage.data || ((uintptr_t)storage.data & 7) ||
      storage.data_length < length) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "service control requires a nonzero incarnation "
                            "and %" PRIhsz " bytes aligned to 8",
                            length);
  }
  memset(storage.data, 0, length);
  loom_runtime_service_control_t* control = (void*)storage.data;
  IREE_RETURN_IF_ERROR(loom_service_control_layout(capacity, control, &length));
  control->words[1] = loom_service_control_word(incarnation);
  *out_control = control;
  return iree_ok_status();
}

loom_runtime_service_control_layout_t loom_runtime_service_control_layout(
    const loom_runtime_service_control_t* control) {
  return (loom_runtime_service_control_layout_t){
      .byte_length = loom_service_control_word(control->words[0]),
      .incarnation = loom_service_control_word(control->words[1]),
      .observation_offset = loom_service_control_word(control->words[2]),
      .offers_offset = loom_service_control_word(control->words[3]),
      .offer_capacity = loom_service_control_word(control->words[4]),
      .operations_offset = loom_service_control_word(control->words[5]),
      .operation_capacity = loom_service_control_word(control->words[6]),
      .reserved = loom_service_control_word(control->words[7]),
  };
}

static const loom_service_control_summary_t* loom_service_control_summary(
    const loom_runtime_service_control_t* control) {
  return (const void*)((const uint8_t*)control +
                       loom_service_control_word(control->words[2]));
}

static const loom_service_control_offer_t* loom_service_control_offer(
    const loom_runtime_service_control_t* control, uint64_t ordinal) {
  const loom_service_control_offer_t* offers =
      (const void*)((const uint8_t*)control +
                    loom_service_control_word(control->words[3]));
  return &offers[ordinal];
}

static const loom_service_control_choice_t* loom_service_control_choice(
    const loom_runtime_service_control_t* control,
    const loom_service_control_offer_t* offer, uint64_t ordinal) {
  const loom_service_control_choice_t* choices =
      (const void*)((const uint8_t*)control +
                    loom_service_control_word(offer->choices_offset));
  return &choices[ordinal];
}

static const loom_service_control_operation_t* loom_service_control_operation(
    const loom_runtime_service_control_t* control, uint64_t ordinal) {
  const loom_service_control_operation_t* operations =
      (const void*)((const uint8_t*)control +
                    loom_service_control_word(control->words[5]));
  return &operations[ordinal];
}

static const loom_service_control_use_t* loom_service_control_uses(
    const loom_runtime_service_control_t* control, uint64_t encoded_offset) {
  return (const void*)((const uint8_t*)control +
                       loom_service_control_word(encoded_offset));
}

loom_runtime_service_read_t loom_runtime_service_read_observation(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_observation_t* out_observation) {
  const loom_service_control_summary_t* summary =
      loom_service_control_summary(control);
  const uint64_t publication = loom_service_control_load(&summary->sequence);
  if (publication & 1) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  uint64_t words[6];
  loom_service_control_load_words(summary->words, IREE_ARRAYSIZE(words), words);
  if (!loom_service_control_unchanged(&summary->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_observation =
      (loom_runtime_service_observation_t){.owned_request_count = words[0],
                                           .waiting_request_count = words[1],
                                           .offer_count = words[2],
                                           .pending_operation_count = words[3],
                                           .offered_devices = words[4],
                                           .accepted_devices = words[5]};
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

loom_runtime_service_read_t loom_runtime_service_read_offer(
    const loom_runtime_service_control_t* control, uint64_t offer_ordinal,
    loom_runtime_service_offer_t* out_offer) {
  const loom_service_control_offer_t* offer =
      loom_service_control_offer(control, offer_ordinal);
  const uint64_t publication = loom_service_control_load(&offer->sequence);
  if (publication & 1) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  uint64_t words[3];
  loom_service_control_load_words(offer->words, IREE_ARRAYSIZE(words), words);
  if (!loom_service_control_unchanged(&offer->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_offer = (loom_runtime_service_offer_t){
      .key = {.ordinal = offer_ordinal, .revision = words[0]},
      .work = words[1],
      .choice_count = words[2]};
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

loom_runtime_service_read_t loom_runtime_service_read_choice(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_offer_key_t key, uint64_t choice_ordinal,
    loom_runtime_service_choice_t* out_choice) {
  const loom_service_control_offer_t* offer =
      loom_service_control_offer(control, key.ordinal);
  const uint64_t publication = loom_service_control_load(&offer->sequence);
  if ((publication & 1) ||
      loom_service_control_load(&offer->words[0]) != key.revision ||
      choice_ordinal >= loom_service_control_load(&offer->words[2])) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  const loom_service_control_choice_t* choice =
      loom_service_control_choice(control, offer, choice_ordinal);
  uint64_t words[6];
  loom_service_control_load_words(choice->words, IREE_ARRAYSIZE(words), words);
  if (!loom_service_control_unchanged(&offer->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_choice = (loom_runtime_service_choice_t){.publication = publication,
                                                .instance_ordinal = words[0],
                                                .request_count = words[1],
                                                .devices = words[2],
                                                .use_count = words[3],
                                                .estimates = words[4],
                                                .duration_ns = words[5]};
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

static loom_runtime_service_use_t loom_service_control_read_use(
    const loom_service_control_use_t* use) {
  uint64_t words[6];
  loom_service_control_load_words(use->words, IREE_ARRAYSIZE(words), words);
  return (loom_runtime_service_use_t){.resource_ordinal = words[0],
                                      .estimates = words[1],
                                      .duration_ns = words[2],
                                      .remaining_duration_ns = words[3],
                                      .first_use_offset_ns = words[4],
                                      .last_use_offset_ns = words[5]};
}

loom_runtime_service_read_t loom_runtime_service_read_choice_use(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_offer_key_t key, uint64_t choice_ordinal,
    uint64_t publication, uint64_t use_ordinal,
    loom_runtime_service_use_t* out_use) {
  const loom_service_control_offer_t* offer =
      loom_service_control_offer(control, key.ordinal);
  if (!loom_service_control_unchanged(&offer->sequence, publication) ||
      loom_service_control_load(&offer->words[0]) != key.revision ||
      choice_ordinal >= loom_service_control_load(&offer->words[2])) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  const loom_service_control_choice_t* choice =
      loom_service_control_choice(control, offer, choice_ordinal);
  if (use_ordinal >= loom_service_control_load(&choice->words[3])) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  const loom_runtime_service_use_t use = loom_service_control_read_use(
      &loom_service_control_uses(control, choice->uses_offset)[use_ordinal]);
  if (!loom_service_control_unchanged(&offer->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_use = use;
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

loom_runtime_service_read_t loom_runtime_service_read_operation(
    const loom_runtime_service_control_t* control, uint64_t operation_ordinal,
    loom_runtime_service_operation_t* out_operation) {
  const loom_service_control_operation_t* operation =
      loom_service_control_operation(control, operation_ordinal);
  const uint64_t publication = loom_service_control_load(&operation->sequence);
  if (publication & 1) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  uint64_t words[10];
  loom_service_control_load_words(operation->words, IREE_ARRAYSIZE(words),
                                  words);
  if (!loom_service_control_unchanged(&operation->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_operation = (loom_runtime_service_operation_t){
      .key = {.ordinal = operation_ordinal, .serial = words[0]},
      .publication = publication,
      .origin = {.ordinal = words[1], .revision = words[2]},
      .choice_ordinal = words[3],
      .work = words[4],
      .waits = words[5],
      .devices = words[6],
      .use_count = words[7],
      .estimates = words[8],
      .remaining_duration_ns = words[9]};
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

loom_runtime_service_read_t loom_runtime_service_read_operation_use(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_operation_key_t key, uint64_t publication,
    uint64_t use_ordinal, loom_runtime_service_use_t* out_use) {
  const loom_service_control_operation_t* operation =
      loom_service_control_operation(control, key.ordinal);
  if (!loom_service_control_unchanged(&operation->sequence, publication) ||
      loom_service_control_load(&operation->words[0]) != key.serial ||
      use_ordinal >= loom_service_control_load(&operation->words[7])) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  const loom_runtime_service_use_t use = loom_service_control_read_use(
      &loom_service_control_uses(control, operation->uses_offset)[use_ordinal]);
  if (!loom_service_control_unchanged(&operation->sequence, publication)) {
    return LOOM_RUNTIME_SERVICE_READ_CHANGED;
  }
  *out_use = use;
  return LOOM_RUNTIME_SERVICE_READ_COMPLETE;
}

void loom_runtime_service_publish_observation(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_observation_t* observation) {
  loom_service_control_summary_t* summary =
      (loom_service_control_summary_t*)loom_service_control_summary(control);
  const uint64_t publication = loom_service_control_begin(&summary->sequence);
  const uint64_t words[] = {
      observation->owned_request_count, observation->waiting_request_count,
      observation->offer_count,         observation->pending_operation_count,
      observation->offered_devices,     observation->accepted_devices};
  loom_service_control_store_words(summary->words, IREE_ARRAYSIZE(words),
                                   words);
  loom_service_control_store(&summary->sequence, publication);
}

static void loom_service_control_publish_uses(
    loom_service_control_use_t* target, uint64_t count,
    const loom_runtime_service_use_t* uses) {
  for (uint64_t i = 0; i < count; ++i) {
    const uint64_t words[] = {
        uses[i].resource_ordinal,    uses[i].estimates,
        uses[i].duration_ns,         uses[i].remaining_duration_ns,
        uses[i].first_use_offset_ns, uses[i].last_use_offset_ns};
    loom_service_control_store_words(target[i].words, IREE_ARRAYSIZE(words),
                                     words);
  }
}

void loom_runtime_service_publish_offer(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_offer_t* offer,
    const loom_runtime_service_choice_publication_t* choices) {
  loom_service_control_offer_t* target =
      (loom_service_control_offer_t*)loom_service_control_offer(
          control, offer->key.ordinal);
  const uint64_t publication = loom_service_control_begin(&target->sequence);
  const uint64_t words[] = {offer->key.revision, offer->work,
                            offer->choice_count};
  loom_service_control_store_words(target->words, IREE_ARRAYSIZE(words), words);
  for (uint64_t i = 0; i < offer->choice_count; ++i) {
    loom_service_control_choice_t* choice =
        (loom_service_control_choice_t*)loom_service_control_choice(control,
                                                                    target, i);
    const loom_runtime_service_choice_t* source = &choices[i].choice;
    const uint64_t choice_words[] = {
        source->instance_ordinal, source->request_count, source->devices,
        source->use_count,        source->estimates,     source->duration_ns};
    loom_service_control_store_words(
        choice->words, IREE_ARRAYSIZE(choice_words), choice_words);
    loom_service_control_publish_uses(
        (loom_service_control_use_t*)loom_service_control_uses(
            control, choice->uses_offset),
        source->use_count, choices[i].uses);
  }
  loom_service_control_store(&target->sequence, publication);
}

void loom_runtime_service_publish_operation(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_operation_t* operation,
    const loom_runtime_service_use_t* uses) {
  loom_service_control_operation_t* target =
      (loom_service_control_operation_t*)loom_service_control_operation(
          control, operation->key.ordinal);
  const uint64_t publication = loom_service_control_begin(&target->sequence);
  const uint64_t words[] = {operation->key.serial,
                            operation->origin.ordinal,
                            operation->origin.revision,
                            operation->choice_ordinal,
                            operation->work,
                            operation->waits,
                            operation->devices,
                            operation->use_count,
                            operation->estimates,
                            operation->remaining_duration_ns};
  loom_service_control_store_words(target->words, IREE_ARRAYSIZE(words), words);
  loom_service_control_publish_uses(
      (loom_service_control_use_t*)loom_service_control_uses(
          control, target->uses_offset),
      operation->use_count, uses);
  loom_service_control_store(&target->sequence, publication);
}
