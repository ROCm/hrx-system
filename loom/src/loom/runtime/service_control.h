// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_SERVICE_CONTROL_H_
#define LOOM_RUNTIME_SERVICE_CONTROL_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Lifetime-retained service observations. Storage belongs to the native
// lifecycle owner or its composition, independently of model residency. Every
// reader borrows the complete control and its immutable resource bindings.
// Retirement of model work alone does not authorize freeing this storage.
typedef struct loom_runtime_service_control_t loom_runtime_service_control_t;

// Native attachment of one serialized contributor to a retained control. The
// composition establishes these exclusive slot assignments before attaching
// readers. Pointers stay outside the shared byte representation.
typedef struct loom_runtime_service_control_binding_t {
  // Borrowed storage, retained through the contributor and all its readers.
  loom_runtime_service_control_t* control;
  // Stable offer slot assigned exclusively to this contributor.
  uint64_t offer_ordinal;
  // Stable accepted-operation slot assigned exclusively to this contributor.
  uint64_t operation_ordinal;
  // Immutable instance coordinate in the composition's binding scope.
  uint64_t instance_ordinal;
} loom_runtime_service_control_binding_t;

// Decoded immutable 64-byte header. All stored words are little-endian u64;
// offsets are relative to the beginning of this control, with 8-byte alignment.
// Native pointers, enum representation, and configurable HAL bitmap widths are
// absent from the byte representation. Shared payloads are accessed atomically
// through the publication/read helpers, never by racing ordinary struct loads.
typedef struct loom_runtime_service_control_layout_t {
  // Complete retained extent, including every bounded child array.
  uint64_t byte_length;
  // Composition-scoped occupant lifetime, unchanged through reader borrows.
  uint64_t incarnation;
  // Relative start of the independently guarded 64-byte summary.
  uint64_t observation_offset;
  // Relative start of stable offer slots.
  uint64_t offers_offset;
  // Addressable offer slots, including currently empty slots.
  uint64_t offer_capacity;
  // Relative start of stable accepted-operation slots.
  uint64_t operations_offset;
  // Addressable operation slots, including currently empty slots.
  uint64_t operation_capacity;
  // Reserved word, initialized to zero.
  uint64_t reserved;
} loom_runtime_service_control_layout_t;

// One source-owned legal opportunity. Timing updates do not change its
// revision.
typedef struct loom_runtime_service_offer_key_t {
  // Stable slot ordinal, independent of request and HAL timeline namespaces.
  uint64_t ordinal;
  // Changes when the legal operation, extent, or alternatives change.
  uint64_t revision;
} loom_runtime_service_offer_key_t;

// One accepted lifetime, independent of the originating offer's publication.
typedef struct loom_runtime_service_operation_key_t {
  // Stable operation slot ordinal.
  uint64_t ordinal;
  // Changes on reuse; zero means the slot has no live operation.
  uint64_t serial;
} loom_runtime_service_operation_key_t;

// Portable result of one attempted choice. The status object stays with its
// actual error owner; only its numeric code enters scheduling/telemetry data.
typedef struct loom_runtime_service_schedule_outcome_t {
  // Acceptance is independent of status: a failed submission can own work.
  bool accepted;
  // Numeric status from this attempt, including an error after acceptance.
  iree_status_code_t status_code;
  // Accepted lifetime, or zero when no obligation was accepted.
  loom_runtime_service_operation_key_t operation;
} loom_runtime_service_schedule_outcome_t;

typedef uint64_t loom_runtime_service_work_flags_t;
enum loom_runtime_service_work_flag_bits_e {
  LOOM_RUNTIME_SERVICE_WORK_PREPARATION = 1u << 0,
  LOOM_RUNTIME_SERVICE_WORK_EXECUTION = 1u << 1,
  LOOM_RUNTIME_SERVICE_WORK_ACQUISITION = 1u << 2,
  LOOM_RUNTIME_SERVICE_WORK_RELEASE = 1u << 3,
};

// Counts are independent facts, not a partition of requests. A request can
// participate in accepted work and a separately offered next unit. Ingress
// that has not transferred requests to this owner retains its own accounting.
typedef struct loom_runtime_service_observation_t {
  // Accepted requests still owned by the service, including native delivery.
  uint64_t owned_request_count;
  // Owned requests awaiting another unit, rather than only its completion.
  uint64_t waiting_request_count;
  // Currently nonempty offers; active slots need not form a dense prefix.
  uint64_t offer_count;
  // Accepted units still requiring progress or final collection.
  uint64_t pending_operation_count;
  // Union of offered group devices; alternatives remain separate choices.
  uint64_t offered_devices;
  // Union of accepted device participation; does not imply exclusive occupancy.
  uint64_t accepted_devices;
} loom_runtime_service_observation_t;

typedef struct loom_runtime_service_offer_t {
  // Source-owned legal opportunity in this service incarnation.
  loom_runtime_service_offer_key_t key;
  // Semantic work descriptions, including combinations for composite units.
  loom_runtime_service_work_flags_t work;
  // Published alternatives; zero means this offer slot is empty.
  uint64_t choice_count;
} loom_runtime_service_offer_t;

typedef uint64_t loom_runtime_service_estimate_flags_t;
enum loom_runtime_service_estimate_flag_bits_e {
  LOOM_RUNTIME_SERVICE_ESTIMATE_DURATION = 1u << 0,
  LOOM_RUNTIME_SERVICE_ESTIMATE_FIRST_USE = 1u << 1,
  LOOM_RUNTIME_SERVICE_ESTIMATE_LAST_USE = 1u << 2,
  LOOM_RUNTIME_SERVICE_ESTIMATE_REMAINING = 1u << 3,
};

typedef struct loom_runtime_service_choice_t {
  // Read token joining this copy to its uses; ignored by the publisher.
  uint64_t publication;
  // Immutable instance/placement coordinate in the composition's binding scope.
  uint64_t instance_ordinal;
  // Source-known request participation in this particular legal choice.
  uint64_t request_count;
  // Group-device participation; zero represents host-only work.
  uint64_t devices;
  // Published resource uses within this choice's retained range.
  uint64_t use_count;
  // Valid estimates; an absent flag means unknown, not zero cost.
  loom_runtime_service_estimate_flags_t estimates;
  // Entire unit duration after known prerequisites, including required native
  // host/feedback/publication phases. Device-only samples cannot fill this.
  uint64_t duration_ns;
} loom_runtime_service_choice_t;

// Participation and forecasts for one resource interval. Several intervals can
// name the same resource. None reserves it, completes work, or returns credit.
typedef struct loom_runtime_service_use_t {
  // Exact resource in the retained host/device/queue/topology binding scope.
  uint64_t resource_ordinal;
  // Independently valid interval forecasts; a flagged zero value is valid.
  loom_runtime_service_estimate_flags_t estimates;
  // Full resource interval, independent of the containing unit's duration.
  uint64_t duration_ns;
  // Remaining interval at the accepted operation's published progress point.
  uint64_t remaining_duration_ns;
  // First use relative to unit start, or to accepted published progress after
  // known prerequisites. A duration alone does not establish this offset.
  uint64_t first_use_offset_ns;
  // Last use at that same origin; independent of physical memory credit return.
  uint64_t last_use_offset_ns;
} loom_runtime_service_use_t;

typedef uint64_t loom_runtime_service_wait_flags_t;
enum loom_runtime_service_wait_flag_bits_e {
  LOOM_RUNTIME_SERVICE_WAIT_CAPACITY = 1u << 0,
  LOOM_RUNTIME_SERVICE_WAIT_CPU = 1u << 1,
  LOOM_RUNTIME_SERVICE_WAIT_DEVICE = 1u << 2,
  LOOM_RUNTIME_SERVICE_WAIT_OUTPUT = 1u << 3,
};

typedef struct loom_runtime_service_operation_t {
  // Accepted lifetime; serial zero denotes an empty slot.
  loom_runtime_service_operation_key_t key;
  // Read token joining progress and uses; ignored by the publisher.
  uint64_t publication;
  // Original opportunity, retained for correlation after offer withdrawal.
  loom_runtime_service_offer_key_t origin;
  // Selected alternative, retained independently of the original choice row.
  uint64_t choice_ordinal;
  // Accepted semantic work, retained without consulting the withdrawn offer.
  loom_runtime_service_work_flags_t work;
  // Prerequisites established by their actual owners.
  loom_runtime_service_wait_flags_t waits;
  // Accepted group-device participation through actual unit retirement.
  uint64_t devices;
  // Published uses in this operation's independently retained range.
  uint64_t use_count;
  // Valid whole-unit forecasts at the published progress point.
  loom_runtime_service_estimate_flags_t estimates;
  // Whole-unit remaining time after known prerequisites. Includes required
  // native collection/publication; a device interval has its own use estimate.
  uint64_t remaining_duration_ns;
} loom_runtime_service_operation_t;

// Cold, native layout inputs. Counts follow the implementer's concurrency and
// established bindings, not the number of queued requests or source stages.
typedef struct loom_runtime_service_offer_capacity_t {
  // Maximum simultaneous alternatives for this stable offer slot.
  iree_host_size_t choice_capacity;
  // One immutable use capacity per choice, borrowed during construction.
  const iree_host_size_t* use_capacities;
} loom_runtime_service_offer_capacity_t;

typedef struct loom_runtime_service_control_capacity_t {
  // Stable offered-unit slots.
  iree_host_size_t offer_capacity;
  // One layout description per offer, borrowed during construction.
  const loom_runtime_service_offer_capacity_t* offers;
  // Independently accepted-unit slots.
  iree_host_size_t operation_capacity;
  // One retained-use capacity per operation, borrowed during construction.
  const iree_host_size_t* operation_use_capacities;
} loom_runtime_service_control_capacity_t;

// Computes checked allocation size. No allocation or publication occurs.
iree_status_t loom_runtime_service_control_size(
    const loom_runtime_service_control_capacity_t* capacity,
    iree_host_size_t* out_byte_length);

// Initializes an aligned, exclusively borrowed region before readers attach.
// Incarnation must be nonzero and distinct when a composition slot is reused.
// The caller retains bytes and immutable bindings through every
// reader/publisher borrow. Reinitialization requires all such accesses to have
// joined.
iree_status_t loom_runtime_service_control_initialize(
    const loom_runtime_service_control_capacity_t* capacity,
    uint64_t incarnation, iree_byte_span_t storage,
    loom_runtime_service_control_t** out_control);

// Immutable native copies, safe through the caller's control borrow.
loom_runtime_service_control_layout_t loom_runtime_service_control_layout(
    const loom_runtime_service_control_t* control);

typedef enum loom_runtime_service_read_e {
  LOOM_RUNTIME_SERVICE_READ_COMPLETE = 0,
  LOOM_RUNTIME_SERVICE_READ_CHANGED,
} loom_runtime_service_read_t;

// One bounded attempt with atomic payload loads. CHANGED leaves output intact
// and never asks the caller to spin. No allocation, clock, source callback,
// history scan, or device query occurs. Summary/detail need not share a global
// transaction. Slot bounds come from the immutable layout, not active counts.
loom_runtime_service_read_t loom_runtime_service_read_observation(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_observation_t* out_observation);
loom_runtime_service_read_t loom_runtime_service_read_offer(
    const loom_runtime_service_control_t* control, uint64_t offer_ordinal,
    loom_runtime_service_offer_t* out_offer);
loom_runtime_service_read_t loom_runtime_service_read_choice(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    loom_runtime_service_choice_t* out_choice);
loom_runtime_service_read_t loom_runtime_service_read_choice_use(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_offer_key_t offer, uint64_t choice_ordinal,
    uint64_t publication, uint64_t use_ordinal,
    loom_runtime_service_use_t* out_use);
loom_runtime_service_read_t loom_runtime_service_read_operation(
    const loom_runtime_service_control_t* control, uint64_t operation_ordinal,
    loom_runtime_service_operation_t* out_operation);
loom_runtime_service_read_t loom_runtime_service_read_operation_use(
    const loom_runtime_service_control_t* control,
    loom_runtime_service_operation_key_t operation, uint64_t publication,
    uint64_t use_ordinal, loom_runtime_service_use_t* out_use);

// Source-owned publication input, borrowed only during the call.
typedef struct loom_runtime_service_choice_publication_t {
  // Choice values; use_count names the following producer-owned input extent.
  loom_runtime_service_choice_t choice;
  // Exactly choice.use_count uses, including NULL when empty.
  const loom_runtime_service_use_t* uses;
} loom_runtime_service_choice_publication_t;

// Each summary/offer/operation region has one publisher. These consume trusted
// producer state within the cold capacities; publication does not validate an
// external plan or retain the passed values. Offer publication covers its
// choices/uses under one guard. Operation publication owns its separate uses
// through retirement, even after the offer disappears. Writer handoff requires
// joining the previous publisher. Readers never acknowledge or delay a writer.
void loom_runtime_service_publish_observation(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_observation_t* observation);
void loom_runtime_service_publish_offer(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_offer_t* offer,
    const loom_runtime_service_choice_publication_t* choices);
void loom_runtime_service_publish_operation(
    loom_runtime_service_control_t* control,
    const loom_runtime_service_operation_t* operation,
    const loom_runtime_service_use_t* uses);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_SERVICE_CONTROL_H_
