// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_SERVING_REGISTRY_H_
#define EXPERIMENTAL_LOOM_SERVE_SERVING_REGISTRY_H_

#include "experimental/loom_serve/http/service.h"
#include "experimental/loom_serve/runtime/compilation.h"
#include "experimental/loom_serve/runtime/residency.h"
#include "experimental/loom_serve/serving/configuration.h"

#ifdef __cplusplus
extern "C" {
#endif

// A modality's owned preparation/retirement operation. All methods run on the
// serialized serving owner and never wait. Wake callbacks only schedule that
// owner. The declaration and factory remain borrowed through destruction.
typedef struct loom_serve_registration_t {
  // Owned modality operation, independent of any management connection.
  void* self;
  // Advances preparation or irrevocable cancellation. A nonterminal result
  // owes a wake after accepted work retires. Only terminal success publishes
  // a service using the supplied stream and cumulative counters; terminal
  // failure still requires retire.
  // Copies compilation evidence on terminal completion, including failure.
  iree_status_t (*prepare)(void* self, bool cancel, bool* out_complete,
                           loom_serve_http_service_t* out_service,
                           loom_serve_compilation_report_t* out_report);
  // Called after preparation terminates and service/tokenizer borrowers retire.
  // Drops service consumers, then waits by ownership wake, not readiness value.
  // Terminal success releases all model resources. Terminal failure retains
  // quiescent ownership for an explicit retry; no callback remains on failure.
  iree_status_t (*retire)(void* self, bool* out_complete);
  // Optional copied parameter state, queried only after successful prepare and
  // before retire. The adapter owns its actual parameter groups; neither the
  // registry nor the HTTP client gains a borrowed model or pool handle.
  loom_serve_residency_snapshot_t (*parameters)(void* self);
  // Releases reloadable parameter storage after admitted service activity has
  // retired. Preserves programs and mutable state. Paired with parameters and
  // called only when its snapshot reports elastic backing. No execution wait.
  iree_status_t (*deactivate)(void* self);
  // Frees the operation after successful retirement; no model resources remain.
  void (*destroy)(void* self);
} loom_serve_registration_t;

typedef struct loom_serve_registration_factory_t {
  // Borrowed application capabilities, live through registry destruction.
  void* self;
  // Supported configuration kinds as bit positions; zero disables management.
  uint32_t kinds;
  // Allocates an unstarted operation; performs no source I/O, model allocation
  // or submission. Failure returns no operation. Invoked only after admission
  // opens, never for metadata-only registration.
  iree_status_t (*create)(void* self,
                          const loom_serve_model_configuration_t* configuration,
                          loom_serve_http_observer_t observer,
                          loom_serve_telemetry_stream_t* stream,
                          loom_serve_registration_t* out_registration,
                          iree_allocator_t allocator);
} loom_serve_registration_factory_t;

typedef enum loom_serve_registration_phase_e {
  LOOM_SERVE_REGISTRATION_DECLARED = 0,
  LOOM_SERVE_REGISTRATION_PREPARING,
  LOOM_SERVE_REGISTRATION_READY,
  LOOM_SERVE_REGISTRATION_RETIRING,
  LOOM_SERVE_REGISTRATION_FAILED,
  LOOM_SERVE_REGISTRATION_RELEASE_FAILED,
} loom_serve_registration_phase_t;

typedef enum loom_serve_removal_e {
  LOOM_SERVE_REMOVAL_NONE = 0,
  LOOM_SERVE_REMOVAL_DRAIN,
  LOOM_SERVE_REMOVAL_ABORT,
} loom_serve_removal_t;

// Stable owner-only slot. Empty service.name marks an unused slot. A slot is
// reused only after every held claim, tokenizer task and model borrower
// retires. Service callbacks exist only while phase is READY. Other phases
// expose just immutable route metadata and copied counters, never model
// pointers.
typedef struct loom_serve_registered_model_t {
  // Borrowed ready interface or metadata-only description for an owned model.
  loom_serve_http_service_t service;
  // Owned declaration for managed entries; empty name denotes borrowed service.
  loom_serve_model_configuration_t configuration;
  // Owned modality operation, created lazily when admission first opens.
  loom_serve_registration_t registration;
  // Preparation/publication/retirement state, independent of admission.
  loom_serve_registration_phase_t phase;
  // Irreversible unregister intent; drain can be escalated explicitly to abort.
  loom_serve_removal_t removal;
  // Reversible admission gate, not a residency or cancellation flag.
  bool closed;
  // Accepted parameter deactivation waiting for this service's admitted work.
  // Independent of the initiating connection; terminal removal supersedes it.
  bool deactivating;
  // Requests parked outside service admission on transport-owned storage.
  iree_host_size_t held_count;
  // Independent tokenizer processes still borrowing the model.
  iree_host_size_t tokenizer_count;
  // Copied scheduler observation for one response; never a readiness signal.
  loom_serve_http_service_snapshot_t snapshot;
  // Final work totals after service destruction, or zero before publication.
  loom_serve_telemetry_counters_t counters;
  // Owned history shared with the live service; identity survives preparation.
  loom_serve_telemetry_stream_t* stream;
  // Terminal load/release evidence, owned until removal or explicit retry.
  iree_status_t status;
  // Copied source diagnostics, independent of compiler and operation lifetimes.
  loom_serve_compilation_report_t compilation;
  // Monotonic registration identity; zero denotes an externally owned service.
  uint64_t incarnation;
} loom_serve_registered_model_t;

typedef struct loom_serve_registry_t loom_serve_registry_t;

// Copies borrowed service interfaces into stable slots and reserves metadata
// capacity for managed registrations. Borrowed services remain externally owned
// and cannot be unregistered here. No source work or model allocation occurs.
iree_status_t loom_serve_registry_create(
    iree_host_size_t service_count, const loom_serve_http_service_t* services,
    iree_host_size_t managed_capacity,
    loom_serve_registration_factory_t factory,
    loom_serve_registry_t** out_registry, iree_allocator_t allocator);

// Requires the router to have retired. A failed model release preserves the
// entire registry and its borrowed factory/device owners; it is not permission
// to destroy their queues or pools. Successful destruction consumes registry.
iree_status_t loom_serve_registry_destroy(loom_serve_registry_t* registry);

iree_host_size_t loom_serve_registry_capacity(
    const loom_serve_registry_t* registry);
iree_host_size_t loom_serve_registry_count(
    const loom_serve_registry_t* registry);
loom_serve_registered_model_t* loom_serve_registry_at(
    loom_serve_registry_t* registry, iree_host_size_t index);

// Moves a validated declaration, including its owning allocator, into a
// stable slot on success, zeroing the caller's value. Failure changes neither.
// Rejects unsupported kinds, duplicate names and exhausted registration credit.
iree_status_t loom_serve_registry_register(
    loom_serve_registry_t* registry,
    loom_serve_model_configuration_t* configuration,
    iree_host_size_t* out_index);

// Seals admission; callers resolve held claims before the next advance.
// Repeated removal is idempotent, with abort escalation and explicit
// release-failure retry.
iree_status_t loom_serve_registry_remove(loom_serve_registered_model_t* model,
                                         loom_serve_removal_t mode);

// Requests parameter deactivation while preserving the prepared model and its
// retained sessions/checkpoints. Requires closed admission and a healthy ready
// elastic residency; an unprepared metadata-only registration is already cold.
// Repeated requests are idempotent. Advance completes after admitted work,
// without waiting for live device work or unrelated services. Physical unmap
// and free remain native synchronous capacity transitions on that owner.
iree_status_t loom_serve_registry_deactivate(
    loom_serve_registered_model_t* model);

// Returns STOPPING only for abort or after graceful admitted work has finished.
// The router still owns held claims and external tokenizer lifetimes.
loom_serve_http_service_flags_t loom_serve_registry_service_flags(
    const loom_serve_registered_model_t* model);

// Advances one lifecycle after the router advances its ready service. Ordinary
// source/load errors are retained in the slot, not returned as server failure.
// true denotes actual progress; false waits for an installed ownership wake.
bool loom_serve_registry_advance(loom_serve_registry_t* registry,
                                 iree_host_size_t index,
                                 loom_serve_http_observer_t observer);
bool loom_serve_registry_pending(const loom_serve_registered_model_t* model);
const char* loom_serve_registry_phase(
    const loom_serve_registered_model_t* model);

// Totals include removed incarnations. False means a borrowed service omitted
// counters; unavailable totals are never presented as a complete partial sum.
bool loom_serve_registry_counters(
    const loom_serve_registry_t* registry,
    loom_serve_telemetry_counters_t* out_counters);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // EXPERIMENTAL_LOOM_SERVE_SERVING_REGISTRY_H_
