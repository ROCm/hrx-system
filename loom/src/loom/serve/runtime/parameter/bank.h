// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_BANK_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_BANK_H_

#include "loom/serve/runtime/device/device.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/parameter/weights.h"
#include "loom/target/arch/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One owned bank of immutable checkpoint roots. Plans and their fixed-buffer
// views belong to the bank; execution residencies borrow its physical group.
// All operations share the device's serialized parameter owner, except for
// exclusive source preparation of individual plans on the shared cold worker.
typedef struct loom_serve_parameter_bank_t loom_serve_parameter_bank_t;

// Allocates roots from a checked command layout without loading checkpoint
// payloads or compiling preparation policy. Parameter-free programs, including
// NULL, return a NULL bank without allocating storage.
// Device outlives the bank and every borrowing residency. Reflection is
// borrowed only during this call. On failure a non-NULL output remains
// caller-owned: fixed backing may already have been accepted and requires
// normal retirement.
iree_status_t loom_serve_parameter_bank_allocate(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    iree_allocator_t host_allocator, loom_serve_parameter_bank_t** out_bank);

// Allocates only the selected domain of a checked program. The resulting bank
// has one domain at index zero, independent of its original binding ordinal.
// The selected ordinal is below fixed_buffer_count. Ownership and partial
// failure follow allocate above; other roots are neither allocated nor
// borrowed. Optional arena selects one independently backed range in its shared
// reservation. The arena outlives this bank; its owner completes asynchronous
// group deactivation before free, or checked whole-arena detach after a failed
// mapping and retirement of every arena consumer.
iree_status_t loom_serve_parameter_bank_allocate_domain(
    loom_serve_device_t* device, const loom_cmd_program_t* program,
    uint32_t domain, loom_serve_memory_arena_t* arena,
    iree_allocator_t host_allocator, loom_serve_parameter_bank_t** out_bank);

// Domain count and borrowed plans in allocation order.
// A NULL bank has zero domains.
// The caller prepares each plan before admitting any execution of the bank.
iree_host_size_t loom_serve_parameter_bank_count(
    const loom_serve_parameter_bank_t* bank);
loom_serve_weights_t* loom_serve_parameter_bank_weights(
    const loom_serve_parameter_bank_t* bank, iree_host_size_t index);

// Borrowed immutable bindings, valid until retirement begins. Recordings retain
// the views independently and must be released before bank retirement can join.
// A NULL bank has no bindings.
iree_hal_buffer_t* const* loom_serve_parameter_bank_buffers(
    const loom_serve_parameter_bank_t* bank);

// Borrowed physical ownership shared by invocation/cohort working sets. Every
// borrowing residency is destroyed before freeing the bank. A NULL bank has no
// group and contributes no entry to an invocation's working set.
loom_serve_residency_group_t* loom_serve_parameter_bank_group(
    const loom_serve_parameter_bank_t* bank);

// A command root bound to independently owned parameter storage. The producer
// resolves checkpoint identity before constructing this mapping. Recordings
// retain buffer views; invocation admission covers each distinct bank's group.
typedef struct loom_serve_parameter_binding_t {
  // Borrowed, unretired owner through layout checking and recording.
  const loom_serve_parameter_bank_t* bank;
  // Selected domain below the bank's count, not the command's binding ordinal.
  uint32_t domain;
} loom_serve_parameter_binding_t;

// Checks mapped roots in the candidate's fixed-buffer binding order. The
// source-selected mapping may reuse domains at different command ordinals.
// Every mapping names an existing domain; count must match the candidate.
// Layout compatibility does not establish checkpoint or numerical identity.
iree_status_t loom_serve_parameter_bindings_check_layout(
    iree_host_size_t count, const loom_serve_parameter_binding_t* bindings,
    const loom_cmd_program_t* program, iree_allocator_t host_allocator);

// Cache-key comparison for a single-domain bank and one checked command root.
// Same checkpoint contents may need distinct packed placement; incompatible
// layout is a cache miss, not an invalid source declaration.
bool loom_serve_parameter_bank_matches_domain(
    const loom_serve_parameter_bank_t* bank, const loom_cmd_program_t* program,
    uint32_t domain);

// Checks a new source program against the bank's retained placement facts.
// Domains, keys, offsets, extents, and alignment must match; row interleaving
// between independent domains is immaterial. This does not establish numerical
// or checkpoint identity, which remains with source preparation. No original
// compiler product is borrowed. NULL denotes a parameter-free bank/program.
iree_status_t loom_serve_parameter_bank_check_layout(
    const loom_serve_parameter_bank_t* bank, const loom_cmd_program_t* program);

// Begins retirement after preparation and recording consumers relinquish the
// bank. Installs the scheduling-only wake and releases owned root views without
// waiting. Accepted allocation, I/O, and execution retain their own resources.
// Repeated calls update the observer without replaying a release. NULL is
// valid.
void loom_serve_parameter_bank_retire(loom_serve_parameter_bank_t* bank,
                                      void (*wake)(void* user_data),
                                      void* user_data);
bool loom_serve_parameter_bank_is_retired(loom_serve_parameter_bank_t* bank);

// Cold outer-thread join after retire; shared workers use the observer/query.
// Workspace progress failures return only after every plan has retired.
iree_status_t loom_serve_parameter_bank_await_retirement(
    loom_serve_parameter_bank_t* bank);

// Destroys a retired bank after all borrowing residencies have been destroyed.
// Native unmapping/release failure preserves the bank and outstanding plans
// for a later cleanup attempt. NULL is accepted.
iree_status_t loom_serve_parameter_bank_free(loom_serve_parameter_bank_t* bank);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_BANK_H_
