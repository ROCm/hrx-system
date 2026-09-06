// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_
#define IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_

#include "iree/base/api.h"
#include "iree/hal/drivers/amdgpu/util/libhsa.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_amdgpu_host_queue_t iree_hal_amdgpu_host_queue_t;
typedef struct iree_hal_amdgpu_logical_device_t
    iree_hal_amdgpu_logical_device_t;

//===----------------------------------------------------------------------===//
// Process-wide HSA system event delivery
//===----------------------------------------------------------------------===//
//
// HSA delivers asynchronous fatal GPU events - memory faults and hardware
// exceptions - to a process-global list of callbacks rather than to the queue
// that faulted. A registration binds one logical device's GPU agents to that
// callback so a fault on any of them becomes a durable failure observable
// through normal HAL operations. Both event types are one-shot per runtime
// instance and both are claimed here. This callback and a queue's own HSA error
// callback can fail the same queue, so both converge on the same terminal queue
// transition, which is idempotent.
//
// Limits the HSA runtime imposes that no amount of code here can escape:
//
// * A fatal event type is delivered at most once per HSA runtime instance.
//   After the first memory fault a second faulting kernel launch produces no
//   callback, no runtime abort and no HAL failure; only a fresh hsa_init
//   following a complete hsa_shut_down re-arms delivery. The instance is the
//   unit, not the device: a fault on a second GPU in the same process is just
//   as silent, and strands that GPU's queues with nothing reported anywhere.
//   That was observed for memory faults; hardware exceptions are believed to
//   behave the same way and have not been observed twice. HSA is reference
//   counted and other components of the process may hold it (see
//   util/libhsa.h), so a runtime instance can outlive any one driver: a driver
//   created after another has already consumed the delivery starts with it
//   already spent.
//
// * Delivery can be disabled with no way to detect it. When the HSA runtime's
//   interrupt path is off it registers neither handler, so no event is ever
//   delivered and no abort happens, while registration still reports success.
//   A fault then strands the queue exactly as it would otherwise, silently.
//   There is no public query for whether delivery is armed, so registering
//   successfully proves nothing about whether events will arrive.
//
// Claiming an event - returning HSA_STATUS_SUCCESS from the callback - is a
// policy decision rather than a report, and what it decides is whether the
// process still gets the runtime's abort: a single success from any registered
// handler suppresses it. It buys nothing back, because the delivery above is
// spent either way. So the callback claims an event only when it wrote the
// failure somewhere that can still be read, and an event nothing claims is left
// to the abort - the honest outcome once nothing remains that could report it.

// Failure-delivery target for one GPU agent of a registration.
//
// Names the queues that are fully initialized, not yet destroyed, and safe to
// fail from the HSA callback thread. That invariant cannot live on the physical
// device, whose host queue count keeps counting queues it has already destroyed
// until its destruction loop ends. Opaque because the only writers are the
// publication and retirement below, both of which take the registry mutex.
typedef struct iree_hal_amdgpu_system_event_agent_target_t
    iree_hal_amdgpu_system_event_agent_target_t;

// An entry in the process-wide registry of devices receiving HSA system events.
typedef struct iree_hal_amdgpu_system_event_registration_t
    iree_hal_amdgpu_system_event_registration_t;

// What pinning the module containing an address did.
typedef enum iree_hal_amdgpu_module_pin_e {
  // The address is in code the process has no way to unload, so no pin was
  // needed: the main program on POSIX, including a link with no dynamic loader.
  IREE_HAL_AMDGPU_MODULE_PIN_NOT_REQUIRED = 0,
  // The module containing the address is now permanently mapped.
  IREE_HAL_AMDGPU_MODULE_PIN_ACQUIRED = 1,
} iree_hal_amdgpu_module_pin_t;

// Makes the code containing |address| stay mapped and callable for the life of
// the process and reports in |out_pin| whether that took a pin or whether the
// containing module was already unloadable-by-nobody.
//
// This is what lets a callback pointer be handed to a component that never
// gives it back. It fails rather than reporting success it cannot support: when
// no loaded module contains |address|, when the module containing it cannot be
// pinned, and on platforms offering no pin at all. A failed pin is never
// reported as "nothing to pin".
//
// Exposed for coverage of that contract; registration below is the only
// production caller.
iree_status_t iree_hal_amdgpu_system_event_pin_module_containing(
    const void* address, iree_hal_amdgpu_module_pin_t* out_pin);

// Registers |logical_device| to receive process-wide HSA system events and
// returns the registration in |out_registration|. Registration is a cold
// device-creation operation and must run after the device's physical devices
// have their agents.
//
// Pins the module implementing the callback into the process before handing its
// pointer to the HSA runtime, and fails if it cannot. The runtime keeps that
// pointer until its final shutdown and offers no unregister, so a module that
// can still be unloaded would leave the runtime able to jump into unmapped
// code; refusing to register is the only honest answer.
//
// The registration copies the device's GPU agent handles and borrows
// |logical_device| until its device status is retired. It starts with no queue
// targets; a fault arriving before frontier assignment latches the device's
// sticky failure status and fails no queues.
//
// The registry mutex serializes registration-list mutation, target publication
// and retirement, and callback traversal. The callback acquires it, so no
// thread may hold it across an HSA entry point an event can be dispatched
// through: that thread would be waiting on the HSA runtime while holding a lock
// the runtime's own callback needs to make progress. Delivery does hold it
// across one HSA entry point, the stop-signal store that recording a queue
// failure performs, because a signal store dispatches nothing. Registration's
// one-time handshake is a call into the very handler registry events are
// dispatched from, so it is serialized by a separate mutex the callback never
// acquires.
iree_status_t iree_hal_amdgpu_system_event_register_device(
    const iree_hal_amdgpu_libhsa_t* libhsa,
    iree_hal_amdgpu_logical_device_t* logical_device,
    iree_allocator_t host_allocator,
    iree_hal_amdgpu_system_event_registration_t** out_registration);

// Retires the sticky device failure status of |registration| as a delivery
// target. Idempotent, and a no-op when |registration| is NULL.
//
// The device status and the queue targets are delivery targets with different
// lifetimes. A queue stops being one when it is about to be destroyed; the
// status stops being one when the logical device stops being reachable through
// the HAL, which is where its last possible reader goes away. Retiring them
// separately is what lets device teardown keep the queue delivery that releases
// its own waits while giving up the delivery nothing would ever read - and a
// registration left holding neither delivers nothing and claims nothing.
//
// Returns once no callback can be inside |registration|'s device status, on the
// same argument as retirement below, so the caller may then empty and free the
// slot. Takes the registry mutex internally; the caller must not hold it.
void iree_hal_amdgpu_system_event_retire_device_status(
    iree_hal_amdgpu_system_event_registration_t* registration);

// Removes |registration| from process-wide HSA system event delivery and frees
// it. No-op when |registration| is NULL.
//
// Returns once no callback can be inside |registration|. A callback holds the
// registry mutex across its whole traversal, so unlinking under that mutex both
// waits out a delivery already reading the list and keeps every later one from
// reaching this entry; a callback delivering to some other registration may
// still be running and is not something this has to exclude.
//
// Must be called once no published queue and no reader of the device's sticky
// failure status remains, and before the logical device allocation is freed.
// After this returns, an event for the device's agents is claimed by nobody.
void iree_hal_amdgpu_system_event_unregister_device(
    iree_hal_amdgpu_system_event_registration_t* registration);

// Returns the delivery target |registration| holds for |agent|, or NULL when
// |registration| is NULL or holds no target for that agent. Agents are 1:1 with
// the registered device's physical devices.
iree_hal_amdgpu_system_event_agent_target_t*
iree_hal_amdgpu_system_event_registration_lookup_agent(
    iree_hal_amdgpu_system_event_registration_t* registration,
    hsa_agent_t agent);

// Publishes the first |live_queue_count| entries of |host_queues| as the
// failure targets for |target|'s agent. No-op when |target| is NULL.
//
// Called as the last step of physical-device frontier assignment. A partially
// assigned physical device is never published, and a callback cannot observe
// an uninitialized queue. Sparse queue additions use the mask form below.
// Takes the registry mutex internally; the caller must not hold it.
void iree_hal_amdgpu_system_event_publish_queue_targets(
    iree_hal_amdgpu_system_event_agent_target_t* target,
    iree_hal_amdgpu_host_queue_t* host_queues,
    iree_host_size_t live_queue_count);

// Publishes the entries selected by |live_queue_mask| from the first
// |queue_capacity| entries of |host_queues| as failure targets for |target|'s
// agent. This is the sparse-slot form used when caller-selected private queues
// are materialized out of ordinal order. No-op when |target| is NULL.
//
// Takes the registry mutex internally; the caller must not hold it.
void iree_hal_amdgpu_system_event_publish_queue_target_mask(
    iree_hal_amdgpu_system_event_agent_target_t* target,
    iree_hal_amdgpu_host_queue_t* host_queues, iree_host_size_t queue_capacity,
    uint64_t live_queue_mask);

// Retires queue failure delivery for |target|'s agent. Idempotent, and a no-op
// when |target| is NULL. Returns once no callback can be inside |target|'s
// queues - the store is made under the same mutex a callback holds across its
// whole traversal - so the caller may then destroy them.
//
// Called after every queue of the physical device has passed its idle/error
// boundary and before any of them is destroyed. Takes the registry mutex
// internally; the caller must not hold it.
void iree_hal_amdgpu_system_event_retire_queue_targets(
    iree_hal_amdgpu_system_event_agent_target_t* target);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_AMDGPU_SYSTEM_EVENT_H_
