// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_DEVICE_DEVICE_H_
#define LOOM_SERVE_RUNTIME_DEVICE_DEVICE_H_

#include "loom/runtime/wake.h"
#include "loom/serve/runtime/device/pool.h"
#include "loom/serve/runtime/execution/execution.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/parameter/residency.h"
#include "loom/serve/runtime/program/jit.h"
#include "loom/serve/runtime/storage/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

// Application owner of all cooperating devices, one immutable HAL topology and
// frontier namespace, profiling session and retained host services. Models
// drain before this owner is destroyed. Group membership creates no execution
// edges and implies no cross-device buffer visibility.
typedef struct loom_serve_device_group_t loom_serve_device_group_t;

// Borrowed execution record within the application group. Queues and compiled
// products belong to the device; model storage belongs to explicitly bound
// pools. One task owner serializes model calls and pool residency transitions.
typedef struct loom_serve_device_t loom_serve_device_t;
typedef struct loom_serve_tokenizer_cache_t loom_serve_tokenizer_cache_t;

typedef struct loom_serve_device_options_t {
  // HAL driver selected by the application, borrowed during creation.
  iree_hal_driver_t* driver;
  // HAL device URI, borrowed during creation.
  iree_string_view_t uri;
  // Application-selected key, copied at creation. Empty is accepted only for
  // a singleton group, where the key is "device". Keys are visible ASCII.
  iree_string_view_t name;
  // Policy for device-owned completion waits, passed to HAL creation. NONE
  // disables optional active observation. The server selects ACTIVE by default.
  iree_async_wait_flags_t completion_wait_flags;
} loom_serve_device_options_t;

// Creates every requested HAL device from its application-selected driver,
// seals one group, then constructs each record's JIT cache and selects its
// first provisioned dispatch/transfer families. The first queue in each family
// serves preparation; model executions share the families' entire fixed
// inventory. Storage pools are constructed separately before models or
// observers attach. Missing roles fail creation. device_count must be nonzero
// and within the HAL topology limit; options are borrowed only during creation.
// This establishes runtime ownership, not compiler support or cross-device
// tensor access. Pass loom_serve_host_task_driver(host) for task devices so CPU
// queues share the host executor instead of creating private workers. Devices
// provision feedback for authored kernel assertions; sanitizer runtime state
// remains a separate driver policy. Runtime-service allocations are driver
// overhead outside the serving-memory budget. Retains host through all device
// and asynchronous service lifetimes. Failure returns no group and releases
// partial state, including earlier devices.
iree_status_t loom_serve_device_group_allocate(
    loom_serve_host_t* host, iree_host_size_t device_count,
    const loom_serve_device_options_t* options, iree_allocator_t host_allocator,
    loom_serve_device_group_t** out_group);

// Releases all device resources and the host reference without an implicit
// device wait. The caller first drains every model domain and releases all
// model/JIT/VM resources while host I/O remains alive. NULL is permitted.
// Outstanding models, banks, results, or elastic reservations preserve the
// entire group and return failure before partial teardown. Idle workspace is
// released before its physical owner; a native cleanup error preserves the
// remaining group resources for outer-owner cleanup.
iree_status_t loom_serve_device_group_free(loom_serve_device_group_t* group);

// Installs the serialized owner's nonblocking wake on every pool's residency,
// workspace, and native capacity work. Pool membership is sealed before this
// call. An empty observer detaches and joins its final callback access. The
// callback only schedules the owner; it cannot reenter the group. NULL group
// is accepted for applications with externally owned execution resources.
void loom_serve_device_group_set_observer(
    loom_serve_device_group_t* group, loom_runtime_wake_callback_t observer);

// Advances accepted workspace and capacity work before services advance.
// Every pool is collected even after a shared failure so independent releases
// can retire. Failures stop new admission but never destroy resource owners.
// Called on the serialized owner, including while draining failed work; no
// device or worker wait is performed. NULL group has no work.
iree_status_t loom_serve_device_group_advance(loom_serve_device_group_t* group);

// Collects and observes outstanding residency, workspace, and native capacity
// borrowers across all pools. The owner joins these obligations independently
// of model readiness or failed semaphores. True requires further owner progress
// before detaching its observer. NULL group has no pending work.
bool loom_serve_device_group_is_pending(loom_serve_device_group_t* group);

// Constructs a named group-owned pool with explicit consumers and policy.
// Called only during application setup, before resource observers and services
// attach. The group owns the returned pool; models borrow it through
// retirement. Pool identity, not device membership or physical location,
// establishes sharing. Qualifies simultaneous access from every selected
// device's dispatch/transfer families before creating payload. Copies the name;
// rejects empty/duplicate names, invalid membership, and incompatible device
// sanitizer policies.
iree_status_t loom_serve_device_group_create_pool(
    loom_serve_device_group_t* group, loom_serve_pool_binding_t binding,
    const loom_serve_pool_options_t* options, loom_serve_pool_t** out_pool);

// Application pool inventory; immutable after setup. Indices are resolved by
// the caller and are independent of execution device indices.
iree_host_size_t loom_serve_device_group_pool_count(
    const loom_serve_device_group_t* group);
loom_serve_pool_t* loom_serve_device_group_pool_at(
    const loom_serve_device_group_t* group, iree_host_size_t pool_index);
// Borrowed immutable pool names and device membership in pool inventory order.
const loom_serve_pool_binding_t* loom_serve_device_group_pool_bindings(
    const loom_serve_device_group_t* group);

// Number of immutable device records owned by the group.
iree_host_size_t loom_serve_device_group_count(
    const loom_serve_device_group_t* group);

// Immutable application keys in native group order, borrowed through teardown.
iree_string_view_list_t loom_serve_device_group_names(
    const loom_serve_device_group_t* group);

// Application key, distinct from the HAL identity or hardware display name.
iree_string_view_t loom_serve_device_name(const loom_serve_device_t* device);

// Borrows a record through group destruction. device_index is a caller-resolved
// ordinal less than the group's count.
loom_serve_device_t* loom_serve_device_group_at(
    loom_serve_device_group_t* group, iree_host_size_t device_index);

// Releases unused device-local compiled products and group-wide immutable
// tokenizers. Live model/load borrowers preserve their entries. These caches
// are independent of model storage and its pool policies.
// Called on the serialized host owner; does not wait for device work.
void loom_serve_device_trim_programs(loom_serve_device_t* device);

// Borrowed handles valid through group destruction. Every device record returns
// the same HAL group; execution bindings retain their shared queue inventory.
iree_hal_device_t* loom_serve_device_handle(const loom_serve_device_t* device);
// Borrowed application host, which outlives this device and its models.
loom_serve_host_t* loom_serve_device_host(const loom_serve_device_t* device);
// Exact immutable tokenizers shared by every model in the application group.
loom_serve_tokenizer_cache_t* loom_serve_device_tokenizer_cache(
    const loom_serve_device_t* device);
iree_hal_device_group_t* loom_serve_device_group(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_dispatch_queue(
    const loom_serve_device_t* device);
iree_hal_queue_t* loom_serve_device_transfer_queue(
    const loom_serve_device_t* device);
// Creates an owned model timeline over the explicitly selected pool. The
// serialized device owner assigns the least borrowed provisioned queues;
// final execution release returns that binding, including retained result and
// VM references. A shared family supplies one exact queue for both roles.
// The domain must retire before the device owner. Sharing physical backing
// introduces no dependency edges between models. Queue placement counts live
// executions, not requests or priority, and adds no native queues.
iree_status_t loom_serve_device_create_execution(
    loom_serve_device_t* device, loom_serve_pool_t* pool,
    loom_serve_execution_t** out_execution);

// Creates a model-local source catalog backed by this device's immutable
// compilation cache. The model retains the returned catalog and its private
// report/callback state; the device owns reusable completed products.
iree_status_t loom_serve_device_create_jit(
    loom_serve_device_t* device, const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_compilation_report_t* report, loom_serve_jit_t** out_jit);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_DEVICE_DEVICE_H_
