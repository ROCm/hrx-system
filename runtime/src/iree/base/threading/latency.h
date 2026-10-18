// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_BASE_THREADING_LATENCY_H_
#define IREE_BASE_THREADING_LATENCY_H_

#include "iree/base/api.h"

#ifdef __cplusplus
extern "C" {
#endif

// Options for a latency-sensitive activity.
typedef struct iree_latency_scope_options_t {
  // Optional CPU idle-state exit latency limit, in microseconds. -1 selects
  // native latency-sensitive policy. Nonnegative values require a platform
  // supporting a numeric limit (currently Linux); other platforms reject them.
  // Zero requests the lowest possible latency and may prevent CPU sleep.
  int32_t maximum_cpu_idle_latency_us;
} iree_latency_scope_options_t;

// Returns options selecting native latency-sensitive policy.
static inline iree_latency_scope_options_t iree_latency_scope_options_default(
    void) {
  iree_latency_scope_options_t options = {-1};
  return options;
}

// An application-owned request for latency-sensitive host power management.
//
// The scope covers a latency-sensitive activity, including time its threads
// spend blocked on device work or I/O. For example, a server can keep one scope
// across a busy period and its idle grace interval, then end it when idle.
// Beginning and ending a scope can invoke the OS and wake idle CPUs; these are
// activity boundaries, not per-operation or per-wait actions.
// The request does not acquire device runtime power or keep a GPU/NPU awake.
//
// Independent scopes may overlap and end in any order. Native policies differ:
// - Linux: a system-wide CPU idle-state latency request. Native policy requests
//   five microseconds; options can tighten or relax that limit. The kernel uses
//   the lowest request across all processes. This does not set CPU or device
//   clocks.
// - Windows: process-wide HighQoS, favoring performance in processor power
//   management and core selection. Scopes in independently embedded runtime
//   copies share ownership of this policy. The first scope saves the
//   execution-speed throttling policy; the last restores it. While scopes are
//   active, changes to that process policy must go through this API.
// - macOS: a latency-critical NSProcessInfo activity that suppresses App Nap
//   and requests precise timer and I/O behavior. System and display sleep
//   policy is unchanged.
// These policies do not guarantee an end-to-end response time or equal timing
// across platforms. Power consumption may increase while a scope is active.
//
// A scope has one owner. Zero initialization produces an inactive scope. A live
// scope must not be copied or begun again. Ownership may be transferred between
// threads with external synchronization; no thread affinity is required.
// The scope must end through the runtime copy that began it, before that copy
// is unloaded. Independent runtime copies coordinate native policy ownership,
// not the ownership of each other's scope storage.
typedef struct iree_latency_scope_t {
  // Opaque native request ownership; zero when inactive.
  uintptr_t handle;
} iree_latency_scope_t;

// Attempts to begin a latency-sensitive activity using |options|.
// |out_scope| must not already own a request and is inactive on failure.
//
// Linux requires permission to open /dev/cpu_dma_latency. The request uses a
// close-on-exec descriptor; forked children inherit it until close or exec.
// Returns true if the OS accepted the request. Invalid options, unsupported
// policies, insufficient permissions, or unavailable native resources return
// false without constructing an error status. The request is an optional hint;
// callers must support running without it.
IREE_API_EXPORT bool iree_latency_scope_begin(
    iree_latency_scope_options_t options, iree_latency_scope_t* out_scope);

// Ends the request owned by |scope| and makes it inactive.
// Safe for an inactive scope, including after a failed begin.
// If restoring native policy fails, returns an error and leaves the scope
// active so its owner can retry. Ending an inactive scope changes no policy.
IREE_API_EXPORT iree_status_t
iree_latency_scope_end(iree_latency_scope_t* scope);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // IREE_BASE_THREADING_LATENCY_H_
