// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_NATIVE_ARTIFACT_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_NATIVE_ARTIFACT_H_

#include "iree/base/byte_sequence.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loomc/launch_config.h"
#include "loomc/target.h"

#ifdef __cplusplus
extern "C" {
#endif

// Host-owned compilation rendezvous. Its indexes are non-owning: catalogs,
// loaded input views, and pending producers retain artifacts. Last release
// removes their keys.
// No HAL handle, device, queue, catalog, or compiler workspace is retained
// here.
typedef struct loom_serve_native_cache_t loom_serve_native_cache_t;
typedef struct loom_serve_native_domain_t loom_serve_native_domain_t;
typedef struct loom_serve_native_artifact_t loom_serve_native_artifact_t;

typedef struct loom_serve_native_domain_options_t {
  // Complete immutable compiler profile, retained by the domain. Its allocator
  // and any allocator context must outlive the host cache.
  loomc_target_profile_t* profile;
  // Process-lifetime emitter name selected by serving target composition.
  iree_string_view_t artifact_format;
  // Exact source-to-low pipeline control-flow policy.
  loomc_target_control_flow_lowering_t control_flow_lowering;
  // Validated assertion classes used to prepare the pipeline.
  loomc_sanitizer_checks_t sanitizer_checks;
  // Validated sanitizer behavior flags.
  loomc_sanitizer_flags_t sanitizer_flags;
  // Validated target assertion reporting behavior.
  loomc_sanitizer_reporting_mode_t sanitizer_reporting_mode;
} loom_serve_native_domain_options_t;

iree_status_t loom_serve_native_cache_allocate(
    iree_allocator_t host_allocator, loom_serve_native_cache_t** out_cache);
// All domains, artifacts, and callbacks have retired before host teardown.
void loom_serve_native_cache_free(loom_serve_native_cache_t* cache);
iree_allocator_t loom_serve_native_cache_allocator(
    const loom_serve_native_cache_t* cache);

// Acquires an exact compilation domain after pipeline option validation.
// Catalog creation is the only profile-comparison path; native lookups do not
// project, compare, or rediscover device facts.
iree_status_t loom_serve_native_domain_acquire(
    loom_serve_native_cache_t* cache,
    const loom_serve_native_domain_options_t* options,
    loom_serve_native_domain_t** out_domain);
void loom_serve_native_domain_release(loom_serve_native_domain_t* domain);
iree_allocator_t loom_serve_native_domain_allocator(
    const loom_serve_native_domain_t* domain);

typedef struct loom_serve_native_result_t {
  // Owned terminal compilation status; each device clones it independently.
  iree_status_t status;
  // Host-owned diagnostic snapshot, or NULL for a clean compilation.
  loom_serve_compilation_diagnostics_t* diagnostics;
  // Owned emitted bytes; publication coalesces and interns them once.
  iree_byte_sequence_t* image;
  // Number of roots allocated with the artifact, in exact input order.
  iree_host_size_t root_count;
  // Host-owned names populated by the producer before publication.
  iree_string_view_t* roots;
  // Direct launch facts belong to this input, not to its interned image.
  loomc_launch_config_t launch_config;
  // Whether this result describes a direct dispatch instead of command roots.
  bool direct_dispatch;
} loom_serve_native_result_t;

typedef struct loom_serve_native_subscription_t {
  // Called after compiler retirement and without registry locks. May release
  // catalog ownership; a producer reference spans the whole fan-out.
  void (*ready)(void* user_data, const loom_serve_native_result_t* result);
  // Borrowed subscriber state kept alive by its pending parent work.
  void* user_data;
  // Registry-owned linkage until detached at publication.
  struct loom_serve_native_subscription_t* next;
} loom_serve_native_subscription_t;

// Acquires one consumer reference by exact framed bytes. The key builder uses
// the domain allocator; a new entry takes its storage, while a hit leaves it
// with the caller. A new entry also owns one producer reference, consumed by
// publish, and reports out_created. Pending consumers are subscribed; a ready
// hit reports out_ready without invoking the callback inline. out_artifact is
// set before a concurrent publisher can invoke the subscription.
iree_status_t loom_serve_native_artifact_acquire(
    loom_serve_native_domain_t* domain, iree_string_builder_t* key,
    iree_host_size_t root_count, loom_serve_native_subscription_t* subscription,
    loom_serve_native_artifact_t** out_artifact, bool* out_created,
    bool* out_ready);
void loom_serve_native_artifact_retain(loom_serve_native_artifact_t* artifact);
void loom_serve_native_artifact_release(loom_serve_native_artifact_t* artifact);

// The producer alone mutates this result before publish. Consumers borrow it
// only after readiness, through their artifact reference. Owned result storage
// uses the domain allocator, independently of the producing catalog.
loom_serve_native_result_t* loom_serve_native_artifact_result(
    loom_serve_native_artifact_t* artifact);
iree_const_byte_span_t loom_serve_native_artifact_key(
    const loom_serve_native_artifact_t* artifact);

// Finishes immutable storage, publishes every pending subscriber, and consumes
// the producer reference. Failure removes the input from lookup before fan-out
// so future attempts can retry without disturbing current consumers.
void loom_serve_native_artifact_publish(loom_serve_native_artifact_t* artifact);

// Stable referenced working-set charge, excluding image bytes charged by the
// device executable cache and host-owned lookup capacity. Charges from several
// catalogs referencing one artifact cannot be summed as physical memory use.
iree_host_size_t loom_serve_native_artifact_storage_size(
    const loom_serve_native_artifact_t* artifact);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_NATIVE_ARTIFACT_H_
