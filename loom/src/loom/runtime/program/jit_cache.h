// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_JIT_CACHE_H_
#define LOOM_RUNTIME_PROGRAM_JIT_CACHE_H_

#include "iree/hal/api.h"
#include "loom/runtime/program/compilation.h"
#include "loom/util/digest.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct loom_runtime_jit_cache_t loom_runtime_jit_cache_t;
typedef struct loom_runtime_jit_generation_t loom_runtime_jit_generation_t;
typedef struct loom_runtime_jit_stage_t loom_runtime_jit_stage_t;

// Allocates an empty process-local compilation cache. The cache owns one
// reference to every exact source generation until trim or destruction.
iree_status_t loom_runtime_jit_cache_allocate(
    iree_hal_device_t* device, iree_hal_queue_t* dispatch,
    iree_allocator_t host_allocator, loom_runtime_jit_cache_t** out_cache);

// Releases cache ownership. Generations still retained by active catalogs
// remain valid independently of the cache and destroy themselves later.
void loom_runtime_jit_cache_free(loom_runtime_jit_cache_t* cache);

// Releases source generations that have no active catalog. Compiled stages
// and VM images retained by such generations retire with them.
void loom_runtime_jit_cache_trim(loom_runtime_jit_cache_t* cache);

// Borrows the cache's fixed device and queue domain through cache lifetime.
iree_hal_device_t* loom_runtime_jit_cache_device(
    const loom_runtime_jit_cache_t* cache);
iree_hal_queue_t* loom_runtime_jit_cache_dispatch(
    const loom_runtime_jit_cache_t* cache);
iree_allocator_t loom_runtime_jit_cache_allocator(
    const loom_runtime_jit_cache_t* cache);

// Acquires one generation for an exact source, configuration, and sanitizer
// identity. A newly created generation contains no compiled products.
iree_status_t loom_runtime_jit_cache_acquire(
    loom_runtime_jit_cache_t* cache, const loom_sha256_t* identity,
    loom_runtime_jit_generation_t** out_generation);
void loom_runtime_jit_generation_retain(
    loom_runtime_jit_generation_t* generation);
void loom_runtime_jit_generation_release(
    loom_runtime_jit_generation_t* generation);

// Releases cached stages with no model consumer. Active stage and executable
// owners survive and keep their exact generation entry reusable.
void loom_runtime_jit_generation_trim(
    loom_runtime_jit_generation_t* generation);

// Looks up or publishes a fully linked immutable stage. Lookup returns a
// retained stage. Publication retains stage only when its key is new; an
// equivalent concurrent producer keeps its independent result.
loom_runtime_jit_stage_t* loom_runtime_jit_generation_lookup_stage(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_compilation_report_t* report);
iree_status_t loom_runtime_jit_generation_publish_stage(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_jit_stage_t* stage,
    const loom_runtime_compilation_report_t* report);

// Looks up or publishes successful VM bytecode. Lookup clones the image for
// transfer to a fresh VM module and merges the original diagnostics. A miss
// leaves out_image empty and sets out_found false. Publication copies both
// image and diagnostics when its key is new.
iree_status_t loom_runtime_jit_generation_lookup_vm(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    loom_runtime_compilation_report_t* report, iree_allocator_t host_allocator,
    bool* out_found, iree_const_byte_span_t* out_image);
iree_status_t loom_runtime_jit_generation_publish_vm(
    loom_runtime_jit_generation_t* generation, const loom_sha256_t* key,
    iree_const_byte_span_t image,
    const loom_runtime_compilation_report_t* report);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_JIT_CACHE_H_
