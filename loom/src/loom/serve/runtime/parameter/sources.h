// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PARAMETER_SOURCES_H_
#define LOOM_SERVE_RUNTIME_PARAMETER_SOURCES_H_

#include "iree/io/parameter_index.h"
#include "iree/io/parameter_provider.h"
#include "loom/serve/runtime/program/sources.h"

#ifdef __cplusplus
extern "C" {
#endif

// Immutable indexed checkpoint sources, independent of resident placement and
// policy input recipes. Each source has its own key namespace and imported-file
// cache. Owners may share this catalog across concurrent preparation and reads.
typedef struct loom_serve_parameter_sources_t loom_serve_parameter_sources_t;

// Opens ordered shard lists on a cold worker using the selected parameter I/O
// mode. Native file/direct modes index metadata without loading payload; the
// explicitly selected preload mode retains file contents in host memory.
// New sources follow the retained prefix; NULL supplies an empty prefix.
// Relative checkpoint paths resolve against sources.base_directory, not its
// source-code package. The namespace and paths are borrowed only for this call.
// A zero count retains prefix without allocating or opening anything.
// The provider permits independent spans in one operation to proceed without a
// catalog-imposed concurrency limit; actual queue/backend credits bound I/O.
iree_status_t loom_serve_parameter_sources_create(
    loom_serve_parameter_sources_t* prefix, const loom_serve_sources_t* sources,
    uint32_t count, const loom_model_file_list_t* paths,
    iree_allocator_t host_allocator,
    loom_serve_parameter_sources_t** out_sources);
void loom_serve_parameter_sources_retain(
    loom_serve_parameter_sources_t* sources);
void loom_serve_parameter_sources_release(
    loom_serve_parameter_sources_t* sources);

// Number of indexed sources, including the prefix. NULL has no sources.
uint32_t loom_serve_parameter_sources_count(
    const loom_serve_parameter_sources_t* sources);

// Borrows one source using an ordinal established by declaration capture. The
// index owns canonical keys and opaque metadata. Accepted reads retain the
// provider independently of the catalog or source program that selected it.
iree_io_parameter_index_t* loom_serve_parameter_sources_index(
    const loom_serve_parameter_sources_t* sources, uint32_t ordinal);
iree_io_parameter_provider_t* loom_serve_parameter_sources_provider(
    const loom_serve_parameter_sources_t* sources, uint32_t ordinal);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PARAMETER_SOURCES_H_
