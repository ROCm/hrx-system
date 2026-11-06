// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PROGRAM_SOURCE_INDEX_H_
#define LOOM_RUNTIME_PROGRAM_SOURCE_INDEX_H_

#include "loom/runtime/program/compilation.h"
#include "loom/runtime/program/sources.h"
#include "loom/util/digest.h"
#include "loomc/context.h"
#include "loomc/link_index.h"
#include "loomc/source.h"

#ifdef __cplusplus
extern "C" {
#endif

// Builds one immutable linker index from a model's selected source catalog.
//
// Builtin Loom text and bytecode remain sources so the link index can preserve
// metadata-first and selective bytecode materialization. Build-configured
// foreign providers are imported once and snapshotted as modules. primary is
// an optional direct input whose matching card provider retains the primary
// role instead of being indexed twice.
//
// When out_identity is non-NULL it receives the exact ordered identity of all
// indexed source bytes, selected formats, provider options, and declared data
// files. No identity work or auxiliary-data I/O occurs when it is NULL.
iree_status_t loom_runtime_source_index_create(
    loomc_context_t* context, const loom_runtime_sources_t* sources,
    loomc_source_t* primary, loom_runtime_compilation_report_t* report,
    iree_allocator_t host_allocator, loom_sha256_t* out_identity,
    loomc_link_index_t** out_index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PROGRAM_SOURCE_INDEX_H_
