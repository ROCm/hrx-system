// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_PROGRAM_H_
#define EXPERIMENTAL_LOOM_SERVE_PROGRAM_H_

#include "iree/vm/environment.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// One source-JIT host program and serialized invocation, shared across
// requests. Native modules own any asynchronous resources submitted by the
// program.
typedef struct loom_serve_program_t loom_serve_program_t;

// Compiles root from source_path, links it as module "model" with libraries,
// and creates one process. The environment's provider scopes must outlive the
// program and any returned references. Libraries are retained by the program.
// No compiler storage or source-file mapping survives this call.
iree_status_t loom_serve_program_create(iree_vm_environment_t* environment,
                                        iree_string_view_t source_path,
                                        iree_string_view_t root,
                                        iree_vm_module_span_t libraries,
                                        iree_allocator_t host_allocator,
                                        loom_serve_program_t** out_program);

// Releases VM ownership only. The caller joins native asynchronous work before
// destroying the program or any host payloads that work borrows.
void loom_serve_program_destroy(loom_serve_program_t* program);

// Borrowed process and invocation for ordinary VM lookup/invoke APIs. Calls
// using this invocation are serialized by its owner; sessions are plain data.
iree_vm_process_t* loom_serve_program_process(loom_serve_program_t* program);
iree_vm_invocation_t* loom_serve_program_invocation(
    loom_serve_program_t* program);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_PROGRAM_H_
