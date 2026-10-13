// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_

#include "iree/vm/environment.h"
#include "iree/vm/invocation.h"
#include "iree/vm/process.h"
#include "iree/vm/program.h"
#include "loom/serve/runtime/program/compilation.h"

#ifdef __cplusplus
extern "C" {
#endif

// One source-JIT host program and serialized invocation, shared across
// requests. Native modules own any asynchronous resources submitted by the
// program.
typedef struct loom_serve_program_t loom_serve_program_t;

// Compiles source into an owned module named "model" without linking native
// libraries or creating a process. The environment supplies reference types;
// its provider scopes outlive the module and escaped references. This finite
// CPU operation may run on a private preparation task. No compiler storage or
// source mapping survives; report accumulates copied diagnostics on failure.
iree_status_t loom_serve_program_compile(
    iree_vm_environment_t* environment, iree_string_view_t source_path,
    iree_host_size_t root_count, const iree_string_view_t* roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);

// Links prepared code to its runtime capabilities and creates one process.
// Retains module and libraries on success; caller ownership never transfers.
// Performs no compilation or source I/O. Native library creation and linking
// run on the owner of the shared resources those capabilities expose.
iree_status_t loom_serve_program_link(iree_vm_module_t* module,
                                      iree_vm_module_span_t libraries,
                                      iree_allocator_t host_allocator,
                                      loom_serve_program_t** out_program);

// Compiles the nonempty roots list from source_path into one module "model",
// links it with libraries, and creates one process. The environment's provider
// scopes must outlive the program and any returned references. Libraries are
// retained by the program.
// No compiler storage or source-file mapping survives this call.
// report accumulates copied source diagnostics, including on failure.
iree_status_t loom_serve_program_create(
    iree_vm_environment_t* environment, iree_string_view_t source_path,
    iree_host_size_t root_count, const iree_string_view_t* roots,
    iree_vm_module_span_t libraries, loom_serve_compilation_report_t* report,
    iree_allocator_t host_allocator, loom_serve_program_t** out_program);

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

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_PROGRAM_H_
