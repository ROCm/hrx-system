// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_

#include "loom/serve/runtime/execution/dispatch.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/program/command.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loomc/config.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-resident source catalog, compiler and live device profile, with a
// shared host compiler sink and catalog-local worker scratch. Compilation calls
// on a catalog are serialized by the caller; native requests run concurrently.
// Execution of already prepared commands does not access this object.
// Native requests with identical source bytes, roots and specialization names
// share loaded executables inside this catalog's fixed target/pipeline domain.
// Prepared stages retain their executables independently of catalog lifetime.
typedef struct loom_serve_jit_t loom_serve_jit_t;
typedef struct loom_serve_jit_stage_t loom_serve_jit_stage_t;

// Loads and indexes the explicit text/bytecode providers once. Source views
// are borrowed only during this call. The selected queue family and device are
// retained until destruction. host supplies the shared compiler sink and worker
// ordinals; it outlives this catalog and every compilation completion.
// Compilation uses the linked backend selected from immutable HAL device
// facts, including its exact loader target and queue affinity. Source target
// declarations are specialized to that profile at each command's kernel roots.
// report receives copied evidence even on failure and outlives the catalog.
// Only the serialized compilation owner accesses it between completions.
iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loom_serve_host_t* host,
                                    const loom_serve_sources_t* sources,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    loom_serve_compilation_report_t* report,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit);
// Releases a quiescent catalog without waiting for other catalogs or closing
// the host compiler queue. No accepted compilation may remain before its
// completion callback. Destruction from that callback is permitted.
void loom_serve_jit_destroy(loom_serve_jit_t* jit);

// Borrowed evidence accumulator for serialized VM/parameter preparation using
// this catalog. Native tasks merge into it before publishing completion.
loom_serve_compilation_report_t* loom_serve_jit_report(loom_serve_jit_t* jit);

// Receives the owned stage on success (NULL on failure) and owns status. Runs
// on a shared task worker after every native request and compiler access has
// retired. May schedule more work or destroy its completed catalog, but never
// blocks or releases the final host owner from this callback.
typedef void (*loom_serve_jit_completion_fn_t)(void* user_data,
                                               iree_status_t status,
                                               loom_serve_jit_stage_t* stage);

// Accepts a compilation without waiting. Product construction, native kernel
// compilation and final binding all run on the caller's shared executor.
// Success invokes completion exactly once, possibly before return. Rejection
// invokes no callback. The caller keeps jit, root, config and its referenced
// storage alive through completion, and submits at most one compile per jit
// at a time. Different catalogs share workers and progress independently.
iree_status_t loom_serve_jit_compile_async(
    loom_serve_jit_t* jit, iree_string_view_t root,
    const loomc_config_options_t* config,
    loom_serve_jit_completion_fn_t completion, void* user_data);

// Compiles a kernel root and its zero-argument launch configuration for direct
// queue submission. Uses the same catalog, exact-input cache, shared workers
// and completion contract as command compilation. The entry takes only buffer
// arguments. Model values belong in specialization or device input packets.
iree_status_t loom_serve_jit_compile_dispatch_async(
    loom_serve_jit_t* jit, iree_string_view_t root,
    const loomc_config_options_t* config,
    loom_serve_jit_completion_fn_t completion, void* user_data);

// Specializes the unique exported definition of a command root and all
// reachable kernels. Declarations may precede its definition in the source
// catalog. Missing definitions and duplicate exports fail before compilation.
// The returned stage owns portable command bytes and loaded executables
// independently of the compiler.
// config is non-null; an empty binding list represents a fully specified root.
// No compiler subprocess, artifact directory, or disk cache is involved.
// All accepted tasks finish before this call returns, including on failure.
// This cold CLI/construction join must not run on a task-executor worker.
iree_status_t loom_serve_jit_compile(loom_serve_jit_t* jit,
                                     iree_string_view_t root,
                                     const loomc_config_options_t* config,
                                     loom_serve_jit_stage_t** out_stage);
void loom_serve_jit_stage_destroy(loom_serve_jit_stage_t* stage);
const loom_cmd_program_t* loom_serve_jit_stage_program(
    const loom_serve_jit_stage_t* stage);

// Borrows the direct stage, or NULL for a command stage. Conversely,
// stage_program returns NULL for a direct stage. Retain before releasing stage.
loom_serve_dispatch_t* loom_serve_jit_stage_dispatch(
    const loom_serve_jit_stage_t* stage);

// Records the compiled stage after the model has allocated its fixed parameter
// roots. The command retains executables and buffers, independently of stage.
// NULL range records the whole program; otherwise it selects complete barrier
// waves with unchanged bindings and workspace, as in command_create.
iree_status_t loom_serve_jit_stage_record(
    const loom_serve_jit_stage_t* stage,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    const loom_cmd_program_command_range_t* range,
    iree_hal_buffer_t* const* fixed_buffers,
    iree_hal_command_buffer_t** out_command);

// Compiles named roots from the model's portable VM source into one image.
// root_count is nonzero. On success, the caller owns
// the image with host_allocator and can transfer it to the VM bytecode module.
iree_status_t loom_serve_jit_compile_vm(iree_string_view_t source_path,
                                        iree_host_size_t root_count,
                                        const iree_string_view_t* roots,
                                        loom_serve_compilation_report_t* report,
                                        iree_allocator_t host_allocator,
                                        iree_const_byte_span_t* out_image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_
