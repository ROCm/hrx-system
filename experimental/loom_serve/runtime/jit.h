// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_JIT_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_JIT_H_

#include "experimental/loom_serve/runtime/command.h"
#include "loomc/config.h"
#include "loomc/sanitizer.h"
#include "loomc/task_pool.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-resident source catalog, compiler and live device profile, with a task
// queue on the caller's executor and worker-local scratch. Compilation calls
// on a catalog are serialized by the caller; native requests run concurrently.
// Execution of already prepared commands does not access this object.
typedef struct loom_serve_jit_t loom_serve_jit_t;
typedef struct loom_serve_jit_stage_t loom_serve_jit_stage_t;

// Loads source_directory/sources.txt: one relative Loom source path per line,
// with blank lines and lines beginning with # ignored. All providers are
// indexed once. The selected queue family, device and task executor are
// retained until destruction. The pool handle is borrowed only during creation.
iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loomc_task_pool_t* task_pool,
                                    iree_string_view_t source_directory,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit);
void loom_serve_jit_destroy(loom_serve_jit_t* jit);

// Receives the owned stage on success (NULL on failure) and owns status. Runs
// on a shared task worker after every native request has retired. May schedule
// more work, but never blocks or destroys the JIT/task queue from this
// callback.
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

// Records the compiled stage after the model has allocated its fixed parameter
// roots. The command retains executables and buffers, independently of stage.
iree_status_t loom_serve_jit_stage_record(
    const loom_serve_jit_stage_t* stage,
    const iree_hal_queue_family_t* queue_family,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_buffer_t* const* fixed_buffers,
    iree_hal_command_buffer_t** out_command);

// Compiles named roots from the model's portable VM source into one image.
// root_count is nonzero. On success, the caller owns
// the image with host_allocator and can transfer it to the VM bytecode module.
iree_status_t loom_serve_jit_compile_vm(iree_string_view_t source_path,
                                        iree_host_size_t root_count,
                                        const iree_string_view_t* roots,
                                        iree_allocator_t host_allocator,
                                        iree_const_byte_span_t* out_image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_JIT_H_
