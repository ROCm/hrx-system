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
// on a catalog may overlap; parents and native requests use distinct worker
// scratch and join shared pending native results without waiting on workers.
// Execution of already prepared commands does not access this object.
// Native requests with identical source bytes and ordered source roots
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
// declarations and launch configurations are specialized to that profile at
// each command's kernel roots before planning and native entry compilation.
// report receives copied evidence even on failure and outlives the catalog.
// Reports publish in submission order. The owner reads the accumulator only
// after all accepted compilations complete; callbacks are serialized.
iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loom_serve_host_t* host,
                                    const loom_serve_sources_t* sources,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    loom_serve_compilation_report_t* report,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit);

// Retains catalog ownership without retaining its borrowed host or report.
void loom_serve_jit_retain(loom_serve_jit_t* jit);

// Releases caller ownership without waiting or closing the host compiler queue.
// Accepted jobs retain the catalog until their compiler/native work and result
// binding retire. Ordered callback state then survives independently, allowing
// catalog/device ownership to retire before completion becomes externally
// visible. Release does not cancel jobs. The host, originating device owner and
// caller-owned root/config storage still outlive all accepted completions. A
// callback may release a separately retained caller reference.
void loom_serve_jit_release(loom_serve_jit_t* jit);

// Borrowed evidence accumulator for VM/parameter preparation while no stage
// compilations are outstanding. Compiler tasks merge before their completions.
loom_serve_compilation_report_t* loom_serve_jit_report(loom_serve_jit_t* jit);

// Borrows the catalog's shared native and VM source grammar through jit
// lifetime.
loomc_target_environment_t* loom_serve_jit_target_environment(
    const loom_serve_jit_t* jit);

// Receives the owned stage on success (NULL on failure) and owns status. Runs
// on a shared task worker after every native request and compiler access has
// retired. May schedule more work or destroy its completed catalog, but never
// blocks or releases the final host owner from this callback.
typedef void (*loom_serve_jit_completion_fn_t)(void* user_data,
                                               iree_status_t status,
                                               loom_serve_jit_stage_t* stage);

// Accepts a compilation without waiting. Product construction, native kernel
// compilation and final binding all run on the caller's shared executor.
// Published native requests are deduplicated and submitted during command
// construction; their compilation does not depend on the completed parent.
// Completion joins all accepted native work even when parent construction
// fails, retaining successfully loaded natives for subsequent compilations.
// Success invokes completion exactly once, possibly before return. Rejection
// invokes no callback. The job retains jit through compiler retirement, while
// callback publication has an independent lifetime. The caller keeps root,
// config and its referenced storage alive through completion. Compilations may
// overlap on the same catalog, including exact-input requests whose native work
// is still pending. Callbacks publish in submission order, after their borrowed
// compiler inputs and catalog ownership retire. Different catalogs share
// workers and progress independently.
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
void loom_serve_jit_stage_free(loom_serve_jit_stage_t* stage);
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

// Links the model's portable VM source against the declared library providers
// and compiles named roots into one image. The entry source is primary input;
// library-private symbols remain private. The entry may also occur in sources.
// target_environment supplies VM compilation and the complete catalog grammar;
// it is borrowed only for this call. Only the named roots target the VM.
// root_count is nonzero. On success, the caller owns
// the image with host_allocator and can transfer it to the VM bytecode module.
iree_status_t loom_serve_jit_compile_vm(
    loomc_target_environment_t* target_environment,
    const loom_serve_sources_t* sources, iree_string_view_t source_path,
    iree_host_size_t root_count, const iree_string_view_t* roots,
    loom_serve_compilation_report_t* report, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_
