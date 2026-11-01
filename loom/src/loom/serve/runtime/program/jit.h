// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_
#define LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_

#include "loom/serve/runtime/execution/dispatch.h"
#include "loom/serve/runtime/host/host.h"
#include "loom/serve/runtime/program/compilation.h"
#include "loom/serve/runtime/program/sources.h"
#include "loom/target/arch/cmd/program.h"
#include "loomc/config.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-resident source catalog, compiler and live device profile, with a
// shared host compiler sink and host-owned worker scratch. Compilation calls
// on a catalog may overlap; parents and native requests use distinct worker
// scratch and join shared pending native results without waiting on workers.
// Execution of already prepared commands does not access this object.
// Native requests with identical sources, diagnostic identifiers, and ordered
// roots share immutable compilation artifacts across the host's matching
// target/pipeline domains. Each catalog loads its own device executable.
// Distinct inputs emitting identical bytes share immutable image storage and,
// within a catalog, loaded code; entry mappings and direct launch facts remain
// independent. Prepared stages retain native artifacts through their loaded
// input views, independently of catalog lifetime. Last catalog/view/producer
// release removes an artifact from lookup.
typedef struct loom_serve_jit_t loom_serve_jit_t;
typedef struct loom_serve_jit_cache_t loom_serve_jit_cache_t;
typedef struct loom_serve_jit_stage_t loom_serve_jit_stage_t;

// Loads and indexes the explicit text/bytecode providers once. Source views
// are borrowed only during this call. The selected queue family and device are
// retained until destruction. host supplies the shared compiler sink and worker
// ordinals; it outlives this catalog, its prepared stages, and every
// compilation completion. Compilation uses the linked backend selected from
// immutable HAL device facts, including its exact loader target and queue
// affinity. Source target declarations and launch configurations are
// specialized to that profile at each command's kernel roots before planning
// and native entry compilation. report receives copied evidence even on failure
// and outlives the catalog. Reports and callbacks publish in readiness order;
// independent compilations never wait for an earlier submission. Callbacks are
// serialized. Direct reads of report require every accepted compilation and
// report producer to retire.
iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    loom_serve_host_t* host,
                                    const loom_serve_sources_t* sources,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    loom_serve_compilation_report_t* report,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit);

// Creates a model-local catalog whose immutable completed products participate
// in the device-owned cache. Source contents, selected configuration,
// sanitizer semantics, stage specializations, and VM entry roots form exact
// identities; authored locator or card pointer identity never authorizes a
// hit. The catalog owns its publication/report and callbacks independently of
// every other model, and borrows scratch from the shared host's workers.
iree_status_t loom_serve_jit_create_cached(
    loom_serve_jit_cache_t* cache, loom_serve_host_t* host,
    const loom_serve_sources_t* sources,
    const loomc_sanitizer_options_t* sanitizer,
    loom_serve_compilation_report_t* report, loom_serve_jit_t** out_jit);

// Retains catalog ownership without retaining its borrowed host or report.
void loom_serve_jit_retain(loom_serve_jit_t* jit);

// Releases caller ownership without waiting or closing the host compiler queue.
// Accepted jobs retain the catalog until their compiler/native work and result
// binding retire. Callback state then survives independently, allowing
// catalog/device ownership to retire before completion becomes externally
// visible. Release does not cancel jobs. The host, originating device owner and
// caller-owned root/config storage still outlive all accepted completions. A
// callback may release a separately retained caller reference.
void loom_serve_jit_release(loom_serve_jit_t* jit);

// Retained exact-input reuse in one immutable source/target generation. These
// counters exclude the fixed compiler/source index, compiler workspaces,
// stages, command recordings, and driver-private allocations made while loading
// code.
typedef struct loom_serve_jit_cache_storage_t {
  // Loaded native inputs still indexed for exact reuse.
  iree_host_size_t count;
  // Distinct emitted images loaded, independent of native-input count.
  iree_host_size_t executable_count;
  // Referenced bytes for keys, images, entries, diagnostics, and local lookup
  // arrays. Shared artifacts are charged fully to each generation so eviction
  // budgets stay stable as peers retire; these charges are not additive
  // physical-memory counts. Host-owned domain/index capacity is excluded.
  iree_host_size_t host_bytes;
  // Lookup capacity included in host_bytes; retained as catalog infrastructure
  // independently of individual native entries.
  iree_host_size_t index_bytes;
  // Artifact bytes supplied to the loader, counted once per cached executable.
  iree_host_size_t code_bytes;
} loom_serve_jit_cache_storage_t;

// Owner-only operations after all accepted compilations have completed.
// Trimming removes only entries unused by compiled stages, retained direct
// dispatches, and HAL consumers. Recordings preserve exact native reuse after
// their compiler stages retire. Concurrent HAL release may defer collection
// until the next trim; no consumer can be created without an existing owner.
// Neither operation allocates, compiles, or waits.
loom_serve_jit_cache_storage_t loom_serve_jit_cache_storage(
    const loom_serve_jit_t* jit);
void loom_serve_jit_trim(loom_serve_jit_t* jit);

// Borrowed evidence accumulator for VM/parameter preparation while no stage
// compilations are outstanding. Compiler tasks merge before their completions.
loom_serve_compilation_report_t* loom_serve_jit_report(loom_serve_jit_t* jit);

// Ordinary catalog compilation queue. An individual request may instead use
// another sink sharing this host's executor and mutually exclusive worker
// ordinals; the request carries that sink through every native child.
loomc_task_sink_t loom_serve_jit_task_sink(const loom_serve_jit_t* jit);

// Copies accumulated evidence or merges private producer evidence under the
// publication lock. Neither waits for outstanding compilation or callbacks.
void loom_serve_jit_report_snapshot(
    loom_serve_jit_t* jit, loom_serve_compilation_report_t* out_report);
void loom_serve_jit_report_merge(loom_serve_jit_t* jit,
                                 const loom_serve_compilation_report_t* report);

// Borrows the catalog's shared native and VM source grammar through jit
// lifetime.
loomc_target_environment_t* loom_serve_jit_target_environment(
    const loom_serve_jit_t* jit);

// Borrows an immutable command layout before its native requests have joined.
// Called at most once, on successful command construction. May overlap other
// layout callbacks and native compilation, but precedes this job's completion.
// The layout lives in the final stage, including when native compilation fails;
// consumers retain it only until they free that stage. Never blocks a worker.
typedef void (*loom_serve_jit_layout_fn_t)(void* user_data,
                                           const loom_cmd_program_t* program);

// Receives the owned stage and status. Failure can carry a partially
// constructed stage owning a previously published layout; the recipient always
// frees it. Runs on a shared task worker after every native request and
// compiler access has retired. May schedule more work or destroy its completed
// catalog, but never blocks or releases the final host owner from this
// callback.
typedef void (*loom_serve_jit_completion_fn_t)(void* user_data,
                                               iree_status_t status,
                                               loom_serve_jit_stage_t* stage);

// Accepts a compilation without waiting. New product/native tasks use sink,
// which shares the catalog's host executor and dense worker ordinals and
// outlives completion. Reused pending natives retain their original scheduling
// domain. Final binding and callbacks run on the shared executor.
// Published native requests are deduplicated and submitted during command
// construction; their compilation does not depend on the completed parent.
// Completion joins all accepted native work even when parent construction
// fails, retaining successfully loaded natives for subsequent compilations.
// Success invokes completion exactly once, possibly before return. Rejection
// invokes no callback. The job retains jit through compiler retirement, while
// callback publication has an independent lifetime. The caller keeps root,
// config and its referenced storage alive through completion. Compilations may
// overlap on the same catalog, including exact-input requests whose native work
// is still pending. Completions publish in readiness order, after their
// borrowed compiler inputs and catalog ownership retire. Different catalogs
// share workers and progress independently.
iree_status_t loom_serve_jit_compile_async(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_view_t root,
    const loomc_config_options_t* config, loom_serve_jit_layout_fn_t layout,
    loom_serve_jit_completion_fn_t completion, void* user_data);

// Compiles a kernel root and its zero-argument launch configuration for direct
// queue submission. Uses the same catalog, exact-input cache, shared workers
// and completion contract as command compilation. The entry takes only buffer
// arguments. Model values belong in specialization or device input packets.
iree_status_t loom_serve_jit_compile_dispatch_async(
    loom_serve_jit_t* jit, loomc_task_sink_t sink, iree_string_view_t root,
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
void loom_serve_jit_stage_retain(loom_serve_jit_stage_t* stage);
bool loom_serve_jit_stage_is_shared(const loom_serve_jit_stage_t* stage);
void loom_serve_jit_stage_free(loom_serve_jit_stage_t* stage);
// Requested host bytes owned by this stage, excluding its shared native code
// and any independently recorded command. Does not inspect a source catalog.
iree_host_size_t loom_serve_jit_stage_storage_size(
    const loom_serve_jit_stage_t* stage);
const loom_cmd_program_t* loom_serve_jit_stage_program(
    const loom_serve_jit_stage_t* stage);

// Borrows the direct stage, or NULL for a command stage. Conversely,
// stage_program returns NULL for a direct stage. Retain before releasing stage.
loom_serve_dispatch_t* loom_serve_jit_stage_dispatch(
    const loom_serve_jit_stage_t* stage);

// Records the compiled stage after the model has allocated its fixed parameter
// roots. The command retains executables and buffers, independently of stage.
// NULL range records the whole program; otherwise it selects complete barrier
// waves with unchanged bindings and workspace, as in the shared HAL command
// buffer materializer.
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

// Compiles or reuses a VM image in jit's exact source generation. Every hit
// returns independently owned bytes for a fresh VM module and replays the
// original diagnostic evidence into jit's model-local report.
iree_status_t loom_serve_jit_compile_vm_cached(
    loom_serve_jit_t* jit, const loom_serve_sources_t* sources,
    iree_string_view_t source_path, iree_host_size_t root_count,
    const iree_string_view_t* roots, iree_allocator_t host_allocator,
    iree_const_byte_span_t* out_image);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_SERVE_RUNTIME_PROGRAM_JIT_H_
