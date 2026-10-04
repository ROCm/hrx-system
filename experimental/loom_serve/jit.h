// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_JIT_H_
#define EXPERIMENTAL_LOOM_SERVE_JIT_H_

#include "experimental/loom_serve/command.h"
#include "loomc/config.h"
#include "loomc/sanitizer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Model-resident source catalog, compiler, live device profile, and standard
// loomc task pool with worker-local scratch. Compilation calls are serialized
// by the caller; independent native requests within a call run concurrently.
// Execution of already prepared commands does not access this object.
typedef struct loom_serve_jit_t loom_serve_jit_t;
typedef struct loom_serve_jit_stage_t loom_serve_jit_stage_t;

// Loads source_directory/sources.txt: one relative Loom source path per line,
// with blank lines and lines beginning with # ignored. All providers are
// indexed once. The selected queue family and device are retained until
// destruction.
iree_status_t loom_serve_jit_create(iree_hal_device_t* device,
                                    iree_hal_queue_t* dispatch,
                                    iree_string_view_t source_directory,
                                    const loomc_sanitizer_options_t* sanitizer,
                                    iree_allocator_t host_allocator,
                                    loom_serve_jit_t** out_jit);
void loom_serve_jit_destroy(loom_serve_jit_t* jit);

// Specializes the unique exported definition of a command root and all
// reachable kernels. Declarations may precede its definition in the source
// catalog. Missing definitions and duplicate exports fail before compilation.
// The returned stage owns portable command bytes and loaded executables
// independently of the compiler.
// config is non-null; an empty binding list represents a fully specified root.
// No compiler subprocess, artifact directory, or disk cache is involved.
// All accepted tasks finish before this call returns, including on failure.
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

#endif  // EXPERIMENTAL_LOOM_SERVE_JIT_H_
