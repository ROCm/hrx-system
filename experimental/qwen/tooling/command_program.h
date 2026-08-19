// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 WITH LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_
#define EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_

#include "experimental/qwen/tooling/runtime.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "loomc/config.h"
#include "loomc/target/cmd/program.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Compiled and materialized command-program root.
typedef struct qwen_tooling_command_program_t qwen_tooling_command_program_t;

// Compiled multi-root command-program set sharing one immutable parameter pack.
typedef struct qwen_tooling_command_program_set_t
    qwen_tooling_command_program_set_t;

// Source and root selection used to prepare one command-program set.
typedef struct qwen_tooling_command_program_set_options_t {
  // Diagnostic identifier for the borrowed source contents.
  iree_string_view_t source_identifier;
  // Borrowed sealed Loom module containing the selected roots and kernels.
  iree_const_byte_span_t source_contents;
  // Public command-program root names to prepare and materialize.
  const iree_string_view_t* root_names;
  // Number of entries in |root_names|.
  iree_host_size_t root_count;
  // Compile-time config values shared by every selected root.
  loomc_config_options_t config;
} qwen_tooling_command_program_set_options_t;

// Compiles and materializes sealed command-program roots for the runtime
// context's device. Logical parameter roots are mapped into subranges of one
// union parameter slab. Repeated keys share their exact payload bytes, and an
// incompatible relative layout fails instead of duplicating storage.
// Parameters are gathered once while distinct kernel units compile, then all
// materialized command buffers retain the shared fixed resources.
iree_status_t qwen_tooling_command_program_set_create(
    qwen_tooling_runtime_context_t* runtime_context,
    const qwen_tooling_command_program_set_options_t* options,
    iree_allocator_t host_allocator,
    qwen_tooling_command_program_set_t** out_program_set);

// Releases a command-program set and every borrowed root. Passing NULL is
// allowed.
void qwen_tooling_command_program_set_release(
    qwen_tooling_command_program_set_t* program_set);

// Looks up a materialized root by canonical public name. Returns NULL when no
// selected root has |root_name|. The result is borrowed from |program_set|.
qwen_tooling_command_program_t* qwen_tooling_command_program_set_lookup(
    const qwen_tooling_command_program_set_t* program_set,
    iree_string_view_t root_name);

// Returns immutable ABI metadata borrowed from program.
const loomc_cmd_program_info_t* qwen_tooling_command_program_info(
    const qwen_tooling_command_program_t* program);

// Returns the reusable command buffer borrowed from program.
iree_hal_command_buffer_t* qwen_tooling_command_program_command_buffer(
    const qwen_tooling_command_program_t* program);

// Populates caller-owned config data for one issue of a dynamic command
// program. |argument_bits| maps positionally to the command root's scalar
// arguments using the loomc launch-config calling convention. |config_data|
// must provide the byte length and alignment published by program info.
iree_status_t qwen_tooling_command_program_populate_config(
    qwen_tooling_command_program_t* program, const uint64_t* argument_bits,
    iree_host_size_t argument_count, iree_byte_span_t config_data);

// Returns the shared packed immutable parameter size in bytes.
iree_device_size_t qwen_tooling_command_program_set_parameter_byte_length(
    const qwen_tooling_command_program_set_t* program_set);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // EXPERIMENTAL_QWEN_TOOLING_COMMAND_PROGRAM_H_
