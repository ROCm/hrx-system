// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef EXPERIMENTAL_LOOM_SERVE_RUNTIME_PREPARATION_H_
#define EXPERIMENTAL_LOOM_SERVE_RUNTIME_PREPARATION_H_

#include "iree/vm/environment.h"
#include "iree/vm/variant.h"
#include "loomc/config.h"

#ifdef __cplusplus
extern "C" {
#endif

// A source-selected checkpoint and preparation policy for one fixed binding.
typedef struct loom_serve_preparation_parameter_t {
  // Fixed-buffer ordinal in the compiled command's parameter reflection.
  uint32_t binding;
  // Owned resolved checkpoint path, independent of the bootstrap VM.
  iree_string_view_t path;
  // Owned preparation-policy path, relative to the bootstrap source directory.
  iree_string_view_t policy;
} loom_serve_preparation_parameter_t;

// Immutable stage declaration produced by a model's cold source program.
typedef struct loom_serve_preparation_stage_t {
  // Exported command root in the model source catalog.
  iree_string_view_t root;
  // Model-defined selection tag. The preparation service does not interpret it.
  int64_t tag;
  // Owned compiler bindings; compilation requires all configuration resolved.
  loomc_config_options_t config;
  // Number of fixed parameter bindings declared by the model.
  iree_host_size_t parameter_count;
  // Owned checkpoint declarations in source registration order.
  const loom_serve_preparation_parameter_t* parameters;
} loom_serve_preparation_stage_t;

// Cold declarations with no device or VM ownership after construction.
typedef struct loom_serve_preparation_t loom_serve_preparation_t;

// Invokes |entry| from |source_path| once with borrowed |arguments|. The source
// imports runner-private "prepare" capabilities:
//   stage(buffer root, i64 tag) -> i32 stage
//   config(i32 stage, buffer key, buffer value)
//   config_i64(i32 stage, buffer key, i64 value)
//   parameter(i32 stage, i32 binding, buffer directory, buffer path,
//             buffer policy)
// Config values use the public loomc textual value syntax; config_i64 formats
// signed decimal values. Parameter paths join directory and path. Policy paths
// are relative to the source directory. Declarations copy their strings and
// retain no references to arguments, source mappings or the VM environment.
// No JIT command compilation, device allocation or parameter IO occurs here.
// On failure all partial declarations are released and *out_preparation is
// NULL.
iree_status_t loom_serve_preparation_create(
    iree_vm_environment_t* environment, iree_string_view_t source_path,
    iree_string_view_t entry, iree_vm_variant_span_t arguments,
    loom_serve_preparation_t** out_preparation,
    iree_allocator_t host_allocator);

void loom_serve_preparation_destroy(loom_serve_preparation_t* preparation);

iree_host_size_t loom_serve_preparation_stage_count(
    const loom_serve_preparation_t* preparation);

// Borrows an immutable declaration; |index| is below stage_count.
const loom_serve_preparation_stage_t* loom_serve_preparation_stage(
    const loom_serve_preparation_t* preparation, iree_host_size_t index);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // EXPERIMENTAL_LOOM_SERVE_RUNTIME_PREPARATION_H_
