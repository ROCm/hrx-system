// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_ABI_V1_H_
#define LOOM_SERVE_SERVICE_TEXT_ABI_V1_H_

#include "iree/vm/buffer.h"
#include "loom/runtime/program/program.h"

#ifdef __cplusplus
extern "C" {
#endif

// Wire operands for one packed submission and its media payload.
typedef struct loom_serve_text_packet_operands_t {
  // Live token count (ABI 2) or prepared stage ordinal (ABI 1).
  int32_t workload;
  // Token records reserved by encode_epoch before projected features.
  int32_t capacity;
} loom_serve_text_packet_operands_t;

typedef struct loom_serve_text_v1_t {
  // Serialized arithmetic-only VM program retained by this legacy model.
  loom_runtime_program_t* program;
  // Translates both live workloads in one call without allocating a table.
  iree_vm_function_t packets;
  // Opaque preparation input retained across cache-sizing invocations.
  iree_vm_buffer_t* catalog;
  // Number of preparation records reported by the compatibility program.
  int64_t catalog_count;
  // Deployment token ceiling used to construct the catalog.
  int32_t token_capacity;
} loom_serve_text_v1_t;

// Creates the compatibility program and its preparation data for a version 1
// service. The caller serializes packet selection with model submission.
iree_status_t loom_serve_text_v1_allocate(loom_runtime_jit_t* jit,
                                          iree_vm_environment_t* environment,
                                          const iree_vm_ref_types_t* types,
                                          int32_t token_capacity,
                                          iree_allocator_t host_allocator,
                                          loom_serve_text_v1_t** out_v1);

void loom_serve_text_v1_free(loom_serve_text_v1_t* v1,
                             iree_allocator_t host_allocator);

// Inputs come from validated, nonempty spans. A missing continuation has zero
// tokens and spans and receives an empty physical reservation.
iree_status_t loom_serve_text_v1_select_packets(
    loom_serve_text_v1_t* v1, int32_t tokens, int32_t spans,
    int32_t next_tokens, int32_t next_spans,
    loom_serve_text_packet_operands_t* first,
    loom_serve_text_packet_operands_t* next);

#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_ABI_V1_H_
