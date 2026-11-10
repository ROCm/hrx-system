// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_SERVE_SERVICE_TEXT_BANK_H_
#define LOOM_SERVE_SERVICE_TEXT_BANK_H_

#include "iree/vm/buffer.h"
#include "loom/serve/runtime/execution/group.h"
#include "loom/serve/runtime/execution/retirement.h"

#ifdef __cplusplus
extern "C" {
#endif

// text_prefill_group_v1 numerical resource ABI. KV/GDN and the complete frozen
// page map are supplied separately and shared by both banks. Resource ABI 1
// supplies residual, metadata, origins, state, tokens, outputs. ABI 2 may also
// bind carry, committed, results, draft_cache, draft_origins, next_results in
// the ordinary epoch order. Its source owns both banks' warm/capture stages.
// Reflected workspace is inserted by the runtime.
enum {
  LOOM_SERVE_TEXT_BANK_RESIDUAL = 0,
  LOOM_SERVE_TEXT_BANK_METADATA,
  LOOM_SERVE_TEXT_BANK_TOKENS,
  LOOM_SERVE_TEXT_BANK_OUTPUTS,
  LOOM_SERVE_TEXT_BANK_WORKSPACE,
  LOOM_SERVE_TEXT_BANK_ROOT_COUNT,
};
enum {
  LOOM_SERVE_TEXT_BANK_PLAN = 0,
  LOOM_SERVE_TEXT_BANK_INPUT,
  LOOM_SERVE_TEXT_BANK_HOST_METADATA,
  LOOM_SERVE_TEXT_BANK_HOST_TOKENS,
  LOOM_SERVE_TEXT_BANK_HOST_OUTPUTS,
  LOOM_SERVE_TEXT_BANK_HOST_COUNT,
};

typedef struct loom_serve_text_bank_t {
  iree_hal_buffer_t* roots[LOOM_SERVE_TEXT_BANK_ROOT_COUNT];
  iree_vm_buffer_t* host[LOOM_SERVE_TEXT_BANK_HOST_COUNT];
  iree_byte_span_t bytes[LOOM_SERVE_TEXT_BANK_HOST_COUNT];
} loom_serve_text_bank_t;

// Clones packet geometry and retains supplied roots. Partial initialization is
// returned to the caller for ordinary retirement cleanup.
iree_status_t loom_serve_text_bank_initialize(
    iree_hal_buffer_t* const* roots, const iree_host_size_t* host_lengths,
    iree_allocator_t allocator, loom_serve_text_bank_t* bank);
void loom_serve_text_bank_retire(loom_serve_text_bank_t* bank);
void loom_serve_text_bank_deinitialize(loom_serve_text_bank_t* bank);
#ifdef __cplusplus
}  // extern "C"
#endif
#endif  // LOOM_SERVE_SERVICE_TEXT_BANK_H_
