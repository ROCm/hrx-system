// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/serve/service/text/bank.h"

iree_status_t loom_serve_text_bank_initialize(iree_hal_buffer_t* const* roots,
                                              const iree_host_size_t* lengths,
                                              iree_allocator_t allocator,
                                              loom_serve_text_bank_t* bank) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_BANK_ROOT_COUNT; ++i) {
    bank->roots[i] = roots[i];
    iree_hal_buffer_retain(roots[i]);
  }
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_BANK_HOST_COUNT; ++i) {
    IREE_RETURN_IF_ERROR(
        iree_vm_buffer_create(lengths[i], 8, allocator, &bank->host[i]));
    IREE_RETURN_IF_ERROR(iree_vm_buffer_map_write(bank->host[i], 0, lengths[i],
                                                  &bank->bytes[i]));
  }
  return iree_ok_status();
}
void loom_serve_text_bank_retire(loom_serve_text_bank_t* bank) {
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_BANK_ROOT_COUNT; ++i) {
    iree_hal_buffer_release(bank->roots[i]);
    bank->roots[i] = NULL;
  }
}
void loom_serve_text_bank_deinitialize(loom_serve_text_bank_t* bank) {
  loom_serve_text_bank_retire(bank);
  for (iree_host_size_t i = 0; i < LOOM_SERVE_TEXT_BANK_HOST_COUNT; ++i) {
    iree_vm_buffer_release(bank->host[i]);
  }
  memset(bank, 0, sizeof(*bank));
}
