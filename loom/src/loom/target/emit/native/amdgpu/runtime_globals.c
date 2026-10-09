// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "loom/target/emit/native/amdgpu/runtime_globals.h"

iree_host_size_t loom_amdgpu_runtime_global_count(
    loom_amdgpu_runtime_global_flags_t flags) {
  iree_host_size_t count = 0;
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG)) {
    ++count;
  }
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG)) {
    count += 2u;
  }
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG)) {
    ++count;
  }
  return count;
}

void loom_amdgpu_runtime_global_symbols(
    loom_amdgpu_runtime_global_flags_t flags,
    loom_amdgpu_hsaco_data_symbol_t* out_symbols,
    iree_host_size_t* out_symbol_count) {
  iree_host_size_t count = 0;
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG)) {
    out_symbols[count++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_NAME,
        .byte_length = LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG_BYTE_LENGTH,
        .alignment = LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT,
        .flags = LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
    };
  }
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG)) {
    out_symbols[count++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_NAME,
        .byte_length = LOOM_AMDGPU_RUNTIME_GLOBAL_TSAN_CONFIG_BYTE_LENGTH,
        .alignment = LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT,
        .flags = LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
    };
  }
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG)) {
    out_symbols[count++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_NAME,
        .byte_length = LOOM_AMDGPU_RUNTIME_GLOBAL_FEEDBACK_CONFIG_BYTE_LENGTH,
        .alignment = LOOM_AMDGPU_RUNTIME_GLOBAL_CONFIG_ALIGNMENT,
        .flags = LOOM_AMDGPU_HSACO_DATA_SYMBOL_FLAG_WRITABLE,
    };
  }
  if (iree_any_bit_set(flags, LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_CONFIG)) {
    out_symbols[count++] = (loom_amdgpu_hsaco_data_symbol_t){
        .name = LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_LAYOUT_MARKER_NAME,
        .byte_length =
            LOOM_AMDGPU_RUNTIME_GLOBAL_ASAN_LAYOUT_MARKER_BYTE_LENGTH,
        .alignment = LOOM_AMDGPU_ASAN_GLOBAL_LAYOUT_V0_GRANULE_SIZE,
    };
  }
  *out_symbol_count = count;
}
