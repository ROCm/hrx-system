// Copyright 2026 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef LOOM_RUNTIME_PARAMETER_I8_PANEL_H_
#define LOOM_RUNTIME_PARAMETER_I8_PANEL_H_

#include "iree/io/parameter_index.h"
#include "loom/runtime/program/sources.h"

#ifdef __cplusplus
extern "C" {
#endif

// One cold catalog's verified pair; indexed files independently retain storage.
typedef struct loom_runtime_i8_panel_cache_t loom_runtime_i8_panel_cache_t;

// Derives or verifies a content-keyed raw pair, then indexes the selected
// output.
iree_status_t loom_runtime_i8_panel_append(
    const loom_runtime_sources_t* sources, const loom_model_file_t* file,
    iree_allocator_t allocator, loom_runtime_i8_panel_cache_t** cache,
    iree_io_parameter_index_t* index);
void loom_runtime_i8_panel_cache_free(loom_runtime_i8_panel_cache_t* cache);

// Decodes complete GGML blocks through the historical FP16 operand boundary.
iree_status_t loom_runtime_i8_panel_decode(uint32_t type,
                                           iree_const_byte_span_t blocks,
                                           iree_host_size_t count,
                                           float* values);

// Quantizes N rows by K columns and writes the v2 128x128 fragment order.
iree_status_t loom_runtime_i8_panel_pack(const float* values, uint32_t n,
                                         uint32_t k, int8_t* panels,
                                         float* scales);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // LOOM_RUNTIME_PARAMETER_I8_PANEL_H_
